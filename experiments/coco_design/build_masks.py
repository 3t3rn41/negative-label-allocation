# SPDX-License-Identifier: MIT
# MIT License
#
# Copyright (c) 2026 Fixed-query experiment contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Construct and audit the two new five-arm COCO20 draws."""

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
G1 = ROOT / 'inputs/reference_coco'
RULES = json.loads((OUT / 'PHASE0_RULES.json').read_text(encoding='utf-8'))
PAIR_MAP = json.loads((G1 / 'pairs.json').read_text())['settings']
SETTINGS = ('S0', 'N1')
TARGETS = [p['target'] for p in PAIR_MAP['S0']]


def load_training():
    with np.load(G1 / 'raw/data/COCO_TRAIN_GOLD.npz') as data:
        ids, truth, classes = (data[k] for k in ('ids', 'truth', 'classes'))
        assert data['valid'].all()
    with np.load(ROOT / 'inputs/coco_difficulty.npz') as data:
        assert np.array_equal(ids, data['ids'])
        cosine = data['cosine']
    assert truth.shape == cosine.shape == (4008, 20)
    assert len(np.unique(ids)) == len(ids) and np.isin(truth, (0, 1)).all()
    assert np.isfinite(cosine).all()
    for setting in SETTINGS:
        for pair in PAIR_MAP[setting]:
            assert classes[pair['target']] == pair['target_name']
            assert classes[pair['partner']] == pair['partner_name']
    return ids, truth, classes, cosine


def strata(ids, truth, cosine, low):
    density = np.minimum(2, (np.delete(low, TARGETS, axis=1) >= 0).sum(axis=1))
    rows = []
    for k, target in enumerate(TARGETS):
        a, b = (truth[:, PAIR_MAP[s][k]['partner']] == 1 for s in SETTINGS)
        negative = np.flatnonzero(truth[:, target] == 0)
        ordered = negative[np.lexsort((ids[negative], cosine[negative, target]))]
        for difficulty, layer in enumerate(np.array_split(ordered, 10)):
            for count in range(3):
                layer_rows = layer[density[layer] == count]
                visible = layer_rows[low[layer_rows, target] == 0]
                hidden = layer_rows[low[layer_rows, target] == -1]
                background = visible[~a[visible] & ~b[visible]]
                s0 = hidden[a[hidden] & ~b[hidden]]
                n1 = hidden[b[hidden] & ~a[hidden]]
                joint = hidden[a[hidden] & b[hidden]]
                max_exclusive = min(len(s0), len(n1)) // 2
                feasible = {
                    m: max(0, m - max_exclusive)
                    for m in range(min(len(background), len(hidden)) + 1)
                    if max(0, m - max_exclusive) <= min(m // 2, len(joint) // 2)
                }
                rows.append(dict(target=target, difficulty=difficulty, density=count,
                                 B=background, E0=s0, E1=n1, J=joint, hidden=hidden,
                                 feasible=feasible, capacity=max(feasible)))
    return rows


def allocate(rows, dose):
    capacities = np.array([row['capacity'] for row in rows])
    total = int(capacities.sum())
    if total < dose:
        return None
    quota, remainder = np.divmod(dose * capacities, total)
    order = np.argsort(-remainder, kind='stable')
    quota[order[:dose - quota.sum()]] += 1
    assert all(m in row['feasible'] for row, m in zip(rows, quota))
    return quota


def construct(rows, quotas, rng, placement_rng, low):
    masks = {'LOW': low}
    for setting in SETTINGS:
        for arm in ('TRUE', 'PLC', 'TRUE_BG', 'TRUE_PARTNER'):
            masks[f'{setting}_{arm}'] = low.copy()
    coordinates = []
    for row, m in zip(rows, quotas):
        target = row['target']
        h = row['feasible'][m]
        background = rng.permutation(row['B'])[:m]
        joint = rng.permutation(row['J'])[:2 * h]
        placebo = placement_rng.permutation(row['hidden'])[:m]
        for setting, pool in zip(SETTINGS, ('E0', 'E1')):
            exclusive = rng.permutation(row[pool])[:2 * (m - h)]
            partner = np.r_[joint[:h], exclusive[:m-h]]
            extra = np.r_[joint[h:], exclusive[m-h:]]
            assert len(np.unique(np.r_[partner, extra])) == 2 * m
            for arm, remove, add in (
                ('TRUE', background, partner),
                ('PLC', background, placebo),
                ('TRUE_BG', [], partner),
                ('TRUE_PARTNER', background, np.r_[partner, extra]),
            ):
                matrix = masks[f'{setting}_{arm}']
                matrix[remove, target] = -1
                matrix[add, target] = 0
            for name, members in (('B', background), ('P', partner), ('U', placebo), ('Pprime', extra)):
                coordinates.extend((setting, target, row['difficulty'], row['density'], name, int(i))
                                   for i in members)
    return masks, coordinates


def audit(truth, cosine, masks, coordinates, dose, details=False):
    low = masks['LOW']
    known = int((low >= 0).sum())
    failures, comparisons = [], []
    for arm, matrix in masks.items():
        visible = matrix >= 0
        expected = known + sum(dose) * arm.endswith(('TRUE_BG', 'TRUE_PARTNER'))
        assert visible.sum() == expected
        assert np.isin(matrix, (-1, 0, 1)).all()
        assert np.array_equal(matrix[visible], truth[visible])
        assert np.array_equal(matrix == 1, low == 1)
        assert np.array_equal(np.delete(matrix, TARGETS, 1), np.delete(low, TARGETS, 1))
        if not ((matrix == 0).any(axis=0) & (matrix == 1).any(axis=0)).all():
            failures.append(f'{arm}:class_support')
    for setting in SETTINGS:
        for pair, m in zip(PAIR_MAP[setting], dose):
            target, partner = pair['target'], pair['partner']
            groups = {name: np.array([i for s, c, _, _, n, i in coordinates
                                       if s == setting and c == target and n == name])
                      for name in ('B', 'P', 'U', 'Pprime')}
            assert all(len(g) == len(np.unique(g)) == m for g in groups.values())
            direct = truth[groups['P'], partner].mean() - truth[groups['U'], partner].mean()
            if direct < 0.2:
                failures.append(f'{setting}:{target}:direction')
            overlap = len(np.intersect1d(groups['P'], groups['U']))
            sd = cosine[truth[:, target] == 0, target].std()
            for a, b in (('B', 'P'), ('B', 'U'), ('B', 'Pprime'), ('P', 'U')):
                x, y = cosine[groups[a], target], cosine[groups[b], target]
                smd = float((x.mean() - y.mean()) / sd)
                if abs(smd) > 0.10:
                    failures.append(f'{setting}:{target}:{a}-{b}:smd')
                if details:
                    comparisons.append(dict(setting=setting, target=target, comparison=f'{a}-{b}',
                        smd=smd, overlap=overlap, partner_gap=float(direct),
                        P_partner_fraction=float(truth[groups['P'], partner].mean()),
                        U_partner_fraction=float(truth[groups['U'], partner].mean()),
                        quantiles_a=np.quantile(x, [0, .25, .5, .75, 1]).tolist(),
                        quantiles_b=np.quantile(y, [0, .25, .5, .75, 1]).tolist(),
                        ks=float(ks_2samp(x, y).statistic)))
        original = ['LOW', f'{setting}_TRUE', f'{setting}_PLC']
        pairs = list(combinations(original, 2)) + [(f'{setting}_TRUE_BG', f'{setting}_TRUE_PARTNER')]
        for a, b in pairs:
            x, y = [(masks[key] >= 0).sum(axis=1) for key in (a, b)]
            active = int(abs(np.count_nonzero(x) - np.count_nonzero(y)))
            tv = np.abs(np.bincount(x, minlength=21) - np.bincount(y, minlength=21)).sum() / (2 * len(x))
            if active > 20 or tv > 0.01:
                failures.append(f'{a}-{b}:density')
            if details:
                comparisons.append(dict(setting=setting, comparison=f'{a}-{b}',
                                        active_difference=active, density_tv=float(tv)))
    return failures, comparisons


def save_csv(name, records, columns=None):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file) if columns else csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writerow(columns) if columns else writer.writeheader()
        writer.writerows(records)


def main():
    ids, truth, classes, cosine = load_training()
    attempts, layers, coordinate_log, summaries = [], [], [], []
    selected = None
    for dose_index, dose in enumerate(RULES['dose_order']):
        accepted = []
        for seed in RULES['draw_seeds']:
            low = np.where(np.random.default_rng(seed).random(truth.shape) < .10, truth, -1)
            rows = strata(ids, truth, cosine, low)
            by_target = [[row for row in rows if row['target'] == target] for target in TARGETS]
            parts = [allocate(r, m) for r, m in zip(by_target, dose)]
            for group, quota in zip(by_target, parts):
                for j, row in enumerate(group):
                    layers.append(dict(draw=seed, dose_index=dose_index, target=row['target'],
                        difficulty=row['difficulty'], density=row['density'],
                        capacity=row['capacity'], quota=None if quota is None else int(quota[j]),
                        joint_per_pack=None if quota is None else int(row['feasible'][quota[j]]),
                        **{f'n_{name}': len(row[name]) for name in ('B', 'E0', 'E1', 'J', 'hidden')}))
            if any(part is None for part in parts):
                attempts.append(dict(draw=seed, dose_index=dose_index, attempt=0, failures=['capacity']))
                continue
            quotas = np.concatenate(parts)
            streams = np.random.SeedSequence([seed, dose_index]).spawn(2)
            rng, placement_rng = map(np.random.default_rng, streams)
            for attempt in range(1, RULES['attempts_per_draw_and_dose'] + 1):
                masks, coords = construct(rows, quotas, rng, placement_rng, low)
                failures, _ = audit(truth, cosine, masks, coords, dose)
                attempts.append(dict(draw=seed, dose_index=dose_index, attempt=attempt, failures=failures))
                if not failures:
                    _, metrics = audit(truth, cosine, masks, coords, dose, details=True)
                    accepted.append((seed, masks, coords, attempt, metrics))
                    break
        if len(accepted) == len(RULES['draw_seeds']):
            selected = dose
            for seed, masks, coords, attempt, metrics in accepted:
                known = int((masks['LOW'] >= 0).sum())
                summaries.append(dict(draw=seed, attempt=attempt, known=known,
                                      learning_rate=.002 * 8976 / known, metrics=metrics))
                np.savez_compressed(OUT / f'MASK_{seed}.npz', ids=ids, classes=classes, **masks)
                coordinate_log.extend((seed, s, c, d, n, k, int(ids[i])) for s, c, d, n, k, i in coords)
            break
    save_csv('MASK_MATCH_QA.csv', layers)
    if coordinate_log:
        save_csv('MASK_COORDINATES.csv', coordinate_log,
                 ['draw', 'setting', 'target', 'difficulty', 'density', 'role', 'coco_id'])
    result = dict(status='PASS' if selected else 'HOLD_MASK', dose=selected, draws=summaries, attempts=attempts)
    (OUT / 'MASK_QA.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'], selected, [(x['draw'], x['attempt'], x['known']) for x in summaries])


if __name__ == '__main__':
    main()
