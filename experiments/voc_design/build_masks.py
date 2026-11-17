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

"""Construct the two VOC draws using frozen fit-only cosine difficulty."""

from collections import Counter
from itertools import combinations
from pathlib import Path
import json

import numpy as np


HERE = Path(__file__).resolve().parent
ARMS = ('P1_TRUE', 'P2_TRUE', 'PLC')


def largest_remainder(rows, dose):
    cap = np.array([min(map(len, r[2])) for r in rows])
    total = cap.sum()
    if total < dose:
        return None
    quota, residual = np.divmod(dose * cap, total)
    quota[np.argsort(-residual, kind='stable')[:dose - quota.sum()]] += 1
    assert np.all(quota <= cap)
    return quota


def stratify(ids, y, scores, low, pairs):
    density = np.minimum(2, (np.delete(low, [t for t, _, _ in pairs], axis=1) >= 0).sum(axis=1))
    for t, p1, p2 in pairs:
        neg = np.flatnonzero(y[:, t] == -1)
        ordered = neg[np.lexsort((ids[neg], scores[neg, t]))]
        rows = []
        for d, group in enumerate(np.array_split(ordered, 10)):
            for k in range(3):
                ix = group[density[group] == k]
                hidden = ix[low[ix, t] == -1]
                known = ix[low[ix, t] == 0]
                background = known[(y[known, p1] == -1) & (y[known, p2] == -1)]
                rows.append((d, k, (background, hidden[y[hidden, p1] == 1],
                                    hidden[y[hidden, p2] == 1], hidden)))
        yield rows


def construct(low, pairs, strata, quotas, main_rng, plc_rng):
    matrices = {arm: low.copy() for arm in ARMS}
    groups = {}
    for (t, _, _), rows, quota in zip(pairs, strata, quotas):
        samples = []
        for (difficulty, density, pools), n in zip(rows, quota):
            rngs = (main_rng, main_rng, main_rng, plc_rng)
            selected = [rng.permutation(pool)[:n] for rng, pool in zip(rngs, pools)]
            remove, *adds = selected
            for matrix, add in zip(matrices.values(), adds):
                matrix[remove, t] = -1
                matrix[add, t] = 0
            samples.append((difficulty, density, selected))
        groups[t] = samples
    return matrices, groups


def audit(y, cosine, low, pairs, doses, matrices, groups):
    failures = []
    metrics = []
    targets = [t for t, _, _ in pairs]
    for arm, matrix in matrices.items():
        observed = matrix >= 0
        assert observed.sum() == (low >= 0).sum() and (matrix == 0).sum() == (low == 0).sum()
        assert np.array_equal(matrix == 1, low == 1)
        assert np.array_equal(np.delete(matrix, targets, 1), np.delete(low, targets, 1))
        assert np.array_equal(matrix[observed], (y[observed] == 1).astype(np.int8))
        if not ((matrix == 0).any(axis=0) & (matrix == 1).any(axis=0)).all():
            failures.append(f'{arm}:class_support')
    for (t, p1, p2), dose in zip(pairs, doses):
        samples = groups[t]
        sets = [np.concatenate([s[2][j] for s in samples]) for j in range(4)]
        assert all(len(g) == len(np.unique(g)) == dose for g in sets)
        scale = cosine[y[:, t] == -1, t].std()
        matches = [[float((y[g, p] == 1).mean()) for g in sets[1:]] for p in (p1, p2)]
        differences = [matches[i][i] - max(matches[i][j] for j in range(3) if i != j)
                       for i in range(2)]
        if min(differences) < .20 - 1e-12:
            failures.append(f'{t}:partner_direction')
        balance = {}
        overlap = {}
        for i, j in combinations(range(4), 2):
            key = f'{i}/{j}'
            smd = float((cosine[sets[i], t].mean() - cosine[sets[j], t].mean()) / scale)
            balance[key] = smd
            overlap[key] = int(np.intersect1d(sets[i], sets[j]).size)
            if abs(smd) > .10:
                failures.append(f'{t}:smd:{key}')
        metrics.append(dict(target=t, dose=dose, partner_fractions=matches,
                            minimum_directional_gaps=differences, smd=balance, shared_coordinates=overlap))
    density = []
    for a, b in combinations(ARMS, 2):
        x, z = [(matrices[arm] >= 0).sum(axis=1) for arm in (a, b)]
        active = int(abs(np.count_nonzero(x) - np.count_nonzero(z)))
        tv = np.abs(np.bincount(x, minlength=21) - np.bincount(z, minlength=21)).sum() / (2 * len(x))
        density.append(dict(arms=[a, b], active_difference=active, density_tv=float(tv)))
        if active > 20 or tv > .01:
            failures.append(f'{a}/{b}:density')
    return failures, metrics, density


def run():
    role = json.loads((HERE / 'DATA_ROLE_LOCK.json').read_text())
    rules = json.loads((HERE / 'RULES.json').read_text())
    assert role['status'] == 'PASS_DATA_QUALIFICATION'
    with np.load(HERE / 'FIT_GOLD.npz') as f, np.load(HERE / 'FIT_COSINE.npz') as s:
        ids, classes, y, cosine = f['ids'], f['classes'].tolist(), f['truth'], s['scores']
        assert np.array_equal(ids, s['ids']) and classes == s['classes'].tolist() == role['classes']
    assert ids.tolist() == role['role_ids']['fit'] and y.shape == cosine.shape == (4008, 20)
    assert np.isfinite(cosine).all() and np.isin(y, (-1, 0, 1)).all()
    pairs = [[classes.index(n) for n in (t, a, b)] for (t, a), (u, b) in
             zip(role['pairs']['P1'], role['pairs']['P2'], strict=True) if t == u]
    assert len(pairs) == 3
    tries = []
    capacities = []
    chosen = None
    for dose_index, doses in enumerate(rules['dose_order']):
        accepted = []
        for draw in rules['draws']:
            sample = np.random.default_rng(draw).random(y.shape) < .10
            low = np.where(sample & (y != 0), y == 1, -1).astype(np.int8)
            strata = list(stratify(ids, y, cosine, low, pairs))
            quotas = [largest_remainder(rows, dose) for rows, dose in zip(strata, doses)]
            for (t, _, _), rows, quota in zip(pairs, strata, quotas):
                for i, (d, k, pools) in enumerate(rows):
                    capacities.append(dict(draw=draw, dose_index=dose_index, target=classes[t],
                        difficulty=d, other_labels=k, counts=list(map(len, pools)),
                        common_capacity=min(map(len, pools)), quota=None if quota is None else int(quota[i])))
            if any(q is None for q in quotas):
                tries.append(dict(draw=draw, dose_index=dose_index, attempts=0, failures={'capacity': 1}))
                continue
            rngs = [np.random.default_rng(s) for s in np.random.SeedSequence([draw, dose_index]).spawn(2)]
            failed = Counter()
            for attempt in range(1, 65):
                matrices, groups = construct(low, pairs, strata, quotas, *rngs)
                errors, metrics, density = audit(y, cosine, low, pairs, doses, matrices, groups)
                if not errors:
                    accepted.append((draw, low, matrices, groups, metrics, density, attempt))
                    break
                failed.update(errors)
            tries.append(dict(draw=draw, dose_index=dose_index, attempts=attempt, failures=dict(failed)))
        if len(accepted) == len(rules['draws']):
            chosen = doses
            break
    summaries = []
    if chosen:
        for draw, low, matrices, groups, metrics, density, attempt in accepted:
            np.savez_compressed(HERE / f'MASK_{draw}.npz', ids=ids.astype(np.int64), classes=classes,
                                LOW=low, **matrices)
            coordinates = {classes[t]: [
                dict(difficulty=d, other_labels=k, ids=[ids[g].tolist() for g in gs])
                for d, k, gs in rows] for t, rows in groups.items()}
            summaries.append(dict(draw=draw, attempt=attempt, known=int((low >= 0).sum()),
                                  comparisons=metrics, density=density, coordinates=coordinates))
    result = dict(status='PASS' if chosen else 'INCOMPLETE_MASK_QUALIFICATION',
                  dose=chosen, draws=summaries, attempts=tries, layers=capacities,
                  coordinate_order=['remove', *ARMS], difficult_visible_count=0)
    (HERE / 'MASK_QA.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'], chosen, [(s['draw'], s['attempt']) for s in summaries], flush=True)


if __name__ == '__main__':
    run()
