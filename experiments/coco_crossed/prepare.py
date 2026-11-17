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

"""Freeze v9.1 candidate roles and compute predictions using sealed v9 scores."""

import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.stats import rankdata, t

HERE = Path(__file__).resolve().parent
P0 = HERE.parent / 'coco_design'
A1 = P0 / 'A1'
S0 = HERE.parent / 'coco_fixed_partner'
sys.path.insert(0, str(P0))
from audit_dev_pool import ARCHIVE, V3
from build_masks import PAIR_MAP
from qualify_evaluation import support


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, payload):
    path = HERE / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def historical_assets():
    split = read(V3 / 'manifests/COCO_split.json')
    with np.load(V3 / 'blind/COCO_dev_gold.npz') as data:
        ids, truth = data['ids'], data['Y']
    assert np.array_equal(ids, split['dev'])
    bindings = read(ARCHIVE / 'category_binding.json')
    lookup = {row['category_id']: j for j, row in enumerate(bindings)}
    columns = [split['category_ids'].index(row['category_id']) for row in bindings]
    with ZipFile(P0 / 'annotations_trainval2014.zip') as archive:
        with archive.open('annotations/instances_train2014.json') as file:
            coco = json.load(file)
    selected = set(ids.tolist())
    metadata = {row['id']: (row['width'], row['height']) for row in coco['images'] if row['id'] in selected}
    positions = {int(i): j for j, i in enumerate(ids)}
    official = np.zeros((len(ids), len(lookup)), dtype=np.int8)
    for annotation in coco['annotations']:
        col = lookup.get(annotation['category_id'])
        row = positions.get(annotation['image_id'])
        if col is not None and row is not None:
            official[row, col] = 1
    assert len(metadata) == len(ids) == 16557 and np.array_equal(official, truth[:, columns])
    return ids, official, [row['class_name'] for row in bindings], metadata


def existing_ids():
    roles = {}
    with (A1 / 'PILOT_PIXEL_MANIFEST.csv').open(encoding='utf-8', newline='') as file:
        for row in csv.DictReader(file):
            roles.setdefault(row['role'], []).append(int(row['id']))
    with np.load(S0 / 'S0_SCORES.npz') as data:
        roles['S0_SCORED'] = data['ids'].tolist()
    for key, file in (
        ('PRIOR_DEV_SCORED', S0.parents[1] / 'inputs/prior_evaluation/dev_pool_scores.npz'),
        ('PRIOR_REFERENCE_SCORED', S0.parents[1] / 'inputs/prior_evaluation/reference_scores.npz'),
    ):
        with np.load(file) as data:
            roles[key] = data['ids'].tolist()
    return roles


def prepare():
    rules = read(HERE / 'RULES.json')
    role_ids = existing_ids()
    excluded = set().union(*map(set, role_ids.values()))
    with (A1 / 'DEV_DISPOSITION.csv').open(encoding='utf-8', newline='') as file:
        source = {int(row['id']) for row in csv.DictReader(file) if row['eligible'] == '1'}
    assert len(source) == 15506
    remaining = sorted(source - excluded)
    assert len(remaining) >= 2*rules['pool_size']
    order = np.random.default_rng(rules['pool_seed']).permutation(remaining).tolist()
    n = rules['pool_size']
    pools = dict(A=order[:n], B=order[n:2*n])
    backups = dict(A=order[2*n::2], B=order[2*n+1::2])
    plan = dict(seed=rules['pool_seed'], initial=pools, reserve=backups)
    if (HERE / 'CANDIDATE_PLAN.json').exists():
        assert read(HERE / 'CANDIDATE_PLAN.json') == plan
    else:
        save('CANDIDATE_PLAN.json', plan)
    ids, truth, classes, sizes = historical_assets()
    lookup = {int(i): j for j, i in enumerate(ids)}
    history = read(P0 / 'HISTORICAL_METRICS_AND_POWER.json')['history']['settings']
    summary = {}
    for pool in ('A', 'B'):
        reserve = backups[pool]
        selected = pools[pool]
        if pool == 'B':
            sequence = selected + reserve
            initial = support(truth[[lookup[i] for i in selected]], history)
            offset = 0
            if not all(row['group_pass'] for row in initial):
                save('B_RESERVE_RULE.json', dict(rule='earliest qualifying contiguous 5000-ID window in the fixed B sequence',
                     recorded_before_reserve_check=now(), initial_counts=initial))
                for j in range(1, len(reserve)+1):
                    window = sequence[j:j+n]
                    if all(x['group_pass'] for x in support(truth[[lookup[i] for i in window]], history)):
                        selected, reserve = window, sequence[:j] + sequence[j+n:]
                        offset = j
                        break
                save('B_RESERVE_DECISION.json', dict(offset=offset, selected_size=len(selected),
                     initial_pass=False, reserve_window_pass=offset>0))
        target = np.array(selected, dtype=np.int64)
        matrix = truth[[lookup[i] for i in target]]
        counts = support(matrix, history)
        if pool == 'A':
            counts = [row for row in counts if row['setting'] == 'S0']
        group_pass = all(row['group_pass'] for row in counts)
        class_pass = bool(((matrix.sum(0) > 0) & (matrix.sum(0) < n)).all())
        pos, neg = rules['ess_minima']
        ess = all(row['positive_ess'] >= pos and all(row['negative_ess'][r] >= neg for r in rules['ratios'])
                  for row in counts)
        summary[pool] = dict(group_pass=group_pass, class_pass=class_pass, ess_pass=ess, pairs=counts,
                             class_positive=matrix.sum(0).tolist(), n=n)
        with (HERE / f'{pool}_PIXEL_CANDIDATES.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['role', 'id', 'width', 'height', 'member'])
            for role, group in [('initial', target), ('reserve', reserve)]:
                writer.writerows((role, i, *sizes[i], f'train2017/{i:012d}.jpg') for i in group)
        np.savez_compressed(HERE / f'{pool}_PREP_GOLD.npz', ids=target, classes=np.array(classes), truth=matrix,
                            backup_ids=reserve, backup_truth=truth[[lookup[i] for i in reserve]])
    save('LABEL_SUPPORT_PREP.json', dict(status='STATIC_CHECK_COMPLETE', candidate_total=len(remaining),
         a1_eligible=len(source), exclusions={key: len(source.intersection(v)) for key, v in role_ids.items()},
         roles=role_ids, pools=summary, full_dev_label_projection_matches=True))
    print('POOLS', len(remaining), [(k, v['group_pass'], v['class_pass'], v['ess_pass']) for k, v in summary.items()])


def freeze():
    rules = read(HERE / 'RULES.json')
    plan = read(HERE / 'CANDIDATE_PLAN.json')
    audit = read(HERE / 'LABEL_SUPPORT_PREP.json')
    pixels = read(HERE / 'DECODE_QA.json')
    history = read(P0 / 'HISTORICAL_METRICS_AND_POWER.json')['history']['settings']
    locked = dict(status='DATA_ROLE_LOCKED', time=now(), seed=plan['seed'], prior_roles=audit['roles'],
                  prior_use='recorded project use audit A1; historical label support', pools={})
    for pool in ('A', 'B'):
        asset = pixels['pools'][pool]
        ids = asset['ids']
        assert len(ids) == len(set(ids)) and set(ids) <= set(plan['initial'][pool] + plan['reserve'][pool])
        with np.load(HERE / f'{pool}_PREP_GOLD.npz') as data:
            full = np.concatenate((data['ids'], data['backup_ids']))
            labels = np.concatenate((data['truth'], data['backup_truth']))
            positions = {int(i): j for j, i in enumerate(full)}
            matrix = labels[[positions[i] for i in ids]]
            classes = data['classes']
        pairs = support(matrix, history)
        if pool == 'A':
            pairs = [x for x in pairs if x['setting']=='S0']
        assert not set(ids) & set().union(*map(set, audit['roles'].values()))
        pos, neg = rules['ess_minima']
        ess = all(row['positive_ess'] >= pos and all(row['negative_ess'][r] >= neg for r in rules['ratios'])
                  for row in pairs)
        size = len(ids)
        pass_pool = size == rules['pool_size'] and all(x['group_pass'] for x in pairs) and bool(
            ((matrix.sum(0) > 0) & (matrix.sum(0) < size)).all())
        np.savez_compressed(HERE / f'{pool}_GOLD.npz', ids=ids, classes=classes, truth=matrix)
        with (HERE / f'{pool}_PIXEL_CANDIDATES.csv').open(encoding='utf-8', newline='') as file:
            reserve = [int(row['id']) for row in csv.DictReader(file) if row['role']=='reserve']
        locked['pools'][pool] = dict(ids=ids, support=pairs, qualified=pass_pool and (ess or pool == 'B'),
                                     fixed_ratios_supported=ess, pixel_receipt=asset,
                                     reserve=[i for i in reserve if i not in set(ids)])
    assert not set(locked['pools']['A']['ids']) & set(locked['pools']['B']['ids'])
    save('DATA_ROLE_LOCK.json', locked)
    print('DATA_ROLE_LOCKED', [(k, v['qualified']) for k, v in locked['pools'].items()])


def four_cells(score, truth, pair):
    y, q = (truth[:, pair[x]] == 1 for x in ('target', 'partner'))
    u, v, w, z = y & ~q, y & q, ~y & ~q, ~y & q
    def auc(pos, neg):
        a, b = int(pos.sum()), int(neg.sum())
        selected = pos | neg
        ranks = rankdata(score[selected])
        return (ranks[pos[selected]].sum() - a*(a+1)/2) / (a*b)
    return np.array([auc(u,w), auc(u,z), auc(v,w), auc(v,z)])


def composition(y, pair):
    target, partner = (y[:, pair[key]] for key in ('target', 'partner'))
    return float(partner[target==1].mean()), float(partner[target==0].mean())


def weight(p, r):
    return np.array([(1-p)*(1-r), (1-p)*r, p*(1-r), p*r])


def interval(vals):
    x = np.asarray(vals)
    m = x.mean(axis=0)
    h = t.ppf(.975, len(x)-1) * x.std(axis=0, ddof=1)/np.sqrt(len(x))
    return dict(mean=m.tolist(), ci95=np.stack([m-h,m+h], axis=-1).tolist())


def predict():
    lock = read(HERE / 'DATA_ROLE_LOCK.json')
    a = lock['pools']['A']
    assert a['qualified'] and not (HERE / 'A_SCORES.npz').exists()
    with np.load(A1 / 'CONFIRM_GOLD.npz') as data:
        old_ids, old_truth, classes = data['ids'], data['truth'], data['classes']
    with np.load(S0 / 'S0_SCORES.npz') as data:
        assert np.array_equal(data['ids'], old_ids) and np.array_equal(data['classes'], classes)
        keys = list(zip(data['draws'].tolist(), data['seeds'].tolist(), data['arms'].tolist()))
        arrays = dict(zip(keys, data['scores']))
    pairs = PAIR_MAP['S0']
    settings = read(P0 / 'HISTORICAL_METRICS_AND_POWER.json')['history']['settings']
    original_lock = read(S0 / 'PROTOCOL_LOCK.json')
    source_p_r = [composition(old_truth, pair) for pair in pairs]
    weights = []
    for pair, row, old in zip(pairs, a['support'], source_p_r):
        n10,n01,n11,n00 = row['counts_10_01_11_00']
        p, r = n11/(n10+n11), n01/(n00+n01)
        pstar = row['pstar']
        assert pstar == settings['S0']['pstar'][pair['pair_id']]
        weights.append([weight(p,r), weight(pstar,.05), weight(pstar,.50), weight(*old)])
    result = {}
    endpoints = []
    for label, control in [('TRUE_MINUS_PLC','S0_PLC'), ('TRUE_MINUS_LOW','LOW')]:
        cells = np.empty((12,2,3,4), np.float64)
        for si, seed in enumerate(original_lock['seeds']):
            for di, draw in enumerate(original_lock['draws']):
                for ti, pair in enumerate(pairs):
                    target = pair['target']
                    first = four_cells(arrays[draw,seed,'S0_TRUE'][:,target],old_truth,pair)
                    second = four_cells(arrays[draw,seed,control][:,target],old_truth,pair)
                    cells[si,di,ti] = 100*(first-second)
        predictions = np.einsum('sdtc,txc->sdtx', cells, np.asarray(weights))
        pooled = predictions.mean(axis=(1,2))
        result[label] = dict(weight_order=['natural','r05','r50','old_natural_baseline'],
            aggregate=interval(pooled), target_intervals=interval(predictions.mean(axis=1)),
            source_cell_intervals=interval(cells.mean(axis=1)),
            by_seed_draw_target_cells_pp=cells.tolist(), by_seed_draw_target_pp=predictions.tolist())
    for draw in original_lock['draws']:
        for seed in original_lock['seeds']:
            endpoints.extend(dict(draw=draw, seed=seed, arm=x) for x in ('S0_TRUE','S0_PLC','LOW'))
    r = result['TRUE_MINUS_PLC']['aggregate']['mean']
    save('A_PREDICTIONS.json', dict(status='PREDICTIONS_LOCKED', locked_at=now(), ids=a['ids'],
         target_names=[p['target_name'] for p in pairs], prediction_direction_pass=r[1]<0<r[2],
         four_cells=['10x00','10x01','11x00','11x01'], weights=np.asarray(weights).tolist(),
         contrasts=result, endpoint_manifest=endpoints, source='sealed v9 S0 confirmation scores'))
    print('PREDICTIONS_LOCKED', r)


if __name__ == '__main__':
    {'prepare':prepare, 'freeze':freeze, 'predict':predict}[sys.argv[1]]()
