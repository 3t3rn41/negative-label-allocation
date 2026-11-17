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

"""Evaluate the complete sealed S0 matrix against the frozen A1 confirmation truth."""

import csv
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, t

HERE = Path(__file__).resolve().parent
PHASE0 = HERE.parent / 'coco_design'
sys.path.insert(0, str(PHASE0))
from build_masks import PAIR_MAP
from historical_metrics import Metric, cell_labels

ARMS = ('LOW', 'S0_TRUE', 'S0_PLC', 'S0_TRUE_BG', 'S0_TRUE_PARTNER')
CELLS = (('10', '00'), ('11', '00'), ('10', '01'), ('11', '01'))
METRICS = ('conditional', 'background', 'natural', 'r_low', 'r_high', 'slope', 'map', 'map_targets', 'map_other')
CONTRASTS = (
    ('H1', 'conditional', 1, 2, 1.0, 0.0), ('H2', 'background', 0, 1, .30, 0.0),
    ('H3', 'background', 3, 1, .30, 0.0), ('H4', 'background', 3, 4, .30, 0.0),
    ('H5', 'conditional', 3, 1, None, -.50),
    ('EVAL_LOW', 'r_low', 1, 2, None, None), ('EVAL_HIGH', 'r_high', 1, 2, None, None),
    ('EVAL_SLOPE', 'slope', 1, 2, None, None), ('TRUE_MINUS_LOW_MAP', 'map', 1, 0, None, None),
    ('TRUE_MINUS_LOW_MAP_TARGETS', 'map_targets', 1, 0, None, None),
    ('TRUE_MINUS_LOW_MAP_OTHER', 'map_other', 1, 0, None, None),
    ('TRUE_BG_MINUS_LOW_BACKGROUND', 'background', 3, 0, None, None),
    ('TRUE_PARTNER_MINUS_TRUE_BACKGROUND', 'background', 4, 1, None, None),
)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def direct_auc(y, score):
    n1, n0 = y.sum(), (1-y).sum()
    ranks = rankdata(score, method='average')
    return (ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0)


def counts_ratio(top, bottom):
    return np.divide(top, bottom, out=np.full_like(top, np.nan, dtype=np.float64), where=bottom > 0)


def init_worker(gold, weights, pstars, ratios):
    global GOLD, WEIGHTS, PSTARS, RATIOS
    GOLD = gold
    WEIGHTS = np.load(weights, mmap_mode='r')
    PSTARS, RATIOS = pstars, ratios


def metrics(scores):
    pairs = PAIR_MAP['S0']
    targets = [row['target'] for row in pairs]
    b = len(WEIGHTS)
    output = np.zeros((b, len(METRICS)), np.float64)
    points = []
    for pair, pstar in zip(pairs, PSTARS):
        target, partner = pair['target'], pair['partner']
        c, q, s = GOLD[:, target], GOLD[:, partner], scores[:, target]
        models = [Metric(s, cell_labels(c, q, a, z)) for a, z in CELLS]
        for first in range(0, b, 16):
            sl = slice(first, min(first+16, b))
            w = WEIGHTS[sl]
            cell = np.array([m.compute_auc(w) for m in models])
            p = counts_ratio(w @ (c*q), w @ c)
            r = counts_ratio(w @ ((1-c)*q), w @ (1-c))
            bg = (1-p)*cell[0] + p*cell[1]
            pn = (1-p)*cell[2] + p*cell[3]
            standardized_bg = (1-pstar)*cell[0] + pstar*cell[1]
            standardized_pn = (1-pstar)*cell[2] + pstar*cell[3]
            low, high = [(1-mix)*standardized_bg + mix*standardized_pn for mix in RATIOS]
            output[sl, :6] += np.stack((cell[2], bg, (1-r)*bg+r*pn, low, high,
                                       (high-low)/(RATIOS[1]-RATIOS[0])), axis=1)
            if first == 0:
                independent_cells = [direct_auc(m.y.astype(np.int8), s[m.order]) for m in models]
                assert np.allclose(cell[:, 0], independent_cells, rtol=0, atol=1e-12)
                natural = ((1-r)*bg+r*pn)[0]
                assert abs(natural-direct_auc(c, s)) < 1e-12
                points.append(dict(target=target, partner=partner, p=float(p[0]), r=float(r[0]),
                    pstar=pstar, natural_auc=float(natural), standardized_bg=float(standardized_bg[0]),
                    standardized_partner=float(standardized_pn[0]),
                    **{f'auc_{a}_{z}': value for (a, z), value in zip(CELLS, cell[:, 0])}))
    output[:, :6] /= len(pairs)
    ap_points = []
    for i, (s, y) in enumerate(zip(scores.T, GOLD.T)):
        metric = Metric(s, y)
        for first in range(0, b, 16):
            sl = slice(first, min(first+16, b))
            aps = metric.compute_ap(WEIGHTS[sl])
            output[sl, 6] += aps / GOLD.shape[1]
            output[sl, 7 if i in targets else 8] += aps / GOLD.shape[1]
            if first == 0:
                ap_points.append(float(aps[0]))
    assert np.allclose(output[:, 6], output[:, 7] + output[:, 8], atol=1e-13, equal_nan=True)
    return output, points, ap_points


def interval(x):
    finite = x[np.isfinite(x)]
    return dict(valid=len(finite), ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                support_pass=len(finite) >= 4500)


def paired_inference(arr, seeds, rng):
    raw = arr[0]
    n = len(seeds)
    assert raw.shape == (n,) and np.isfinite(raw).all()
    mean, sd = float(raw.mean()), float(raw.std(ddof=1))
    se = sd / np.sqrt(n)
    one, two = (float(t.ppf(p, n-1))*se for p in (.95, .975))
    resample = rng.integers(n, size=(len(arr)-1, n))
    combined = arr[np.arange(1, len(arr))[:, None], resample].mean(1)
    return dict(mean_pp=mean, sd_pp=sd, median_pp=float(np.median(raw)),
                one_sided_lower=float(mean-one), one_sided_upper=float(mean+one),
                ci95=[float(mean-two), float(mean+two)],
                leave_one_out_pp=((raw.sum()-raw)/(n-1)).tolist(), initialization_points=dict(zip(seeds, raw.tolist())),
                image_bootstrap=interval(arr[1:].mean(1)), image_and_init_bootstrap=interval(combined))


def analyze(lock, receipt, scores_path, out, bootstrap=5000, workers=4):
    assert receipt['status'] == 'COMPLETE' and receipt['truth_loaded'] is False
    with np.load(scores_path) as data:
        scores, ids, classes = data['scores'], data['ids'], data['classes']
        key = list(zip(data['seeds'].tolist(), data['draws'].tolist(), data['arms'].tolist()))
    seeds, draws = lock['seeds'], lock['draws']
    expected = list(itertools.product(seeds, draws, ARMS))
    assert len(key) == len(set(key)) == len(expected) == receipt['endpoints'] == lock['endpoints']
    assert set(key) == set(expected) and ids.tolist() == lock['confirmation_ids']
    assert scores.shape == (len(key), len(ids), len(classes)) and scores.dtype == np.float32 and np.isfinite(scores).all()
    with np.load(PHASE0 / 'A1/CONFIRM_GOLD.npz') as data:
        assert np.array_equal(data['ids'], ids) and np.array_equal(data['classes'], classes)
        gold = data['truth']
    history = read(PHASE0 / 'HISTORICAL_METRICS_AND_POWER.json')
    pstars = history['history']['settings']['S0']['pstar']
    ratios = lock['evaluation_ratios']
    rng = np.random.default_rng(lock['bootstrap_seed'])
    out.mkdir(exist_ok=True)
    wfile = out / 'SHARED_IMAGE_BOOTSTRAP.npy'
    w = np.lib.format.open_memmap(wfile, mode='w+', dtype=np.uint16, shape=(bootstrap+1, len(ids)))
    w[0] = 1
    for b in range(1, bootstrap+1):
        w[b] = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
    w.flush()
    stats = np.lib.format.open_memmap(out / 'BOOTSTRAP_ENDPOINT_TERMS.npy', mode='w+', dtype=np.float64,
              shape=(bootstrap+1, len(seeds), len(draws), len(ARMS), len(METRICS)))
    points, aps = [], []
    order = {k: j for j, k in enumerate(key)}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(gold, wfile, pstars, ratios)) as pool:
        for count, ((seed, draw, arm), (result, cell, ap)) in enumerate(
                zip(expected, pool.map(metrics, (scores[order[k]] for k in expected)), strict=True), start=1):
            stats[:, seeds.index(seed), draws.index(draw), ARMS.index(arm)] = result
            points.extend(dict(seed=seed, draw=draw, arm=arm, **row) for row in cell)
            aps.extend(dict(seed=seed, draw=draw, arm=arm, category=c, ap=x) for c, x in zip(classes, ap))
            if count % 10 == 0:
                stats.flush()
                print('EVALUATED', count, '/', len(expected), flush=True)
    stats.flush()
    comparisons = {}
    for name, metric, left, right, point, lower in CONTRASTS:
        slice_ = stats[..., METRICS.index(metric)]
        result = paired_inference(100*(slice_[..., left]-slice_[..., right]).mean(axis=2), seeds, rng)
        if lower is not None:
            result['pass'] = bool((point is None or result['mean_pp'] >= point) and result['one_sided_lower'] > lower)
        comparisons[name] = result
    joint = all(comparisons[f'H{i}']['pass'] for i in range(1, 6))
    mix = bool(comparisons['EVAL_LOW']['one_sided_upper'] < 0 and comparisons['EVAL_HIGH']['one_sided_lower'] > 0)
    result = dict(status='S0_ANALYSIS_COMPLETE', primary_mechanism_pass=joint, mixture_sign_pass=mix,
                  n=len(seeds), bootstrap=bootstrap, comparisons=comparisons)
    write_csv(out / 'AUC_CELLS.csv', points)
    write_csv(out / 'PER_CLASS_AP.csv', aps)
    curve = []
    for seed in seeds:
        point = [row for row in points if row['seed'] == seed]
        averages = {
            arm: np.array([[x['standardized_bg'], x['standardized_partner']]
                           for x in point if x['arm'] == arm]).mean(0)
            for arm in ('S0_TRUE', 'S0_PLC')
        }
        bg, partner = 100*(averages['S0_TRUE'] - averages['S0_PLC'])
        for ratio in sorted(set(np.linspace(0, 1, 101).tolist() + ratios)):
            curve.append(dict(seed=seed, partner_ratio=ratio, true_minus_plc_pp=(1-ratio)*bg+ratio*partner))
    write_csv(out / 'MIXTURE_CURVE.csv', curve)
    write_csv(out / 'PRIMARY_CONTRASTS.csv', [dict(contrast=k, mean_pp=r['mean_pp'], sd_pp=r['sd_pp'],
        lower95=r['one_sided_lower'], upper95=r['one_sided_upper'], passed=r.get('pass'))
        for k, r in comparisons.items()])
    temporary = out / 'S0_RESULT.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(out / 'S0_RESULT.json')
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(read(HERE / 'PROTOCOL_LOCK.json'), read(HERE / 'S0_SCORE_RECEIPT.json'),
                             HERE / 'S0_SCORES.npz', HERE / 'analysis'), indent=2))
