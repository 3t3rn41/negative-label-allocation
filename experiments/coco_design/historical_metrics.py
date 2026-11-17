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

"""Reproduce the development effect sizes and verify the scoring arithmetic."""

import json
import re
import sys

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, norm, rankdata, t

from build_masks import G1, OUT, PAIR_MAP, SETTINGS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'metrics'))
from pr_metrics import Metric, cell_labels


def compare_cells(scores, target, partner, weights=None):
    comparisons = (('10', '00'), ('11', '00'), ('10', '01'), ('11', '01'))
    return np.array([float(np.asarray(Metric(scores, cell_labels(target, partner, a, b))
                                     .compute_auc(weights)).item())
                     for a, b in comparisons])


def components(cells, p):
    return (1-p) * cells[0] + p * cells[1], (1-p) * cells[2] + p * cells[3]


def rank_auc(scores, pos, neg):
    npos, nneg = int(pos.sum()), int(neg.sum())
    if not (npos and nneg):
        return np.nan
    ranks = rankdata(scores[pos | neg], method='average')
    return (ranks[pos[pos | neg]].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def toy_check():
    target = np.array([1, 1, 1, 0, 0, 0])
    partner = np.array([0, 1, 0, 1, 0, 0])
    scores = np.array([.9, .6, .3, .3, .6, .1])
    mult = np.array([1, 2, 3, 2, 3, 1])
    receipts = []
    for weights in (np.ones(6), mult):
        p = weights[(target == 1) & (partner == 1)].sum() / weights[target == 1].sum()
        r = weights[(target == 0) & (partner == 1)].sum() / weights[target == 0].sum()
        bg, pn = components(compare_cells(scores, target, partner, weights), p)
        auc = float(Metric(scores, target).compute_auc(weights))
        assert np.isclose(auc, (1-r)*bg+r*pn, atol=1e-13, rtol=0)
        for mix in (.05, .50):
            standardized = weights.astype(float).copy()
            for c, q, mass in ((1, 0, 1-p), (1, 1, p), (0, 0, 1-mix), (0, 1, mix)):
                group = (target == c) & (partner == q)
                standardized[group] *= mass / weights[group].sum()
            positives, negatives = scores[target == 1], scores[target == 0]
            wins = (positives[:, None] > negatives) + 0.5 * (positives[:, None] == negatives)
            direct = standardized[target == 1] @ wins @ standardized[target == 0]
            reconstructed = (1-mix)*bg + mix*pn
            assert np.isclose(direct, reconstructed, atol=1e-13, rtol=0)
            receipts.append(dict(weights=weights.tolist(), r=mix, direct=direct,
                                 reconstructed=reconstructed, natural_auc=auc))
    assert Metric(np.array([.5, .5]), np.array([1, 0])).compute_auc() == .5
    assert np.isnan(Metric(scores, np.full(6, -1)).compute_auc())
    assert np.isnan(Metric(scores, target).compute_auc(np.zeros(6)))
    assert np.isclose(Metric(np.array([.9, .8, .7, .6]), np.array([0, 1, 1, 0])).compute_ap(), 2/3)
    return dict(status='PASS', mixture=receipts, all_ties_auc=.5, empty_support='NaN', envelope_ap=2/3)


def history():
    with np.load(G1 / 'raw/data/COCO_CONFIRM_GOLD.npz') as gold:
        ids, truth, classes = (gold[k] for k in ('ids', 'truth', 'classes'))
    settings = {}
    max_error = 0.0
    for setting in SETTINGS:
        endpoints = {}
        pairs = PAIR_MAP[setting]
        pstar = [float(truth[truth[:, p['target']] == 1, p['partner']].mean()) for p in pairs]
        for file in sorted((G1 / 'raw/models' / setting).glob('*/native_scores.npz')):
            job = re.fullmatch(rf'DC0_G1R2_{setting}_d(\d+)_i(\d+)_(TRUE|PLC)_(LOW|MID)', file.parent.name)
            assert job
            draw, init, arm, level = job.groups()
            with np.load(file) as data:
                assert np.array_equal(data['ids'], ids) and np.array_equal(data['classes'], classes)
                scores = data['scores']
            assert scores.shape == truth.shape and np.isfinite(scores).all()
            per_target = []
            for pair, p in zip(pairs, pstar):
                c, q = pair['target'], pair['partner']
                y, z = truth[:, c], truth[:, q]
                groups = {(a, b): (y == a) & (z == b) for a in (0, 1) for b in (0, 1)}
                cell_pairs = (((1,0),(0,0)), ((1,1),(0,0)), ((1,0),(0,1)), ((1,1),(0,1)))
                cells = np.array([rank_auc(scores[:,c], groups[a], groups[b]) for a, b in cell_pairs])
                reference = compare_cells(scores[:,c], y, z)
                max_error = max(max_error, float(np.max(np.abs(cells-reference))))
                bg, pn = components(cells, p)
                r = z[y == 0].mean()
                assert np.isclose((1-r)*bg+r*pn, rank_auc(scores[:,c], y == 1, y == 0), atol=1e-13, rtol=0)
                per_target.append([cells[2], bg, pn])
            ap = np.array([Metric(scores[:,c], truth[:,c]).compute_ap() for c in range(20)])
            endpoints[init, draw, arm, level] = (np.array(per_target), ap)
        assert len(endpoints) == 36
        inits = []
        for init in sorted({key[0] for key in endpoints}):
            rows = []
            for draw in sorted({key[1] for key in endpoints}):
                tm, tl, pm, pl = [endpoints[init, draw, arm, level]
                                  for arm, level in (('TRUE','MID'), ('TRUE','LOW'), ('PLC','MID'), ('PLC','LOW'))]
                net = (tm[0]-tl[0])-(pm[0]-pl[0])
                rows.append([net[:,0].mean(), (tl[0][:,1]-tm[0][:,1]).mean(),
                             net[:,1].mean(), net[:,2].mean(), (tm[1]-tl[1]).mean()])
            inits.append(dict(init=init, effect_pp=(100*np.mean(rows, axis=0)).tolist()))
        values = np.array([x['effect_pp'] for x in inits])
        means = values.mean(axis=0)
        settings[setting] = dict(pstar=pstar, initializations=inits, mean_pp=means.tolist(),
            sd_pp=values.std(axis=0, ddof=1).tolist(), zero_crossing=float(-means[2]/(means[3]-means[2])))
    assert max_error < 1e-13
    return dict(metrics=['conditional_net', 'background_loss', 'background_net', 'partner_negative_net',
                         'map_true_mid_minus_low'], max_independent_auc_error=max_error, settings=settings)


def power():
    rows = []
    point = [1.0, .3, .3, .3, -np.inf]
    bound = [0., 0., 0., 0., -.5]
    sd = [1.5, .6, .5, .5, .5]
    for n in (6, 8, 12, 16):
        df = n-1
        critical = t.ppf(.95, df)
        for label, means in (('moderate', [2., .65, .55, .55, 0.]),
                             ('weak', [1.5, .5, .4, .4, -.15])):
            probabilities = []
            for mu, sigma, floor, limit in zip(means, sd, point, bound):
                se = sigma / np.sqrt(n)
                def integrand(x):
                    threshold = max(floor, limit + critical * se * np.sqrt(x/df))
                    return norm.sf((threshold-mu)/se) * chi2.pdf(x, df)
                probabilities.append(quad(integrand, 0, np.inf, epsabs=1e-9)[0])
            rows.append(dict(n=n, scenario=label, marginal_power=probabilities,
                             joint_lower=max(0, 1-sum(1-v for v in probabilities)),
                             s0_endpoints=10*n, two_config_endpoints=18*n))
    return rows


if __name__ == '__main__':
    result = dict(toy=toy_check(), history=history(), power=power())
    (OUT / 'HISTORICAL_METRICS_AND_POWER.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for key, row in result['history']['settings'].items():
        print(key, np.round(row['mean_pp'], 4), round(row['zero_crossing'], 5))
    print([(r['n'], r['scenario'], round(r['joint_lower'], 3)) for r in result['power']])
