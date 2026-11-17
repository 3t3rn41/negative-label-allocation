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

"""Construct the frozen fallback masks using binary feasibility constraints."""

import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'voc_design'
sys.path.insert(0, str(PARENT))
from build_masks import ARMS, audit, largest_remainder, stratify


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class Constraints:
    def __init__(self, n):
        self.n = n
        self.rows, self.columns, self.values = [], [], []
        self.lower, self.upper = [], []

    def add(self, indices, weights, lower, upper):
        self.rows.extend([len(self.lower)]*len(indices))
        self.columns.extend(indices)
        self.values.extend(weights)
        self.lower.append(lower)
        self.upper.append(upper)

    def solve(self, objective):
        matrix = sparse.coo_matrix((self.values, (self.rows, self.columns)),
                                  shape=(len(self.lower), self.n)).tocsc()
        return milp(objective, integrality=np.ones(self.n), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, self.lower, self.upper),
                    options={'time_limit': 45})


def neutral_placebo(rows, quota, value, ids, rng, offset=0):
    selected = []
    for (_, _, pools), n in zip(rows, quota):
        # Difficulty ranks come exclusively from the common hidden-negative pool.
        # Choose an approximately even sample across each layer, with seeded
        # tie-breaking and a common fractional offset on the rank grid.
        hidden = pools[3]
        shuffled = rng.permutation(hidden)
        ordered = shuffled[np.argsort(value[shuffled], kind='stable')]
        if n:
            fraction = (np.arange(n) + (.5 + offset)) / n
            selected.append(ordered[np.floor(fraction*len(ordered)).astype(int)])
        else:
            selected.append(np.array([], dtype=np.int64))
    return selected


def directed_coordinates(rows, quota, dose, t, a, b, truth, score, plc, rng, rules):
    inputs = [[[], [], []] for _ in rows]
    group_columns, source_rows = [[], [], []], [[], [], []]
    sizes = []
    column = 0
    for li, ((_, _, groups), n) in enumerate(zip(rows, quota)):
        if not n:
            continue
        for k, group in enumerate(groups[:3]):
            columns = np.arange(column, column+len(group))
            inputs[li][k] = columns
            group_columns[k].extend(columns)
            source_rows[k].extend(group)
            sizes.append((columns, n))
            column += len(group)
    problem = Constraints(column)
    for indices, n in sizes:
        problem.add(indices, np.ones(len(indices)), n, n)
    y = truth[:, t] == -1
    standardized = (score[:, t].astype(float)-score[y, t].mean())/score[y, t].std()
    maxdiff = rules['max_abs_smd'] * dose - 1e-5
    placebo_rows = np.concatenate(plc)
    placebo_sum = standardized[placebo_rows].sum()
    for i, j in combinations(range(3), 2):
        idx = group_columns[i]+group_columns[j]
        weight = np.r_[standardized[source_rows[i]], -standardized[source_rows[j]]]
        problem.add(idx, weight, -maxdiff, maxdiff)
    for idx, group in zip(group_columns, source_rows):
        problem.add(idx, standardized[group], placebo_sum-maxdiff, placebo_sum+maxdiff)
    gap = int(np.ceil(dose*rules['minimum_directional_gap'] - 1e-10))
    for group, partner in ((1, a), (2, b)):
        other = 3-group
        if (truth[placebo_rows, partner] == 1).sum() > dose-gap:
            return None, 'PLC_DIRECTION'
        problem.add(group_columns[other], (truth[source_rows[other], partner] == 1).astype(float),
                    -np.inf, dose-gap)
    fit = problem.solve(rng.uniform(0, 1, column))
    if fit.x is None:
        return None, str(fit.message)
    assert np.max(abs(fit.x-np.rint(fit.x))) < 1e-5
    selected = []
    for (d, k, groups), slots, ctrl in zip(rows, inputs, plc):
        output = [g[np.rint(fit.x[np.asarray(idx, dtype=int)]).astype(bool)] if len(idx)
                  else np.array([], dtype=np.int64) for g,idx in zip(groups[:3], slots)]
        selected.append((d, k, [*output, ctrl]))
    return selected, str(fit.message)


def main():
    rules = read(HERE/'RULES.json')
    role = read(PARENT/'DATA_ROLE_LOCK.json')
    with np.load(PARENT/'FIT_GOLD.npz') as f, np.load(PARENT/'FIT_COSINE.npz') as s:
        ids, classes, y, cosine = f['ids'], f['classes'].tolist(), f['truth'], s['scores']
        assert np.array_equal(ids, s['ids']) and classes == s['classes'].tolist()
    pairs = [(classes.index(t), classes.index(a), classes.index(b)) for (t, a), (u, b) in
             zip(role['pairs']['P1'], role['pairs']['P2'], strict=True) if t == u]
    assert len(pairs) == 3 and ids.tolist() == role['role_ids']['fit']
    dose = rules['fixed_dose']
    qa, draw_results = [], []
    for draw in rules['draws']:
        sample = np.random.default_rng(draw).random(y.shape) < rules['low_probability']
        low = np.where(sample & (y!=0), y==1, -1).astype(np.int8)
        strata = list(stratify(ids,y,cosine,low,pairs))
        quotas = [largest_remainder(s,d) for s,d in zip(strata,dose)]
        assert all(q is not None for q in quotas)
        # A fixed, partner-blind PLC rank grid is generated before each
        # feasibility solve. The LP never has PLC decision variables.
        rngs = np.random.SeedSequence([draw, 922]).spawn(2)
        tie, solver = map(np.random.default_rng, rngs)
        for offset in rules['plc_rank_offsets']:
            groups = {}
            statuses = {}
            masks = {arm: low.copy() for arm in ARMS}
            for (t,a,b), d, rows, q in zip(pairs,dose,strata,quotas):
                plc = neutral_placebo(rows,q,cosine[:,t],ids,tie,offset)
                selected, status = directed_coordinates(rows,q,d,t,a,b,y,cosine,plc,solver,rules)
                statuses[classes[t]] = status
                if selected is None:
                    continue
                groups[t] = selected
                for _, _, (remove,*add) in selected:
                    for matrix, members in zip(masks.values(),add):
                        matrix[remove,t] = -1
                        matrix[members,t] = 0
            if len(groups) == len(pairs):
                failures, metrics, density = audit(y,cosine,low,pairs,dose,masks,groups)
            else:
                failures, metrics, density = ['NO_FEASIBLE_COORDINATES'], [], []
            qa.append(dict(draw=draw, offset=offset, failures=failures, statuses=statuses))
            print(draw, offset, failures, statuses, flush=True)
            if not failures:
                np.savez_compressed(HERE/f'MASK_{draw}.npz', ids=ids.astype(np.int64), classes=classes,
                                    LOW=low, **masks)
                coordinates = {classes[t]: [dict(difficulty=d, other_labels=k,
                               ids=[ids[g].tolist() for g in gs]) for d,k,gs in rows]
                               for t,rows in groups.items()}
                draw_results.append(dict(draw=draw, known=int((low>=0).sum()), offset=offset,
                                   comparisons=metrics,density=density,coordinates=coordinates))
                break
    result = dict(status='PASS' if len(draw_results)==2 else 'INCOMPLETE_MASK_QUALIFICATION',
                  dose=dose, draws=draw_results, attempts=qa,
                  coordinate_order=['remove',*ARMS], solver='scipy.optimize.milp / HiGHS')
    (HERE/'MASK_QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['status']=='PASS'


if __name__ == '__main__':
    main()
