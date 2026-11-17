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

"""Independently read back the generated masks and their coordinate receipts."""

import csv
import json
from collections import defaultdict

import numpy as np

from build_masks import OUT, PAIR_MAP, RULES, SETTINGS, TARGETS, load_training


def main():
    ids, truth, classes, cosine = load_training()
    row_lookup = {int(i): j for j, i in enumerate(ids)}
    summary = json.loads((OUT / 'MASK_QA.json').read_text())
    assert summary['status'] == 'PASS'
    coordinates = defaultdict(set)
    with (OUT / 'MASK_COORDINATES.csv').open(encoding='utf-8-sig', newline='') as file:
        for row in csv.DictReader(file):
            key = (int(row['draw']), row['setting'], int(row['target']),
                   int(row['difficulty']), int(row['density']), row['role'])
            index = row_lookup[int(row['coco_id'])]
            assert index not in coordinates[key]
            coordinates[key].add(index)
    results = []
    for seed in RULES['draw_seeds']:
        with np.load(OUT / f'MASK_{seed}.npz') as data:
            assert np.array_equal(data['ids'], ids) and np.array_equal(data['classes'], classes)
            arms = {k: data[k] for k in data.files if k not in ('ids', 'classes')}
        low = arms['LOW']
        assert np.array_equal(low >= 0, np.random.Generator(np.random.PCG64(seed)).random(truth.shape) < .10)
        density = (low[:, [c for c in range(20) if c not in TARGETS]] >= 0).sum(1).clip(max=2)
        expected = {key: low.copy() for key in arms}
        for i, target in enumerate(TARGETS):
            partners = [PAIR_MAP[setting][i]['partner'] for setting in SETTINGS]
            ordered = sorted(np.flatnonzero(truth[:, target] == 0), key=lambda r: (cosine[r, target], ids[r]))
            totals = 0
            for difficulty, bucket in enumerate(np.array_split(ordered, 10)):
                for count in range(3):
                    allowed = {int(j) for j in bucket if density[j] == count}
                    prefix = (seed, 'S0', target, difficulty, count)
                    shared_b = coordinates[(*prefix, 'B')]
                    totals += len(shared_b)
                    joint_packs = []
                    for setting, partner in zip(SETTINGS, partners):
                        prefix = (seed, setting, target, difficulty, count)
                        b, p, u, extra = [coordinates[(*prefix, role)] for role in ('B', 'P', 'U', 'Pprime')]
                        assert b == shared_b and len(b) == len(p) == len(u) == len(extra)
                        assert (b | p | u | extra) <= allowed and not p & extra
                        assert all(low[j, target] == 0 and not truth[j, partners].any() for j in b)
                        assert all(low[j, target] == -1 for j in p | u | extra)
                        assert all(truth[j, partner] == 1 for j in p | extra)
                        joint_packs.append(({j for j in p if truth[j, partners].all()},
                                            {j for j in extra if truth[j, partners].all()}))
                        for arm, remove, add in (('TRUE', b, p), ('PLC', b, u),
                                                 ('TRUE_BG', set(), p), ('TRUE_PARTNER', b, p | extra)):
                            expected[f'{setting}_{arm}'][list(remove), target] = -1
                            expected[f'{setting}_{arm}'][list(add), target] = 0
                    assert joint_packs[0] == joint_packs[1]
                    assert all(len(joint) <= len(shared_b)//2 for joint in joint_packs[0])
            assert totals == summary['dose'][i]
        class_counts = {}
        for arm, matrix in arms.items():
            assert np.array_equal(expected[arm], matrix)
            assert np.array_equal(truth[matrix >= 0], matrix[matrix >= 0])
            positives, negatives = (matrix == 1).sum(0), (matrix == 0).sum(0)
            assert (positives > 0).all() and (negatives > 0).all()
            class_counts[arm] = dict(positive=positives.tolist(), negative=negatives.tolist())
        results.append(dict(draw=seed, exact_matrix_reconstruction=True, shared_low=True, class_counts=class_counts))
    (OUT / 'MASK_INDEPENDENT_QA.json').write_text(json.dumps(dict(status='PASS', draws=results), indent=2))
    print('PASS', len(results), 'draws, all 18 stored matrices reconstructed')


if __name__ == '__main__':
    main()
