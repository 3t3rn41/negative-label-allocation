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

"""Screen P2 pottedplant partners using VOC labels only."""

from pathlib import Path
import json

import numpy as np

import qualify as q


def main():
    trainval = q.listed('trainval')
    test = q.listed('test')
    history = json.loads(q.MANIFEST.read_text())
    fit, dev = [set(str(i).zfill(6) for i in history[k]) for k in ('fit', 'dev')]
    assert fit.isdisjoint(dev) and fit | dev == set(trainval)
    all_tv = q.labels(trainval, 'trainval')
    yfit = all_tv[[i in fit for i in trainval]]
    ydev = all_tv[[i in dev for i in trainval]]
    ytest = q.labels(test, 'test')
    t = q.CLASSES.index('pottedplant')
    p1 = q.CLASSES.index('person')
    rows = []
    for j, candidate in enumerate(q.CLASSES):
        if j in (t, p1):
            continue
        counts = {}
        for split, matrix in (('fit', yfit), ('dev', ydev), ('test', ytest)):
            a, b = matrix[:, t], matrix[:, j]
            counts[split] = [int(((a == u) & (b == v)).sum())
                             for u, v in ((1, -1), (-1, 1), (1, 1), (-1, -1))]
        n10, n01, n11, n00 = counts['test']
        p = counts['dev'][2] / (counts['dev'][0] + counts['dev'][2])
        peff = 1 / ((1-p)**2/n10 + p**2/n11) if min(n10, n11) else 0.
        neff = [1 / (w*w/n01 + (1-w)**2/n00) if min(n01, n00) else 0. for w in (.05, .50)]
        target, person, alternative = yfit[:, t], yfit[:, p1], yfit[:, j]
        exclusive = [int(((target == -1) & (a == 1) & (b == -1)).sum())
                     for a, b in ((person, alternative), (alternative, person))]
        joint = int(((target == -1) & (person == 1) & (alternative == 1)).sum())
        eligible = min(counts['test']) >= 30 and peff >= 80 and min(neff) >= 250
        rows.append(dict(candidate=candidate, counts_10_01_11_00=counts,
                         effective_positive=peff, effective_negative=neff,
                         fit_exclusive_01=exclusive, fit_shared_01=joint, eligible=eligible))
    # Favor the candidate with the largest pool of distinct training
    # negatives in the more constrained direction. Test counts only gate.
    selected = max((r for r in rows if r['eligible']),
                   key=lambda r: (min(r['fit_exclusive_01']), -q.CLASSES.index(r['candidate'])))
    q.save('PARTNER_SCREEN.json', dict(status='PASS', selected=selected,
           rule='test four-cell and secondary ESS eligibility; maximize minimum fit exclusive 01 count; VOC class order breaks ties',
           scores_accessed=False, candidates=rows))
    print(json.dumps(selected), flush=True)


if __name__ == '__main__':
    main()
