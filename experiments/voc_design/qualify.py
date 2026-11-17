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

"""Verify the VOC2007 roles and the official classification label policy."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("FQ_VOC_INPUTS", Path(__file__).resolve().parents[2] / "inputs/voc2007"))
VOC = SOURCE / 'VOCdevkit/VOC2007'
MANIFEST = SOURCE / 'VOC_split.json'
MAIN = VOC / 'ImageSets/Main'
CLASSES = 'aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'.split()
PAIRS = {'P1': [('chair', 'sofa'), ('diningtable', 'bottle'), ('pottedplant', 'person')],
         'P2': [('chair', 'tvmonitor'), ('diningtable', 'person'), ('pottedplant', 'sofa')]}

if (ROOT / 'PARTNER_SCREEN.json').is_file():
    selection = json.loads((ROOT / 'PARTNER_SCREEN.json').read_text(encoding='utf-8'))
    assert selection['status'] == 'PASS' and not selection['scores_accessed']
    PAIRS['P2'][2] = ('pottedplant', selection['selected']['candidate'])


def listed(name):
    rows = [line.split() for line in (MAIN / f'{name}.txt').read_text().splitlines()]
    assert all(len(row) == 1 for row in rows)
    ids = [row[0] for row in rows]
    assert len(ids) == len(set(ids))
    return ids


def labels(ids, split):
    matrix = []
    for cls in CLASSES:
        rows = [line.split() for line in (MAIN / f'{cls}_{split}.txt').read_text().splitlines()]
        values = {i: int(y) for i, y in rows}
        assert len(values) == len(rows) == len(ids) and set(values) == set(ids)
        assert set(values.values()) <= {-1, 0, 1}
        matrix.append([values[i] for i in ids])
    return np.array(matrix, dtype=np.int8).T


def decode(image_id):
    xml = ET.parse(VOC / 'Annotations' / f'{image_id}.xml').getroot()
    assert xml.findtext('filename') == f'{image_id}.jpg'
    expected = (int(xml.findtext('size/width')), int(xml.findtext('size/height')))
    path = VOC / 'JPEGImages' / f'{image_id}.jpg'
    with Image.open(path) as image:
        image.load()
        assert image.format == 'JPEG' and image.size == expected
        image.convert('RGB').load()
    return image_id, [*expected, path.stat().st_size]


def save(name, payload):
    (ROOT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    trainval, test = listed('trainval'), listed('test')
    assert len(trainval) == 5011 and len(test) == 4952 and set(trainval).isdisjoint(test)
    assert set(listed('train')).isdisjoint(listed('val'))
    assert set(listed('train')) | set(listed('val')) == set(trainval)
    history = json.loads(MANIFEST.read_text())
    assert history['classes'] == CLASSES
    fit, dev = [[str(i).zfill(6) for i in history[role]] for role in ('fit', 'dev')]
    assert len(fit) == len(set(fit)) == 4008 and len(dev) == len(set(dev)) == 1003
    assert set(fit).isdisjoint(dev) and set(fit) | set(dev) == set(trainval)
    indexed = np.array(sorted(trainval))
    order = np.random.default_rng(history['split_seed']).permutation(len(indexed))
    assert set(indexed[order[:4008]]) == set(fit) and set(indexed[order[4008:]]) == set(dev)
    tv_y = labels(trainval, 'trainval')
    test_y = labels(test, 'test')
    index = {i: j for j, i in enumerate(trainval)}
    datasets = {'fit': (fit, tv_y[[index[i] for i in fit]]),
                'dev': (dev, tv_y[[index[i] for i in dev]]), 'test': (test, test_y)}
    support, ap, failures, composition = {}, {}, [], []
    for role, (ids, y) in datasets.items():
        support[role] = {}
        ap[role] = {c: {str(v): int((y[:, j] == v).sum()) for v in (1, -1, 0)}
                    for j, c in enumerate(CLASSES)}
        for setting, pairs in PAIRS.items():
            for t, p in pairs:
                a, b = (y[:, CLASSES.index(c)] for c in (t, p))
                cells = {f'{int(u == 1)}{int(v == 1)}': int(((a == u) & (b == v)).sum())
                         for u, v in ((1, -1), (-1, 1), (1, 1), (-1, -1))}
                key = f'{setting}:{t}/{p}'
                support[role][key] = dict(cells=cells, excluded=int(((a == 0) | (b == 0)).sum()))
                if role == 'test':
                    failures.extend(dict(key=key, cell=cell, count=n, required=30)
                                    for cell, n in cells.items() if n < 30)
        if role == 'test':
            failures.extend(dict(class_name=c, counts=vals, reason='AP_POS_NEG_SUPPORT')
                            for c, vals in ap[role].items() if min(vals['1'], vals['-1']) == 0)
    for key, row in support['test'].items():
        n = row['cells']
        d = support['dev'][key]['cells']
        p = d['11'] / (d['10'] + d['11'])
        for q in (.05, .50):
            pos = 1 / ((1-p)**2/n['10'] + p**2/n['11'])
            neg = 1 / (q**2/n['01'] + (1-q)**2/n['00'])
            composition.append(dict(key=key, dev_positive_partner_fraction=p, test_negative_partner_fraction=q,
                                    effective_positive=pos, effective_negative=neg,
                                    status='AVAILABLE' if pos >= 80 and neg >= 250 else 'INSUFFICIENT_SUPPORT'))
    print('VOC_SUPPORT', json.dumps(support['test']), 'FAILURES', failures, flush=True)
    with ThreadPoolExecutor(max_workers=16) as workers:
        pixels = dict(workers.map(decode, trainval + test))
    assert len(pixels) == 9963
    result = dict(status='INCOMPLETE_DATA_QUALIFICATION' if failures else 'PASS_DATA_QUALIFICATION',
                  source=str(VOC), history_manifest=str(MANIFEST), history_split_seed=history['split_seed'],
                  classes=CLASSES, pairs=PAIRS, official_label_policy='1 positive, -1 negative, 0 difficult/ignored',
                  role_ids={k: v[0] for k, v in datasets.items()}, role_overlap=0,
                  historical_roles={'fit': 'v3 fit and prior VOC trainval experiments',
                                    'dev': 'v3 development and prior VOC trainval experiments',
                                    'test': 'historical VOC2007/RN101 evaluation; v9.2 prespecified reuse',
                                    'source_scope': 'VOC2007 official images; separate named source from COCO'},
                  decoded_pixels=len(pixels), pixel_policy='decode all official JPEGs and verify XML dimensions',
                  ap_label_support=ap, support=support, formal_support_failures=failures,
                  composition=composition, test_model_scores_accessed=False)
    save('DATA_ROLE_LOCK.json' if not failures else 'DATA_ROLE_QA.json', result)
    save('PIXEL_QA.json', dict(source=str(VOC / 'JPEGImages'), decoded=pixels))
    if not failures:
        for role, (ids, y) in datasets.items():
            np.savez_compressed(ROOT / f'{role.upper()}_GOLD.npz', ids=ids, classes=CLASSES, truth=y)
    print(result['status'], 'decoded', len(pixels), flush=True)


if __name__ == '__main__':
    main()
