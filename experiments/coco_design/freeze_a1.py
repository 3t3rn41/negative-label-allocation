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

"""Cross-check and freeze the confirmed data and the isolated pilot inputs."""

import csv
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

from audit_dev_pool import ARCHIVE, DEST, OUT, ROOT, V3, save
from build_masks import G1, RULES
from qualify_evaluation import read_json, support


def csv_rows(name):
    with (DEST / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def prepare_pilot(ids):
    with np.load(G1 / 'raw/data/COCO_CONFIRM_GOLD.npz') as old:
        development = old['ids'].tolist()
    with np.load(OUT / f"MASK_{RULES['draw_seeds'][0]}.npz") as masks:
        training = masks['ids'].tolist()
        known = int((masks['LOW'] >= 0).sum())
    assert not set(ids) & set(development) and not set(ids) & set(training)
    assets = read_json(ROOT / 'inputs/coco_pixel_sources.json')
    by_id = {row['image_id']: row for row in assets}
    assert len(by_id) == len(assets) == len(training) + len(development)
    with (DEST / 'PILOT_PIXEL_MANIFEST.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['role', 'id', 'existing_subdirectory', 'file_name'])
        for role, rows in (('TRAIN', training), ('DEVELOPMENT', development)):
            writer.writerows((role, i, by_id[i]['subset'], f'{i:012d}.jpg') for i in rows)
    save('PILOT_INPUT_QA.json', dict(status='INPUTS_PREPARED', initialization=RULES['development_initialization'],
        draw=RULES['draw_seeds'][0], development_count=len(development), train_count=len(training),
        overlap_with_confirmation=0, shared_base_lr=.002*8976/known, native_epochs=50,
        cache_parity='PENDING_PER_EPOCH_VIEWS', gpu_benchmark='PENDING', formal_budget='PENDING'))


def main():
    manifest = csv_rows('CONFIRM_PIXEL_BINDING.csv')
    pixel = read_json(DEST / 'PIXEL_DECODE_QA.json')
    assert pixel['status'] == 'PASS' and pixel['full_decode_pass'] == len(manifest) == 5000
    ids = [int(row['id']) for row in manifest]
    candidate = read_json(DEST / 'CANDIDATE_ORDER.json')
    assert ids == candidate['ids'][:5000] and len(set(ids)) == len(ids)
    source = csv_rows('CONFIRM_PIXELS.csv')
    for original, decoded in zip(source, manifest):
        assert (original['id'], original['width'], original['height']) == (
            decoded['id'], decoded['width'], decoded['height'])
        assert decoded['member'].rsplit('/', 1)[-1] == f"{int(decoded['id']):012d}.jpg"
        assert decoded['asset'] in {pixel['source'], *pixel.get('fallback_sources', [])}
        assert int(decoded['bytes']) > 0
    assert sum(int(row['bytes']) for row in manifest) == pixel['total_bytes']
    dispositions = csv_rows('DEV_DISPOSITION.csv')
    eligibility = sorted(int(row['id']) for row in dispositions if row['eligible'] == '1')
    assert len(eligibility) == candidate['eligible']
    assert np.random.default_rng(candidate['seed']).permutation(eligibility)[:8000].tolist() == candidate['ids']
    with np.load(DEST / 'CONFIRM_GOLD.npz') as gold:
        y, classes = gold['truth'], gold['classes']
        assert gold['ids'].tolist() == ids and y.shape == (5000, 20)
    split = read_json(V3 / 'manifests/COCO_split.json')
    mapping = read_json(ARCHIVE / 'category_binding.json')
    columns = [split['category_ids'].index(row['category_id']) for row in mapping]
    assert list(classes) == [row['class_name'] for row in mapping]
    with np.load(V3 / 'blind/COCO_dev_gold.npz') as old:
        lookup = {int(i): j for j, i in enumerate(old['ids'])}
        assert np.array_equal(old['Y'][np.ix_([lookup[i] for i in ids], columns)], y)
    old_records = ARCHIVE / 'prior_confirmation/CONFIRM_IDS_FROZEN.json'
    assert not set(ids).intersection(read_json(old_records)['NATIVE_CONFIRM'])
    history = read_json(OUT / 'HISTORICAL_METRICS_AND_POWER.json')['history']['settings']
    stats = read_json(DEST / 'SUPPORT_5000.json')
    low, high = stats['low'], stats['high']
    pairs = support(y, history)
    assert [row['counts_10_01_11_00'] for row in pairs] == [
        row['counts_10_01_11_00'] for row in stats['pairs']]
    assert all(row['group_pass'] and row['positive_ess'] >= 80 and
               min(row['negative_ess'][low], row['negative_ess'][high]) >= 250 for row in pairs)
    prepare_pilot(ids)
    save('CONFIRM_DATA_LOCK.json', dict(status='PASS_CONFIRMATION_POOL_LOCKED',
        timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
        protocol_revision='coco-a1-qualification', eligible=candidate['eligible'], n=len(ids),
        seed=candidate['seed'], order='CANDIDATE_ORDER.json', official_annotation='COCO 2014 instances',
        prior_use='historical label support; recorded training, model use and unresolved artifact hits excluded',
        annotation_cross_check='all 100000 selected binary labels equal historical dev gold projection',
        class_names=classes.tolist(), gold='CONFIRM_GOLD.npz', pixel_manifest='CONFIRM_PIXEL_BINDING.csv',
        pixel_decode_receipt='PIXEL_DECODE_QA.json', proportions=[low, high],
        four_cell_minimum=[min(row['counts_10_01_11_00'][j] for row in pairs) for j in range(4)],
        min_positive_ess=min(row['positive_ess'] for row in pairs),
        min_negative_ess=min(min(row['negative_ess'][low], row['negative_ess'][high]) for row in pairs),
        formal_training='PENDING_ENGINEERING_AND_BUDGET'))
    print('PASS_CONFIRMATION_POOL_LOCKED', len(ids), low, high, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--pilot-only']:
        with np.load(DEST / 'CONFIRM_GOLD.npz') as gold:
            prepare_pilot(gold['ids'].tolist())
        print('PASS_PILOT_INPUTS_ISOLATED')
    else:
        main()
