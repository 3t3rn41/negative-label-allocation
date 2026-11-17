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

"""Screen nested COCO confirmation candidates before model scoring."""

import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from build_masks import G1, OUT, PAIR_MAP, ROOT, RULES, SETTINGS, load_training

OUTPUT = ROOT / 'inputs'
ARCHIVE = OUTPUT / 'reference_experiments'
AUDIT = OUTPUT / 'independent_coco_pool'
BINDING = ARCHIVE / 'category_binding.json'
ID_FIELDS = {
    'id', 'ids', 'image_id', 'image_ids', 'img_ids', 'coco_id', 'coco_ids',
    'train_ids', 'eval_ids', 'confirm_ids', 'native_confirm', 'exclusion_ids',
    'union_ids', 'selected_ids', 'development_ids', 'dev_ids', 'fit_ids',
    'coco_train', 'coco_dev', 'coco_test', 'coco_confirm', 'image_paths', 'pixel_paths',
}
ALIAS = re.compile(r'(?:COCO_(?:train|val)2014_)?(\d{1,12})(?:\.(?:jpg|jpeg|png))?$', re.I)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def normalize(value):
    if isinstance(value, (int, str, np.integer)) and not isinstance(value, bool):
        match = ALIAS.fullmatch(str(value).replace('\\', '/').rsplit('/', 1)[-1])
        if match:
            return int(match.group(1))
    return None


def ids_from_json(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in ID_FIELDS:
                values = child if isinstance(child, list) else [child]
                yield from (i for x in values if (i := normalize(x)) is not None)
            if isinstance(child, (list, dict)):
                yield from ids_from_json(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (list, dict)):
                yield from ids_from_json(child)


def historical_exclusions(universe):
    prior = read_json(AUDIT / 'EXPOSURE_AND_ELIGIBILITY.json')
    assert prior['universe_available_COCO_images'] == len(universe) == 123287
    eligible_prior = set(prior['eligible_ids'])
    assert eligible_prior <= universe and len(eligible_prior) == prior['eligible_n']
    sources = {}
    issues = []
    scans = {'npz': 0, 'npy': 0, 'json': 0, 'csv': 0}
    selectors = ('ids', 'manifest', 'exposure', 'split', 'candidate', 'data_files',
                 'pixel', 'inventory', 'pool', 'holdout', 'resource', 'roster')
    for folder in (OUTPUT, ROOT / 'V8', ROOT / 'work'):
        for path in folder.rglob('*'):
            if any(part.lower() in ('site-packages', 'runtime_deps', 'packages', 'manuscript_deps',
                                    'vendor', '__pycache__', 'fixtures', 'seals', 'pipeline_selftest')
                   for part in path.parts):
                continue
            ext = path.suffix.lower()
            name = path.stem.lower()
            if ext == '.npz':
                scans['npz'] += 1
                try:
                    with np.load(path, allow_pickle=False) as data:
                        found = set()
                        for key in ID_FIELDS.intersection(data.files):
                            try:
                                arr = data[key]
                            except ValueError as exc:
                                issues.append([str(path.relative_to(ROOT)), key, str(exc)])
                                continue
                            if arr.ndim == 1:
                                found.update(i for v in arr if (i := normalize(v)) in eligible_prior)
                except (OSError, ValueError, zipfile.BadZipFile) as exc:
                    issues.append([str(path.relative_to(ROOT)), str(exc)])
                    continue
            elif ext == '.npy' and (name in ID_FIELDS or name.endswith('_ids')):
                scans['npy'] += 1
                try:
                    arr = np.load(path, mmap_mode='r', allow_pickle=False)
                    found = {i for v in arr if (i := normalize(v)) in eligible_prior} if arr.ndim == 1 else set()
                except (OSError, ValueError) as exc:
                    issues.append([str(path.relative_to(ROOT)), str(exc)])
                    continue
            elif ext == '.json' and any(token in name for token in selectors) and name != 'exposure_and_eligibility':
                scans['json'] += 1
                try:
                    found = set(ids_from_json(read_json(path))) & eligible_prior
                except (OSError, ValueError) as exc:
                    issues.append([str(path.relative_to(ROOT)), str(exc)])
                    continue
            elif ext in ('.csv', '.tsv') and any(token in name for token in selectors):
                scans['csv'] += 1
                with path.open(encoding='utf-8-sig', errors='replace', newline='') as file:
                    reader = csv.DictReader(file, delimiter='\t' if ext == '.tsv' else ',')
                    keys = [key for key in reader.fieldnames or [] if key.lower() in ID_FIELDS]
                    found = {i for row in reader for key in keys
                             if (i := normalize(row[key])) in eligible_prior} if keys else set()
            elif ext in ('.jpg', '.jpeg') and (i := normalize(path.name)) in eligible_prior:
                found = {i}
            else:
                continue
            if found:
                sources[str(path.relative_to(ROOT))] = sorted(found)
    later = set().union(*(set(found) for found in sources.values()))
    assert set(read_json(AUDIT / 'CANDIDATE_MEASUREMENT_6000.json')['ids']) <= later
    receipt = dict(inherited_exclusion_count=len(universe - eligible_prior), supplemental_exclusion_count=len(later),
                   eligible_count=len(eligible_prior - later), scans=scans, source_hits=sources, issues=issues)
    (OUT / 'HISTORICAL_EXCLUSIONS.json').write_text(json.dumps(receipt, ensure_ascii=False), encoding='utf-8')
    return eligible_prior - later, receipt


def direct_exposure_audit(images, classes):
    old_manifest = ARCHIVE / 'prior_confirmation/CONFIRM_IDS_FROZEN.json'
    old = read_json(old_manifest)
    old_ids = set(old['exclusion_IDs']) | set(old['NATIVE_CONFIRM']) | set(old['TRAIN_ids'])
    assert len(old_ids) == 102283 and old_ids <= images.keys()
    covered = set(old_ids)
    sources = [dict(path=str(old_manifest.relative_to(ROOT)), count=len(old_ids), newly_excluded=len(old_ids))]
    for relative, expected in (
        ('prior_evaluation/dev_pool_scores.npz', 6000),
        ('prior_evaluation/reference_scores.npz', 12826),
    ):
        path = OUTPUT / relative
        with np.load(path) as data:
            ids = set(map(int, data['ids']))
            scores = data['scores']
            assert len(ids) == len(data['ids']) == expected and ids <= images.keys()
            assert np.array_equal(data['classes'], classes)
            assert scores.ndim == 3 and scores.shape[1:] == (expected, 20)
            assert np.isfinite(scores).all()
        sources.append(dict(path=str(path.relative_to(ROOT)), count=len(ids), newly_excluded=len(ids-covered),
                            scored_models=scores.shape[0]))
        covered.update(ids)
    remaining = images.keys() - covered
    result = dict(annotated_universe=len(images), covered=len(covered), sources=sources,
                  remaining_upper_bound=len(remaining), remaining_ids_upper_bound=sorted(remaining))
    (OUT / 'EVAL_DIRECT_EXPOSURE_QA.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_coco(classes):
    mapping = read_json(BINDING)
    assert list(classes) == [row['class_name'] for row in mapping]
    index = {row['category_id']: c for c, row in enumerate(mapping)}
    images, labels = {}, {}
    with zipfile.ZipFile(OUT / 'annotations_trainval2014.zip') as archive:
        for split in ('train2014', 'val2014'):
            with archive.open(f'annotations/instances_{split}.json') as file:
                content = json.load(file)
            categories = {c['id']: c['name'] for c in content['categories']}
            assert all(categories[row['category_id']] == row['display_name'] for row in mapping)
            for image in content['images']:
                i = image['id']
                assert i not in images and normalize(image['file_name']) == i
                images[i] = dict(split=split, file_name=image['file_name'], width=image['width'], height=image['height'])
                labels[i] = np.zeros(20, dtype=np.int8)
            for annotation in content['annotations']:
                col = index.get(annotation['category_id'])
                if col is not None:
                    labels[annotation['image_id']][col] = 1
            del content
    return images, labels


def effective_sample_size(n0, n1, mix):
    return 1 / ((1-mix)**2/n0 + mix**2/n1) if n0 and n1 else 0.


def support(truth, history):
    result = []
    for setting in SETTINGS:
        for k, pair in enumerate(PAIR_MAP[setting]):
            target, partner = truth[:, pair['target']], truth[:, pair['partner']]
            n10, n01, n11, n00 = (int(np.count_nonzero((target == t) & (partner == p)))
                                  for t, p in ((1, 0), (0, 1), (1, 1), (0, 0)))
            counts = [n10, n01, n11, n00]
            pstar = history[setting]['pstar'][k]
            mixes = [RULES['low_mixture'], *RULES['high_mixture_order']]
            result.append(dict(setting=setting, target=pair['target_name'], partner=pair['partner_name'],
                counts_10_01_11_00=counts, pstar=pstar, natural_r=n01/(n01+n00),
                group_pass=all(x >= minimum for x, minimum in zip(counts, RULES['evaluation_group_minimums_10_01_11_00'])),
                positive_ess=effective_sample_size(n10, n11, pstar),
                negative_ess={r: effective_sample_size(n00, n01, r) for r in mixes}))
    return result


def main():
    train_ids, train_truth, classes, _ = load_training()
    with np.load(G1 / 'raw/data/COCO_CONFIRM_GOLD.npz') as gold:
        old_ids, old_truth = gold['ids'], gold['truth']
    images, labels = load_coco(classes)
    assert np.array_equal(np.stack([labels[i] for i in train_ids]), train_truth)
    assert np.array_equal(np.stack([labels[i] for i in old_ids]), old_truth)
    direct = direct_exposure_audit(images, classes)
    minimum = min(RULES['evaluation_sizes'])
    if direct['remaining_upper_bound'] < minimum:
        status = dict(status='HOLD_EVALUATION_POOL', reason='eligible upper bound below smallest protocol size',
                      annotated_universe=len(images), known_gold_agreement=[4008, 5000],
                      eligible_upper_bound=direct['remaining_upper_bound'],
                      minimum_shortfall=minimum-direct['remaining_upper_bound'],
                      sizes=[dict(n=n, minimum_shortfall=n-direct['remaining_upper_bound'],
                                  group_support='NOT_RUN_INSUFFICIENT_CANDIDATES')
                             for n in RULES['evaluation_sizes']],
                      candidate_order='NOT_GENERATED', pixel_decode='NOT_RUN', selected_n=None,
                      composition_high=None)
        (OUT / 'EVAL_SUPPORT_QA.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(status['status'], 'at most', status['eligible_upper_bound'], 'IDs; minimum', minimum)
        return
    eligible, exclusion = historical_exclusions(set(images))
    assert not set(train_ids) & eligible and not set(old_ids) & eligible
    largest = max(RULES['evaluation_sizes'])
    if len(eligible) < minimum:
        status = dict(status='HOLD_EVALUATION_POOL', eligible_count=len(eligible),
                      minimum_shortfall=minimum-len(eligible), selected_n=None)
        (OUT / 'EVAL_SUPPORT_QA.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        return
    ids = np.random.default_rng(RULES['evaluation_seed']).permutation(sorted(eligible))[:largest].tolist()
    file = OUT / 'EVAL_CANDIDATE_ORDER.json'
    candidate = dict(status='CANDIDATE_ORDER_FIXED', seed=RULES['evaluation_seed'], ids=ids,
                     annotated_universe=len(images), eligible_count=len(eligible),
                     history_scope='D4E inherited historical exposure plus local archived manifests and pixels')
    if file.exists():
        assert read_json(file) == candidate
    else:
        file.write_text(json.dumps(candidate), encoding='utf-8')
    history = read_json(OUT / 'HISTORICAL_METRICS_AND_POWER.json')['history']['settings']
    attempts, selected = [], None
    for n in RULES['evaluation_sizes']:
        if len(ids) < n:
            attempts.append(dict(n=n, group_pass=False, reason='insufficient eligible candidates',
                                 pairs=[], candidate_high=None))
            continue
        truth = np.stack([labels[i] for i in ids[:n]])
        pairs = support(truth, history)
        pos_min, neg_min = RULES['positive_and_negative_ess_minimums']
        low = RULES['low_mixture']
        supported = []
        for high in RULES['high_mixture_order']:
            if not all(row['positive_ess'] >= pos_min and
                       min(row['negative_ess'][low], row['negative_ess'][high]) >= neg_min for row in pairs):
                continue
            if all((1-low)*h['mean_pp'][2] + low*h['mean_pp'][3] < 0 <
                   (1-high)*h['mean_pp'][2] + high*h['mean_pp'][3] for h in history.values()):
                supported.append(high)
        positive = truth.sum(axis=0)
        group_pass = all(row['group_pass'] for row in pairs) and bool(((positive > 0) & (positive < n)).all())
        attempts.append(dict(n=n, group_pass=group_pass, pairs=pairs, class_positive=positive.tolist(),
                             candidate_high=supported[0] if supported else None))
        if group_pass:
            selected = attempts[-1]
            break
    if selected:
        n = selected['n']
        np.savez_compressed(OUT / 'EVAL_CANDIDATE_GOLD.npz', ids=ids[:n], truth=truth, classes=classes)
        with (OUT / 'EVAL_PIXEL_CANDIDATES.csv').open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, ['id', 'split', 'file_name', 'width', 'height'])
            writer.writeheader()
            writer.writerows(dict(id=i, **images[i]) for i in ids[:n])
    result = dict(status='PASS_LABEL_SUPPORT' if selected else 'HOLD_EVALUATION_SUPPORT',
                  known_gold_agreement=[4008, 5000], exclusion_scan_issues=exclusion['issues'],
                  pixel_decode='PENDING' if selected else 'NOT_RUN',
                  sizes=attempts, selected_n=selected['n'] if selected else None,
                  composition_high=selected['candidate_high'] if selected else None)
    (OUT / 'EVAL_SUPPORT_QA.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'], 'eligible', len(eligible), 'scan_issues', len(exclusion['issues']))
    for item in attempts:
        print(item['n'], item['group_pass'], 'high', item['candidate_high'])
        for pair in item['pairs']:
            print(pair['setting'], pair['target'], pair['partner'], pair['counts_10_01_11_00'],
                  pair['group_pass'], round(pair['positive_ess'], 1))


if __name__ == '__main__':
    main()
