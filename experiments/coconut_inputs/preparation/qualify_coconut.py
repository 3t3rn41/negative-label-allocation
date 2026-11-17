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

"""Qualify fixed prefixes of COCO unlabeled2017 linked to released COCONut-B labels."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREFIXES = (5000, 6000, 8000, 10000)
CELLS = ("10", "01", "11", "00")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count(a, b):
    bins = np.bincount(2 * a + b, minlength=4)
    return [int(bins[k]) for k in (2, 1, 3, 0)]


def ess(a, b, share):
    return 1 / ((1 - share)**2 / a + share**2 / b) if a and b else 0.


def support(y, dev, t, p, names):
    n10, n01, n11, n00 = count(y[:, t], y[:, p])
    d10, _, d11, _ = count(dev[:, t], dev[:, p])
    p_dev = d11 / (d10 + d11)
    p_conf = n11 / (n10 + n11)
    r_conf = n01 / (n01 + n00)
    pts = [(p_conf, r_conf), (p_dev, .05), (p_dev, .5)]
    points = [dict(p=pa, r=ra, pos_ess=ess(n10, n11, pa), neg_ess=ess(n00, n01, ra)) for pa, ra in pts]
    return dict(target=names[t], partner=names[p], counts=dict(zip(CELLS, (n10, n01, n11, n00))),
                natural_fixed_points=points,
                passed=all(x >= gate for x, gate in zip((n10, n01, n11, n00), (50, 50, 30, 200)))
                and all(x["pos_ess"] >= 80 and x["neg_ess"] >= 250 for x in points))


def load_labels(paths, category_ids, sources):
    rows = {}
    intersection = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            keep = np.isin(data["ids"], list(sources))
            ids = data["ids"][keep].tolist()
            labels = np.unpackbits(data["categories"][keep], axis=1)[:, category_ids]
            assert all((labels.sum(1) >= 0) & (labels.sum(1) <= len(category_ids)))
            for i, width, height, y in zip(ids, data["width"][keep], data["height"][keep], labels, strict=True):
                assert i not in rows and sources[i][:2] == (int(width), int(height))
                rows[i] = y
            intersection[path.name] = len(ids)
    return rows, intersection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--expanded-dev", action="store_true")
    parser.add_argument("--larger-pool-diagnostic", action="store_true")
    args = parser.parse_args()
    folder = HERE / ("data_results_v3" if args.expanded_dev else "data_results")
    folder.mkdir(exist_ok=True)
    mask_folder = HERE / ("mask_results_v3" if args.expanded_dev else "mask_results")
    mask = read_json(mask_folder / "MASK_GATE_STATUS.json")
    qa = read_json(mask_folder / "MASK_INDEPENDENT_QA.json")
    assert mask["status"] == "FIT_MASKS_CONSTRUCTED" and qa["status"] == "PASS"
    with np.load(HERE / "inputs/historical_ids.npz", allow_pickle=False) as old:
        dev, names, cats = old["dev_truth"], old["classes"].tolist(), old["category_ids"]
    if args.expanded_dev:
        with np.load(HERE / "inputs/EXPANDED_DEV_28595.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["classes"], names)
            dev = data["truth"]
    with (HERE / "inputs/UNLABELED2017_SOURCE_ROWS.csv").open(encoding="utf-8", newline="") as stream:
        sources = {int(r["id"]): (int(r["width"]), int(r["height"]), r["flickr_id"])
                   for r in csv.DictReader(stream) if r["disposition"] == "CANDIDATE"}
    order = np.load(HERE / "inputs/FROZEN_SOURCE_ORDER.npy", allow_pickle=False)
    assert set(sources) == set(order.tolist())
    paths = sorted(args.labels_dir.glob("train-*-of-00004_labels.npz"))
    if not args.partial:
        assert len(paths) == 4
    rows, overlaps = load_labels(paths, cats, sources)
    selected = [int(i) for i in order if i in rows]
    assert len(selected) == len(rows)
    details = {}
    choice = None
    sizes = PREFIXES if not args.larger_pool_diagnostic else (
        *PREFIXES, 16000, 24000, 32000, 40000, 60000, 80000, 100000, len(selected))
    for size in sizes:
        ids = selected[:size]
        if len(ids) < size:
            break
        y = np.stack([rows[i] for i in ids])
        pairs = [support(y, dev, t, pair, names)
                 for (t, a, b) in mask["panel_before_confirm_support"] for pair in (a, b)]
        counts = y.sum(0)
        class_gate = bool(((counts > 0) & (counts < size)).all())
        passed = class_gate and all(row["passed"] for row in pairs)
        details[size] = dict(pass_label_support=passed, pairs=pairs,
                             positives=dict(zip(names, counts.tolist())))
        if passed and choice is None:
            choice = size
        print(size, "PASS" if passed else "HOLD", [(p["target"], p["partner"]) for p in pairs if not p["passed"]],
              flush=True)
    receipt = dict(status="PARTIAL_SHARD_DIAGNOSTIC" if args.partial else (
        "LABEL_SUPPORT_PASS_PIXEL_PENDING" if choice else "HOLD_DATA"),
        source="COCONut-B segments with isthing=1 projected to the original COCO category IDs",
        panel_frozen_from_fit=True, image_order_seed=90931102, eligible_images_with_labels=len(rows),
        first_qualifying_size=choice, shard_intersections=overlaps, support=details)
    name = ("LARGER_POOL_SUPPORT_DIAGNOSTIC.json" if args.larger_pool_diagnostic else
            ("PARTIAL_SHARD_QA.json" if args.partial else "COCONUT_LABEL_SUPPORT.json"))
    if args.larger_pool_diagnostic:
        receipt["status"] = "LARGER_POOL_LABEL_SUPPORT_DIAGNOSTIC"
        # Exact first supported prefix: 11 is monotone in the fixed ID order.
        # Evaluate ESS and all other gates on the first prefix meeting every count.
        y_all = np.stack([rows[i] for i in selected])
        required = []
        for (t, a, b) in mask["panel_before_confirm_support"]:
            for p in (a, b):
                keys = 2 * y_all[:, t] + y_all[:, p]
                minima = ((2, 50), (1, 50), (3, 30), (0, 200))
                size = max(int(np.flatnonzero(keys == cell)[minimum - 1] + 1)
                           for cell, minimum in minima)
                required.append(dict(target=names[t], partner=names[p], minimum_prefix_from_counts=size))
        size = max(r["minimum_prefix_from_counts"] for r in required)
        result = [support(y_all[:size], dev, t, p, names)
                  for (t, a, b) in mask["panel_before_confirm_support"] for p in (a, b)]
        receipt["exact_first_count_eligible_prefix"] = size
        receipt["exact_first_prefix_passes_ess"] = all(r["passed"] for r in result)
        receipt["exact_first_prefix_pairs"] = result
        receipt["pair_minimum_sizes"] = required
    (folder / name).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if args.partial or args.larger_pool_diagnostic or choice is None:
        return
    # This order fixes membership. Pixel retry uses these same IDs.
    ids = selected[:choice]
    np.savez_compressed(folder / "CONF_M_LABEL_CANDIDATE.npz", ids=ids, classes=names,
                        truth=np.stack([rows[i] for i in ids]))
    with (folder / "CONF_M_PIXEL_REQUEST.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "width", "height", "flickr_id", "zip_member"))
        writer.writerows((i, *sources[i], f"unlabeled2017/{i:012d}.jpg") for i in ids)


if __name__ == "__main__":
    main()
