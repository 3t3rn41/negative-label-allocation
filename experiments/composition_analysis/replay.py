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

"""CPU replay of the sealed v9 matrices for Z1, Z2, Z4 and Z6."""

import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
V9 = HERE.parent
ROOT = V9.parent
S0 = V9 / "coco_fixed_partner"
AB = V9 / "coco_crossed"
VOC = V9 / "voc_crossed"
ROLE = V9 / "voc_design"
P0 = V9 / "coco_design"
G1 = ROOT / "inputs/reference_coco"
sys.path.insert(0, str(HERE / ".deps"))
from numba import njit, prange, set_num_threads, __version__ as numba_version
sys.path.insert(0, str(ROOT / "metrics"))
from pr_metrics import Metric

N_BOOT = 5000
GROUPS = ("00", "01", "10", "11")
CELLS = ((2, 0), (2, 1), (3, 0), (3, 1))
PAIR_MAP = json.loads((G1 / "pairs.json").read_text())["settings"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name, rows):
    with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


@njit(parallel=True, cache=True)
def paired_group_auc(weights, order, groups, starts):
    """Count wins and half ties in ascending-score order with integer weights."""
    out = np.empty((len(weights), 5), dtype=np.float64)
    for b in prange(len(weights)):
        c0 = c1 = c2 = c3 = 0
        w20 = w21 = w30 = w31 = 0
        for k in range(len(starts) - 1):
            g0 = g1 = g2 = g3 = 0
            for j in range(starts[k], starts[k + 1]):
                g = groups[j]
                w = np.int64(weights[b, order[j]])
                if g == 0:
                    g0 += w
                elif g == 1:
                    g1 += w
                elif g == 2:
                    g2 += w
                else:
                    g3 += w
            w20 += g2 * (2 * c0 + g0)
            w21 += g2 * (2 * c1 + g1)
            w30 += g3 * (2 * c0 + g0)
            w31 += g3 * (2 * c1 + g1)
            c0 += g0
            c1 += g1
            c2 += g2
            c3 += g3
        out[b, 0] = w20 / (2 * c2 * c0) if c2 * c0 else np.nan
        out[b, 1] = w21 / (2 * c2 * c1) if c2 * c1 else np.nan
        out[b, 2] = w30 / (2 * c3 * c0) if c3 * c0 else np.nan
        out[b, 3] = w31 / (2 * c3 * c1) if c3 * c1 else np.nan
        denominator = 2 * (c2 + c3) * (c0 + c1)
        out[b, 4] = (w20 + w21 + w30 + w31) / denominator if denominator else np.nan
    return out


def replay_auc(score, g, weights):
    valid = np.flatnonzero(g >= 0)
    order = valid[np.argsort(score[valid], kind="stable")]
    starts = np.r_[0, np.flatnonzero(np.diff(score[order]) != 0) + 1, len(order)]
    return paired_group_auc(weights, order, g[order], starts)


def group_codes(gold, target, partner, voc=False):
    a, b = gold[:, target], gold[:, partner]
    if voc:
        return np.where((a != 0) & (b != 0), 2*(a == 1) + (b == 1), -1).astype(np.int8)
    return 2*a + b


def bootstrap_for(stage, n, limit):
    rules = read(AB / "RULES.json")
    voc_rules = read(VOC / "RULES.json")
    sources = {
        "S0": (S0 / "analysis/SHARED_IMAGE_BOOTSTRAP.npy", read(S0 / "PROTOCOL_LOCK.json")["bootstrap_seed"]),
        "A": (AB / "A_analysis/A_SHARED_WEIGHTS.npy", rules["a_bootstrap_seed"]),
        "B": (AB / "B_analysis/B_SHARED_WEIGHTS.npy", rules["b_bootstrap_seed"]),
        "VOC": (VOC / "BOOTSTRAP_IMAGE_WEIGHTS.npy", voc_rules["bootstrap_seed"]),
    }
    path, seed = sources[stage]
    w = np.load(path, mmap_mode="r")
    assert w.shape == (N_BOOT + 1, n) and w.dtype == np.uint16 and (w[0] == 1).all()
    rng = np.random.default_rng(seed)
    for index in range(1, len(w)):
        recreated = np.bincount(rng.integers(n, size=n), minlength=n)
        np.testing.assert_array_equal(w[index], recreated)
    n_streams = 13 if stage == "S0" else 1
    streams = np.stack([rng.integers(12, size=(N_BOOT, 12)) for _ in range(n_streams)])
    if stage == "VOC":
        np.testing.assert_array_equal(streams[0], np.load(VOC / "BOOTSTRAP_INIT_INDICES.npy"))
    return w[:limit+1], streams[:, :limit]


def design_for(stage, classes):
    if stage == "VOC":
        role = read(ROLE / "DATA_ROLE_LOCK.json")
        return {
            key: [(classes.index(t), classes.index(p), next(
                row["dev_positive_partner_fraction"] for row in role["composition"]
                if row["key"] == f"{key}:{t}/{p}")) for t, p in role["pairs"][key]]
            for key in ("P1", "P2")
        }
    if stage == "S0":
        ratios = read(P0 / "HISTORICAL_METRICS_AND_POWER.json")["history"]["settings"]["S0"]["pstar"]
        return {"S0": [(p["target"], p["partner"], r) for p, r in zip(PAIR_MAP["S0"], ratios, strict=True)]}
    rows = read(AB / "DATA_ROLE_LOCK.json")["pools"][stage]["support"]
    return {
        key: [(p["target"], p["partner"], row["pstar"]) for p, row in zip(
            PAIR_MAP[key], [row for row in rows if row["setting"] == key], strict=True)]
        for key in (("S0",) if stage == "A" else ("S0", "N1"))
    }


def auc_hybrid(g, first, second):
    positive, negative = first[g == 2], second[g == 1]
    if len(positive) == 0 or len(negative) == 0:
        return np.nan
    ranks = rankdata(np.concatenate((positive, negative)), method="average")
    n = len(positive)
    return (ranks[:n].sum() - n*(n+1)/2) / (n*len(negative))


def draw_effect(effect, indices):
    return effect[np.arange(1, len(effect))[:, None], indices].mean(axis=1)


def bounds(values):
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (None, None)
    return tuple(map(float, np.quantile(values, (.025, .975))))


def summarize(x, indices):
    sampled = draw_effect(x, indices)
    lo, hi = bounds(sampled)
    return dict(point_pp=float(x[0].mean()), low_pp=lo, high_pp=hi, valid=int(np.isfinite(sampled).sum()))


def root(intercept, slope):
    return np.divide(-intercept, slope, out=np.full_like(intercept, np.nan), where=np.abs(slope) > 1e-12)


def zero_band(boot_bg, boot_slope):
    """Find percentile zero acceptance on a fixed grid, including disconnected regions.

    Quantile signs can only change when a sampled affine line crosses zero,
    or when linear interpolation of the two order statistics crosses zero.
    A 0.001 grid gives a descriptive projection of the pointwise band.
    """
    grid = np.linspace(0, 1, 1001)
    inside = np.zeros(len(grid), dtype=bool)
    for section in np.array_split(np.arange(len(grid)), 20):
        lines = boot_bg[:, None] + boot_slope[:, None]*grid[section]
        lo, hi = np.quantile(lines, [.025, .975], axis=0)
        inside[section] = (lo <= 0) & (hi >= 0)
    transitions = np.diff(np.r_[False, inside, False].astype(int))
    return [[float(grid[a]), float(grid[b-1])] for a, b in zip(
        np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1), strict=True)]


def run_stage(stage, limit):
    started = perf_counter()
    inputs = {
        "S0": (S0 / "S0_SCORES.npz", P0 / "A1/CONFIRM_GOLD.npz", "S0_PLC", [("S0_TRUE", "S0_PLC")]),
        "A": (AB / "A_SCORES.npz", AB / "A_GOLD.npz", "S0_PLC", [("S0_TRUE", "S0_PLC"), ("S0_TRUE", "LOW")]),
        "B": (AB / "B_SCORES.npz", AB / "B_GOLD.npz", "S0_PLC", [("S0_TRUE", "S0_PLC"), ("N1_TRUE", "S0_PLC")]),
        "VOC": (VOC / "SCORES.npz", ROLE / "TEST_GOLD.npz", "PLC", [("P1_TRUE", "PLC"), ("P2_TRUE", "PLC")]),
    }
    score_path, truth_path, plc, contrasts = inputs[stage]
    with np.load(score_path) as s, np.load(truth_path) as g:
        ids, classes = s["ids"], s["classes"].tolist()
        np.testing.assert_array_equal(ids, g["ids"])
        np.testing.assert_array_equal(classes, g["classes"])
        gold, scores = g["truth"], s["scores"]
        keys = list(zip(s["seeds"].tolist(), s["draws"].tolist(), s["arms"].tolist()))
    seeds, masks, arms = map(sorted, map(set, zip(*keys)))
    assert len(seeds) == 12 and len(masks) == 2 and len(set(keys)) == len(keys) == 24*len(arms)
    assert len(set(ids)) == len(ids) and scores.shape == (len(keys), len(ids), len(classes))
    assert np.isfinite(scores).all() and scores.dtype == np.float32
    w, streams = bootstrap_for(stage, len(ids), limit)
    indices = streams[0]
    designs = design_for(stage, classes)
    effects = np.empty((len(w), 12, 2, len(arms), len(designs), 3, 4))
    effects.fill(np.nan)
    shifts, compositions, endpoint_cells = [], [], []
    used = {arm for pair in contrasts for arm in pair}
    lookup = dict(zip(keys, range(len(keys)), strict=True))
    max_reference_error = 0.
    subset = np.unique(np.linspace(0, limit, min(limit+1, 9), dtype=int))
    for evaluation, (setting, pairs) in enumerate(designs.items()):
        for j, (t, p, pstar) in enumerate(pairs):
            group = group_codes(gold, t, p, stage == "VOC")
            n00, n01, n10, n11 = (int((group == code).sum()) for code in range(4))
            r_nat, p_nat = n01/(n00+n01), n11/(n10+n11)
            composition = dict(stage=stage, evaluation=setting, target=classes[t], partner=classes[p],
                               pstar=pstar, p_natural=p_nat, r_natural=r_nat,
                               n00=n00, n01=n01, n10=n10, n11=n11, excluded=int((group < 0).sum()))
            compositions.append(composition)
            for k, ((seed, mask, arm), matrix) in enumerate(zip(keys, scores, strict=True)):
                ii, mm, aa = seeds.index(seed), masks.index(mask), arms.index(arm)
                current = matrix[:, t]
                reference = scores[lookup[seed, mask, plc], :, t]
                if arm in used:
                    cell = replay_auc(current, group, w)
                    for c, (pos, neg) in enumerate((*CELLS, (-1, -1))):
                        positive = group == pos if pos != -1 else group >= 2
                        negative = group == neg if neg != -1 else (group >= 0) & (group <= 1)
                        labels = np.where(positive, 1, np.where(negative, 0, -1))
                        oracle = Metric(current, labels).compute_auc(w[subset])
                        np.testing.assert_allclose(cell[subset, c], oracle, atol=1e-12, rtol=0)
                        max_reference_error = max(max_reference_error, float(np.nanmax(np.abs(cell[subset,c]-oracle))))
                    bg = (1-pstar)*cell[:, 0]+pstar*cell[:, 2]
                    partner = (1-pstar)*cell[:, 1]+pstar*cell[:, 3]
                    effects[:, ii, mm, aa, evaluation, j] = np.stack((cell[:, 1], cell[:, 4], bg, partner-bg), axis=1)
                    endpoint_cells.append(dict(stage=stage, evaluation=setting, target=classes[t], partner=classes[p],
                                               seed=seed, mask=mask, arm=arm, auc_10_00=cell[0,0], auc_10_01=cell[0,1],
                                               auc_11_00=cell[0,2], auc_11_01=cell[0,3], natural_auc=cell[0,4]))
                if arm != plc:
                    s00, s10, s01, s11 = [100*auc_hybrid(group, a, b) for a, b in (
                        (reference, reference), (current, reference), (reference, current), (current, current))]
                    phi10 = ((s10-s00)+(s11-s01))/2
                    phi01 = ((s01-s00)+(s11-s10))/2
                    assert abs((phi10+phi01)-(s11-s00)) < 1e-12
                    if arm in used:
                        assert abs(s11 - 100*effects[0,ii,mm,aa,evaluation,j,0]) < 1e-12
                    shifts.append(dict(stage=stage, evaluation=setting, target=classes[t], partner=classes[p],
                                       seed=seed, mask=mask, arm=arm, plc=plc,
                                       **{f"score_shift_{name}": float((current[group == c].astype(float)
                                          - reference[group == c]).mean()) for c, name in enumerate(GROUPS)},
                                       auc_plc_pp=s00, auc_only_10_replaced_pp=s10, auc_only_01_replaced_pp=s01,
                                       auc_arm_pp=s11, contribution_10_pp=phi10, contribution_01_pp=phi01,
                                       conditional_delta_pp=s11-s00))
        print(stage, setting, "elapsed", round(perf_counter()-started, 1), flush=True)
    return dict(stage=stage, indices=indices, streams=streams, contrasts=contrasts, design=designs,
                classes=classes, masks=masks, seeds=seeds, arms=arms, effects=effects,
                shifts=shifts, compositions=compositions, endpoint_cells=endpoint_cells,
                qa=dict(stage=stage, endpoints=len(keys), images=len(ids), reference_max_abs=max_reference_error,
                        weight_file_verified=True, image_replicates=limit, seconds=perf_counter()-started))

def cache_replay(stages, limit):
    for stage in stages:
        result = run_stage(stage, limit)
        print(result["qa"], flush=True)
        if limit == N_BOOT:
            destination = HERE / (stage + "_REPLAY.npz")
            keys = ("effects", "indices", "streams")
            np.savez_compressed(destination, **{key: result.pop(key) for key in keys})
            save(stage + "_REPLAY.json", result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("S0", "A", "B", "VOC"), default=("S0", "A", "B", "VOC"))
    parser.add_argument("--replicates", type=int, choices=(8, N_BOOT), default=N_BOOT)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    set_num_threads(args.threads)
    print("Python", sys.version.split()[0], "NumPy", np.__version__, "Numba", numba_version, flush=True)
    cache_replay(args.stages, args.replicates)

