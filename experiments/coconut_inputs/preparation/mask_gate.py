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

"""FIT-only partner screening and deterministic v9.3 swap feasibility."""

import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver"))
from solve_masks import Constraints, neutral_placebo, stratify

ARMS = ("LOW", "DIR1", "DIR2", "PLC")
MASK_SEEDS = tuple(range(90931201, 90931209))


def load_inputs(expand=False):
    with np.load(HERE / "inputs/historical_ids.npz", allow_pickle=False) as data:
        fit, dev, ids, classes = (data[key] for key in ("fit_truth", "dev_truth", "fit", "classes"))
    cosine = np.load(HERE / "inputs/FIT_COSINE.npy", allow_pickle=False)
    assert cosine.shape == fit.shape and np.isfinite(cosine).all()
    if expand:
        with np.load(HERE / "inputs/EXPANDED_DEV_28595.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["classes"], classes)
            assert len(data["ids"]) == 28595 and not set(ids).intersection(data["ids"])
            dev = data["truth"]
    history = json.loads((HERE / "inputs/PANEL_HISTORICAL_EXPOSURE.json").read_text(encoding="utf-8"))
    return ids, classes.tolist(), fit, dev, cosine, history["v9_partner_interventions"]


def choose_quotas(rows, dose, attempt):
    common = np.array([min(map(len, pools)) for _, _, pools in rows])
    if common.sum() < dose:
        return None
    weights = common.astype(np.float64)
    if attempt:
        weights *= np.where(np.arange(len(rows)) // 3 == 9, [.1, 0][attempt - 1], 1)
        if common[weights > 0].sum() < dose:
            return None
    desired = dose * weights / weights.sum()
    quotas = np.zeros(len(rows), dtype=int)
    for _ in range(dose):
        available = np.where((quotas < common) & (weights > 0), desired - quotas, -np.inf)
        quotas[int(np.argmax(available))] += 1
    assert quotas.sum() == dose and np.all(quotas <= common)
    return quotas


def solve_target(ids, truth, cosine, pair, layers, dose, seed):
    t, p1, p2 = pair
    s = cosine[:, t].astype(np.float64)
    negatives = s[truth[:, t] == -1]
    sd = negatives.std()
    assert sd > 0
    scale = (s - negatives.mean()) / sd
    tail = s > np.quantile(negatives, .90)
    reason = "CAPACITY"
    for attempt in range(3):
        quotas = choose_quotas(layers, dose, attempt)
        if quotas is None:
            continue
        tie, solver = map(np.random.default_rng, np.random.SeedSequence([seed, t, p1, p2, dose, attempt]).spawn(2))
        placebo = neutral_placebo(layers, quotas, scale, ids, tie)
        placebo_rows = np.concatenate(placebo)
        gap = int(np.ceil(.20 * dose))
        if any((truth[placebo_rows, p] == 1).sum() > dose - gap for p in (p1, p2)):
            reason = "PLC_DIRECTION"
            continue
        chosen = [[] for _ in range(3)]
        groups = []
        size = 0
        for (_, _, pools), n in zip(layers, quotas, strict=True):
            active = []
            for i, pool in enumerate(pools[:3]):
                rows = solver.permutation(pool) if n else np.empty(0, dtype=int)
                columns = np.arange(size, size + len(rows))
                size += len(rows)
                active.append((columns, rows, int(n)))
                chosen[i].append((columns, rows))
            groups.extend(x for x in active if x[2])
        problem = Constraints(size)
        for columns, _, n in groups:
            problem.add(columns, np.ones(len(columns)), n, n)
        all_cols, all_rows = zip(*[(np.concatenate([a for a, _ in parts]),
                                    np.concatenate([b for _, b in parts])) for parts in chosen])
        allowed = (.10 - 2e-7) * dose
        plc_sum = scale[placebo_rows].sum()
        for columns, rows in zip(all_cols, all_rows):
            problem.add(columns, scale[rows], plc_sum - allowed, plc_sum + allowed)
        for a, b in combinations(range(3), 2):
            problem.add(np.r_[all_cols[a], all_cols[b]], np.r_[scale[all_rows[a]], -scale[all_rows[b]]],
                        -allowed, allowed)
        for i, cross_partner in ((1, p2), (2, p1)):
            members = all_rows[i]
            problem.add(all_cols[i], (truth[members, cross_partner] == 1).astype(float), 0, dose - gap)
            problem.add(all_cols[i], tail[members].astype(float), tail[placebo_rows].sum(), tail[placebo_rows].sum())
        result = problem.solve(np.zeros(size))
        reason = str(result.message)
        if result.x is None:
            continue
        selected = np.rint(result.x).astype(bool)
        assert np.max(abs(result.x - selected)) < 1e-6
        layer_coordinates = [
            (d, k, [chosen[j][i][1][selected[chosen[j][i][0]]] for j in range(3)] + [placebo[i]])
            for i, (d, k, _) in enumerate(layers)
        ]
        return layer_coordinates, "PASS"
    return None, reason


def cells(yt, yp):
    result = np.bincount(2 * yt + yp, minlength=4)
    return [int(result[k]) for k in (2, 1, 3, 0)]


def partner_candidates(classes, fit, dev, interventions):
    prior = {t for _, t, _ in interventions}
    partners = {p for _, _, p in interventions}
    by_target = {}
    for t, target in enumerate(classes):
        if target == "chair":
            continue
        dev_support = {p: cells(dev[:, t], dev[:, p]) for p in range(len(classes)) if p != t}
        qualified = [p for p, row in dev_support.items()
                     if all(n >= m for n, m in zip(row, (50, 50, 30, 200)))]
        nominations = []
        for a, b in combinations(qualified, 2):
            first = int(((fit[:, t] == 0) & (fit[:, a] == 1) & (fit[:, b] == 0)).sum())
            second = int(((fit[:, t] == 0) & (fit[:, a] == 0) & (fit[:, b] == 1)).sum())
            nominations.append(dict(pair=[t, a, b], exclusive=[first, second],
                                    dev_cells=[dev_support[a], dev_support[b]]))
        nominations.sort(key=lambda item: (-min(item["exclusive"]), item["pair"][1:]))
        by_target[t] = dict(target=target, prior_target=target in prior, prior_partner=target in partners,
                            dev_frequency=int(dev[:, t].sum()), nominations=nominations)
    return by_target


def evaluate_draw(fit, score, low, ids, panel, dose, draw):
    selections = {}
    truth = 2 * fit - 1
    layers = list(stratify(ids, truth, score, low, panel))
    for pair, target_layers, d in zip(panel, layers, dose, strict=True):
        found, reason = solve_target(ids, truth, score, pair, target_layers, d, draw)
        if found is None:
            return None, f"{pair[0]}: {reason}"
        selections[pair[0]] = found
    return selections, "PASS"


def feasibility(ids, classes, fit, dev, score, interventions):
    by_target = partner_candidates(classes, fit, dev, interventions)
    masks = {seed: np.where(np.random.default_rng(seed).random(fit.shape) < .10, fit, -1)
             for seed in MASK_SEEDS}
    preliminary = {}
    for t, record in by_target.items():
        record["tests"] = []
        for row in record["nominations"]:
            pair = row["pair"]
            for dose in (40, 20):
                if min(row["exclusive"]) < dose // 5:
                    continue
                statuses = []
                for draw in MASK_SEEDS[:6]:
                    _, status = evaluate_draw(fit, score, masks[draw], ids, [pair], [dose], draw)
                    statuses.append(status)
                    if status != "PASS":
                        break
                record["tests"].append(dict(pair=pair, dose=dose, statuses=statuses))
                if len(statuses) == 6 and set(statuses) == {"PASS"}:
                    preliminary[t] = dict(**row, dose=dose, prior_target=record["prior_target"],
                                          dev_frequency=record["dev_frequency"])
                    break
            if t in preliminary:
                break
        if t in preliminary:
            print(classes[t], preliminary[t], flush=True)
    return by_target, preliminary, masks


def target_order(preliminary):
    freq_order = sorted(preliminary, key=lambda t: (preliminary[t]["dev_frequency"], t))
    layers = [list(group) for group in np.array_split(freq_order, 3)]
    rng = np.random.default_rng(90931101)
    ordered = {}
    for key, members in zip(("low", "mid", "high"), layers, strict=True):
        draw = [int(t) for t in rng.permutation(members)]
        ordered[key] = sorted(draw, key=lambda t: preliminary[t]["prior_target"])
    novel_layers = {key: [t for t in layer if not preliminary[t]["prior_target"]]
                    for key, layer in ordered.items()}
    new = []
    earlier = []
    for role in ("high", "mid", "low"):
        new.extend(novel_layers[role][:2])
    for j in range(2, max(map(len, novel_layers.values()), default=0)):
        for role in ("mid", "high", "low"):
            if j < len(novel_layers[role]):
                new.append(novel_layers[role][j])
    for role in ("mid", "high", "low"):
        earlier.extend(t for t in ordered[role] if preliminary[t]["prior_target"])
    return new, earlier, ordered


def density(masks):
    metrics = []
    for a, b in combinations(ARMS, 2):
        x, y = [(masks[arm] >= 0).sum(1) for arm in (a, b)]
        active = abs(int((x > 0).sum()) - int((y > 0).sum()))
        tv = abs(np.bincount(x, minlength=21) - np.bincount(y, minlength=21)).sum() / (2 * len(x))
        metrics.append(dict(arms=[a, b], active=active, density_tv=float(tv)))
    return metrics


def materialize(low, selections):
    result = {arm: low.copy() for arm in ARMS}
    for target, layers in selections.items():
        for _, _, (remove, *adds) in layers:
            for arm, add in zip(ARMS[1:], adds, strict=True):
                result[arm][remove, target] = -1
                result[arm][add, target] = 0
    return result


def joint_assay(ids, fit, score, base_masks, panel):
    first_failure = None
    vectors = list(np.ndindex(*(2,) * len(panel)))
    vectors.sort(key=lambda q: (sum(q), tuple(sorted(zip([p[0] for p in panel], q)))))
    for q in vectors:
        doses = [40 - 20 * bit for bit in q]
        accepted = {}
        for draw in MASK_SEEDS:
            low = base_masks[draw]
            choices, how = evaluate_draw(fit, score, low, ids, panel, doses, draw)
            if choices is None:
                first_failure = how
                if draw in MASK_SEEDS[:6]:
                    break
                continue
            metrics = density(materialize(low, choices))
            if any(row["active"] > 20 or row["density_tv"] > .01 for row in metrics):
                first_failure = f"{draw}: joint density or active-image bound"
                if draw in MASK_SEEDS[:6]:
                    break
                continue
            accepted[draw] = (choices, metrics)
        if set(MASK_SEEDS[:6]).issubset(accepted):
            return doses, accepted, None
    return None, {}, first_failure


def write_result(folder, ids, names, panel, doses, accepted, base_masks):
    coordinate_file = folder / "MASK_COORDINATES.csv"
    with coordinate_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "target", "difficulty", "density", "operation", "id"))
        for seed, (choices, _) in accepted.items():
            for target, groups in choices.items():
                for decile, count, members in groups:
                    for operation, positions in zip(("REMOVE", "DIR1", "DIR2", "PLC"), members, strict=True):
                        writer.writerows((seed, target, decile, count, operation, int(ids[i])) for i in positions)
            np.savez_compressed(folder / f"MASK_{seed}.npz", ids=ids, classes=names,
                                **materialize(base_masks[seed], choices))
    store(folder / "FIT_MASK_MATRIX.json", dict(pairs=panel, doses=doses, feasible_seeds=list(accepted),
          density={seed: metrics for seed, (_, metrics) in accepted.items()}))


def store(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    expand = "--expanded-dev" in sys.argv
    folder = HERE / ("mask_results_v3" if expand else (
        "mask_results_final" if "--all-history" in sys.argv else (
            "mask_results_v2" if "--corrected-history" in sys.argv else "mask_results")))
    folder.mkdir(exist_ok=True)
    ids, classes, fit, dev, score, history = load_inputs(expand)
    screen, preliminary, base_masks = feasibility(ids, classes, fit, dev, score, history)
    new, previously_used, strata = target_order(preliminary)
    store(folder / "FIT_PAIR_SCREEN.json", dict(classes=classes, all_candidates=screen,
         paired_capacity_candidates=preliminary, new_target_order=new, previously_used_order=previously_used,
         frequency_layers=strata))
    print("new order", [classes[t] for t in new], "previously used", [classes[t] for t in previously_used], flush=True)
    if len(new) < 4:
        store(folder / "MASK_GATE_STATUS.json", dict(status="HOLD_MASK", eligible_new_targets=len(new),
                                                   reason="fewer than four newly intervened targets on FIT and DEV"))
        return
    panel = [preliminary[t]["pair"] for t in new[:6]]
    doses, accepted, reason = joint_assay(ids, fit, score, base_masks, panel)
    if doses is None:
        store(folder / "MASK_GATE_STATUS.json", dict(status="HOLD_MASK", eligible_new_targets=len(new),
                                                   reason=reason))
        return
    write_result(folder, ids, classes, panel, doses, accepted, base_masks)
    store(folder / "MASK_GATE_STATUS.json", dict(status="FIT_MASKS_CONSTRUCTED",
          eligible_new_targets=len(new), panel_before_confirm_support=panel,
          reserve=[preliminary[t]["pair"] for t in new[6:8]], doses=doses, masks=len(accepted)))


if __name__ == "__main__":
    main()
