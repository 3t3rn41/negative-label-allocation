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

"""Build the KBS manuscript figures from the archived v9 results.

Run with Python, Matplotlib, and NumPy from the fixed-query workspace.
The experiment workspace is resolved relative to this file.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.font_manager as fonts
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MultipleLocator


OUTPUT = Path(__file__).resolve().parent
DATA = OUTPUT.parent / "experiments"
SUPPLEMENT = "composition_analysis"
GPU = "readout_comparisons"
COCO = "coco_crossed"
VOC = "voc_crossed"
PANOPTIC = "coconut_panoptic/formal"

INK = "#182631"
BLUE = "#006896"
LIGHT_BLUE = "#e0eff6"
ORANGE = "#9c460c"
LIGHT_ORANGE = "#f8eadc"
GRAY = "#536372"
LINE = "#a7b5bf"
GRID = "#e6ebef"
WHITE = "#ffffff"
WIDTH_MM = 168


def configure_style():
    family = next(name for name in ("Arial", "Liberation Sans", "DejaVu Sans")
                  if name in {font.name for font in fonts.fontManager.ttflist})
    matplotlib.rcParams.update({
        "font.family": family,
        "font.size": 8.0,
        "font.weight": "normal",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "text.color": INK,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "legend.fontsize": 8,
    })


def read_csv(relative):
    with (DATA / relative).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def read_json(relative):
    with (DATA / relative).open(encoding="utf-8") as source:
        return json.load(source)


def single(rows, **criteria):
    matches = [row for row in rows if all(row[key] == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one source row for {criteria}, got {len(matches)}")
    return matches[0]


def new_figure(height_mm):
    return plt.figure(figsize=(WIDTH_MM / 25.4, height_mm / 25.4))


def panel(fig, x, y, label, title):
    fig.text(x, y, f"({label})  {title}", fontsize=9.2, fontweight="bold", va="top")


def save(fig, filename, title):
    fig.savefig(OUTPUT / filename, metadata={"Title": title, "CreationDate": None})
    plt.close(fig)


def diagram_axes(fig):
    ax = fig.add_axes((0, 0, 1, 1), frameon=False)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return ax


def label(ax, x, y, text, *, weight="normal", color=INK, size=8, align="left"):
    ax.text(x, y, text, ha=align, va="center", fontsize=size, color=color, fontweight=weight)


def box(ax, left, bottom, width, height, *, fill=WHITE, edge=LINE, linewidth=0.7):
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor=fill,
                           edgecolor=edge, linewidth=linewidth))


def arrow(ax, x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                color=color, mutation_scale=8, lw=0.9))


def design_figure():
    fig = new_figure(81)
    ax = diagram_axes(fig)
    panel(fig, 0.018, 0.963, "a", "Equal-budget negative-label reallocation")
    panel(fig, 0.573, 0.963, "b", "Crossed training and evaluation")

    box(ax, 0.018, 0.729, 0.510, 0.123, fill="#f1f5f7")
    label(ax, 0.273, 0.801, "LOW: original visible-label mask", weight="bold", align="center")
    label(ax, 0.273, 0.752, "Select known background negatives to hide", align="center")

    ax.plot([0.273, 0.273], [0.729, 0.662], lw=0.9, c=GRAY)
    ax.plot([0.143, 0.405], [0.662, 0.662], lw=0.9, c=GRAY)
    arrow(ax, 0.143, 0.662, 0.143, 0.584, BLUE)
    arrow(ax, 0.405, 0.662, 0.405, 0.584, GRAY)

    box(ax, 0.018, 0.333, 0.248, 0.238, fill=LIGHT_BLUE, edge=BLUE)
    label(ax, 0.142, 0.521, "TRUE", weight="bold", color=BLUE, align="center", size=8.8)
    label(ax, 0.142, 0.429, "Reveal an equal number\nof target negatives on\npartner-only images", align="center")

    box(ax, 0.282, 0.333, 0.246, 0.238, fill="#f4f6f8", edge=GRAY)
    label(ax, 0.405, 0.521, "PLC", weight="bold", align="center", size=8.8)
    label(ax, 0.405, 0.429, "Reveal an equal number\nof target negatives from\nmatched strata", align="center")

    box(ax, 0.018, 0.084, 0.510, 0.165, fill=WHITE)
    label(ax, 0.273, 0.205, "Shared images and observed positives", weight="bold", align="center")
    label(ax, 0.273, 0.130, "Matched counts, difficulty and annotation density", align="center")

    label(ax, 0.829, 0.814, "Evaluation partner", weight="bold", align="center")
    left, bottom, cw, ch = 0.674, 0.337, 0.153, 0.160
    for col, heading in enumerate(("Partner 1", "Partner 2")):
        label(ax, left + (col + 0.5) * cw, bottom + 2 * ch + 0.069, heading, align="center")
    for row, heading in enumerate(("Train on\npartner 1", "Train on\npartner 2")):
        y = bottom + (1 - row) * ch
        label(ax, 0.656, y + ch / 2, heading, align="right")
        for col in range(2):
            same = row == col
            x = left + col * cw
            box(ax, x, y, cw, ch, fill=LIGHT_BLUE if same else WHITE)
            label(ax, x + cw / 2, y + ch / 2, "Matched" if same else "Crossed",
                  weight="bold" if same else "normal", color=BLUE if same else INK, align="center")

    label(ax, 0.800, 0.229, "Shared evaluation images and PLC", align="center")
    label(ax, 0.800, 0.149, "Subtract crossed from matched in each column", align="center")
    label(ax, 0.800, 0.090, "Average the two column contrasts", align="center")
    save(fig, "design.pdf", "Equal-budget negative-label reallocation and crossed evaluation")


def matrix_from_csv(relative, codes):
    rows = read_csv(relative)
    matrix = []
    for train in codes:
        matrix.append([float(single(rows, train=train, evaluation=evaluation,
                                    group="aggregate", metric="conditional")["mean_pp"])
                       for evaluation in codes])
    return np.asarray(matrix)


def matrix_from_vit(result, codes):
    return np.asarray([[single(result["cross"], train=f"{train}_TRUE", evaluation=evaluation)
                        ["aggregate"]["conditional"]["point_pp"] for evaluation in codes]
                       for train in codes])


def signed(value, precision=3):
    return f"{value:+.{precision}f}".replace("-", "−")


def luminance(color):
    srgb = np.asarray(colors.to_rgb(color))
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.asarray([0.2126, 0.7152, 0.0722])


def cross_figure():
    fig = new_figure(71)
    ax = diagram_axes(fig)
    s0_codes, voc_codes = ("S0", "N1"), ("P1", "P2")
    vit = read_json(f"{GPU}/T5/T5_RESULT.json")
    configs = (
        ("a", "COCO B / RN50", s0_codes, matrix_from_csv(f"{COCO}/B_CROSS_RESULTS.csv", s0_codes)),
        ("b", "VOC / ViT-B/16", voc_codes, matrix_from_csv(f"{VOC}/CROSS_RESULTS.csv", voc_codes)),
        ("c", "COCO B / ViT-B/16", s0_codes, matrix_from_vit(vit, s0_codes)),
    )
    cmap = colors.LinearSegmentedColormap.from_list("ranking", [ORANGE, WHITE, BLUE])
    normalize = colors.CenteredNorm(vcenter=0, halfrange=2.5)
    cw, ch = 0.102, 0.190
    bottom = 0.330
    for (letter, title, codes, matrix), start in zip(configs, (0.023, 0.351, 0.679), strict=True):
        panel(fig, start, 0.963, letter, title)
        left = start + 0.060
        label(ax, left + cw, 0.851, "Evaluation partner", align="center")
        for col, code in enumerate(codes):
            label(ax, left + (col + 0.5) * cw, 0.758, code, align="center", weight="bold")
        for row, code in enumerate(codes):
            y = bottom + (1 - row) * ch
            label(ax, left - 0.012, y + ch / 2, code, align="right", weight="bold")
            for col, value in enumerate(matrix[row]):
                match = row == col
                x = left + col * cw
                fill = cmap(normalize(value))
                box(ax, x, y, cw, ch, fill=fill, edge=WHITE, linewidth=1.3)
                if match:
                    ax.add_patch(Rectangle((x + 0.003, y + 0.009), cw - 0.006, ch - 0.018,
                                           facecolor="none", edgecolor=INK, linewidth=0.8))
                background = luminance(fill) + 0.05
                fg = WHITE if 1.05 / background > background / 0.05 else "#000000"
                label(ax, x + cw / 2, y + ch / 2, signed(value), color=fg,
                      weight="bold" if match else "normal", align="center", size=9)
        interaction = (matrix[0, 0] - matrix[1, 0] + matrix[1, 1] - matrix[0, 1]) / 2
        label(ax, left + cw, 0.262, f"Mean match advantage {signed(interaction)} pp", align="center")

    ax.text(0.022, bottom + ch, "Training partner", ha="center", va="center",
            rotation=90, fontsize=8)
    x, y, width, height = 0.344, 0.146, 0.330, 0.035
    color_steps = 55
    for index in range(color_steps):
        step_width = width / color_steps
        padding = 0.001 if index < color_steps - 1 else 0
        box(ax, x + index * step_width, y, step_width + padding, height,
            fill=cmap((index + 0.5) / color_steps), edge="none", linewidth=0)
    ax.add_patch(Rectangle((x, y), width, height, fill=False, edgecolor=LINE, linewidth=0.5))
    for tick, fraction in zip(("−2.5", "0", "+2.5"), (0, 0.5, 1), strict=True):
        label(ax, x + width * fraction, 0.116, tick, align="center")
    label(ax, x + width / 2, 0.047, "Conditional AUC change vs. PLC (pp)", align="center")
    save(fig, "interaction.pdf", "Partner-specific conditional AUC in crossed evaluations")


def composition_figure():
    fig = new_figure(74)
    panel(fig, 0.020, 0.957, "a", "COCO A1 composition curve")
    panel(fig, 0.566, 0.957, "b", "COCO A held-out evaluation")
    first = fig.add_axes((0.116, 0.210, 0.390, 0.570))
    second = fig.add_axes((0.659, 0.210, 0.303, 0.570))

    curve = [row for row in read_csv(f"{SUPPLEMENT}/Z2_curve.csv")
             if row["stage"] == "S0" and row["target"] == "aggregate"]
    curve.sort(key=lambda row: float(row["r"]))
    ratio = np.asarray([100 * float(row["r"]) for row in curve])
    point = np.asarray([float(row["point_pp"]) for row in curve])
    low = np.asarray([float(row["low_pp"]) for row in curve])
    high = np.asarray([float(row["high_pp"]) for row in curve])
    root_row = single(read_csv(f"{SUPPLEMENT}/Z2_roots.csv"), stage="S0", target="aggregate")
    root = 100 * float(root_row["root_in_domain"])

    for ax in (first, second):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.axhline(0, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.tick_params(length=2.6)
        ax.set_ylabel("Full AUC change vs. PLC (pp)", labelpad=4)
    first.fill_between(ratio, low, high, color=LIGHT_BLUE, linewidth=0)
    first.plot(ratio, point, c=BLUE, linewidth=1.5)
    first.plot(root, 0, marker="o", ls="none", color=BLUE, markerfacecolor=WHITE, markersize=4.4)
    first.annotate(f"Zero crossing\n{root:.2f}%", xy=(root, 0), xytext=(43, -0.30),
                   fontsize=8, ha="center", va="center",
                   arrowprops={"arrowstyle": "-", "lw": 0.7, "color": GRAY})
    first.set(xlim=(0, 100), ylim=(-0.53, 1.88), xlabel="Partner-only share of target negatives (%)")
    first.xaxis.set_major_locator(MultipleLocator(25))
    first.yaxis.set_major_locator(MultipleLocator(0.5))
    first.text(0.03, 0.94, "Pointwise 95% interval", transform=first.transAxes,
               fontsize=8, ha="left", va="top", color=GRAY)

    observed_rows = read_csv(f"{COCO}/A_RESULTS.csv")
    observations = [single(observed_rows, contrast="TRUE_MINUS_PLC", group="aggregate", mode=mode)
                    for mode in ("r05", "r50")]
    observed = np.asarray([float(row["observed_pp"]) for row in observations])
    predicted = np.asarray([float(row["predicted_pp"]) for row in observations])
    limits = np.asarray([[float(row["bootstrap_lower_95"]), float(row["bootstrap_upper_95"])]
                         for row in observations])
    x = np.asarray([0, 1])
    second.plot(x - 0.11, predicted, marker="D", linestyle="none", markersize=4.6,
                markerfacecolor=WHITE, color=ORANGE, label="Prediction")
    second.errorbar(x + 0.11, observed, yerr=(observed - limits[:, 0], limits[:, 1] - observed),
                    marker="o", markersize=4.8, capsize=2.6, linewidth=1.1,
                    linestyle="none", color=BLUE, label="Observed")
    second.set(xlim=(-0.42, 1.42), ylim=(-0.53, 1.88), xlabel="Partner-only share")
    second.set_xticks(x, ("5%", "50%"))
    second.yaxis.set_major_locator(MultipleLocator(0.5))
    second.legend(loc="upper left", frameon=False, handlelength=1.55, handletextpad=0.35, borderpad=0)
    save(fig, "composition.pdf", "Full AUC composition curve and held-out predictions")


def pretty_class(name):
    return {"diningtable": "Dining table", "pottedplant": "Potted plant",
            "tvmonitor": "TV/monitor"}.get(name, name.capitalize())


def forest(ax, labels, estimates):
    y = np.arange(len(labels))[::-1]
    for idx, (value, low, high) in enumerate(estimates):
        pooled = idx == 0
        ax.errorbar(value, y[idx], xerr=((value - low,), (high - value,)), fmt="D" if pooled else "o",
                    color=BLUE if pooled else GRAY, markerfacecolor=BLUE if pooled else WHITE,
                    markersize=5.0 if pooled else 4.1, linewidth=1.1, capsize=2.3, zorder=3)
    ax.set(xlim=(-0.55, 7.55), ylim=(-0.7, len(labels) - 0.3), xlabel="Match advantage (pp)")
    ax.set_yticks(y, labels)
    ax.get_yticklabels()[0].set_weight("bold")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.spines["left"].set_visible(True)
    ax.axvline(0, color=GRAY, linewidth=0.8, linestyle=(0, (4, 3)), zorder=2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=2.6)


def point_and_ci(result):
    return (result["point_pp"], *result["ci95_pp"])


def replication_figure():
    fig = new_figure(82)
    panel(fig, 0.018, 0.953, "a", "COCO target–partner pairs")
    panel(fig, 0.626, 0.953, "b", "VOC target heterogeneity")
    first = fig.add_axes((0.218, 0.170, 0.275, 0.654))
    second = fig.add_axes((0.692, 0.170, 0.275, 0.654))

    t1 = read_json(f"{GPU}/COCO/T1_RESULT.json")
    pairs = sorted(t1["per_pair"], key=lambda row: row["interaction"]["point_pp"], reverse=True)
    pair_names = ["Seven-pair mean"]
    pair_names.extend(f"{pretty_class(row['target'])} / {pretty_class(row['anchor'])}" for row in pairs)
    pair_stats = [point_and_ci(t1["interaction"])]
    pair_stats.extend(point_and_ci(row["interaction"]) for row in pairs)
    forest(first, pair_names, pair_stats)

    voc = read_csv(f"{VOC}/MATCHED_ADVANTAGE.csv")
    classes = ("aggregate", "chair", "pottedplant", "diningtable")
    voc_rows = [single(voc, group=target, component="I") for target in classes]
    voc_stats = [(float(row["mean_pp"]), float(row["lower95_two_sided_pp"]),
                  float(row["upper95_two_sided_pp"])) for row in voc_rows]
    forest(second, ["Three-target mean", *(pretty_class(target) for target in classes[1:])], voc_stats)
    save(fig, "replication.pdf", "Extended target–partner pairs and VOC target heterogeneity")


def panoptic_confirmation_figure():
    """Plot the sealed matching estimates and aggregate intervals for both readouts."""
    protocol = read_json(f"{PANOPTIC}/PROTOCOL_LOCK.json")
    target_names = {168: "Towel", 171: "Brick wall", 109: "Curtain", 193: "Grass"}
    names = ["Four-target mean", *(target_names[row["target_id"]] for row in protocol["targets"])]
    positions = np.array([4.60, 3.43, 2.42, 1.41, 0.40])

    fig = new_figure(65)
    ax = fig.add_axes((0.196, 0.208, 0.520, 0.664))
    estimates = fig.add_axes((0.734, 0.208, 0.255, 0.664), sharey=ax)
    ax.set(xlim=(-0.04, 4.18), ylim=(-0.10, 5.11), xticks=np.arange(5),
           yticks=positions, yticklabels=names, xlabel="Matching interaction, I (pp)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=0.65, color=GRID)
    ax.axvline(0, color=GRAY, linewidth=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=9)
    ax.get_yticklabels()[0].set_fontweight("bold")
    estimates.set_axis_off()

    for pane in (ax, estimates):
        pane.axhspan(4.10, 5.10, facecolor="#f1f5f7", linewidth=0, zorder=0)

    for model, name, color, marker, shift, column in (
            ("M1", "DualCoOp", BLUE, "o", 0.16, 0.24),
            ("M2", "Linear readout", ORANGE, "D", -0.16, 0.74)):
        result = read_json(f"{PANOPTIC}/{model}_RESULT.json")
        if result["run_id"] != protocol["run_id"] or len(result["I_by_target_pp"]) != len(names) - 1:
            raise ValueError(f"{model} results do not match the panoptic protocol")
        mean = result["I_pp"]
        bounds = dict(zip(result["interval_order"], result["exact_bounds_pp"], strict=True))
        values = [mean, *result["I_by_target_pp"]]
        xerr = [[mean - bounds["two_sided_lower"]], [bounds["two_sided_upper"] - mean]]
        ax.errorbar(mean, positions[0] + shift, xerr=xerr, fmt="none", ecolor=color,
                    elinewidth=1.2, capsize=2.5, zorder=3)
        ax.plot(values, positions + shift, linestyle="none", marker=marker, color=color,
                markersize=4.4, zorder=4)

        header = estimates.get_yaxis_transform()
        estimates.plot(column, 5.35, linestyle="none", marker=marker, color=color,
                       markersize=4.6, transform=header, clip_on=False)
        estimates.text(column, 5.58, name, ha="center", va="bottom",
                       fontsize=8.4, fontweight="bold", transform=header)
        for position, value in zip(positions, values, strict=True):
            estimates.text(column, position, f"{value:+.3f}", ha="center", va="center",
                           fontsize=9, fontweight="bold" if position == positions[0] else "normal",
                           color=color, transform=header)

    save(fig, "panoptic_confirmation.pdf", "Partner matching on four panoptic targets")


if __name__ == "__main__":
    configure_style()
    design_figure()
    cross_figure()
    composition_figure()
    replication_figure()
    panoptic_confirmation_figure()
