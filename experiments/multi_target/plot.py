# ported from editdiff/experiments/paper_integration/figure_redesign/fig_finalize.py @ 51db142 (Figure 5, with the A_avg label) and experiments/sec4_diag/sd_three_row.py (mechanism), experiments/paper_integration/arxiv_curation/appB/appB_crosstask_figure.py, experiments/paper_integration/agent_c/c_figures.py (appA_mj_power) @ d276fa6
"""Figure 6 (Section 4.2) and Figures 9, 10 (Appendix B.3) and 12 (Appendix B.6) from the tables in
data/tables/multi_target (CPU only). Figure and appendix numbers are those of the arXiv version (v1); the code
writes M for the paper's D.

    multi_target.pdf            Figure 6. List Functions–Easy, columns J in {1, 2, 4, 8}, x = D in {2, 4, 8, 16},
                                three diffusion models: unguided A_all, unguided A_avg (= A_target) and the paired
                                gain in A_all from example guidance at beta = 0.5 (cells.csv). Means only, as in the
                                paper.
    multi_target_mechanism.pdf  Figure 9 (Appendix B.3, Multi-Target Experiments). Reference NLL per target token
                                under P1 (solid) and P0 (dashed) by position in the reveal order; cross-benchmark
                                study, D = 4, J = 8 (mechanism.csv).
    crossbench_d4.pdf           Figure 10 (Appendix B.3). The cross-benchmark study, D = 4, K = 64: unguided A_all
                                and its paired gain at beta = 0.5 against J, with 95 % family-bootstrap bands
                                (cells_crossbench.csv).
    sharpening.pdf              Figure 12 (Appendix B.6, Likelihood Sharpening). The D x J grid: gain in A_all of
                                the sharpening control (target P1^{3/2}) and guidance (beta = 0.5) minus sharpening,
                                with bands (sharpening.csv).

Rendered at 150 dpi, each PDF is pixel-identical to the arXiv figure when TeX Gyre Pagella is used.

    python experiments/multi_target/plot.py [--tables data/tables/multi_target] [--out experiments/multi_target/out]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from fewshot_dlms import figstyle as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

MODELS = [("llada", "LLaDA-1.5"), ("nemotron_pt", "Nemotron prefix/tail"), ("dqwen", "dQwen3.5")]
M_GRID = [2, 4, 8, 16]
J_GRID = [1, 2, 4, 8]
BAND = Patch(facecolor="0.55", alpha=0.25, linewidth=0)  # legend key of a 95 % family-bootstrap band


# ---------------------------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------------------------


def curve(ax, xs, ys, style, dx, lo=None, hi=None, dashed=False, hollow=False):
    """One model's series: line and markers, shifted by dx so that the models do not overlap; a band if lo/hi."""
    x = [v + dx for v in xs]
    color = style["color"]
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0, zorder=1)
    ax.plot(
        x,
        ys,
        color=color,
        marker=style["marker"],
        markersize=3.2,
        linewidth=1.1,
        linestyle="--" if dashed else "-",
        markeredgewidth=0.9,
        markerfacecolor="white" if hollow else color,
        markeredgecolor=color,
        clip_on=False,
        zorder=3,
    )


def tidy(ax, title, xlabels, ylim, yticks, left, bottom):
    """Ticks, limits and title of one panel. Only the bottom row shows x tick labels and reserves the x label."""
    F.panel(ax, title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels if bottom else [])
    ax.set_xlim(-0.4, len(xlabels) - 0.6)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.minorticks_off()
    ax.tick_params(length=2, pad=1.5)
    if bottom:
        ax.set_xlabel(" ")
    if not left:
        ax.tick_params(labelleft=False)


def model_key(model, label):
    """Legend entry of a model: its line and marker."""
    style = F.MODEL_STYLE[model]
    return Line2D([], [], color=style["color"], marker=style["marker"], markersize=3.2, lw=1.1), label


def shared_legend(fig, items, ncol, rows, base=0.10):
    """A legend centred under the grid; the layout leaves (base + 0.16 * rows) inches for it."""
    pad = (base + 0.16 * rows) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, pad, 1, 1 - pad))
    return fig.legend(
        [handle for handle, _ in items],
        [label for _, label in items],
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=F.PT["legend"],
        handlelength=2.2,
        columnspacing=1.1,
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path} ({fig.get_figwidth() * 72:.1f} x {fig.get_figheight() * 72:.1f} pt)")


# ---------------------------------------------------------------------------------------------
# Figure 6: three rows, no bands, row tags placed clear of the J = 1 markers
# ---------------------------------------------------------------------------------------------

MULTI_ROWS = [
    # (cells.csv column, beta, y limits, y ticks, row tag, tag corner, y label)
    ("A_all", 0.0, (0, 100), [0, 25, 50, 75, 100], "All targets correct", "top", "unguided\n$A_{\\mathrm{all}}$ (%)"),
    ("A_target", 0.0, (0, 100), [0, 25, 50, 75, 100], "Per-target accuracy", "bottom", "unguided\n$A_{\\mathrm{avg}}$ (%)"),
    ("gain_all", 0.5, (-10, 25), [-10, 0, 10, 20], r"Guidance gain, $\beta = 0.5$", "bottom", "$\\Delta A_{\\mathrm{all}}$ over\nunguided (pp)"),
]
MULTI_OFFSET = {"llada": -0.08, "nemotron_pt": 0.0, "dqwen": 0.08}
MULTI_PANEL_H = 0.88  # inches per row
MULTI_FURNITURE = 0.99  # titles, x label and the one-row legend, inches
MARKER_HALF_WIDTH = 0.06  # x extent of a marker in data units, for the tag placement
TAG_TRIES = {"top": [(0.93, 0.6), (0.97, 0.4)], "bottom": [(0.07, 0.6), (0.03, 0.4)]}


def corner_tag(ax, text, where, y, pad):
    """The row's name in a white box at the top-left or bottom-left of a panel (axes fraction y)."""
    return ax.text(
        0.03,
        y,
        text,
        transform=ax.transAxes,
        fontsize=F.PT["annot"],
        va="top" if where == "top" else "bottom",
        ha="left",
        color=F.INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": pad, "alpha": 0.85},
        zorder=4,
    )


def overlaps(ax, artist, boxes):
    """True if the drawn box of `artist` (its white background box) meets any of `boxes` (x0, y0, x1, y1, data units)."""
    renderer = ax.figure.canvas.get_renderer()
    extent = artist.get_bbox_patch().get_window_extent(renderer)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(extent.x0, extent.y0), (extent.x1, extent.y1)])
    return any(not (b[2] < x0 or b[0] > x1 or b[3] < y0 or b[1] > y1) for b in boxes)


def place_row_tag(fig, ax, text, where, boxes):
    """Put the row tag at its corner; if it covers a marker, move it once closer to the frame with a smaller box."""
    tries = TAG_TRIES[where]
    for attempt, (y, pad) in enumerate(tries):
        tag = corner_tag(ax, text, where, y, pad)
        fig.canvas.draw()
        if not overlaps(ax, tag, boxes) or attempt == len(tries) - 1:
            return tag
        tag.remove()


def multi_target_figure(cells, path):
    """Figure 6 of Section 4.2 from cells.csv (64 families per cell, K = 16, oracle lengths)."""
    fig, axes = F.grid(len(MULTI_ROWS), len(J_GRID), height=len(MULTI_ROWS) * MULTI_PANEL_H + MULTI_FURNITURE)
    boxes = {}
    for j, J in enumerate(J_GRID):
        for r, (column, beta, ylim, yticks, _, _, ylabel) in enumerate(MULTI_ROWS):
            ax = axes[r, j]
            boxes[(r, j)] = []
            half_height = 0.03 * (ylim[1] - ylim[0])  # vertical extent of a marker
            for model, _ in MODELS:
                rows = cells[(cells.model == model) & (cells.J == J) & (cells.beta == beta)].set_index("M").loc[M_GRID]
                values = 100.0 * rows[column].values
                curve(ax, range(len(M_GRID)), values, F.MODEL_STYLE[model], MULTI_OFFSET[model])
                for i, value in enumerate(values):
                    x = i + MULTI_OFFSET[model]
                    boxes[(r, j)].append((x - MARKER_HALF_WIDTH, value - half_height, x + MARKER_HALF_WIDTH, value + half_height))
            if column == "gain_all":
                ax.axhline(0, color=F.NEUTRAL, lw=1.0, zorder=2)
            tidy(ax, f"$J = {J}$" if r == 0 else None, [str(M) for M in M_GRID], ylim, yticks, j == 0, r == len(MULTI_ROWS) - 1)
            if j == 0:
                ax.set_ylabel(ylabel)
    items = [model_key(model, label) for model, label in MODELS] + [(Line2D([], [], color=F.NEUTRAL, lw=1.0), "no gain (0 pp)")]
    shared_legend(fig, items, ncol=len(items), rows=1, base=0.08)
    F.row_xlabel(fig, axes[-1], "demonstrations $D$ (List Functions–Easy, $K = 16$)")
    for r, (_, _, _, _, text, where, _) in enumerate(MULTI_ROWS):
        place_row_tag(fig, axes[r, 0], text, where, boxes[(r, 0)])
    save(fig, path)


# ---------------------------------------------------------------------------------------------
# Figure 9 (Appendix B.3): reference NLL by reveal position (cross-benchmark study, J = 8)
# ---------------------------------------------------------------------------------------------

TASKS = [("sst2_fam", "SST-2"), ("list_easy_fam", "List Functions–Easy"), ("salt_rep_fam", "SALT"), ("movie_dir_fam", "Movie Director")]
MECHANISM_OFFSET = {"llada": -0.12, "nemotron_pt": -0.04, "dqwen": 0.04}


def mechanism_figure(mechanism, path):
    """Figure 9 from mechanism.csv: P0 (dashed, hollow) and P1 (solid) reference NLL per token."""
    rows = mechanism[mechanism.J == 8]
    fig, axes = F.grid(1, len(TASKS), height=1.05 + 1.25)
    top = float(np.ceil(rows.gold_nll0_per_token.max()))
    for j, (task, label) in enumerate(TASKS):
        ax = axes[0, j]
        for model, _ in MODELS:
            e = rows[(rows.model == model) & (rows.source == task)].sort_values("position")
            if not len(e):
                continue
            positions = list(e.position - 1)
            style = F.MODEL_STYLE[model]
            curve(ax, positions, e.gold_nll0_per_token.values, style, MECHANISM_OFFSET[model], dashed=True, hollow=True)
            curve(ax, positions, e.gold_nll1_per_token.values, style, MECHANISM_OFFSET[model])
        tidy(ax, label, [str(k) for k in range(1, 9)], (0, top), list(np.arange(0, top + 1, 2)), j == 0, True)
    axes[0, 0].set_ylabel("reference NLL / target token")
    p1 = Line2D([], [], color="0.3", marker="o", markersize=3.2, lw=1.1)
    p0 = Line2D([], [], color="0.3", marker="o", markerfacecolor="white", markersize=3.2, lw=1.1, ls="--")
    items = [model_key(model, label) for model, label in MODELS]
    items += [(p1, "$P_1$ (demonstrations visible)"), (p0, "$P_0$ (demonstration outputs masked)")]
    shared_legend(fig, items, ncol=3, rows=2)
    F.row_xlabel(fig, axes[0], "target position in the reveal order ($J = 8$, $D = 4$; earlier targets revealed with their reference answers)")
    save(fig, path)


# ---------------------------------------------------------------------------------------------
# Figure 10 (Appendix B.3): the cross-benchmark study, D = 4, K = 64
# ---------------------------------------------------------------------------------------------

CROSSBENCH_TITLES = {"sst2_fam": "SST-2", "list_easy_fam": "List Functions–Easy", "salt_rep_fam": "SALT", "movie_dir_fam": "Movie Director"}
CROSSBENCH_OFFSET = {"llada": -0.08, "nemotron_pt": 0.0, "dqwen": 0.08}


def crossbench_figure(crossbench, path):
    """Figure 10 from cells_crossbench.csv: unguided A_all (beta = 0) and its gain at beta = 0.5."""
    report = crossbench[crossbench.bank == "report"]
    ordinary = report[np.isclose(report.beta, 0.0)]
    guided = report[np.isclose(report.beta, 0.5)]
    gain_lo = 100 * guided.paired_gain_lo.min()
    gain_hi = 100 * guided.paired_gain_hi.max()
    gain_ylim = (np.floor(gain_lo / 10) * 10, np.ceil(gain_hi / 10) * 10)
    fig, axes = F.grid(2, len(TASKS), height=2 * 1.0 + 1.25)
    for j, (task, _) in enumerate(TASKS):
        title = CROSSBENCH_TITLES[task]
        for row in range(2):
            ax = axes[row, j]
            for model, _ in MODELS:
                if row == 0:
                    e = ordinary[(ordinary.model == model) & (ordinary.source == task)].set_index("J").loc[J_GRID]
                    values, lo, hi = e.A_all.values, e.ci_lo.values, e.ci_hi.values
                else:
                    e = guided[(guided.model == model) & (guided.source == task)].set_index("J").loc[J_GRID]
                    values, lo, hi = e.paired_gain.values, e.paired_gain_lo.values, e.paired_gain_hi.values
                curve(ax, range(len(J_GRID)), 100 * values, F.MODEL_STYLE[model], CROSSBENCH_OFFSET[model], lo=100 * lo, hi=100 * hi)
            if row == 1:
                ax.axhline(0, color=F.NEUTRAL, lw=1.0, zorder=2)
            if row == 0:
                tidy(ax, title, [str(J) for J in J_GRID], (0, 100), (0, 25, 50, 75, 100), j == 0, False)
            else:
                yticks = [v for v in range(int(gain_ylim[0]), int(gain_ylim[1]) + 1, 10)]
                tidy(ax, None, [str(J) for J in J_GRID], gain_ylim, yticks, j == 0, True)
    axes[0, 0].set_ylabel("unguided $A_{\\mathrm{all}}$ (%)")
    axes[1, 0].set_ylabel("$A_{\\mathrm{all}}$ gain at\n$\\beta=0.5$ (pp)")
    items = [model_key(model, label) for model, label in MODELS]
    items += [(Line2D([], [], color=F.NEUTRAL, lw=1.0), "no gain (0 pp)"), (BAND, "95 % family bootstrap")]
    shared_legend(fig, items, ncol=3, rows=2)
    F.row_xlabel(fig, axes[-1], "unresolved targets $J$ (original study: $D=4$, $K=64$, seeded 12-slot interleaving)")
    save(fig, path)


# ---------------------------------------------------------------------------------------------
# Figure 12 (Appendix B.6): the sharpening control in the D x J grid
# ---------------------------------------------------------------------------------------------

SHARPENING_ROWS = [
    # (sharpening.csv column, y label, y limits, y ticks)
    ("gain_A_all_power_matched", "$A_{\\mathrm{all}}$ gain of\nsharpening (pp)", (-5, 30), (0, 10, 20, 30)),
    ("tilt_minus_power_A_all", "guidance $(\\beta=0.5)$\nminus sharpening (pp)", (-34, 8), (-30, -20, -10, 0)),
]
SHARPENING_OFFSET = {"llada": -0.12, "nemotron_pt": -0.04, "dqwen": 0.04}


def sharpening_figure(sharpening, path):
    """Figure 12 from sharpening.csv (64 families per cell, K = 16).

    Bands are 95 % family-bootstrap intervals.
    """
    fig, axes = F.grid(len(SHARPENING_ROWS), len(J_GRID), height=len(SHARPENING_ROWS) * 1.0 + 1.35)
    for row, (column, ylabel, ylim, yticks) in enumerate(SHARPENING_ROWS):
        for j, J in enumerate(J_GRID):
            ax = axes[row, j]
            for model, _ in MODELS:
                e = sharpening[(sharpening.model == model) & (sharpening.J == J)].set_index("M").loc[M_GRID]
                lo = 100 * e[column + "_lo"].values
                hi = 100 * e[column + "_hi"].values
                curve(ax, list(range(len(M_GRID))), 100 * e[column].values, F.MODEL_STYLE[model], SHARPENING_OFFSET[model], lo=lo, hi=hi)
            ax.axhline(0.0, color=F.NEUTRAL, lw=1.0, zorder=2)
            tidy(ax, f"$J={J}$" if row == 0 else None, [str(M) for M in M_GRID], ylim, yticks, j == 0, row == len(SHARPENING_ROWS) - 1)
        axes[row, 0].set_ylabel(ylabel)
    items = [model_key(model, label) for model, label in MODELS]
    items += [(Line2D([], [], color=F.NEUTRAL, lw=1.0), "zero"), (BAND, "95 % family bootstrap")]
    shared_legend(fig, items, ncol=3, rows=2)
    F.row_xlabel(fig, axes[-1], "demonstrations $D$ (List Functions–Easy, $K=16$, 64 families per cell)")
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables" / "multi_target"))
    parser.add_argument("--out", default=str(HERE / "out"))
    args = parser.parse_args()
    tables = Path(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    F.use_font()
    multi_target_figure(pd.read_csv(tables / "cells.csv"), out / "multi_target.pdf")
    mechanism_figure(pd.read_csv(tables / "mechanism.csv"), out / "multi_target_mechanism.pdf")
    crossbench_figure(pd.read_csv(tables / "cells_crossbench.csv"), out / "crossbench_d4.pdf")
    sharpening_figure(pd.read_csv(tables / "sharpening.csv"), out / "sharpening.pdf")


if __name__ == "__main__":
    main()
