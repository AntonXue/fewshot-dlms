# ported from editdiff/experiments/paper_integration/figure_redesign/fig_finalize.py::{fig4, fig4_gold_row} @ 51db142 and experiments/beta_sweep/bs_figure.py @ 1aeba87
"""Figure 4 (Section 4.1), its gold-reward row and Figure 11 (Appendix B, Guidance-Strength Sensitivity) from the tables in data/tables/fig4 (CPU only).

    fig4.pdf              Figure 4. Columns: SST-2, List Functions-Easy, SALT, Movie Director; x = M in {1, 2, 4, 8, 16}
                          demonstrations, one target, K = 64; four model interfaces. Top row: ordinary accuracy (beta = 0);
                          bottom row: the paired accuracy gain of example guidance at beta = 0.5 (sec41_accuracy.csv).
                          Means only, as in the paper.
    gold_reward_row.pdf   The gold example reward per target token, (log P1 - log P0) / T of the reference answer, per cell
                          with 95 % family-bootstrap bands (gold_reward_cells.csv); a row of the appendix.
    fig11_beta_sweep.pdf  Figure 11 (Appendix B, Guidance-Strength Sensitivity). List Functions-Easy, M = 4, K = 64: the paired gain in A_all over beta = 0
                          at J = 1 and J = 4, and the effective support exp(H) / K of the weights, against beta in
                          {0, .25, .5, 1, 2} on categorical spacing, three diffusion models, 95 % bands (beta_sweep.csv).

Rendered at 150 dpi, each PDF is pixel-identical to the paper's figure when TeX Gyre Pagella is used.

    python experiments/fig4_single_target/plot.py [--tables data/tables/fig4] [--out experiments/fig4_single_target/out]
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

TASKS = [("sst2_fam", "SST-2"), ("list_easy_fam", "List Functions"), ("salt_rep_fam", "SALT"), ("movie_dir_fam", "Movie Director")]
MODELS = [("llada", "LLaDA-1.5"), ("nemotron_ar", "Nemotron causal"), ("dqwen", "dQwen3.5"), ("qwen", "Qwen3-8B")]
OFFSET = {"llada": -0.12, "nemotron_ar": -0.04, "dqwen": 0.04, "qwen": 0.12}  # x shift of each model, so markers do not overlap
M_GRID = [1, 2, 4, 8, 16]
BETA = 0.5  # the common guidance strength of Figure 4
BAND = Patch(facecolor="0.55", alpha=0.25, linewidth=0)  # legend key of a 95 % family-bootstrap band
LEGEND_ROW = 0.16  # inches per legend row under the panels


# ---------------------------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------------------------


def curve(ax, xs, ys, style, dx, lo=None, hi=None):
    """One model's series: line and filled markers, shifted by dx; a 10 % band between lo and hi if given."""
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
        markeredgewidth=0.9,
        markerfacecolor=color,
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


def model_key(model):
    """Legend handle of a model: its line and marker."""
    style = F.MODEL_STYLE[model]
    return Line2D([], [], color=style["color"], marker=style["marker"], markersize=3.2, lw=1.1)


def shared_legend(fig, items, ncol, rows):
    """A legend centred under the grid; the layout leaves (0.08 + 0.16 * rows) inches for it."""
    pad = (0.08 + LEGEND_ROW * rows) / fig.get_figheight()
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


def legend_fits(fig, legend):
    """True if the legend lies within the figure's width."""
    fig.canvas.draw()
    box = legend.get_window_extent(fig.canvas.get_renderer())
    return box.x0 >= 0 and box.x1 <= fig.get_figwidth() * fig.dpi


def corner_tag(ax, text, y, pad):
    """The row's name in a white box at the top left of a panel (axes fraction y)."""
    return ax.text(
        0.03,
        y,
        text,
        transform=ax.transAxes,
        fontsize=F.PT["annot"],
        va="top",
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


def place_row_tag(fig, ax, text, boxes):
    """Put the row tag at the top left; if it covers a marker, move it once closer to the frame with a smaller box."""
    tries = [(0.93, 0.6), (0.97, 0.4)]
    for attempt, (y, pad) in enumerate(tries):
        tag = corner_tag(ax, text, y, pad)
        fig.canvas.draw()
        if not overlaps(ax, tag, boxes) or attempt == len(tries) - 1:
            return tag
        tag.remove()


def save(fig, path, **kwargs):
    fig.savefig(path, **kwargs)
    plt.close(fig)
    print(f"wrote {path} ({fig.get_figwidth() * 72:.1f} x {fig.get_figheight() * 72:.1f} pt)")


# ---------------------------------------------------------------------------------------------
# Figure 4: two rows, no bands, one-row legend
# ---------------------------------------------------------------------------------------------

FIG4_PANEL_H = 0.85  # inches per row
FIG4_FURNITURE = 0.99  # titles, x label and the one-row legend, inches
MARKER_HALF_WIDTH = 0.06  # x extent of a marker in data units, for the tag placement


def figure4(accuracy, path):
    """Figure 4 of Section 4.1 from sec41_accuracy.csv (report bank, 64 families per cell, K = 64).

    Top row: ordinary accuracy (beta = 0), y up to 105 when a mean exceeds 97 % (whitespace only; ticks stop at 100).
    Bottom row: paired gain at beta = 0.5 over ordinary sampling, in percentage points.
    """
    ordinary = accuracy[(accuracy.bank == "report") & (accuracy.beta == 0.0)]
    guided = accuracy[(accuracy.bank == "report") & (accuracy.beta == BETA)]
    top = (0, 105) if 100 * ordinary.accuracy.max() > 97 else (0, 100)
    rows = [
        # (cells of the row, column, y limits, y ticks)
        (ordinary, "accuracy", top, [0, 25, 50, 75, 100]),
        (guided, "paired_gain", (-10, 30), [-10, 0, 10, 20, 30]),
    ]
    fig, axes = F.grid(2, len(TASKS), height=2 * FIG4_PANEL_H + FIG4_FURNITURE)
    boxes = {}
    for j, (task, label) in enumerate(TASKS):
        for r, (cells, column, ylim, yticks) in enumerate(rows):
            ax = axes[r, j]
            boxes[(r, j)] = []
            half_height = 0.03 * (ylim[1] - ylim[0])  # vertical extent of a marker
            for model, _ in MODELS:
                e = cells[(cells.model == model) & (cells.source == task)].set_index("M").loc[M_GRID]
                values = 100.0 * e[column].values
                curve(ax, range(len(M_GRID)), values, F.MODEL_STYLE[model], OFFSET[model])
                for i, value in enumerate(values):
                    x = i + OFFSET[model]
                    boxes[(r, j)].append((x - MARKER_HALF_WIDTH, value - half_height, x + MARKER_HALF_WIDTH, value + half_height))
            if r == 1:
                ax.axhline(0, color=F.NEUTRAL, lw=1.0, zorder=2)
            tidy(ax, label if r == 0 else None, [str(M) for M in M_GRID], ylim, yticks, j == 0, r == 1)
    axes[0, 0].set_ylabel("ordinary\naccuracy (%)")
    axes[1, 0].set_ylabel("gain over\nordinary (pp)")
    items = [(model_key(model), name) for model, name in MODELS] + [(Line2D([], [], color=F.NEUTRAL, lw=1.0), "no gain (0 pp)")]
    legend = shared_legend(fig, items, len(items), 1)
    F.row_xlabel(fig, axes[1], "demonstrations $M$ (single target, $K = 64$)")
    if not legend_fits(fig, legend):  # fall back to the four models (the zero line explains itself)
        legend.remove()
        shared_legend(fig, items[:4], 4, 1)
    fig.canvas.draw()
    # The row tags sit in the SST-2 panels, both at the top left (at this height a bottom box would cover the near-zero
    # SST-2 gains).
    for r, text in enumerate(["Ordinary accuracy", r"Guidance gain, $\beta = 0.5$"]):
        place_row_tag(fig, axes[r, 0], text, boxes[(r, 0)])
    save(fig, path)


def gold_reward_row(gold, path):
    """The gold example reward per target token of every Section 4.1 cell (gold_reward_cells.csv), with 95 % bands."""
    top = float(np.ceil(gold.gold_rpt_hi.max() / 2) * 2)
    fig, axes = F.grid(1, len(TASKS), height=1.0 + 1.15)
    for j, (task, label) in enumerate(TASKS):
        ax = axes[0, j]
        for model, _ in MODELS:
            e = gold[(gold.model == model) & (gold.source == task)].set_index("M").loc[M_GRID]
            curve(ax, range(len(M_GRID)), e.gold_reward_per_token.values, F.MODEL_STYLE[model], OFFSET[model], lo=e.gold_rpt_lo.values, hi=e.gold_rpt_hi.values)
        tidy(ax, label, [str(M) for M in M_GRID], (0, top), list(np.arange(0, top + 1, 4)), j == 0, True)
        if j == 0:
            corner_tag(ax, "Gold example reward", 0.93, 0.6)
    axes[0, 0].set_ylabel("gold example reward\n/ token (nats)")
    items = [(model_key(model), name) for model, name in MODELS] + [(BAND, "95 % family bootstrap")]
    shared_legend(fig, items, 3, 2)
    F.row_xlabel(fig, axes[0], "demonstrations $M$ (single target, $K = 64$)")
    save(fig, path)


# ---------------------------------------------------------------------------------------------
# Figure 11: guidance-strength sweep (Appendix B, Guidance-Strength Sensitivity)
# ---------------------------------------------------------------------------------------------

SWEEP_BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]
SWEEP_TICKS = ["0", ".25", ".5", "1", "2"]
SWEEP_MODELS = [("llada", "LLaDA-1.5"), ("nemotron_pt", "Nemotron prefix/tail"), ("dqwen", "dQwen3.5")]
# The model hues and markers of Figures 5 and 9, plus a line style per model for grayscale.
SWEEP_LINE = {"llada": "-", "nemotron_pt": (0, (4.0, 1.6)), "dqwen": (0, (1.3, 1.3))}
SWEEP_OFFSET = {"llada": -0.07, "nemotron_pt": 0.0, "dqwen": 0.07}


def sweep_curve(ax, model, ys, lo, hi):
    """One model's series of Figure 11: band, line in the model's style, filled markers."""
    style = F.MODEL_STYLE[model]
    x = np.arange(len(SWEEP_BETAS)) + SWEEP_OFFSET[model]
    ax.fill_between(x, lo, hi, color=style["color"], alpha=0.11, linewidth=0, zorder=1)
    ax.plot(
        x,
        ys,
        color=style["color"],
        marker=style["marker"],
        ls=SWEEP_LINE[model],
        markersize=3.3,
        linewidth=1.15,
        markeredgewidth=0.9,
        markerfacecolor=style["color"],
        markeredgecolor=style["color"],
        clip_on=False,
        zorder=3,
    )


def figure11(sweep, path):
    """Figure 11 of Appendix B, Guidance-Strength Sensitivity from beta_sweep.csv: gain in A_all over beta = 0 (pp) at J = 1 and J = 4, then the effective
    support exp(H) / K at J = 1 and J = 4. beta on categorical spacing (0, .25 and .5 would collide on a linear axis)."""
    gain_lo = sweep.gain_lo_pp.min()
    gain_hi = sweep.gain_hi_pp.max()
    gain_ylim = (np.floor(gain_lo / 10) * 10, np.ceil(gain_hi / 5) * 5)  # 5 pp headroom at the top, ticks every 10 pp
    fig, axes = F.grid(1, 4, height=1.98)
    axes = axes[0]
    for ax, (kind, J) in zip(axes, [("gain", 1), ("gain", 4), ("support", 1), ("support", 4)]):
        for model, _ in SWEEP_MODELS:
            e = sweep[(sweep.model == model) & (sweep.J == J)].set_index("beta").loc[SWEEP_BETAS]
            if kind == "gain":
                sweep_curve(ax, model, e.gain_pp.values, e.gain_lo_pp.values, e.gain_hi_pp.values)
            else:
                sweep_curve(ax, model, e.D_eff.values, e.D_eff_lo.values, e.D_eff_hi.values)
        title = f"$A_{{\\mathrm{{all}}}}$ gain, $J={J}$" if kind == "gain" else f"support $e^{{H}}/K$, $J={J}$"
        F.panel(ax, title)
        ax.set_xticks(np.arange(len(SWEEP_BETAS)))
        ax.set_xticklabels(SWEEP_TICKS)
        ax.set_xlim(-0.35, len(SWEEP_BETAS) - 0.65)
        ax.minorticks_off()
        ax.tick_params(length=2, pad=1.5)
        ax.set_xlabel(" ")
        if kind == "gain":
            ax.axhline(0, color=F.NEUTRAL, lw=1.0, zorder=2)
            ax.set_ylim(*gain_ylim)
            ax.set_yticks(list(range(int(gain_ylim[0]), int(np.floor(gain_ylim[1] / 10) * 10) + 1, 10)))
        else:
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0", ".25", ".5", ".75", "1"])
    axes[1].tick_params(labelleft=False)
    axes[3].tick_params(labelleft=False)
    axes[0].set_ylabel("paired gain (pp)")
    axes[2].set_ylabel("$\\exp(H)/K$")
    items = []
    for model, name in SWEEP_MODELS:
        style = F.MODEL_STYLE[model]
        items.append((Line2D([], [], color=style["color"], marker=style["marker"], ls=SWEEP_LINE[model], markersize=3.3, lw=1.15), name))
    items.append((BAND, "95 % family bootstrap"))
    pad = (0.10 + 0.16) / fig.get_figheight()
    fig.get_layout_engine().set(rect=(0, pad, 1, 1 - pad))
    fig.legend(
        [handle for handle, _ in items],
        [label for _, label in items],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=F.PT["legend"],
        handlelength=2.6,
        columnspacing=1.3,
    )
    F.row_xlabel(fig, list(axes), "guidance strength $\\beta$ (the same stored pools at every $\\beta$; categorical spacing)")
    save(fig, path, metadata={"CreationDate": None})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables" / "fig4"))
    parser.add_argument("--out", default=str(HERE / "out"))
    args = parser.parse_args()
    tables = Path(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    F.use_font()
    figure4(pd.read_csv(tables / "sec41_accuracy.csv"), out / "fig4.pdf")
    gold_reward_row(pd.read_csv(tables / "gold_reward_cells.csv"), out / "gold_reward_row.pdf")
    figure11(pd.read_csv(tables / "beta_sweep.csv"), out / "fig11_beta_sweep.pdf")


if __name__ == "__main__":
    main()
