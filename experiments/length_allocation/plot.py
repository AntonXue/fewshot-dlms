# ported from editdiff/experiments/paper_integration/figure_redesign/fig_finalize.py::{fig6, fig6_fidelity_row} (the production Figure 6) and experiments/length_project/lpx_figs43d.py @ d276fa6
"""Figure 6 (Section 4.3) and the appendix fidelity row (Appendix B, Length Allocation) from the tables in
data/tables/length_allocation (CPU only).

    length_allocation.pdf  Figure 6. Three panels per row, one per diffusion model, M = 4, beta = 0, List Functions-Easy.
                      Top row: per-target accuracy A_avg (A_target) with sampled learned lengths vs oracle lengths,
                      against the number of targets J (sec43_downstream.csv; oracle K = 64, learned K = 64 / J).
                      Bottom row: A_avg with forced offsets L = L* + Delta at every target, J = 1 and 2, on the
                      families valid at every Delta (sec43_length_perturbation.csv, "common to all deltas").
                      Means only, as in the paper.
    length_fidelity.pdf    Appendix: the learned head's probability mass on the exact length and within 1 and 2 tokens,
                      against J, with 95 % family-bootstrap bands (sec43_length_metrics.csv).

Rendered at 150 dpi, each PDF is pixel-identical to the paper's figure when TeX Gyre Pagella is used.

    python experiments/length_allocation/plot.py [--tables data/tables/length_allocation] [--out experiments/length_allocation/out]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

from fewshot_dlms import figstyle as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

BACKBONES = [("llada", "LLaDA-1.5"), ("nemotron", "Nemotron prefix/tail"), ("dqwen", "dQwen3.5")]
J_GRID = [1, 2, 4, 8]
DELTAS = [-2, -1, 0, 1, 2]
BAND = Patch(facecolor="0.55", alpha=0.25, linewidth=0)  # legend key of a 95 % family-bootstrap band

# Series styles: (colour, marker, line style, hollow marker).
ORACLE_COLOR = "#64748B"  # slate-500: the oracle-length reference series
LENGTHS = {
    "learned": {"color": F.TEAL, "marker": "D", "ls": "-", "hollow": False},
    "oracle": {"color": ORACLE_COLOR, "marker": "D", "ls": "--", "hollow": True},
}
TARGETS = {
    1: {"color": F.INK, "marker": "o", "ls": "-", "hollow": False},
    2: {"color": F.INK, "marker": "s", "ls": "--", "hollow": True},
}
TOLERANCES = [
    # (legend label, sec43_length_metrics.csv column, style)
    ("exact", "P0", {"color": F.TEAL, "marker": "o", "ls": "-", "hollow": False}),
    ("within 1 token", "P1", {"color": F.TEAL, "marker": "s", "ls": "--", "hollow": True}),
    ("within 2 tokens", "P2", {"color": F.TEAL, "marker": "^", "ls": ":", "hollow": True}),
]

ACCURACY_YLIM = (0, 80)
ACCURACY_YTICKS = [0, 20, 40, 60, 80]
OFFSET_YLIM = (-3, 80)  # whitespace below 0 so that the markers at Delta = +-2 (means 0.7-4 %) clear the lower spine
FIDELITY_YLIM = (0, 70)
FIDELITY_YTICKS = [0, 20, 40, 60]
PANEL_HEIGHT = 1.02  # inches per row
ALLOCATION_FURNITURE = 1.20  # titles, the two x titles and the two one-line legends, inches
FIDELITY_FURNITURE = 0.9


def curve(ax, xs, ys, style, dx=0.0, lo=None, hi=None, alpha=0.10):
    """One series: line and markers shifted by dx so that series do not overlap; a band if lo/hi are given."""
    x = [v + dx for v in xs]
    if lo is not None:
        ax.fill_between(x, lo, hi, color=style["color"], alpha=alpha, linewidth=0, zorder=1)
    ax.plot(
        x,
        ys,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["ls"],
        markersize=3.4,
        linewidth=1.2,
        markeredgewidth=1.0,
        markerfacecolor="white" if style["hollow"] else style["color"],
        clip_on=False,
        zorder=3,
    )


def handle(style):
    """Legend key of a series style."""
    return Line2D(
        [],
        [],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["ls"],
        markersize=3.4,
        lw=1.2,
        markerfacecolor="white" if style["hollow"] else style["color"],
    )


def tidy(ax, title, xlabels, ylim, yticks, left, xlabel_lines):
    """Ticks, limits and title of one panel; xlabel_lines blank lines under it reserve room for a row's x title and legend."""
    F.panel(ax, title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.4, len(xlabels) - 0.6)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.minorticks_off()
    ax.tick_params(length=2, pad=1.5)
    ax.set_xlabel("\n".join([" "] * xlabel_lines))
    if not left:
        ax.tick_params(labelleft=False)


def under_row(fig, axes, r, xtitle, items):
    """The x title of row r and its one-line legend, written into the space the panels reserved under the row."""
    point = 1.0 / (72.0 * fig.get_figheight())
    tick = 2 + 1.5 + F.PT["tick"]  # tick length + pad + tick label height, in points
    boxes = [axes[r, j].get_position() for j in range(axes.shape[1])]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y = min(box.y0 for box in boxes) - tick * point
    fig.text((x0 + x1) / 2, y - 2 * point, xtitle, ha="center", va="top", fontsize=F.PT["label"])
    y -= (F.PT["label"] + 4) * point
    fig.legend(
        [h for h, _ in items],
        [label for _, label in items],
        loc="upper center",
        bbox_to_anchor=((x0 + x1) / 2, y - 1 * point),
        bbox_transform=fig.transFigure,
        ncol=len(items),
        frameon=False,
        fontsize=F.PT["legend"],
        handlelength=2.4,
        columnspacing=2.0,
        borderaxespad=0,
        borderpad=0,
    )


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path} ({fig.get_figwidth() * 72:.1f} x {fig.get_figheight() * 72:.1f} pt)")


def length_allocation_figure(downstream, perturbation, path):
    """Figure 6 of Section 4.3: downstream accuracy with learned vs oracle lengths (top), forced offsets (bottom)."""
    common = perturbation[perturbation.family_set == "common to all deltas"]
    fig, axes = F.grid(2, len(BACKBONES), height=2 * PANEL_HEIGHT + ALLOCATION_FURNITURE)
    for j, (backbone, name) in enumerate(BACKBONES):
        ax = axes[0, j]
        for condition, dx in (("oracle", -0.06), ("learned", 0.06)):
            rows = downstream[(downstream.backbone == backbone) & (downstream.condition == condition)].set_index("J").loc[J_GRID]
            curve(ax, range(len(J_GRID)), 100 * rows.A_target.values, LENGTHS[condition], dx=dx)
        tidy(ax, name, [str(J) for J in J_GRID], ACCURACY_YLIM, ACCURACY_YTICKS, j == 0, 2)
        ax = axes[1, j]
        for J, style in TARGETS.items():
            rows = common[(common.backbone == backbone) & (common.J == J)].set_index("delta").loc[DELTAS]
            curve(ax, range(len(DELTAS)), 100 * rows.A_target.values, style, dx=-0.05 if J == 1 else 0.05)
        tidy(ax, None, ["$-2$", "$-1$", "$0$", "$+1$", "$+2$"], OFFSET_YLIM, ACCURACY_YTICKS, j == 0, 2)
    axes[0, 0].set_ylabel("$\\mathbf{Downstream}$\ntarget accuracy (%)")
    axes[1, 0].set_ylabel("$\\mathbf{Forced\\ offset}$\ntarget accuracy (%)")
    fig.canvas.draw()  # settle the layout before measuring the panels
    under_row(fig, axes, 0, "unresolved targets $J$", [(handle(LENGTHS["learned"]), "sampled learned lengths"), (handle(LENGTHS["oracle"]), "oracle lengths")])
    under_row(
        fig,
        axes,
        1,
        "length offset $\\Delta$ in $L = L^\\star + \\Delta$ ($\\Delta = 0$: oracle allocation)",
        [(handle(TARGETS[1]), "$J = 1$"), (handle(TARGETS[2]), "$J = 2$")],
    )
    save(fig, path)


def length_fidelity_row(metrics, path):
    """The appendix fidelity row: the learned head's mass within 0, 1 and 2 tokens of L*, with family-bootstrap bands."""
    fig, axes = F.grid(1, len(BACKBONES), height=PANEL_HEIGHT + FIDELITY_FURNITURE)
    for j, (backbone, name) in enumerate(BACKBONES):
        ax = axes[0, j]
        learned = metrics[(metrics.backbone == backbone) & (metrics.run != "visible_demo_prior")].set_index("J").loc[J_GRID]
        for (_, column, style), dx in zip(TOLERANCES, (-0.06, 0.0, 0.06)):
            lo = 100 * learned[f"{column}_lo"].values
            hi = 100 * learned[f"{column}_hi"].values
            curve(ax, range(len(J_GRID)), 100 * learned[column].values, style, dx=dx, lo=lo, hi=hi, alpha=0.07)
        tidy(ax, name, [str(J) for J in J_GRID], FIDELITY_YLIM, FIDELITY_YTICKS, j == 0, 2)
    axes[0, 0].set_ylabel("$\\mathbf{Fidelity}$\nlearned allocator\nmass (%)")
    fig.canvas.draw()
    items = [(handle(style), label) for label, _, style in TOLERANCES] + [(BAND, "95 % family bootstrap")]
    under_row(fig, axes, 0, "unresolved targets $J$", items)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables" / "length_allocation"))
    parser.add_argument("--out", default=str(HERE / "out"))
    args = parser.parse_args()
    tables = Path(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    F.use_font()
    downstream = pd.read_csv(tables / "sec43_downstream.csv")
    perturbation = pd.read_csv(tables / "sec43_length_perturbation.csv")
    metrics = pd.read_csv(tables / "sec43_length_metrics.csv")
    length_allocation_figure(downstream, perturbation, out / "length_allocation.pdf")
    length_fidelity_row(metrics, out / "length_fidelity.pdf")


if __name__ == "__main__":
    main()
