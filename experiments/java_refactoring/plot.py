# ported from editdiff/experiments/paper_integration/figure_redesign/fig_finalize.py::fig7 (the production Figure 7, no error bars) @ d276fa6
"""Figure 8 (Section 5) from the tables in data/tables/java_refactoring (CPU only). Figure numbers are those of the
arXiv version (v1); the code writes M for the paper's D.

    java_refactoring.pdf   left: length allocation on the 80 unique target sites of the Java episodes, by the length
                           of the reference replacement: the mass that the Java predictor (head s5hc_C) puts within
                           +-2 tokens of the reference length, next to the training-label prior
                           (length_fidelity_by_bin.csv); right: joint LLM-judged success per cell (D, J) with oracle
                           and learned target lengths, unguided sampling (beta = 0) and example guidance
                           (beta = 0.5), commit-level means (summary_by_cell.csv, primary stratum). Means only, as in
                           the paper; the 95 % commit-bootstrap intervals are in the tables.

The tables name the two rules `ordinary` (unguided, beta = 0) and `tilting` (example guidance, beta = 0.5). Rendered
at 150 dpi, the PDF is pixel-identical to the arXiv figure when TeX Gyre Pagella is used.

    python experiments/java_refactoring/plot.py [--tables data/tables/java_refactoring] [--out experiments/java_refactoring/out]
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fewshot_dlms import figstyle as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PRIOR_COLOR = "#CBD5E1"  # the training-label prior (slate-300)
HEAD = "s5hc_C"  # the Java predictor (length head C)
CELLS = ["M1J1", "M2J1", "M1J2", "M2J2"]  # left to right: J = 1, then J = 2
# (regime, rule, colour, marker, filled): oracle lengths are circles, learned lengths diamonds; guidance is filled.
CONDITIONS = [
    ("oracle", "ordinary", F.EPMUTED, "o", False),
    ("oracle", "tilting", F.DIFFADD, "o", True),
    ("learned", "ordinary", F.EPMUTED, "D", False),
    ("learned", "tilting", F.DIFFADD, "D", True),
]
OFFSET = {"ordinary": -0.13, "tilting": 0.13}  # x shift of the two rules within a cell
HEIGHT = 2.25  # inches
BAR_WIDTH = 0.36
RIGHT_LIMITS = (0, 100)
MARKER_HALF_HEIGHT = 0.03  # vertical extent of a marker, as a fraction of the y range (legend placement)


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def length_panel(fig, ax, bins):
    """Left panel: % of the Java predictor's length mass within +-2 tokens of the reference length, and the prior's,
    per bin."""
    for i, row in enumerate(bins):
        head = 100 * float(row["F2"])
        prior = 100 * float(row["prior_F2"])
        ax.bar(i - BAR_WIDTH / 2, head, BAR_WIDTH, color=F.TEAL, zorder=2, label="Java predictor" if i == 0 else None)
        ax.bar(i + BAR_WIDTH / 2, prior, BAR_WIDTH, color=PRIOR_COLOR, zorder=2, label="training-label prior" if i == 0 else None)
        ax.text(i, max(head, prior) + 2.5, f"n={row['unique_sites']}", ha="center", va="bottom", fontsize=F.PT["annot"])
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([row["bin"].replace("-", "–") for row in bins])
    ax.set_xlabel("reference replacement length (LLaDA tokens)")
    ax.set_ylabel(r"% length mass within $\pm2$ tokens")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", visible=False)
    F.panel(ax, "Length allocation (80 unique sites)")
    legend = ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8, handletextpad=0.4)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = legend.get_window_extent(renderer)
    panel = ax.get_window_extent(renderer)
    if box.x0 < panel.x0 + 0.3 * panel.width:  # a one-row key would run over the first bars: two rows instead
        legend.remove()
        ax.legend(loc="upper right", handlelength=1.2)


def success_panel(fig, ax, summary):
    """Right panel: joint judged success per cell, regime and rule; the legend sits in the gap between the oracle
    and the learned points."""
    marker_extent = MARKER_HALF_HEIGHT * (RIGHT_LIMITS[1] - RIGHT_LIMITS[0])
    learned_top = RIGHT_LIMITS[0]
    oracle_bottom = RIGHT_LIMITS[1]
    for regime, rule, color, marker, filled in CONDITIONS:
        for j, cell in enumerate(CELLS):
            row = next(r for r in summary if r["cell"] == cell and r["regime"] == regime and r["rule"] == rule)
            value = 100 * float(row["U_commit"])
            ax.errorbar(
                j + OFFSET[rule],
                value,
                yerr=None,
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=1.8,
                ms=4.0 if marker == "o" else 3.6,
                mfc=color if filled else "white",
                mec=color,
                mew=0.9,
                zorder=3,
            )
            if regime == "learned":
                learned_top = max(learned_top, value + marker_extent)
            else:
                oracle_bottom = min(oracle_bottom, value - marker_extent)
    commits = {cell: next(r["n_commits"] for r in summary if r["cell"] == cell and r["regime"] == "oracle" and r["rule"] == "ordinary") for cell in CELLS}
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([f"({cell[1]},{cell[3]})\n{commits[cell]} commits" for cell in CELLS])
    ax.set_xlabel(r"($D$ demonstrations, $J$ targets)")  # no y label, as in Figure 8 of arXiv v1 (the authors' script lost it); kept for pixel parity
    ax.set_ylim(*RIGHT_LIMITS)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.5, len(CELLS) - 0.5)
    ax.grid(axis="x", visible=False)
    handles = [
        Line2D([], [], ls="", marker="o", color=F.EPMUTED, mfc="white", mec=F.EPMUTED, ms=4.0, label="unguided"),
        Line2D([], [], ls="", marker="o", color=F.DIFFADD, mfc=F.DIFFADD, mec=F.DIFFADD, ms=4.0, label="guided"),
        Line2D([], [], ls="", marker="o", color=F.INK, mfc="white", mec=F.INK, ms=4.0, label="oracle lengths"),
        Line2D([], [], ls="", marker="D", color=F.INK, mfc="white", mec=F.INK, ms=3.6, label="learned lengths"),
    ]
    F.panel(ax, r"Helper propagation, LLaDA-1.5, $K=16$")
    fig.canvas.draw()
    centre = (learned_top + oracle_bottom) / 2
    anchor = (centre - RIGHT_LIMITS[0]) / (RIGHT_LIMITS[1] - RIGHT_LIMITS[0])
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, anchor), ncol=2, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    fig.canvas.draw()


def java_refactoring_figure(bins, summary, path):
    """Figure 8 of Section 5 (two panels, full text width)."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(F.FIGW, HEIGHT), layout="constrained", gridspec_kw={"width_ratios": [1.0, 1.32]})
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06)
    length_panel(fig, left, bins)
    success_panel(fig, right, summary)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path} ({fig.get_figwidth() * 72:.1f} x {fig.get_figheight() * 72:.1f} pt)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables" / "java_refactoring"))
    parser.add_argument("--out", default=str(HERE / "out"))
    args = parser.parse_args()
    tables = Path(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    F.use_font()
    bins = [row for row in read_rows(tables / "length_fidelity_by_bin.csv") if row["population"] == "B" and row["head"] == HEAD]
    summary = [row for row in read_rows(tables / "summary_by_cell.csv") if row["stratum"] == "primary" and row["cell"] != "all"]
    java_refactoring_figure(bins, summary, out / "java_refactoring.pdf")


if __name__ == "__main__":
    main()
