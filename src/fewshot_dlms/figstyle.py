# ported from the paper repository's scripts/figstyle.py and scripts/figgrid.py @ fd2a8a9
"""One shared style for the paper's figures.

Width is an input, not an output: every figure is authored at the paper's full text width (FIGW) with a fixed
figsize and constrained layout, never bbox_inches="tight", so the point sizes below are the sizes on the
page. The figures are set in TeX Gyre Pagella (a Palatino), which ships with this package (fonts/, under
the GUST Font License); `use_font` registers it with matplotlib. If the font files are missing it falls
back to DejaVu Serif, which matplotlib always has: the figures then show the same data in another face.

Every plot.py calls `use_font()` once before drawing and builds its panels with `grid`, `panel`,
`row_tag` and `row_xlabel`.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FIGW_PT = 455.24411  # the paper's \textwidth in points (two-column layout): the authoring width
FIGW = FIGW_PT / 72.0  # in inches

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FILES = [
    "texgyrepagella-regular.otf",
    "texgyrepagella-bold.otf",
    "texgyrepagella-italic.otf",
    "texgyrepagella-bolditalic.otf",
]
FONT_NAME = "TeX Gyre Pagella"
FALLBACK_FONT = "DejaVu Serif"

# On-page point sizes of the figure text: tick labels, legends, in-panel tags, axis labels, panel titles.
PT = {"tick": 9.0, "legend": 9.0, "annot": 8.5, "label": 9.3, "title": 9.3}

# Colours. Each model has a hue and a marker, so the figures stay readable in grayscale.
INK = "#1f2328"  # text and axes
NEUTRAL = "0.45"  # zero lines and reference lines
TEAL = "#0F766E"
SLATE = "#94A3B8"
DIFFADD = "#1D4ED8"  # example guidance
EPMUTED = "#64748B"  # unguided sampling
MODEL_STYLE = {
    "llada": {"color": "#2563EB", "marker": "o"},
    "nemotron_ar": {"color": "#0F9D8A", "marker": "D"},
    "nemotron_pt": {"color": "#0F9D8A", "marker": "D"},
    "dqwen": {"color": "#D97706", "marker": "^"},
    "qwen": {"color": "#A21CAF", "marker": "s"},
}

# Grid spacing of `grid`, in figure-relative units of constrained layout.
GRID_PADS = {"h_pad": 0.02, "w_pad": 0.02, "hspace": 0.04, "wspace": 0.03}


def use_font():
    """Register TeX Gyre Pagella (or fall back to DejaVu Serif) and install the house rcParams.

    Returns the name of the font in use. Text is embedded as real (Type 42) fonts, not outlines.
    """
    paths = [FONT_DIR / name for name in FONT_FILES]
    if all(path.exists() for path in paths):
        for path in paths:
            font_manager.fontManager.addfont(str(path))
        name = FONT_NAME
        math = {
            "mathtext.fontset": "custom",
            "mathtext.rm": name,
            "mathtext.it": f"{name}:italic",
            "mathtext.bf": f"{name}:bold",
            "mathtext.cal": name,
            "mathtext.tt": name,
        }
    else:
        name = FALLBACK_FONT
        math = {"mathtext.fontset": "dejavuserif"}
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [name],
        **math,
        "pdf.fonttype": 42,
        "font.size": PT["tick"],
        "axes.labelsize": PT["label"],
        "axes.titlesize": PT["title"],
        "xtick.labelsize": PT["tick"],
        "ytick.labelsize": PT["tick"],
        "legend.fontsize": PT["legend"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": True,
        "grid.color": "0.90",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderpad": 0.0,
        "legend.columnspacing": 1.2,
    })
    return name


def grid(nrows, ncols, height):
    """A figure of the full text width and `height` inches with nrows x ncols panels (constrained layout)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIGW, height), layout="constrained", squeeze=False)
    fig.get_layout_engine().set(**GRID_PADS)
    return fig, axes


def panel(ax, title=None):
    """House tick lengths, and the panel title if there is one."""
    ax.tick_params(length=2.5, pad=2)
    if title:
        ax.set_title(title, pad=4)


def row_tag(ax, text):
    """A small label in the top-left corner of a panel, on a white box: names the row of a grid."""
    ax.text(
        0.03,
        0.93,
        text,
        transform=ax.transAxes,
        fontsize=PT["annot"],
        va="top",
        ha="left",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
        zorder=4,
    )


def row_xlabel(fig, row_axes, text, gap_pt=3.0):
    """One x label centred under a row of panels, in place of one label per panel.

    Each panel of the row must already carry ax.set_xlabel(" "), so that constrained layout reserves the
    height of one label under the row; the text is then written into that gap. The position is measured
    from the laid-out axes, so the gap is the same under every row and in every figure.
    """
    fig.canvas.draw()  # the tick labels must exist before they are measured
    inverse = fig.transFigure.inverted()
    low = min(inverse.transform((0, ax.get_tightbbox().y0))[1] for ax in row_axes)
    centres = [inverse.transform((ax.get_tightbbox().x0 + ax.get_tightbbox().width / 2, 0))[0] for ax in row_axes]
    middle = sum(centres) / len(row_axes)
    fig.text(middle, low + gap_pt / (fig.get_figheight() * 72), text, ha="center", va="bottom", fontsize=PT["label"], color=INK)
