"""Shared matplotlib styling, so every figure across the notebooks reads the same way.

matplotlib is used rather than plotly/seaborn for these first notebooks: the figures are
static, log-scaled convergence curves and sparsity patterns destined for a written
narrative, and matplotlib gives the tightest control over log axes, spy plots and
annotation placement. Interactive libraries earn their place later, when we start
exploring high-dimensional training results.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind-safe qualitative palette (Okabe-Ito), used consistently for methods.
PALETTE = {
    "newton":  "#0072B2",   # blue    - Newton / direct LU
    "gmres":   "#D55E00",   # vermil. - plain GMRES
    "precond": "#009E73",   # green   - preconditioned GMRES
    "learned": "#CC79A7",   # pink    - learned iterations (later notebooks)
    "ref":     "#7F7F7F",   # grey    - references, tolerances, guides
    "accent":  "#E69F00",   # orange  - highlights
}

_RC = {
    # Figures are stacked one panel per row rather than side by side, so titles,
    # axis labels and legends never collide. Width is fixed; height grows with
    # the number of panels via the `stacked` helper below.
    "figure.figsize": (8.6, 5.2),
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "lines.linewidth": 1.9,
    "lines.markersize": 5.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
}


def use_style() -> None:
    """Apply the project figure style. Call once at the top of a notebook."""
    mpl.rcParams.update(_RC)


def stacked(n_panels: int, panel_height: float = 5.0, width: float = 8.6,
            square: bool = False):
    """Create ``n_panels`` axes stacked vertically, one per row.

    Vertical stacking is the house style here: side-by-side panels force titles and
    tick labels into each other at any readable font size.  ``constrained`` layout
    keeps the suptitle clear of the top axes automatically.

    Parameters
    ----------
    square
        Use a 1:1 panel aspect, for sparsity patterns and other matrix plots.
    """
    h = width if square else panel_height
    fig, axes = plt.subplots(n_panels, 1, figsize=(width, h * n_panels),
                             layout="constrained")
    return fig, ([axes] if n_panels == 1 else list(axes))


def suptitle(fig, text: str) -> None:
    """Figure-level title, positioned consistently for stacked layouts."""
    fig.suptitle(text, fontsize=13, fontweight="semibold")


def save(fig, name: str, folder: str = "../figures") -> str:
    """Save a figure as PNG next to the notebooks and return its path."""
    from pathlib import Path

    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    return str(path)


def annotate_tolerance(ax, tol: float, label: str | None = None,
                       x: float = 0.015, ha: str = "left") -> None:
    """Draw a horizontal convergence-tolerance guide line with a legible label.

    The label sits on an opaque background so it stays readable where it crosses
    data, which a bare text call does not.
    """
    ax.axhline(tol, color=PALETTE["ref"], ls=":", lw=1.2)
    ax.text(x, tol, label or f"tol = {tol:g}", transform=ax.get_yaxis_transform(),
            ha=ha, va="bottom", fontsize=9, color=PALETTE["ref"],
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))


__all__ = ["PALETTE", "use_style", "save", "annotate_tolerance", "stacked",
           "suptitle", "plt"]
