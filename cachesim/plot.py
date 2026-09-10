"""Presentation for the analysis commands: every plot, and shared labels.

matplotlib is an optional dependency (``pip install "cachesim[plots]"``).
Nothing here imports it at module import time: the import happens inside the
drawing functions, so importing ``cachesim.plot`` costs nothing and the
analysis commands still run, tables and all, on an installation without it.
Callers check ``available()`` or catch ``MatplotlibUnavailable``.

House style, applied by every function so a directory of plots looks like
one set:

* Agg backend, so no display is needed and output does not depend on the
  environment.
* Figures 7.0 x 4.5 inches at 120 dpi, tight layout.
* Cache parameters that double (capacity, size, associativity) get a
  base-2 logarithmic x-axis with ticks at the data points and plain
  decimal labels, never ``2^n`` or scientific notation.
* Miss ratios are plotted as fractions and formatted as percentages, with
  the y-axis anchored at 0 so bar heights and gaps are comparable between
  plots.
* Grid at alpha 0.3, top and right spines removed.

Output is deterministic: the style is applied through an explicit
``rc_context`` rather than the user's matplotlibrc, no timestamp is written
into the file metadata, and the SVG hash salt is fixed, so re-running a
command byte-for-byte reproduces its plots.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from typing import Any, Protocol

#: Figure size, resolution and grid opacity shared by every plot.
FIGSIZE = (7.0, 4.5)
DPI = 120
GRID_ALPHA = 0.3

_RC: dict[str, Any] = {
    "figure.figsize": FIGSIZE,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": GRID_ALPHA,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "svg.hashsalt": "cachesim",
}


class MatplotlibUnavailable(RuntimeError):
    """Raised when a plot is requested but matplotlib is not installed."""


def available() -> bool:
    """True if matplotlib can be imported, without importing it."""
    return importlib.util.find_spec("matplotlib") is not None


def format_bytes(nbytes: int) -> str:
    """``32768`` -> ``'32 KB'``, ``2097152`` -> ``'2 MB'``, ``64`` -> ``'64 B'``.

    Used for axis labels and for the size columns of the analysis tables, so
    the two always agree. Fractional values keep one decimal (``'1.5 KB'``).
    """
    for unit, scale in (("MB", 1 << 20), ("KB", 1 << 10)):
        if nbytes >= scale:
            value = nbytes / scale
            return f"{value:.0f} {unit}" if value == int(value) else f"{value:.1f} {unit}"
    return f"{nbytes} B"


def _pyplot() -> Any:
    """Import pyplot with the Agg backend, or raise ``MatplotlibUnavailable``."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise MatplotlibUnavailable(
            'matplotlib is not installed; install it with: pip install "cachesim[plots]"'
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _log2_x_axis(ax: Any, ticks: Sequence[float]) -> None:
    """Base-2 log x-axis with ticks at ``ticks`` and plain decimal labels."""
    from matplotlib.ticker import ScalarFormatter

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(ticks))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def _percent_y_axis(ax: Any) -> None:
    """Percent-formatted y-axis anchored at zero (values are fractions)."""
    from matplotlib.ticker import PercentFormatter

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))


def _save(plt: Any, fig: Any, out_path: str) -> str:
    """Tight-layout, write and close ``fig``; returns ``out_path``."""
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    # PNG is the only format whose metadata matplotlib would otherwise stamp
    # with its own version string; pin it so output is byte-reproducible.
    metadata = {"Software": "cachesim"} if out_path.lower().endswith(".png") else None
    fig.savefig(out_path, dpi=DPI, metadata=metadata)
    plt.close(fig)
    return out_path


def plot_mrc(
    capacities: Sequence[int],
    miss_ratios: Sequence[float],
    out_path: str,
    title: str | None = None,
    extra: Sequence[tuple[str, Sequence[int], Sequence[float]]] = (),
) -> str:
    """Plot a miss-ratio curve: miss ratio against cache capacity in blocks.

    ``miss_ratios`` are fractions in [0, 1], one per capacity. ``extra``
    holds additional labelled curves ``(label, capacities, miss_ratios)``
    drawn on the same axes -- used to show a set-associative curve above the
    fully-associative lower bound, both against capacity, so the vertical
    gap between them is the conflict miss ratio.

    Returns the path written. Raises ``MatplotlibUnavailable`` if matplotlib
    is missing, and ``ValueError`` if a series is empty or ragged.
    """
    if len(capacities) != len(miss_ratios):
        raise ValueError(
            f"capacities and miss_ratios differ in length: {len(capacities)} vs {len(miss_ratios)}"
        )
    if not capacities:
        raise ValueError("nothing to plot: no capacities given")

    plt = _pyplot()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        ax.plot(list(capacities), list(miss_ratios), marker="o", label="fully associative")
        ticks = set(capacities)
        for label, xs, ys in extra:
            if len(xs) != len(ys):
                raise ValueError(f"series {label!r} is ragged: {len(xs)} vs {len(ys)}")
            ax.plot(list(xs), list(ys), marker="s", linestyle="--", label=label)
            ticks.update(xs)
        _log2_x_axis(ax, sorted(ticks))
        _percent_y_axis(ax)
        ax.set_xlabel("cache capacity (blocks)")
        ax.set_ylabel("miss ratio")
        ax.set_title(title or "Miss-ratio curve (LRU)")
        if extra:
            ax.legend()
        return _save(plt, fig, out_path)


class SweepRowLike(Protocol):
    """The part of a sweep row a plot needs.

    Declared structurally so this module stays independent of
    ``cachesim.sweep``: anything with a swept value and a miss rate can be
    plotted. The members are read-only properties, which a frozen dataclass
    satisfies.
    """

    @property
    def value(self) -> int | str: ...

    @property
    def miss_rate(self) -> float: ...


def plot_sweep(
    rows: Sequence[SweepRowLike],
    param: str,
    out_path: str,
    title: str | None = None,
) -> str:
    """Plot miss rate against the swept parameter.

    Numeric parameters (size, associativity, block size) are drawn as a line
    on a base-2 log x-axis with a tick at each swept value, which is the
    right shape for knees and cliffs. A parameter whose values are names
    (policy) is drawn as a bar chart with categorical labels instead,
    because interpolating between "lru" and "fifo" would be meaningless.

    Returns the path written. Raises ``MatplotlibUnavailable`` if matplotlib
    is missing and ``ValueError`` if ``rows`` is empty.
    """
    if not rows:
        raise ValueError(f"nothing to plot: no rows for parameter {param!r}")
    values = [row.value for row in rows]
    miss_rates = [row.miss_rate for row in rows]
    numeric = all(isinstance(v, int) for v in values)

    plt = _pyplot()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        if numeric:
            xs = [int(v) for v in values]
            ax.plot(xs, miss_rates, marker="o")
            _log2_x_axis(ax, xs)
        else:
            labels = [str(v) for v in values]
            ax.bar(labels, miss_rates)
            ax.grid(axis="x", visible=False)
        _percent_y_axis(ax)
        ax.set_xlabel(param.replace("_", " "))
        ax.set_ylabel("miss rate")
        ax.set_title(title or f"Miss rate vs {param.replace('_', ' ')}")
        return _save(plt, fig, out_path)


def plot_grid(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str,
    title: str | None = None,
    row_axis_label: str = "cache size",
    col_axis_label: str = "associativity (ways)",
) -> str:
    """Plot a two-parameter miss-rate grid as an annotated heat map.

    ``matrix[r][c]`` is a miss rate in [0, 1]; use ``float('nan')`` for a
    combination that cannot be built (a size that is not a multiple of
    block_size * ways), which is drawn as an empty cell. Each cell is
    labelled with its percentage, in white on dark cells and black on light
    ones, so the plot is readable without the colour bar.

    Returns the path written. Raises ``MatplotlibUnavailable`` if matplotlib
    is missing and ``ValueError`` if the matrix is empty or ragged.
    """
    if not matrix or not matrix[0]:
        raise ValueError("nothing to plot: the grid is empty")
    if any(len(row) != len(col_labels) for row in matrix):
        raise ValueError("grid rows must all have one cell per column label")
    if len(matrix) != len(row_labels):
        raise ValueError("the grid must have one row per row label")

    plt = _pyplot()
    with plt.rc_context(_RC):
        fig, ax = plt.subplots()
        finite = [v for row in matrix for v in row if v == v]  # NaN != NaN
        vmax = max(finite) if finite else 1.0
        image = ax.imshow(
            [list(row) for row in matrix],
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=vmax or 1.0,
        )
        ax.set_xticks(range(len(col_labels)), [str(c) for c in col_labels])
        ax.set_yticks(range(len(row_labels)), [str(r) for r in row_labels])
        ax.set_xlabel(col_axis_label)
        ax.set_ylabel(row_axis_label)
        ax.set_title(title or "Miss rate")
        ax.grid(visible=False)
        threshold = (vmax or 1.0) * 0.55
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                if value != value:  # NaN: this geometry does not exist
                    continue
                ax.text(
                    c,
                    r,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value < threshold else "black",
                )
        bar = fig.colorbar(image, ax=ax)
        bar.set_label("miss rate")
        from matplotlib.ticker import PercentFormatter

        bar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        return _save(plt, fig, out_path)
