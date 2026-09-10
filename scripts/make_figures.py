"""Regenerate every figure used by ``docs/results.md``.

Run from the repository root::

    PYTHONPATH=. python scripts/make_figures.py --out docs/figures

The script writes the six PNG files ``docs/results.md`` refers to and prints
the numbers behind each one to stdout as a Markdown table, so the document
can be rewritten by hand from this output alone. Nothing is read from a
previous run: every figure is recomputed from the sample traces in
``traces/``, which are generated first if they are missing.

Output is deterministic. Plots are drawn through ``cachesim.plot`` where that
module has the chart type (single-series sweep, heat map) and by the small
helpers below where it does not (two-series line chart, twin-axis chart,
horizontal bar chart), all under the same house style: Agg backend, 7.0 x 4.5
inches at 120 dpi, no timestamp in the file metadata. Two runs of this script
produce byte-identical files.

``--quick`` truncates every trace to its first 40,000 accesses. It exists to
smoke-test the plotting path; the numbers it prints do NOT match the document
and must not be copied into it.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from typing import Any

from cachesim import CacheSpec, ConstantMemorySpec, Hierarchy, HierarchySpec
from cachesim.opt import opt_misses
from cachesim.plot import (
    DPI,
    FIGSIZE,
    GRID_ALPHA,
    MatplotlibUnavailable,
    format_bytes,
    plot_grid,
    plot_sweep,
)
from cachesim.policycmp import PolicyResult, compare_policies
from cachesim.stackdist import (
    StackDistanceProfile,
    block_stream,
    power_of_two_capacities,
    stack_distance_profile,
)
from cachesim.sweep import SweepRow, monotonicity_note, sweep, sweep_grid
from cachesim.trace import load_trace
from cachesim.workloads import SAMPLE_NAMES, write_traces

Accesses = list[tuple[int, bool]]

#: Traces every figure is built from.
TRACES = ("sequential", "matmul_naive", "matmul_blocked", "conflict")

#: Geometry shared by the sweeps: an L1-sized level in front of DRAM.
BLOCK_SIZE = 64
HIT_TIME = 4
MEM_TIME = 100

#: Axis values. The size and associativity lists are the ones ``cachesim
#: sweep`` uses by default, so the tables here and the command's own output
#: are the same numbers.
SIZES = tuple(kb * 1024 for kb in (1, 2, 4, 8, 16, 32, 64))
WAYS = (1, 2, 4, 8, 16)
BLOCK_SIZES = (16, 32, 64, 128, 256, 512)

#: The bus width used for the block-size figure, in bytes per cycle.
BUS_WIDTH = 16

#: Accesses kept per trace under --quick.
QUICK_ACCESSES = 40_000


# -- plotting ------------------------------------------------------------------

#: The house style of ``cachesim.plot``, rebuilt from its public constants so
#: that figures drawn here are indistinguishable from the ones the analysis
#: commands write.
RC: dict[str, Any] = {
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


def _pyplot() -> Any:
    """pyplot on the Agg backend, or ``MatplotlibUnavailable``."""
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - only without matplotlib
        raise MatplotlibUnavailable(
            'matplotlib is not installed; install it with: pip install "cachesim[plots]"'
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(plt: Any, fig: Any, out_path: str) -> str:
    """Tight-layout, write and close ``fig``; returns ``out_path``."""
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, metadata={"Software": "cachesim"})
    plt.close(fig)
    return out_path


def _log2_axis(ax: Any, ticks: Sequence[float]) -> None:
    """Base-2 log x-axis with a plain decimal label at each tick."""
    from matplotlib.ticker import ScalarFormatter

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(ticks))
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def _percent_axis(ax: Any) -> None:
    """Percent-formatted y-axis anchored at zero (values are fractions)."""
    from matplotlib.ticker import PercentFormatter

    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))


def plot_lines(
    series: Sequence[tuple[str, Sequence[int], Sequence[float]]],
    out_path: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> str:
    """Two or more labelled miss-ratio series on one base-2 log x-axis.

    ``cachesim.plot.plot_sweep`` draws one series and ``plot_mrc`` labels its
    first series "fully associative", so neither can carry two workloads
    compared against each other; this is that chart.
    """
    plt = _pyplot()
    with plt.rc_context(RC):
        fig, ax = plt.subplots()
        ticks: set[int] = set()
        for marker, (label, xs, ys) in zip("os^Dv", series, strict=False):
            ax.plot(list(xs), list(ys), marker=marker, label=label)
            ticks.update(xs)
        _log2_axis(ax, sorted(ticks))
        _percent_axis(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        return _save(plt, fig, out_path)


def plot_miss_rate_and_amat(
    xs: Sequence[int],
    miss_rates: Sequence[tuple[str, Sequence[float]]],
    amats: Sequence[tuple[str, Sequence[float]]],
    out_path: str,
    xlabel: str,
    title: str,
) -> str:
    """Miss rate (left axis, solid) against AMAT (right axis, dashed).

    The two quantities have different units and the point of the figure is
    that they need not move together, so they share an x-axis and nothing
    else. Colours pair a workload's two lines. An AMAT series whose minimum
    is interior -- the block size past which transfer time costs more than
    the misses it saves -- is annotated at that point, because the turn is
    small next to the range of the axis and is the finding.
    """
    plt = _pyplot()
    with plt.rc_context(RC):
        fig, ax = plt.subplots()
        right = ax.twinx()
        right.grid(visible=False)
        right.spines["right"].set_visible(True)
        colours = [f"C{i}" for i in range(len(miss_rates))]
        for colour, (label, ys) in zip(colours, miss_rates, strict=True):
            ax.plot(list(xs), list(ys), marker="o", color=colour, label=f"{label}: miss rate")
        for colour, (label, ys) in zip(colours, amats, strict=True):
            right.plot(
                list(xs),
                list(ys),
                marker="s",
                linestyle="--",
                color=colour,
                label=f"{label}: AMAT",
            )
            best = min(range(len(ys)), key=lambda i: ys[i])
            if 0 < best < len(ys) - 1:
                right.annotate(
                    f"minimum {ys[best]:.3f} cycles at {xs[best]} B",
                    xy=(xs[best], ys[best]),
                    xytext=(-8, 26),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                    color=colour,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                    arrowprops={"arrowstyle": "->", "color": colour, "linewidth": 0.8},
                )
        _log2_axis(ax, xs)
        _percent_axis(ax)
        right.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("miss rate")
        right.set_ylabel("AMAT (cycles)")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        extra = right.get_legend_handles_labels()
        ax.legend(handles + extra[0], labels + extra[1], loc="upper center", ncol=2)
        return _save(plt, fig, out_path)


def plot_policy_misses(
    results: Sequence[PolicyResult],
    bound: int,
    out_path: str,
    title: str,
) -> str:
    """Misses per replacement policy as horizontal bars, best at the top.

    OPT is drawn in a different colour because it is a bound rather than a
    policy, and the fully-associative Belady bound is drawn as a vertical
    line: the distance from it to the ``opt`` bar is the cost of the set
    mapping, which no replacement policy can recover.
    """
    plt = _pyplot()
    with plt.rc_context(RC):
        fig, ax = plt.subplots()
        labels = [r.policy for r in results][::-1]
        values = [r.misses for r in results][::-1]
        colours = ["C1" if label == "opt" else "C0" for label in labels]
        ax.barh(labels, values, color=colours)
        ax.axvline(bound, color="C3", linestyle="--", linewidth=1)
        ax.annotate(
            f"fully-associative OPT: {bound:,}",
            xy=(bound, len(labels) - 0.5),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="C3",
        )
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(lambda value, _pos: f"{int(value):,}")
        ax.set_xlabel("misses")
        ax.set_title(title)
        ax.set_xlim(right=max(values) * 1.08)
        for label, value in zip(labels, values, strict=True):
            ax.annotate(
                f"{value:,}",
                xy=(value, label),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        return _save(plt, fig, out_path)


# -- tables --------------------------------------------------------------------


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A Markdown table, one row per sequence of preformatted cells."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def heading(figure: str, description: str) -> None:
    """Announce the figure whose numbers follow."""
    print(f"\n## {figure}\n")
    print(f"{description}\n")


def percent(fraction: float) -> str:
    return f"{fraction * 100:.2f}%"


# -- the experiments -----------------------------------------------------------


def base_spec(size: int, associativity: int, **kwargs: Any) -> CacheSpec:
    """One cache level of the geometry the sweeps vary around."""
    return CacheSpec(
        name="L1",
        size=size,
        block_size=BLOCK_SIZE,
        associativity=associativity,
        hit_time=HIT_TIME,
        policy="lru",
        **kwargs,
    )


def figure_size_sweep(traces: dict[str, Accesses], out_dir: str) -> None:
    """L1 size sweep, 4-way, naive against blocked matrix multiply."""
    heading(
        "miss_rate_vs_size_matmul.png",
        "One 4-way level of each size, 64 B blocks, LRU, hit time 4, DRAM 100. "
        "Equivalent to `cachesim sweep --param size traces/matmul_naive.trace` "
        "and the same command on matmul_blocked.trace.",
    )
    rows: dict[str, list[SweepRow]] = {}
    for name in ("matmul_naive", "matmul_blocked"):
        rows[name] = sweep(traces[name], base_spec(SIZES[0], 4), "size", list(SIZES), MEM_TIME)
    print(
        table(
            [
                "size",
                "naive misses",
                "naive miss rate",
                "naive AMAT",
                "blocked misses",
                "blocked miss rate",
                "blocked AMAT",
            ],
            [
                [
                    format_bytes(size),
                    f"{naive.misses:,}",
                    percent(naive.miss_rate),
                    f"{naive.amat:.2f}",
                    f"{blocked.misses:,}",
                    percent(blocked.miss_rate),
                    f"{blocked.amat:.2f}",
                ]
                for size, naive, blocked in zip(
                    SIZES, rows["matmul_naive"], rows["matmul_blocked"], strict=True
                )
            ],
        )
    )
    written = plot_lines(
        [
            (name, [int(row.value) for row in rows[name]], [row.miss_rate for row in rows[name]])
            for name in ("matmul_naive", "matmul_blocked")
        ],
        os.path.join(out_dir, "miss_rate_vs_size_matmul.png"),
        xlabel="cache size (bytes)",
        ylabel="miss rate",
        title="Miss rate vs cache size (4-way, 64 B blocks, LRU)",
    )
    print(f"\nwrote {written}")


def figure_associativity(traces: dict[str, Accesses], out_dir: str) -> None:
    """Associativity sweep at a fixed 8 KB on the conflict workload."""
    heading(
        "miss_rate_vs_associativity_conflict.png",
        "One 8 KB level, 64 B blocks, LRU, hit time 4, DRAM 100. "
        "Equivalent to `cachesim sweep --param associativity traces/conflict.trace`. "
        "Capacity and conflict are the Hill and Smith aggregate decomposition.",
    )
    rows = sweep(
        traces["conflict"], base_spec(8 * 1024, WAYS[0]), "associativity", list(WAYS), MEM_TIME
    )
    print(
        table(
            ["ways", "sets", "misses", "miss rate", "AMAT", "compulsory", "capacity", "conflict"],
            [
                [
                    f"{row.value}",
                    f"{row.num_sets}",
                    f"{row.misses:,}",
                    percent(row.miss_rate),
                    f"{row.amat:.2f}",
                    f"{row.compulsory:,}",
                    f"{row.capacity:,}",
                    f"{row.conflict:,}",
                ]
                for row in rows
            ],
        )
    )
    note = monotonicity_note(rows, unit="-way")
    print(f"\nmonotonicity: {note or 'misses never rose with associativity'}")
    written = plot_sweep(
        rows,
        "associativity",
        os.path.join(out_dir, "miss_rate_vs_associativity_conflict.png"),
        title="Miss rate vs associativity (8 KB, 64 B blocks, LRU)\nconflict.trace",
    )
    print(f"wrote {written}")


def figure_grid(traces: dict[str, Accesses], out_dir: str) -> None:
    """Size against associativity on the naive matrix multiply."""
    heading(
        "grid_matmul_naive.png",
        "One level per cell, 64 B blocks, LRU. Equivalent to "
        "`cachesim sweep --grid traces/matmul_naive.trace`. A cell is empty where the "
        "size is not a whole multiple of block_size x ways. The 3-C shadow cache is "
        "off here: it does not affect a miss rate, and every cell replays the whole trace.",
    )
    grid = sweep_grid(
        traces["matmul_naive"],
        base_spec(SIZES[0], WAYS[0], track_3c=False),
        list(SIZES),
        list(WAYS),
        MEM_TIME,
    )
    matrix = grid.miss_rate_matrix()
    print(
        table(
            ["size", *[f"{ways}-way" for ways in WAYS]],
            [
                [format_bytes(size), *["-" if v != v else percent(v) for v in row]]
                for size, row in zip(SIZES, matrix, strict=True)
            ],
        )
    )
    print()
    for size, size_row in zip(SIZES, grid.rows, strict=True):
        note = monotonicity_note([cell for cell in size_row if cell is not None], unit="-way")
        if note:
            print(f"note: not monotonic at {format_bytes(size)} -- {note}")
    written = plot_grid(
        matrix,
        [format_bytes(size) for size in SIZES],
        [str(ways) for ways in WAYS],
        os.path.join(out_dir, "grid_matmul_naive.png"),
        title="Miss rate by size and associativity (64 B blocks, LRU)\nmatmul_naive.trace",
    )
    print(f"\nwrote {written}")


def figure_policies(traces: dict[str, Accesses], out_dir: str) -> None:
    """Every replacement policy against Belady's OPT, at one geometry."""
    heading(
        "policies_matmul_naive.png",
        "One 32 KB 4-way level, 64 B blocks (128 sets), hit time 4, DRAM 100. "
        "Equivalent to `cachesim policies traces/matmul_naive.trace`. "
        "The `opt` row is Belady's rule inside each set; the vertical line is the "
        "fully-associative bound, which no set-associative policy can reach.",
    )
    accesses = traces["matmul_naive"]
    results = compare_policies(accesses, 32 * 1024, BLOCK_SIZE, 4, HIT_TIME, MEM_TIME)
    blocks = [addr // BLOCK_SIZE for addr, _ in accesses]
    bound = opt_misses(blocks, 32 * 1024 // BLOCK_SIZE)
    print(
        table(
            ["policy", "misses", "miss rate", "AMAT", "x OPT"],
            [
                [
                    r.policy,
                    f"{r.misses:,}",
                    percent(r.miss_rate),
                    f"{r.amat:.2f}",
                    f"{r.gap_to_opt:.2f}",
                ]
                for r in results
            ],
        )
    )
    total = results[0].accesses
    print(f"\nfully-associative Belady bound: {bound:,} misses ({percent(bound / total)})")
    written = plot_policy_misses(
        results,
        bound,
        os.path.join(out_dir, "policies_matmul_naive.png"),
        title="Misses by replacement policy (32 KB, 4-way, 64 B blocks)\nmatmul_naive.trace",
    )
    print(f"wrote {written}")


def figure_block_size(traces: dict[str, Accesses], out_dir: str) -> None:
    """Block size against miss rate and AMAT on a bandwidth-limited link."""
    heading(
        "block_size_amat.png",
        "One 32 KB 4-way level, LRU, hit time 4, DRAM 100, `bus_width` 16 bytes per "
        "cycle, so a fill costs ceil(block_size / 16) cycles on top of the latency.",
    )
    measured: dict[str, list[tuple[float, float]]] = {}
    for name in ("sequential", "matmul_naive"):
        points = []
        for block in BLOCK_SIZES:
            spec = HierarchySpec(
                levels=(
                    CacheSpec(
                        name="L1",
                        size=32 * 1024,
                        block_size=block,
                        associativity=4,
                        hit_time=HIT_TIME,
                        policy="lru",
                        bus_width=BUS_WIDTH,
                        track_3c=False,
                    ),
                ),
                memory=ConstantMemorySpec(MEM_TIME),
            )
            hierarchy = Hierarchy.from_spec(spec)
            for addr, is_write in traces[name]:
                hierarchy.access(addr, is_write)
            points.append((hierarchy.levels[0].cache.miss_rate, hierarchy.amat()))
        measured[name] = points
    print(
        table(
            [
                "block size",
                "transfer cycles",
                "sequential miss rate",
                "sequential AMAT",
                "matmul_naive miss rate",
                "matmul_naive AMAT",
            ],
            [
                [
                    f"{block} B",
                    f"{math.ceil(block / BUS_WIDTH)}",
                    f"{seq[0] * 100:.4f}%",
                    f"{seq[1]:.3f}",
                    f"{naive[0] * 100:.4f}%",
                    f"{naive[1]:.3f}",
                ]
                for block, seq, naive in zip(
                    BLOCK_SIZES, measured["sequential"], measured["matmul_naive"], strict=True
                )
            ],
        )
    )
    best = min(range(len(BLOCK_SIZES)), key=lambda i: measured["matmul_naive"][i][1])
    print(
        f"\nlowest matmul_naive AMAT at {BLOCK_SIZES[best]} B "
        f"({measured['matmul_naive'][best][1]:.3f} cycles); "
        f"sequential AMAT falls to the last point ({measured['sequential'][-1][1]:.3f})"
    )
    written = plot_miss_rate_and_amat(
        BLOCK_SIZES,
        [(name, [p[0] for p in measured[name]]) for name in ("sequential", "matmul_naive")],
        [(name, [p[1] for p in measured[name]]) for name in ("sequential", "matmul_naive")],
        os.path.join(out_dir, "block_size_amat.png"),
        xlabel="block size (bytes)",
        title="Miss rate and AMAT vs block size (32 KB, 4-way, bus_width 16)",
    )
    print(f"wrote {written}")


def figure_mrc(traces: dict[str, Accesses], out_dir: str) -> None:
    """Fully-associative LRU miss-ratio curves from the stack-distance profiler."""
    heading(
        "mrc_matmul.png",
        "Exact fully-associative LRU miss ratios at every capacity, from one pass of "
        "the Mattson stack-distance profiler over each trace, 64 B blocks. "
        "Equivalent to `cachesim mrc traces/matmul_naive.trace` and the same command "
        "on matmul_blocked.trace.",
    )
    profiles: dict[str, StackDistanceProfile] = {}
    for name in ("matmul_naive", "matmul_blocked"):
        profiles[name] = stack_distance_profile(block_stream(traces[name], BLOCK_SIZE))
    capacities = sorted(
        set().union(*(set(power_of_two_capacities(p.max_distance + 1)) for p in profiles.values()))
    )
    print(
        table(
            [
                "capacity (blocks)",
                "size",
                "naive misses",
                "naive miss ratio",
                "blocked misses",
                "blocked miss ratio",
            ],
            [
                [
                    f"{capacity:,}",
                    format_bytes(capacity * BLOCK_SIZE),
                    f"{profiles['matmul_naive'].misses(capacity):,}",
                    percent(profiles["matmul_naive"].miss_ratio(capacity)),
                    f"{profiles['matmul_blocked'].misses(capacity):,}",
                    percent(profiles["matmul_blocked"].miss_ratio(capacity)),
                ]
                for capacity in capacities
            ],
        )
    )
    print()
    for name, profile in profiles.items():
        working_set = profile.working_set_size()
        print(
            f"{name}: {profile.distinct_blocks:,} distinct blocks, "
            f"compulsory floor {percent(profile.compulsory_ratio)}, "
            f"working set {working_set:,} blocks "
            f"({format_bytes(working_set * BLOCK_SIZE)})"
        )
    written = plot_lines(
        [
            (
                name,
                capacities,
                [profiles[name].miss_ratio(capacity) for capacity in capacities],
            )
            for name in ("matmul_naive", "matmul_blocked")
        ],
        os.path.join(out_dir, "mrc_matmul.png"),
        xlabel="cache capacity (blocks)",
        ylabel="miss ratio",
        title="Fully-associative LRU miss-ratio curve (64 B blocks)",
    )
    print(f"\nwrote {written}")


FIGURES = (
    figure_size_sweep,
    figure_associativity,
    figure_grid,
    figure_policies,
    figure_block_size,
    figure_mrc,
)


# -- driver --------------------------------------------------------------------


def load_traces(trace_dir: str, quick: bool) -> dict[str, Accesses]:
    """Read every trace a figure needs, generating the files if absent."""
    missing = [
        name
        for name in SAMPLE_NAMES
        if not os.path.exists(os.path.join(trace_dir, f"{name}.trace"))
    ]
    if missing:
        print(f"generating {len(missing)} missing trace(s) in {trace_dir}:")
        write_traces(trace_dir)
    traces = {}
    for name in TRACES:
        accesses = load_trace(os.path.join(trace_dir, f"{name}.trace"))
        traces[name] = accesses[:QUICK_ACCESSES] if quick else accesses
        print(f"loaded {name}: {len(traces[name]):,} accesses")
    return traces


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="make_figures.py",
        description="Regenerate the figures used by docs/results.md.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("docs", "figures"),
        metavar="DIR",
        help="directory for the PNG files (default: docs/figures)",
    )
    parser.add_argument(
        "--traces",
        default="traces",
        metavar="DIR",
        help="directory holding the sample traces (default: traces; generated if missing)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"use only the first {QUICK_ACCESSES:,} accesses of each trace: a smoke "
        "test of the plotting path, whose numbers do not match docs/results.md",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    traces = load_traces(args.traces, args.quick)
    if args.quick:
        print(f"\n**--quick: every trace truncated to {QUICK_ACCESSES:,} accesses.**")
    for figure in FIGURES:
        figure(traces, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
