"""Sweeps one cache parameter, or a grid of two, and tabulates the result.

Every point is a SINGLE cache level backed directly by memory, so the effect
of the knob being turned is not blurred by an L2 or an L3 absorbing its
misses. The same parsed access list is replayed for every point, so the only
difference between rows is the parameter.

Four parameters can be swept:

size
    Capacity in bytes. Watch capacity misses disappear once the cache holds
    the working set.
associativity
    Ways per set. Watch conflict misses melt away as ways increase, then
    flatten: past the knee, extra associativity buys nothing.
block_size
    Bytes per block. Larger blocks prefetch spatially adjacent data, which
    cuts compulsory misses but coarsens replacement and can raise conflict
    misses.
policy
    Replacement policy by name. Values are not ordered, so this sweep is
    tabulated and plotted as categories.

More of a good thing is not always better, and a sweep that goes the wrong
way is a finding rather than a bug: increasing associativity can increase
misses (LRU is not a stack algorithm across DIFFERENT set counts, and a
power-of-two stride can alias worse at one shape than another), and larger
blocks can increase misses when the extra data is never used. Every sweep is
checked for that and prints a one-line note summarising the regression.

With no trace argument the command runs the two sweeps that show their
effect most clearly: associativity on conflict.trace for the knee, size on
matmul_naive.trace for the cliff. Pass a trace to run both on it instead --
NOTE that sweeping associativity on matmul_naive is worth doing precisely
because it does NOT flatten nicely: its power-of-two stride (N=64 doubles =
512 B rows) keeps aliasing to the same sets however the cache is shaped.

Usage::

    cachesim sweep                                  # the two default sweeps
    cachesim sweep traces/pointer_chase.trace       # both sweeps on one trace
    cachesim sweep --size 16384 --policy fifo       # change the fixed knobs
    cachesim sweep --param block_size traces/sequential.trace
    cachesim sweep --param size --values 4k,16k,64k traces/matmul_naive.trace
    cachesim sweep --grid traces/matmul_naive.trace # size x associativity
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from cachesim.cache import Cache
from cachesim.config import CacheSpec
from cachesim.hierarchy import Hierarchy, Level
from cachesim.plot import (
    MatplotlibUnavailable,
    available,
    format_bytes,
    plot_grid,
    plot_sweep,
)
from cachesim.policies import POLICIES
from cachesim.trace import parse_trace

#: The parameters ``sweep`` knows how to vary.
PARAMS = ("size", "associativity", "block_size", "policy")

#: Default value list for each parameter, used when --values is omitted.
DEFAULT_VALUES: dict[str, tuple[int | str, ...]] = {
    "size": tuple(kb * 1024 for kb in (1, 2, 4, 8, 16, 32, 64)),
    "associativity": (1, 2, 4, 8, 16),
    "block_size": (16, 32, 64, 128, 256),
    "policy": tuple(sorted(POLICIES)),
}

#: Legacy shorthands kept so the two default sweeps read the same as before.
ASSOCIATIVITIES: tuple[int, ...] = (1, 2, 4, 8, 16)
SIZES_KB: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

#: The base geometry every sweep starts from; one knob is then replaced.
DEFAULT_BASE = CacheSpec(
    name="L1", size=8 * 1024, block_size=64, associativity=4, hit_time=4, policy="lru"
)

Accesses = Sequence[tuple[int, bool]]


@dataclass(frozen=True)
class SweepRow:
    """One point of a sweep: the geometry simulated and what it cost.

    ``value`` is the swept parameter's value at this point; the geometry
    fields record the whole shape, so a row is self-describing in a CSV.
    ``capacity`` and ``conflict`` are the Hill-Smith AGGREGATE decomposition
    (capacity = fully-associative LRU misses - compulsory, conflict = misses
    - fully-associative LRU misses), not the per-reference labels, so
    ``conflict`` is negative when the set mapping beats fully-associative
    LRU. See ``cachesim.cache`` for why the two taxonomies differ.
    """

    param: str
    value: int | str
    size: int
    block_size: int
    associativity: int
    policy: str
    num_sets: int
    accesses: int
    hits: int
    misses: int
    miss_rate: float
    compulsory: int
    capacity: int
    conflict: int
    shadow_misses: int
    amat: float

    @property
    def label(self) -> str:
        """The swept value formatted for a table cell (sizes in KB/MB)."""
        if self.param in ("size", "block_size") and isinstance(self.value, int):
            return format_bytes(self.value)
        return f"{self.value}"


@dataclass(frozen=True)
class SweepGrid:
    """A size x associativity grid; ``rows[s][w]`` is None when infeasible.

    A cell exists only when ``size`` is a whole multiple of
    ``block_size * ways``; the rest of the rectangle is genuinely
    unbuildable, not merely uninteresting, so it is left empty rather than
    filled with a substitute geometry.
    """

    sizes: tuple[int, ...]
    associativities: tuple[int, ...]
    block_size: int
    policy: str
    rows: tuple[tuple[SweepRow | None, ...], ...]

    def miss_rate_matrix(self) -> list[list[float]]:
        """Miss rates as a plain matrix, ``nan`` for infeasible cells."""
        return [
            [math.nan if row is None else row.miss_rate for row in size_row]
            for size_row in self.rows
        ]

    def flat_rows(self) -> list[SweepRow]:
        """Every feasible cell, row-major, for CSV output."""
        return [row for size_row in self.rows for row in size_row if row is not None]

    def non_monotonic_rows(self) -> list[tuple[int, SweepRow, SweepRow]]:
        """Sizes at which adding ways ADDED misses, as (size, before, after).

        Each grid row is an associativity sweep at a fixed size, so the
        1-D check applies to it unchanged.
        """
        found: list[tuple[int, SweepRow, SweepRow]] = []
        for size, size_row in zip(self.sizes, self.rows, strict=True):
            feasible = [cell for cell in size_row if cell is not None]
            found.extend((size, earlier, later) for earlier, later in non_monotonic_pairs(feasible))
        return found


# -- the engine ----------------------------------------------------------------


def _apply(base: CacheSpec, param: str, value: int | str) -> CacheSpec:
    """``base`` with ``param`` set to ``value``; validates the parameter name."""
    if param == "policy":
        if not isinstance(value, str):
            raise ValueError(f"policy value must be a string, got {value!r}")
        if value not in POLICIES:
            raise ValueError(f"unknown policy {value!r}; choose from {', '.join(sorted(POLICIES))}")
        return replace(base, policy=value)
    if param not in ("size", "associativity", "block_size"):
        raise ValueError(f"unknown sweep parameter {param!r}; choose from {', '.join(PARAMS)}")
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{param} value must be an integer, got {value!r}")
    if param == "size":
        return replace(base, size=value)
    if param == "associativity":
        return replace(base, associativity=value)
    return replace(base, block_size=value)


def run_point(accesses: Accesses, spec: CacheSpec, param: str, mem_time: int) -> SweepRow:
    """Simulate ``accesses`` through one cache level and snapshot the result.

    EVERY field of ``spec`` is honoured, including ``rng_seed`` and
    ``track_3c``, exactly as ``Hierarchy.from_spec`` honours them: a spec
    that is silently partly ignored would make a sweep over seeds of the
    ``random`` policy return identical rows with no error.

    ``track_3c`` defaults to true on ``CacheSpec`` and should normally stay
    there: with it off the shadow cache never runs, so ``compulsory``,
    ``capacity`` and ``shadow_misses`` are 0 and ``conflict``, being
    defined as ``misses - shadow_misses``, degenerates to the total miss
    count. Turning it off buys about a fifth of the run time (measured on
    matmul_naive.trace) for a sweep that only wants miss rates and AMAT.
    """
    try:
        cache = Cache(
            name=spec.name,
            size=spec.size,
            block_size=spec.block_size,
            associativity=spec.associativity,
            policy=spec.policy,
            track_3c=spec.track_3c,
            rng_seed=spec.rng_seed,
        )
    except ValueError as exc:
        raise ValueError(f"{param}: {exc}") from None
    hierarchy = Hierarchy([Level(cache, spec.hit_time)], mem_time)
    for addr, is_write in accesses:
        hierarchy.access(addr, is_write)
    value: int | str = spec.policy if param == "policy" else getattr(spec, param)
    return SweepRow(
        param=param,
        value=value,
        size=spec.size,
        block_size=spec.block_size,
        associativity=spec.associativity,
        policy=spec.policy,
        num_sets=cache.num_sets,
        accesses=cache.accesses,
        hits=cache.hits,
        misses=cache.misses,
        miss_rate=cache.miss_rate,
        compulsory=cache.compulsory_misses,
        capacity=cache.capacity_misses_aggregate,
        conflict=cache.conflict_misses_aggregate,
        shadow_misses=cache.shadow_misses,
        amat=hierarchy.amat(),
    )


def sweep(
    accesses: Accesses,
    base: CacheSpec,
    param: str,
    values: Sequence[int | str],
    mem_time: int = 100,
) -> list[SweepRow]:
    """Replay ``accesses`` once per value of ``param``; one row per value.

    ``base`` supplies every field except the one being swept: the fixed
    geometry, the hit time (``base.hit_time``), the RNG seed
    (``base.rng_seed``, which is what the ``random`` policy draws from) and
    ``base.track_3c``. ``mem_time`` is the memory access time behind the
    single level. Raises ``ValueError`` naming the parameter if a value
    produces a geometry that cannot be built, for example a size that is
    not a multiple of ``block_size * associativity``.
    """
    return [run_point(accesses, _apply(base, param, value), param, mem_time) for value in values]


def sweep_grid(
    accesses: Accesses,
    base: CacheSpec,
    sizes: Sequence[int],
    associativities: Sequence[int],
    mem_time: int = 100,
) -> SweepGrid:
    """Sweep size against associativity; infeasible combinations stay empty.

    The two axes are not independent -- a 1 KB cache cannot be 16-way with
    64 B blocks -- so the grid is triangular in practice. ``param`` on each
    row is reported as ``"size x associativity"``.
    """
    rows: list[tuple[SweepRow | None, ...]] = []
    for size in sizes:
        row: list[SweepRow | None] = []
        for ways in associativities:
            if size % (base.block_size * ways) != 0:
                row.append(None)
                continue
            spec = replace(base, size=size, associativity=ways)
            # Each row of the grid is an associativity sweep at a fixed size,
            # so cells carry associativity as their swept value: the 1-D
            # tables, CSV schema and monotonicity check all apply unchanged.
            row.append(run_point(accesses, spec, "associativity", mem_time))
        rows.append(tuple(row))
    return SweepGrid(
        sizes=tuple(sizes),
        associativities=tuple(associativities),
        block_size=base.block_size,
        policy=base.policy,
        rows=tuple(rows),
    )


def non_monotonic_pairs(rows: Sequence[SweepRow]) -> list[tuple[SweepRow, SweepRow]]:
    """Consecutive pairs where a LARGER parameter value caused MORE misses.

    Returns an empty list for an unordered parameter (policy), where the
    notion does not apply.
    """
    if not rows or rows[0].param == "policy":
        return []
    pairs = []
    for earlier, later in itertools.pairwise(rows):
        if not isinstance(earlier.value, int) or not isinstance(later.value, int):
            return []
        if later.value > earlier.value and later.misses > earlier.misses:
            pairs.append((earlier, later))
    return pairs


def monotonicity_note(rows: Sequence[SweepRow], unit: str = "") -> str:
    """One line summarising a sweep that went the wrong way, or "".

    Reports how many steps increased misses out of how many, and the best
    and worst points, rather than one line per offending pair: a sweep can
    regress at every step and the per-pair detail adds nothing. ``unit`` is
    appended to the value labels (``"-way"`` for a grid row).
    """
    offenders = non_monotonic_pairs(rows)
    if not offenders:
        return ""
    best = min(rows, key=lambda row: row.misses)
    worst = max(rows, key=lambda row: row.misses)
    return (
        f"{len(offenders)} of {len(rows) - 1} steps increased misses; "
        f"best {best.label}{unit} ({best.misses:,}), "
        f"worst {worst.label}{unit} ({worst.misses:,})"
    )


# -- CSV -----------------------------------------------------------------------

CSV_COLUMNS = (
    "param",
    "value",
    "size",
    "block_size",
    "associativity",
    "policy",
    "num_sets",
    "accesses",
    "hits",
    "misses",
    "miss_rate",
    "compulsory",
    "capacity",
    "conflict",
    "shadow_misses",
    "amat",
)


def write_csv(path: str, rows: Sequence[SweepRow]) -> None:
    """Write sweep rows as CSV; a grid uses the same schema as a 1-D sweep.

    Every counter travels with the row rather than only the miss rate, so a
    grid CSV can be re-analysed without re-running the sweep.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row.param,
                    row.value,
                    row.size,
                    row.block_size,
                    row.associativity,
                    row.policy,
                    row.num_sets,
                    row.accesses,
                    row.hits,
                    row.misses,
                    f"{row.miss_rate:.6f}",
                    row.compulsory,
                    row.capacity,
                    row.conflict,
                    row.shadow_misses,
                    f"{row.amat:.4f}",
                ]
            )


# -- printing ------------------------------------------------------------------


def _fixed_description(base: CacheSpec, param: str) -> str:
    """The knobs held fixed during a sweep, for the table heading."""
    parts = []
    if param != "size":
        parts.append(format_bytes(base.size))
    if param != "associativity":
        parts.append(f"{base.associativity}-way")
    if param != "block_size":
        parts.append(f"{base.block_size} B blocks")
    if param != "policy":
        parts.append(base.policy.upper())
    return ", ".join(parts)


def print_sweep(rows: Sequence[SweepRow], base: CacheSpec, param: str) -> None:
    """Print one sweep as a table, followed by any non-monotonicity note."""
    if not rows:
        return
    print(f"\n=== {param} sweep ({_fixed_description(base, param)}) ===")
    print(
        f"{param:>10} {'miss rate':>10} {'AMAT':>8}   "
        f"{'compulsory':>10} {'capacity':>9} {'conflict':>9} {'FA-LRU':>9}"
    )
    for row in rows:
        print(
            f"{row.label:>10} {row.miss_rate:>9.2%} {row.amat:>8.2f}   "
            f"{row.compulsory:>10,} {row.capacity:>9,} "
            f"{row.conflict:>9,} {row.shadow_misses:>9,}"
        )
    note = monotonicity_note(rows)
    if note:
        print(f"note: not monotonic -- {note}; a larger {param} made this trace worse")


def print_grid(grid: SweepGrid) -> None:
    """Print a size x associativity grid of miss rates; '-' where infeasible."""
    print(
        f"\n=== size x associativity grid "
        f"({grid.block_size} B blocks, {grid.policy.upper()}, miss rate) ==="
    )
    header = "".join(f"{f'{ways}-way':>10}" for ways in grid.associativities)
    print(f"{'size':>10}{header}")
    for size, row in zip(grid.sizes, grid.rows, strict=True):
        cells = "".join(
            f"{'-':>10}" if cell is None else f"{cell.miss_rate:>9.2%} " for cell in row
        )
        print(f"{format_bytes(size):>10}{cells}")
    for size, size_row in zip(grid.sizes, grid.rows, strict=True):
        note = monotonicity_note([cell for cell in size_row if cell is not None], unit="-way")
        if note:
            print(
                f"note: not monotonic at {format_bytes(size)} -- {note}; "
                "at a fixed size, more ways means fewer sets"
            )


# -- CLI -----------------------------------------------------------------------


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text}")
    return value


def parse_size(text: str) -> int:
    """``'4k'`` -> 4096, ``'2M'`` -> 2097152, ``'512'`` -> 512.

    A bare number is bytes; a ``k``/``K`` or ``m``/``M`` suffix (with an
    optional trailing ``b``/``B``) multiplies by 1024 or 1024*1024.
    """
    cleaned = text.strip().lower().removesuffix("b")
    scale = 1
    if cleaned.endswith("k"):
        scale, cleaned = 1 << 10, cleaned[:-1]
    elif cleaned.endswith("m"):
        scale, cleaned = 1 << 20, cleaned[:-1]
    try:
        value = int(cleaned) * scale
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (try 4096, 4k or 1M)") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive, got {text!r}")
    return value


def parse_values(param: str, text: str) -> list[int | str]:
    """Parse a comma-separated --values list according to ``param``.

    Sizes and block sizes accept the k/M suffixes; associativity takes plain
    integers; policy takes names checked against the registry.
    """
    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--values must list at least one value")
    if param == "policy":
        for item in items:
            if item.lower() not in POLICIES:
                raise argparse.ArgumentTypeError(
                    f"unknown policy {item!r}; choose from {', '.join(sorted(POLICIES))}"
                )
        return [item.lower() for item in items]
    if param == "associativity":
        return [_positive_int(item) for item in items]
    return [parse_size(item) for item in items]


def _values_or_error(parser: argparse.ArgumentParser, param: str, text: str) -> list[int | str]:
    """``parse_values``, reporting a bad list as a CLI error.

    ``--values`` cannot use argparse's ``type=`` hook because how it parses
    depends on ``--param``, so the ArgumentTypeError is turned into a
    ``parser.error`` here instead of escaping as a traceback.
    """
    try:
        return parse_values(param, text)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))


def _load_trace(parser: argparse.ArgumentParser, path: str) -> list[tuple[int, bool]]:
    """Parse a whole trace into memory, turning problems into CLI errors."""
    try:
        trace = list(parse_trace(path))
    except FileNotFoundError:
        parser.error(f"trace not found: {path} (run `cachesim gen-traces` to create the samples)")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not trace:
        parser.error(f"no accesses in {path}")
    return trace


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the sweep options to ``parser``."""
    parser.add_argument(
        "trace",
        nargs="?",
        default=None,
        help="trace file (default: conflict.trace for the associativity sweep, "
        "matmul_naive.trace for the size sweep)",
    )
    parser.add_argument(
        "--param",
        choices=PARAMS,
        default=None,
        help="sweep this one parameter instead of the two default sweeps",
    )
    parser.add_argument(
        "--values",
        default=None,
        metavar="LIST",
        help="comma-separated values for --param, or the size axis of --grid "
        "(sizes accept a k/M suffix, e.g. 4k,16k,64k); "
        "default depends on the parameter",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="sweep size against associativity and print a grid of miss rates",
    )
    parser.add_argument(
        "--ways",
        default=None,
        metavar="LIST",
        help="comma-separated associativity axis for --grid (default 1,2,4,8,16)",
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        default=8 * 1024,
        help="fixed size for the associativity sweep (default 8192 = 8 KB)",
    )
    parser.add_argument(
        "--assoc",
        type=_positive_int,
        default=4,
        help="fixed associativity for the size sweep (default 4)",
    )
    parser.add_argument(
        "--block-size", type=parse_size, default=64, help="block size in bytes (default 64)"
    )
    parser.add_argument(
        "--policy",
        default="lru",
        # Offline policies need the whole reference stream of the level up
        # front, which a sweep cannot supply: use the `policies` command.
        choices=sorted(name for name, cls in POLICIES.items() if not cls.sees_references),
        help="replacement policy (default lru; for the offline optimum see `cachesim policies`)",
    )
    parser.add_argument(
        "--hit-time", type=_non_negative_int, default=4, help="cache hit time in cycles (default 4)"
    )
    parser.add_argument(
        "--mem-time",
        type=_non_negative_int,
        default=100,
        help="memory access time in cycles (default 100)",
    )
    parser.add_argument("--csv", metavar="FILE", help="write every row to FILE as CSV")
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="directory for the plots (default: plots, when matplotlib is installed)",
    )
    parser.add_argument(
        "--out-dir",
        dest="plot_dir",
        help=argparse.SUPPRESS,  # kept as an alias for --plot-dir
    )
    parser.add_argument("--no-plot", action="store_true", help="skip plotting entirely")


def _base_spec(args: argparse.Namespace, size: int, associativity: int) -> CacheSpec:
    return CacheSpec(
        name="L1",
        size=size,
        block_size=args.block_size,
        associativity=associativity,
        hit_time=args.hit_time,
        policy=args.policy,
    )


def _trace_base(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _maybe_plot(
    args: argparse.Namespace,
    draw: Any,
    *draw_args: Any,
    **draw_kwargs: Any,
) -> None:
    """Draw a plot unless plotting is off, reporting what was written.

    Plots are drawn when --plot-dir was given explicitly, or when matplotlib
    is installed; --no-plot always wins. A missing matplotlib is a note, not
    an error: the tables above already tell the story.
    """
    if args.no_plot:
        return
    if args.plot_dir is None and not available():
        return
    try:
        print(f"saved {draw(*draw_args, **draw_kwargs)}")
    except MatplotlibUnavailable as exc:
        print(f"({exc})")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute a sweep from parsed arguments."""
    if args.grid and args.param:
        parser.error("--grid and --param are mutually exclusive")
    if args.ways and not args.grid:
        parser.error("--ways only applies to --grid")
    plot_dir = args.plot_dir if args.plot_dir is not None else "plots"

    if args.grid:
        return _run_grid(args, parser, plot_dir)
    if args.param:
        return _run_single(args, parser, plot_dir)
    return _run_default(args, parser, plot_dir)


def _run_single(args: argparse.Namespace, parser: argparse.ArgumentParser, plot_dir: str) -> int:
    """--param: one sweep over one parameter on one trace."""
    if not args.trace:
        parser.error(f"--param {args.param} needs a trace argument")
    values = (
        _values_or_error(parser, args.param, args.values)
        if args.values
        else list(DEFAULT_VALUES[args.param])
    )
    accesses = _load_trace(parser, args.trace)
    print(f"trace: {args.trace} ({len(accesses):,} accesses)")
    base = _base_spec(args, args.size, args.assoc)
    try:
        rows = sweep(accesses, base, args.param, values, args.mem_time)
    except ValueError as exc:
        parser.error(str(exc))
    print_sweep(rows, base, args.param)
    if args.csv:
        write_csv(args.csv, rows)
        print(f"\nwrote {args.csv}")
    name = _trace_base(args.trace)
    _maybe_plot(
        args,
        plot_sweep,
        rows,
        args.param,
        os.path.join(plot_dir, f"miss_rate_vs_{args.param}_{name}.png"),
        title=f"Miss rate vs {args.param.replace('_', ' ')}\n"
        f"{name} trace, {_fixed_description(base, args.param)}",
    )
    return 0


def _run_grid(args: argparse.Namespace, parser: argparse.ArgumentParser, plot_dir: str) -> int:
    """--grid: size against associativity."""
    trace_path = args.trace or "traces/matmul_naive.trace"
    axis = args.values or ",".join(str(v) for v in DEFAULT_VALUES["size"])
    sizes = [int(v) for v in _values_or_error(parser, "size", axis)]
    ways_axis = args.ways or ",".join(str(v) for v in ASSOCIATIVITIES)
    ways = [int(v) for v in _values_or_error(parser, "associativity", ways_axis)]
    accesses = _load_trace(parser, trace_path)
    print(f"trace: {trace_path} ({len(accesses):,} accesses)")
    base = _base_spec(args, args.size, args.assoc)
    grid = sweep_grid(accesses, base, sizes, ways, args.mem_time)
    print_grid(grid)
    if args.csv:
        write_csv(args.csv, grid.flat_rows())
        print(f"\nwrote {args.csv}")
    name = _trace_base(trace_path)
    _maybe_plot(
        args,
        plot_grid,
        grid.miss_rate_matrix(),
        [format_bytes(size) for size in grid.sizes],
        [str(w) for w in grid.associativities],
        os.path.join(plot_dir, f"miss_rate_grid_{name}.png"),
        title=f"Miss rate: size x associativity\n"
        f"{name} trace, {grid.block_size} B blocks, {grid.policy.upper()}",
    )
    return 0


def _run_default(args: argparse.Namespace, parser: argparse.ArgumentParser, plot_dir: str) -> int:
    """No --param and no --grid: the two classic sweeps."""
    assoc_path = args.trace or "traces/conflict.trace"
    size_path = args.trace or "traces/matmul_naive.trace"

    for ways in ASSOCIATIVITIES:
        if args.size % (args.block_size * ways):
            parser.error(
                f"--size {args.size} is not a multiple of "
                f"block_size*ways ({args.block_size}*{ways}); pick a power-of-two size"
            )
    for size in DEFAULT_VALUES["size"]:
        if int(size) % (args.block_size * args.assoc):
            parser.error(
                f"size {format_bytes(int(size))} is not a multiple of block_size*--assoc "
                f"({args.block_size}*{args.assoc}); pick a power-of-two associativity"
            )

    assoc_accesses = _load_trace(parser, assoc_path)
    print(f"associativity sweep trace: {assoc_path} ({len(assoc_accesses):,} accesses)")
    assoc_base = _base_spec(args, args.size, args.assoc)
    assoc_rows = sweep(
        assoc_accesses, assoc_base, "associativity", list(ASSOCIATIVITIES), args.mem_time
    )
    print_sweep(assoc_rows, assoc_base, "associativity")

    size_accesses = assoc_accesses if size_path == assoc_path else _load_trace(parser, size_path)
    print(f"\nsize sweep trace: {size_path} ({len(size_accesses):,} accesses)")
    size_base = _base_spec(args, args.size, args.assoc)
    size_rows = sweep(size_accesses, size_base, "size", list(DEFAULT_VALUES["size"]), args.mem_time)
    print_sweep(size_rows, size_base, "size")

    if args.csv:
        write_csv(args.csv, [*assoc_rows, *size_rows])
        print(f"\nwrote {args.csv}")

    assoc_name, size_name = _trace_base(assoc_path), _trace_base(size_path)
    _maybe_plot(
        args,
        plot_sweep,
        assoc_rows,
        "associativity",
        os.path.join(plot_dir, f"miss_rate_vs_associativity_{assoc_name}.png"),
        title=f"Miss rate vs associativity\n{assoc_name} trace, "
        f"{_fixed_description(assoc_base, 'associativity')}",
    )
    _maybe_plot(
        args,
        plot_sweep,
        size_rows,
        "size",
        os.path.join(plot_dir, f"miss_rate_vs_size_{size_name}.png"),
        title=f"Miss rate vs size\n{size_name} trace, {_fixed_description(size_base, 'size')}",
    )
    return 0


def register(subparsers: Any) -> None:
    """Register the ``sweep`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "sweep", help="sweep a cache parameter (or a size x associativity grid)"
    )
    add_arguments(parser)
    parser.set_defaults(func=lambda args: run(args, parser))


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.sweep``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim sweep",
        description="Sweep cache parameters; tabulate, plot and export miss rates.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


# -- backwards-compatible convenience wrappers ---------------------------------


def run_single_level(
    trace: Accesses,
    size: int,
    block_size: int,
    associativity: int,
    policy: str,
    hit_time: int,
    mem_time: int,
) -> tuple[float, float, Cache]:
    """Simulate ``trace`` through one cache level.

    Returns ``(miss_rate, amat, cache)``. Kept as the lowest-level entry
    point for callers that want the ``Cache`` object itself rather than a
    ``SweepRow``.
    """
    cache = Cache(
        name="L1", size=size, block_size=block_size, associativity=associativity, policy=policy
    )
    hierarchy = Hierarchy([Level(cache, hit_time)], mem_time)
    for addr, is_write in trace:
        hierarchy.access(addr, is_write)
    return cache.miss_rate, hierarchy.amat(), cache


def sweep_associativity(
    trace: Accesses,
    size: int,
    block_size: int,
    policy: str,
    hit_time: int,
    mem_time: int,
    values: Sequence[int] = ASSOCIATIVITIES,
) -> list[SweepRow]:
    """Sweep associativity at a fixed size, printing the table."""
    base = CacheSpec(
        name="L1",
        size=size,
        block_size=block_size,
        associativity=1,
        hit_time=hit_time,
        policy=policy,
    )
    rows = sweep(trace, base, "associativity", list(values), mem_time)
    print_sweep(rows, base, "associativity")
    return rows


def sweep_size(
    trace: Accesses,
    associativity: int,
    block_size: int,
    policy: str,
    hit_time: int,
    mem_time: int,
    values: Sequence[int] | None = None,
) -> list[SweepRow]:
    """Sweep size at a fixed associativity, printing the table.

    ``values`` are sizes in BYTES (the default list is 1 KB to 64 KB).
    """
    sizes = list(values) if values is not None else [int(v) for v in DEFAULT_VALUES["size"]]
    base = CacheSpec(
        name="L1",
        size=sizes[0],
        block_size=block_size,
        associativity=associativity,
        hit_time=hit_time,
        policy=policy,
    )
    rows = sweep(trace, base, "size", list(sizes), mem_time)
    print_sweep(rows, base, "size")
    return rows


if __name__ == "__main__":
    sys.exit(main())
