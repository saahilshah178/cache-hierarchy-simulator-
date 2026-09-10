"""Runs one trace through several hierarchy configurations side by side.

Answering "is this configuration better?" from two separate ``cachesim run``
reports means reading two screens of numbers and doing the subtraction by
hand. This command replays ONE trace through every configuration and prints
one aligned table, so the columns that matter -- per-level local miss rates,
DRAM traffic, AMAT and total cycles -- sit next to each other, with the
difference from the first configuration spelled out.

The first ``--config`` is the baseline: its delta columns are blank and
every other row is measured against it. Deltas are signed, and negative is
better (fewer cycles, lower AMAT).

The trace is parsed ONCE and the resulting access list is replayed through
each hierarchy, so the comparison cannot be distorted by a trace that
changed between runs, and parsing is not paid for repeatedly.

Configurations may differ in any way, including in how many levels they
have. The level columns are the union of level names in the order they
first appear, and a configuration without a given level shows ``-`` for it.
Because the level column is a NAME, comparing a two-level configuration
against a three-level one lines up sensibly as long as the names agree.

Usage::

    cachesim compare traces/matmul_naive.trace --config a.json --config b.json
    cachesim compare traces/sequential.trace --config default --config big-l1.json
    cachesim compare traces/random.trace --config a.json --config b.json --format json

``--config default`` is the built-in hierarchy, so a candidate can be
compared against the shipped defaults without keeping a copy of them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cachesim.cliargs import load_trace_or_error, non_negative_int
from cachesim.config import ConfigError, default_config, load_config, parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.opt import simulate_with_opt, uses_opt
from cachesim.stats import HierarchyStats

#: ``--config default`` means the built-in hierarchy rather than a file.
DEFAULT_CONFIG_NAME = "default"

Accesses = Sequence[tuple[int, bool]]


@dataclass(frozen=True)
class Comparison:
    """One configuration's result: how it was named, and what it measured."""

    label: str
    source: str
    stats: HierarchyStats

    def miss_rate(self, level_name: str) -> float | None:
        """Local miss rate of the named level, or None if it has no such level."""
        for level in self.stats.levels:
            if level.name == level_name:
                return level.local_miss_rate
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON form: the statistics snapshot, tagged with its configuration."""
        return {"label": self.label, "config": self.source, **self.stats.to_dict()}


def simulate(accesses: Accesses, config: Mapping[str, Any], warmup: int = 0) -> HierarchyStats:
    """Replay ``accesses`` through one configuration and snapshot the result.

    ``warmup`` accesses are simulated first and the statistics then reset, so
    the reported figures describe steady state rather than cold caches. The
    cache contents survive the reset; see ``Hierarchy.reset_stats``.
    """
    spec = parse_config(config)
    if uses_opt(spec):
        return simulate_with_opt(list(accesses), spec, warmup=warmup).stats()
    hierarchy = Hierarchy.from_spec(spec)
    reset = warmup <= 0
    for index, (addr, is_write) in enumerate(accesses):
        if not reset and index == warmup:
            hierarchy.reset_stats()
            reset = True
        hierarchy.access(addr, is_write)
    if not reset:
        # The warm-up swallowed the whole trace: nothing was measured.
        hierarchy.reset_stats()
    return hierarchy.stats()


def compare(
    accesses: Accesses,
    configs: Sequence[tuple[str, str, Mapping[str, Any]]],
    warmup: int = 0,
) -> list[Comparison]:
    """Run ``accesses`` through each ``(label, source, config)`` in order.

    Returns one ``Comparison`` per configuration, in the order given; the
    first is the baseline the table's delta columns refer to.
    """
    return [
        Comparison(label=label, source=source, stats=simulate(accesses, config, warmup))
        for label, source, config in configs
    ]


def level_names(results: Sequence[Comparison]) -> list[str]:
    """Every level name across ``results``, in order of first appearance."""
    names: list[str] = []
    for result in results:
        for level in result.stats.levels:
            if level.name not in names:
                names.append(level.name)
    return names


def format_table(results: Sequence[Comparison]) -> str:
    """Render the comparison as one aligned table.

    Columns: the local miss rate of every level, DRAM reads and writes,
    analytic AMAT with its signed difference from the baseline in cycles,
    and total simulated cycles with its signed difference as a percentage.
    """
    if not results:
        return ""
    names = level_names(results)
    width = max(len(result.label) for result in results)
    width = max(width, len("config"))
    baseline = results[0]

    header = f"{'config':<{width}}"
    for name in names:
        header += f"{name + ' miss':>10}"
    header += f"{'dram rd':>11}{'dram wr':>11}{'AMAT':>9}{'d AMAT':>9}"
    header += f"{'cycles':>14}{'d cycles':>10}"
    lines = [header]

    for result in results:
        stats = result.stats
        line = f"{result.label:<{width}}"
        for name in names:
            rate = result.miss_rate(name)
            line += f"{'-':>10}" if rate is None else f"{rate:>9.2%} "
        line += f"{stats.dram_reads:>11,}{stats.dram_writes:>11,}{stats.amat:>9.3f}"
        if result is baseline:
            line += f"{'-':>9}"
        else:
            line += f"{stats.amat - baseline.stats.amat:>+9.3f}"
        line += f"{stats.total_cycles:>14,}"
        if result is baseline:
            line += f"{'-':>10}"
        elif baseline.stats.total_cycles:
            delta = stats.total_cycles / baseline.stats.total_cycles - 1.0
            line += f"{delta:>+9.2%} "
        else:
            line += f"{'n/a':>10}"
        lines.append(line)
    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------


def _label_for(path: str) -> str:
    if path == DEFAULT_CONFIG_NAME:
        return DEFAULT_CONFIG_NAME
    return os.path.splitext(os.path.basename(path))[0]


def load_configs(paths: Sequence[str]) -> list[tuple[str, str, Mapping[str, Any]]]:
    """Load each path into ``(label, source, config)``.

    The label is the file's base name, or ``default`` for the built-in
    hierarchy. If two paths share a base name the full paths are used
    instead, so every row of the table is identifiable.
    """
    labels = [_label_for(path) for path in paths]
    if len(set(labels)) != len(labels):
        labels = list(paths)
    loaded: list[tuple[str, str, Mapping[str, Any]]] = []
    for label, path in zip(labels, paths, strict=True):
        config = default_config() if path == DEFAULT_CONFIG_NAME else load_config(path)
        loaded.append((label, path, config))
    return loaded


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the compare options to ``parser``."""
    parser.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="FILE",
        dest="configs",
        help="a hierarchy config to compare; repeat for each one (at least two). "
        f"Use {DEFAULT_CONFIG_NAME!r} for the built-in hierarchy. "
        "The first is the baseline the deltas are measured against.",
    )
    parser.add_argument(
        "--warmup",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="simulate the first N accesses, then reset all statistics before "
        "counting the rest (default 0: report the whole trace, cold caches)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format: aligned table (default) or a JSON list of stats",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Execute the ``compare`` command from parsed arguments."""
    paths = args.configs or []
    if len(paths) < 2:
        parser.error("compare needs at least two --config values (the first is the baseline)")

    try:
        configs = load_configs(paths)
    except (OSError, json.JSONDecodeError, ConfigError) as exc:
        parser.error(f"cannot load config: {exc}")

    accesses = load_trace_or_error(parser, args.trace)
    if args.warmup >= len(accesses):
        parser.error(
            f"--warmup {args.warmup} leaves nothing to measure "
            f"in {args.trace} ({len(accesses):,} accesses)"
        )

    try:
        results = compare(accesses, configs, args.warmup)
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0

    measured = len(accesses) - args.warmup
    print(f"trace: {args.trace} ({measured:,} accesses measured of {len(accesses):,})")
    print(f"baseline: {results[0].label}\n")
    print(format_table(results))
    return 0


def register(subparsers: Any) -> None:
    """Register the ``compare`` subcommand with the ``cachesim`` CLI."""
    parser = subparsers.add_parser(
        "compare", help="run one trace through several configs and tabulate the differences"
    )
    add_arguments(parser)
    parser.set_defaults(func=lambda args: run(args, parser))


def main(argv: Sequence[str] | None = None) -> int:
    """Stand-alone entry point (``python -m cachesim.compare``)."""
    parser = argparse.ArgumentParser(
        prog="cachesim compare",
        description="Compare hierarchy configurations on one trace.",
    )
    add_arguments(parser)
    return run(parser.parse_args(argv), parser)


if __name__ == "__main__":
    sys.exit(main())
