"""Command-line interface: ``cachesim <subcommand> ...``.

Each subcommand lives in the module that implements it and exposes a
``register(subparsers)`` function that adds its parser and sets a ``func``
default; ``COMMANDS`` lists them in display order.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import TypeAlias

from cachesim import __version__, run_trace
from cachesim.config import DEFAULT_CONFIG, ConfigError, load_config
from cachesim.invariants import check_hierarchy
from cachesim.report import print_report
from cachesim.workloads import SAMPLE_TRACES, write_sample_traces

Subparsers: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"


def _cmd_run(args: argparse.Namespace) -> int:
    config = DEFAULT_CONFIG
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, json.JSONDecodeError, ConfigError) as exc:
            sys.exit(f"error: cannot load config {args.config}: {exc}")
    try:
        hierarchy = run_trace(args.trace, config, warmup=args.warmup)
    except ConfigError as exc:
        sys.exit(f"error: invalid config: {exc}")
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if hierarchy.accesses == 0:
        detail = " after the warm-up" if args.warmup else ""
        sys.exit(f"error: no accesses found in {args.trace}{detail}")
    if args.format == "json":
        print(json.dumps(hierarchy.stats().to_dict(), indent=2))
    else:
        print_report(hierarchy, trace_name=args.trace)
    if args.check:
        # Diagnostics go to stderr so that --format json stays parseable.
        report = check_hierarchy(hierarchy)
        for skipped in report.skipped:
            print(f"invariants: skipped {skipped}", file=sys.stderr)
        if report.violations:
            for violation in report.violations:
                print(f"error: invariant violated: {violation}", file=sys.stderr)
            return 1
        print(f"invariants: {report.checked} checks passed", file=sys.stderr)
    return 0


def register_run(subparsers: Subparsers) -> None:
    p = subparsers.add_parser("run", help="simulate one trace and print a report")
    p.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    p.add_argument(
        "--config",
        metavar="FILE",
        help="JSON hierarchy config (default: built-in L1/L2/L3 config, see configs/default.json)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        metavar="N",
        help="simulate the first N accesses, then reset all statistics before "
        "counting the rest (default 0: report the whole trace, cold caches)",
    )
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format: human-readable text (default) or JSON",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="after the run, verify the simulator's internal invariants "
        "(see cachesim.invariants) and exit non-zero listing any violation",
    )
    p.set_defaults(func=_cmd_run)


def _cmd_gen_traces(args: argparse.Namespace) -> int:
    print(f"generating traces in {args.out_dir}:")
    write_sample_traces(args.out_dir, args.names or None)
    return 0


def register_gen_traces(subparsers: Subparsers) -> None:
    p = subparsers.add_parser("gen-traces", help="write the sample traces")
    p.add_argument(
        "names",
        nargs="*",
        choices=sorted(SAMPLE_TRACES),
        help="which traces to write (default: all)",
    )
    p.add_argument("--out-dir", default="traces", help="output directory (default: traces)")
    p.set_defaults(func=_cmd_gen_traces)


def _register_sweep(subparsers: Subparsers) -> None:
    from cachesim.sweep import register

    register(subparsers)


#: Subcommand registration functions, in the order shown by --help.
COMMANDS: list[Callable[[Subparsers], None]] = [
    register_run,
    _register_sweep,
    register_gen_traces,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cachesim",
        description="Trace-driven cache and memory-hierarchy simulator.",
    )
    parser.add_argument("--version", action="version", version=f"cachesim {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for register in COMMANDS:
        register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
