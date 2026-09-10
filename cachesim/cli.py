"""Command-line interface: ``cachesim <subcommand> ...``.

Each subcommand lives in the module that implements it and exposes a
``register(subparsers)`` function that adds its parser and sets a ``func``
default; ``COMMANDS`` lists them in display order.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections.abc import Callable, Sequence
from typing import TypeAlias

from cachesim import __version__, run_trace
from cachesim.benchcmd import register as register_bench
from cachesim.cliargs import non_negative_int
from cachesim.config import DEFAULT_CONFIG, ConfigError, load_config
from cachesim.invariants import check_hierarchy
from cachesim.report import print_report
from cachesim.trace import FORMATS
from cachesim.tracecmd import register as register_trace_stats
from cachesim.workloads import SAMPLE_NAMES, WORKLOADS, write_traces

Subparsers: TypeAlias = "argparse._SubParsersAction[argparse.ArgumentParser]"


def _cmd_run(args: argparse.Namespace) -> int:
    config = DEFAULT_CONFIG
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, json.JSONDecodeError, ConfigError) as exc:
            sys.exit(f"error: cannot load config {args.config}: {exc}")
    try:
        hierarchy = run_trace(args.trace, config, warmup=args.warmup, fmt=args.trace_format)
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
    p.add_argument("trace", help="trace file (native, Dinero IV .din, or Valgrind Lackey)")
    p.add_argument(
        "--trace-format",
        choices=("auto", *FORMATS),
        default="auto",
        help="format of the input trace (default: auto, chosen by extension: "
        ".din is Dinero IV, .lackey/.vg is Valgrind Lackey, anything else is "
        "the native 'ADDR R|W' format; a .gz suffix is decompressed first)",
    )
    p.add_argument(
        "--config",
        metavar="FILE",
        help="JSON hierarchy config (default: built-in L1/L2/L3 config, see configs/default.json)",
    )
    p.add_argument(
        "--warmup",
        type=non_negative_int,
        default=0,
        metavar="N",
        help="simulate the first N accesses, then reset all statistics before "
        "counting the rest (default 0: report the whole trace, cold caches)",
    )
    p.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="report format: human-readable text (default) or JSON "
        "(this is the output format; --trace-format selects the input format)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="after the run, verify the simulator's internal invariants "
        "(see cachesim.invariants) and exit non-zero listing any violation",
    )
    p.set_defaults(func=_cmd_run)


def _cmd_gen_traces(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    # The names are validated here rather than with argparse ``choices``:
    # before Python 3.12 argparse checks the empty default of a ``nargs="*"``
    # positional against the choices and rejects a bare ``gen-traces``.
    unknown = [name for name in args.names if name not in WORKLOADS]
    if unknown:
        parser.error(
            f"unknown workload(s) {', '.join(map(repr, unknown))} "
            f"(choose from {', '.join(sorted(WORKLOADS))})"
        )
    if args.list:
        for workload in WORKLOADS.values():
            print(workload.name)
            print(
                textwrap.fill(
                    workload.description, width=78, initial_indent="  ", subsequent_indent="  "
                )
            )
            print(
                textwrap.fill(
                    f"expectation: {workload.expectation}",
                    width=78,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )
            )
        return 0
    print(f"generating traces in {args.out_dir}:")
    try:
        write_traces(args.out_dir, args.names or None)
    except OSError as exc:
        # A directory that cannot be created, or a full disk partway through:
        # the same one-line report the other commands give a bad path.
        sys.exit(f"error: cannot write traces in {args.out_dir}: {exc}")
    return 0


def register_gen_traces(subparsers: Subparsers) -> None:
    p = subparsers.add_parser("gen-traces", help="write synthetic workload traces")
    p.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help=f"workloads to write (default: the {len(SAMPLE_NAMES)} samples "
        f"{', '.join(SAMPLE_NAMES)}); see --list",
    )
    p.add_argument("--out-dir", default="traces", help="output directory (default: traces)")
    p.add_argument(
        "--list",
        action="store_true",
        help="print every workload's name, description and expectation, and write nothing",
    )
    p.set_defaults(func=lambda args: _cmd_gen_traces(args, p))


def _register_sweep(subparsers: Subparsers) -> None:
    from cachesim.sweep import register

    register(subparsers)


def _register_policies(subparsers: Subparsers) -> None:
    from cachesim.policycmp import register

    register(subparsers)


def _register_mrc(subparsers: Subparsers) -> None:
    from cachesim.stackdist import register

    register(subparsers)


def _register_compare(subparsers: Subparsers) -> None:
    from cachesim.compare import register

    register(subparsers)


def _register_sets(subparsers: Subparsers) -> None:
    from cachesim.setpressure import register

    register(subparsers)


#: Subcommand registration functions, in the order shown by --help.
COMMANDS: list[Callable[[Subparsers], None]] = [
    register_run,
    register_trace_stats,
    _register_sweep,
    _register_policies,
    _register_mrc,
    _register_compare,
    _register_sets,
    register_gen_traces,
    register_bench,
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
