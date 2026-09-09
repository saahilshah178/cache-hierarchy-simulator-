"""Command-line interface: ``cachesim <subcommand> ...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from cachesim import __version__, run_trace
from cachesim.config import DEFAULT_CONFIG, ConfigError, load_config
from cachesim.report import print_report
from cachesim.workloads import SAMPLE_TRACES, write_sample_traces


def _cmd_run(args: argparse.Namespace) -> int:
    config = DEFAULT_CONFIG
    if args.config:
        try:
            config = load_config(args.config)
        except (OSError, json.JSONDecodeError, ConfigError) as exc:
            sys.exit(f"error: cannot load config {args.config}: {exc}")
    try:
        hierarchy = run_trace(args.trace, config)
    except ConfigError as exc:
        sys.exit(f"error: invalid config: {exc}")
    except (OSError, ValueError) as exc:
        sys.exit(f"error: {exc}")
    if hierarchy.accesses == 0:
        sys.exit(f"error: no accesses found in {args.trace}")
    print_report(hierarchy, trace_name=args.trace)
    return 0


def _cmd_gen_traces(args: argparse.Namespace) -> int:
    print(f"generating traces in {args.out_dir}:")
    write_sample_traces(args.out_dir, args.names or None)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from cachesim.sweep import main as sweep_main

    return sweep_main(args.sweep_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cachesim", description="Trace-driven cache and memory-hierarchy simulator."
    )
    parser.add_argument("--version", action="version", version=f"cachesim {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="simulate one trace and print a report")
    p_run.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    p_run.add_argument(
        "--config",
        metavar="FILE",
        help="JSON hierarchy config (default: built-in L1/L2/L3 config, see configs/default.json)",
    )
    p_run.set_defaults(func=_cmd_run)

    p_gen = sub.add_parser("gen-traces", help="write the sample traces")
    p_gen.add_argument(
        "names",
        nargs="*",
        choices=sorted(SAMPLE_TRACES),
        help="which traces to write (default: all)",
    )
    p_gen.add_argument("--out-dir", default="traces", help="output directory (default: traces)")
    p_gen.set_defaults(func=_cmd_gen_traces)

    p_sweep = sub.add_parser(
        "sweep", help="sweep associativity and size; plot miss rates", add_help=False
    )
    p_sweep.add_argument("sweep_args", nargs=argparse.REMAINDER)
    p_sweep.set_defaults(func=_cmd_sweep)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
