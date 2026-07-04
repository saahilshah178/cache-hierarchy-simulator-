"""simulator.py — reads a memory-access trace and drives it through a hierarchy.

Trace format (one access per line)::

    0x00400000 R
    0x00400008 R
    0x7fff0010 W

  * first field:  the byte address, hex ("0x" prefix optional).
  * second field: R (read/load) or W (write/store), case-insensitive.
  * blank lines and lines starting with '#' are ignored.

Usage::

    python3 simulator.py traces/sequential.trace
    python3 simulator.py traces/matmul_naive.trace --config myconfig.json

Without --config you get a realistic default three-level hierarchy
(see DEFAULT_CONFIG below). A config file is JSON with the same shape.
"""

import argparse
import json
import sys

from hierarchy import Hierarchy
from report import print_report


#: The default hierarchy: sizes, latencies, and shapes typical of a modern
#: desktop core. Copy this into a .json file and edit to experiment.
DEFAULT_CONFIG = {
    "memory_access_time": 100,     # cycles to reach DRAM
    "levels": [
        {"name": "L1", "size": 32 * 1024,   "block_size": 64,
         "associativity": 4,  "policy": "lru", "hit_time": 4},
        {"name": "L2", "size": 256 * 1024,  "block_size": 64,
         "associativity": 8,  "policy": "lru", "hit_time": 12},
        {"name": "L3", "size": 2 * 1024 * 1024, "block_size": 64,
         "associativity": 16, "policy": "lru", "hit_time": 40},
    ],
}


def parse_trace(path):
    """Yield (address, is_write) tuples from a trace file.

    Malformed lines raise with the offending line number so bad traces are
    easy to debug.
    """
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'ADDR R|W', got {line!r}")
            addr_str, op = parts
            try:
                addr = int(addr_str, 16)
            except ValueError:
                raise ValueError(f"{path}:{lineno}: bad hex address {addr_str!r}") from None
            op = op.upper()
            if op not in ("R", "W"):
                raise ValueError(f"{path}:{lineno}: op must be R or W, got {op!r}")
            yield addr, op == "W"


def run_trace(trace_path, config=None):
    """Simulate one trace through one hierarchy; returns the Hierarchy."""
    hierarchy = Hierarchy.from_config(config or DEFAULT_CONFIG)
    for addr, is_write in parse_trace(trace_path):
        hierarchy.access(addr, is_write)
    return hierarchy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Trace-driven cache hierarchy simulator.")
    parser.add_argument("trace", help="trace file (one 'ADDR R|W' per line)")
    parser.add_argument("--config", metavar="FILE",
                        help="JSON hierarchy config (default: built-in "
                             "L1/L2/L3 config, see simulator.DEFAULT_CONFIG)")
    args = parser.parse_args(argv)

    config = DEFAULT_CONFIG
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    hierarchy = run_trace(args.trace, config)
    if hierarchy.accesses == 0:
        sys.exit(f"error: no accesses found in {args.trace}")
    print_report(hierarchy, trace_name=args.trace)


if __name__ == "__main__":
    main()
