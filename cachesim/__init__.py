"""cachesim: a trace-driven cache and memory-hierarchy simulator."""

from typing import Any

from cachesim.cache import Cache, VictimBuffer
from cachesim.config import (
    DEFAULT_CONFIG,
    INCLUSION_POLICIES,
    WRITE_POLICIES,
    CacheSpec,
    ConfigError,
    ConstantMemorySpec,
    HierarchySpec,
    MemorySpec,
    RowBufferMemorySpec,
    default_config,
    load_config,
    parse_config,
)
from cachesim.dram import ConstantMemory, MemoryModel, RowBufferMemory
from cachesim.hierarchy import Hierarchy, Level
from cachesim.invariants import CheckReport, check_hierarchy, check_invariants

# Importing cachesim.opt registers the offline "opt" policy in POLICIES, so
# that a config or a --policy flag can name it wherever the package is used.
from cachesim.opt import OPTPolicy, opt_misses, run_opt, simulate_with_opt
from cachesim.policies import POLICIES, ReplacementPolicy
from cachesim.prefetch import PREFETCHER_NAMES, PREFETCHERS, Prefetcher, make_prefetcher
from cachesim.report import build_report, format_report, print_report
from cachesim.stats import HierarchyStats, LevelStats, MemoryStats, ThreeCStats
from cachesim.trace import parse_trace

__version__ = "0.9.0"

__all__ = [
    "DEFAULT_CONFIG",
    "INCLUSION_POLICIES",
    "POLICIES",
    "PREFETCHERS",
    "PREFETCHER_NAMES",
    "WRITE_POLICIES",
    "Cache",
    "CacheSpec",
    "CheckReport",
    "ConfigError",
    "ConstantMemory",
    "ConstantMemorySpec",
    "Hierarchy",
    "HierarchySpec",
    "HierarchyStats",
    "Level",
    "LevelStats",
    "MemoryModel",
    "MemorySpec",
    "MemoryStats",
    "OPTPolicy",
    "Prefetcher",
    "ReplacementPolicy",
    "RowBufferMemory",
    "RowBufferMemorySpec",
    "ThreeCStats",
    "VictimBuffer",
    "__version__",
    "build_report",
    "check_hierarchy",
    "check_invariants",
    "default_config",
    "format_report",
    "load_config",
    "make_prefetcher",
    "opt_misses",
    "parse_config",
    "parse_trace",
    "print_report",
    "run_opt",
    "run_trace",
    "simulate_with_opt",
]


def run_trace(
    trace_path: str,
    config: dict[str, Any] | None = None,
    warmup: int = 0,
    fmt: str = "auto",
) -> Hierarchy:
    """Simulate one trace through one hierarchy; returns the Hierarchy.

    ``warmup`` accesses are simulated first and then the statistics are
    reset, so the returned counters describe only the remainder. ``fmt``
    selects the trace format and defaults to choosing it by extension; see
    ``cachesim.trace``.
    """
    hierarchy = Hierarchy.from_config(config or DEFAULT_CONFIG)
    accesses = parse_trace(trace_path, fmt)
    if warmup > 0:
        for n, (addr, is_write) in enumerate(accesses, start=1):
            hierarchy.access(addr, is_write)
            if n == warmup:
                break
        hierarchy.reset_stats()
    for addr, is_write in accesses:
        hierarchy.access(addr, is_write)
    return hierarchy
