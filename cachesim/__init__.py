"""cachesim: a trace-driven cache and memory-hierarchy simulator."""

from typing import Any

from cachesim.cache import Cache
from cachesim.config import (
    DEFAULT_CONFIG,
    CacheSpec,
    ConfigError,
    HierarchySpec,
    default_config,
    load_config,
    parse_config,
)
from cachesim.hierarchy import Hierarchy, Level
from cachesim.invariants import CheckReport, check_hierarchy, check_invariants
from cachesim.policies import POLICIES, ReplacementPolicy
from cachesim.report import build_report, format_report, print_report
from cachesim.stats import HierarchyStats, LevelStats, ThreeCStats
from cachesim.trace import parse_trace

__version__ = "0.9.0"

__all__ = [
    "DEFAULT_CONFIG",
    "POLICIES",
    "Cache",
    "CacheSpec",
    "CheckReport",
    "ConfigError",
    "Hierarchy",
    "HierarchySpec",
    "HierarchyStats",
    "Level",
    "LevelStats",
    "ReplacementPolicy",
    "ThreeCStats",
    "__version__",
    "build_report",
    "check_hierarchy",
    "check_invariants",
    "default_config",
    "format_report",
    "load_config",
    "parse_config",
    "parse_trace",
    "print_report",
    "run_trace",
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
