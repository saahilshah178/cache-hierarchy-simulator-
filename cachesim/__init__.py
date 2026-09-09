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
from cachesim.policies import POLICIES, ReplacementPolicy
from cachesim.report import build_report, print_report
from cachesim.trace import parse_trace

__version__ = "0.9.0"

__all__ = [
    "DEFAULT_CONFIG",
    "POLICIES",
    "Cache",
    "CacheSpec",
    "ConfigError",
    "Hierarchy",
    "HierarchySpec",
    "Level",
    "ReplacementPolicy",
    "__version__",
    "build_report",
    "default_config",
    "load_config",
    "parse_config",
    "parse_trace",
    "print_report",
    "run_trace",
]


def run_trace(trace_path: str, config: dict[str, Any] | None = None) -> Hierarchy:
    """Simulate one trace through one hierarchy; returns the Hierarchy."""
    hierarchy = Hierarchy.from_config(config or DEFAULT_CONFIG)
    for addr, is_write in parse_trace(trace_path):
        hierarchy.access(addr, is_write)
    return hierarchy
