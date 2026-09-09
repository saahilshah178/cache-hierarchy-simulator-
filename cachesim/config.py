"""Hierarchy configuration: the built-in default and JSON loading."""

from __future__ import annotations

import copy
import json
from typing import Any

#: The default hierarchy: sizes, latencies, and shapes typical of a modern
#: desktop core. Copy this into a .json file and edit to experiment.
DEFAULT_CONFIG: dict[str, Any] = {
    "memory_access_time": 100,  # cycles to reach DRAM
    "levels": [
        {
            "name": "L1",
            "size": 32 * 1024,
            "block_size": 64,
            "associativity": 4,
            "policy": "lru",
            "hit_time": 4,
        },
        {
            "name": "L2",
            "size": 256 * 1024,
            "block_size": 64,
            "associativity": 8,
            "policy": "lru",
            "hit_time": 12,
        },
        {
            "name": "L3",
            "size": 2 * 1024 * 1024,
            "block_size": 64,
            "associativity": 16,
            "policy": "lru",
            "hit_time": 40,
        },
    ],
}


def default_config() -> dict[str, Any]:
    """Return a fresh copy of ``DEFAULT_CONFIG`` that is safe to mutate."""
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: str) -> dict[str, Any]:
    """Read a hierarchy configuration from a JSON file."""
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object at top level")
    return config
