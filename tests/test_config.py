"""Tests for configuration parsing and validation."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from typing import Any

from cachesim.config import (
    DEFAULT_CONFIG,
    CacheSpec,
    ConfigError,
    HierarchySpec,
    default_config,
    load_config,
    parse_config,
)
from cachesim.hierarchy import Hierarchy


def one_level(**overrides: Any) -> dict[str, Any]:
    level: dict[str, Any] = {
        "name": "L1",
        "size": 1024,
        "block_size": 64,
        "associativity": 2,
        "hit_time": 1,
    }
    level.update(overrides)
    return {"memory_access_time": 10, "levels": [level]}


class TestParseConfig(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        spec = parse_config(DEFAULT_CONFIG)
        self.assertEqual([level.name for level in spec.levels], ["L1", "L2", "L3"])
        self.assertEqual(spec.memory_access_time, 100)
        self.assertEqual(spec.levels[0], CacheSpec("L1", 32768, 64, 4, 4))

    def test_round_trip_through_to_dict(self) -> None:
        spec = parse_config(DEFAULT_CONFIG)
        self.assertEqual(parse_config(spec.to_dict()), spec)
        self.assertIsInstance(spec, HierarchySpec)

    def test_optional_keys_are_applied(self) -> None:
        spec = parse_config(one_level(policy="FIFO", track_3c=False, rng_seed=7))
        level = spec.levels[0]
        self.assertEqual((level.policy, level.track_3c, level.rng_seed), ("fifo", False, 7))

    def test_default_config_copy_is_independent(self) -> None:
        cfg = default_config()
        cfg["levels"][0]["size"] = 1
        self.assertEqual(DEFAULT_CONFIG["levels"][0]["size"], 32768)

    def test_errors_are_path_qualified(self) -> None:
        cases: list[tuple[dict[str, Any], str]] = [
            (one_level(assoc=4), "levels[0] (L1): unknown key(s) 'assoc'"),
            ({"levels": [{"size": 1}]}, "missing required top-level key(s) 'memory_access_time'"),
            ({"memory_access_time": 1, "levels": []}, "non-empty list"),
            ({"memory_access_time": 1, "levels": [{"size": 1}]}, "levels[0]: missing required"),
            (one_level(size="32768"), "'size' must be an integer"),
            (one_level(size=True), "'size' must be an integer"),
            (one_level(size=0), "'size' must be >= 1"),
            (one_level(associativity=-4), "'associativity' must be >= 1"),
            (one_level(hit_time=-1), "'hit_time' must be >= 0"),
            (one_level(policy="clairvoyant"), "unknown policy 'clairvoyant'"),
            (one_level(track_3c="yes"), "'track_3c' must be true or false"),
            (one_level(rng_seed=None), "'rng_seed' must be an integer"),
            (one_level(name=""), "'name' must be a non-empty string"),
            ({**one_level(), "dram": 1}, "unknown top-level key(s) 'dram'"),
            ({**one_level(), "memory_access_time": -5}, "'memory_access_time' must be >= 0"),
            (one_level(size=1000), "must be a multiple of"),
            (one_level(block_size=48), "power of two"),
        ]
        for config, fragment in cases:
            with self.subTest(fragment=fragment), self.assertRaises(ConfigError) as ctx:
                Hierarchy.from_config(config)
            self.assertIn(fragment, str(ctx.exception))

    def test_duplicate_level_names_rejected(self) -> None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["levels"][1]["name"] = "L1"
        with self.assertRaises(ConfigError):
            parse_config(cfg)

    def test_mixed_block_sizes_rejected(self) -> None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["levels"][1]["block_size"] = 128
        with self.assertRaises(ConfigError) as ctx:
            parse_config(cfg)
        self.assertIn("block_size", str(ctx.exception))

    def test_config_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(ConfigError, ValueError))


class TestLoadConfig(unittest.TestCase):
    def test_loads_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            with open(path, "w") as f:
                json.dump(DEFAULT_CONFIG, f)
            self.assertEqual(load_config(path), DEFAULT_CONFIG)

    def test_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.json")
            with open(path, "w") as f:
                f.write("[1, 2, 3]")
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_repo_default_config_matches_builtin(self) -> None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "configs", "default.json")
        self.assertEqual(parse_config(load_config(path)), parse_config(DEFAULT_CONFIG))


if __name__ == "__main__":
    unittest.main(verbosity=2)
