"""Tests for the multi-configuration comparison command.

The central case is a hand-checkable one: three blocks cycled through a
two-block cache miss every time, and through a four-block cache miss only
once each. Every number in the table follows from that, so the whole
pipeline -- simulation, statistics, deltas and formatting -- is pinned to
arithmetic that can be done on paper.

Every pair of configurations compared here differs in exactly ONE key, and
that is asserted rather than assumed: a table whose two rows differ in two
places cannot attribute its deltas to either of them, which would defeat
the point of the command. ``TestCompare`` isolates capacity on the
three-block cycle; ``TestLargerL1`` isolates the L1 size in the full
three-level default hierarchy.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from typing import Any

from cachesim.compare import (
    Comparison,
    compare,
    format_table,
    level_names,
    load_configs,
    main,
    simulate,
)
from cachesim.config import DEFAULT_CONFIG
from cachesim.workloads import sequential, write_trace

#: Three 64 B blocks, cycled four times: 12 accesses, 3 distinct blocks.
CYCLE = [(addr, False) for _ in range(4) for addr in (0, 64, 128)]


def one_level(size: int, associativity: int = 2, hit_time: int = 4) -> dict[str, Any]:
    """A single-level hierarchy of ``size`` bytes over 64 B blocks."""
    return {
        "memory_access_time": 100,
        "levels": [
            {
                "name": "L1",
                "size": size,
                "block_size": 64,
                "associativity": associativity,
                "hit_time": hit_time,
            }
        ],
    }


def differing_keys(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    """The L1 keys on which two ``one_level`` configurations disagree."""
    left, right = a["levels"][0], b["levels"][0]
    return {key for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


#: The comparison fixture: two configurations one parameter apart. Only
#: ``size`` differs -- 2 blocks against 4, both 2-way -- so every delta the
#: table reports is attributable to capacity and nothing else. A 128 B 2-way
#: cache is one set of two ways; 256 B is two sets of two ways, which holds
#: blocks 0 and 2 in set 0 and block 1 in set 1, so all three fit.
SMALL = one_level(128)
BIG = one_level(256)


class TestSimulate(unittest.TestCase):
    def test_two_blocks_miss_every_access(self) -> None:
        """Stack distance 2 in a 2-block cache: nothing is ever reused in time."""
        stats = simulate(CYCLE, one_level(128, associativity=2))
        self.assertEqual(stats.accesses, 12)
        self.assertEqual(stats.levels[0].misses, 12)
        self.assertEqual(stats.levels[0].local_miss_rate, 1.0)
        self.assertEqual(stats.dram_reads, 12)
        self.assertEqual(stats.dram_writes, 0)
        self.assertEqual(stats.amat, 104.0)
        self.assertEqual(stats.total_cycles, 12 * 104)

    def test_four_blocks_miss_only_once_per_block(self) -> None:
        stats = simulate(CYCLE, one_level(256, associativity=4))
        self.assertEqual(stats.levels[0].misses, 3)
        self.assertEqual(stats.levels[0].local_miss_rate, 0.25)
        self.assertEqual(stats.dram_reads, 3)
        self.assertEqual(stats.amat, 29.0)
        self.assertEqual(stats.total_cycles, 3 * 104 + 9 * 4)

    def test_warmup_excludes_the_prefix(self) -> None:
        """Warming up over the first 3 accesses hides the compulsory misses."""
        stats = simulate(CYCLE, one_level(256, associativity=4), warmup=3)
        self.assertEqual(stats.accesses, 9)
        self.assertEqual(stats.levels[0].misses, 0)
        self.assertEqual(stats.total_cycles, 9 * 4)

    def test_warmup_covering_the_whole_trace_measures_nothing(self) -> None:
        stats = simulate(CYCLE, one_level(256, associativity=4), warmup=12)
        self.assertEqual(stats.accesses, 0)
        self.assertEqual(stats.total_cycles, 0)


class TestCompare(unittest.TestCase):
    def results(self) -> list[Comparison]:
        return compare(CYCLE, [("small", "small.json", SMALL), ("big", "big.json", BIG)])

    def test_the_fixture_really_is_one_parameter_apart(self) -> None:
        """Guards the premise of every other test in this class.

        A comparison whose configurations differ in two places cannot
        attribute its deltas to either, so the fixture is asserted to be
        minimal rather than assumed to be.
        """
        self.assertEqual(differing_keys(SMALL, BIG), {"size"})

    def test_order_and_labels_are_preserved(self) -> None:
        results = self.results()
        self.assertEqual([r.label for r in results], ["small", "big"])
        self.assertEqual([r.source for r in results], ["small.json", "big.json"])

    def test_doubling_the_size_alone_moves_every_column(self) -> None:
        """128 B holds 2 of the 3 blocks and thrashes; 256 B holds all three."""
        small, big = self.results()
        self.assertEqual(small.stats.levels[0].misses, 12)
        self.assertEqual(big.stats.levels[0].misses, 3)
        self.assertEqual(small.miss_rate("L1"), 1.0)
        self.assertEqual(big.miss_rate("L1"), 0.25)
        self.assertIsNone(big.miss_rate("L2"))
        # Hit time is identical, so the whole AMAT delta is missed traffic:
        # 104.0 - 29.0 = 75.0 cycles, all of it 0.75 * 100 cycles of DRAM.
        self.assertEqual(small.stats.amat - big.stats.amat, 75.0)
        self.assertEqual(small.stats.dram_reads - big.stats.dram_reads, 9)

    def test_to_dict_tags_the_configuration(self) -> None:
        small = self.results()[0]
        payload = small.to_dict()
        self.assertEqual(payload["label"], "small")
        self.assertEqual(payload["config"], "small.json")
        self.assertEqual(payload["accesses"], 12)
        self.assertIn("schema_version", payload)


class TestLevelNames(unittest.TestCase):
    def test_union_in_order_of_first_appearance(self) -> None:
        results = compare(
            CYCLE,
            [
                ("one", "one.json", one_level(256, associativity=4)),
                ("three", "three.json", DEFAULT_CONFIG),
            ],
        )
        self.assertEqual(level_names(results), ["L1", "L2", "L3"])

    def test_empty(self) -> None:
        self.assertEqual(level_names([]), [])
        self.assertEqual(format_table([]), "")


class TestFormatTable(unittest.TestCase):
    def test_baseline_has_blank_deltas_and_others_are_signed(self) -> None:
        results = compare(CYCLE, [("small", "small.json", SMALL), ("big", "big.json", BIG)])
        table = format_table(results).splitlines()
        self.assertEqual(len(table), 3)
        header, baseline, candidate = table
        self.assertIn("L1 miss", header)
        self.assertIn("d cycles", header)
        self.assertTrue(baseline.startswith("small"))
        self.assertIn("100.00%", baseline)
        # 348 cycles against 1,248 is a 72.12% reduction.
        self.assertIn("-72.12%", candidate)
        self.assertIn("-75.000", candidate)  # AMAT 29.0 - 104.0
        self.assertIn("1,248", baseline)
        self.assertIn("348", candidate)

    def test_absent_levels_show_a_dash(self) -> None:
        """A one-level config compared against a three-level one keeps its columns."""
        results = compare(
            CYCLE,
            [
                ("three", "three.json", DEFAULT_CONFIG),
                ("one", "one.json", one_level(256, associativity=4)),
            ],
        )
        table = format_table(results).splitlines()
        # Columns are fixed width: the label field, then 10 chars per level.
        label_width = len("config")
        single = table[2]
        self.assertTrue(single.startswith("one"))
        columns = [single[label_width + 10 * i : label_width + 10 * (i + 1)] for i in range(3)]
        self.assertEqual(columns[0].strip(), "25.00%")  # L1 exists
        self.assertEqual(columns[1].strip(), "-")  # no L2
        self.assertEqual(columns[2].strip(), "-")  # no L3


class TestLargerL1(unittest.TestCase):
    """The default hierarchy against the same hierarchy with a 64 KB L1.

    This is the comparison the command exists for: a full three-level
    configuration and a candidate that changes exactly one number in it.
    The workload is a 48 KB buffer scanned twice, chosen so the answer is
    arithmetic rather than an observation.

    * 48 KB is 768 blocks of 64 B, so the first pass takes 768 compulsory
      misses however large L1 is.
    * The 32 KB L1 is 128 sets of 4 ways. Block ``b`` lands in set
      ``b % 128``, so each set is asked for 6 blocks and holds 4; a
      forward scan evicts each one just before the next pass wants it, and
      the second pass misses all 768 again.
    * The 64 KB L1 is 256 sets of 4 ways: 3 blocks per set, all resident,
      so the second pass takes no L1 misses at all.

    L1 misses therefore halve, 1,536 to 768, and every L1 miss that
    remains is compulsory. The trace fits in L2 and L3, so no access
    reaches DRAM twice and the DRAM columns are identical.
    """

    def setUp(self) -> None:
        self.trace = [(addr, op == "W") for addr, op in sequential(48 * 1024, passes=2)]
        self.bigger = copy.deepcopy(DEFAULT_CONFIG)
        self.bigger["levels"][0]["size"] = 64 * 1024
        self.results = compare(
            self.trace,
            [
                ("default", "default", DEFAULT_CONFIG),
                ("l1-64k", "l1-64k.json", self.bigger),
            ],
        )

    def test_the_candidate_changes_exactly_one_number(self) -> None:
        levels = zip(DEFAULT_CONFIG["levels"], self.bigger["levels"], strict=True)
        changed = {
            (index, key)
            for index, (base, candidate) in enumerate(levels)
            for key in base.keys() | candidate.keys()
            if base.get(key) != candidate.get(key)
        }
        self.assertEqual(changed, {(0, "size")})

    def test_the_second_pass_stops_missing_in_l1(self) -> None:
        default, bigger = self.results
        self.assertEqual(len(self.trace), 12288)
        self.assertEqual(default.stats.levels[0].misses, 1536)  # both passes miss
        self.assertEqual(bigger.stats.levels[0].misses, 768)  # compulsory only
        self.assertEqual(default.miss_rate("L1"), 0.125)
        self.assertEqual(bigger.miss_rate("L1"), 0.0625)

    def test_dram_traffic_is_unchanged_and_the_deltas_are_negative(self) -> None:
        """A bigger L1 filters L2, it does not change what DRAM must supply."""
        default, bigger = self.results
        self.assertEqual(default.stats.dram_reads, bigger.stats.dram_reads)
        self.assertEqual(default.stats.dram_writes, bigger.stats.dram_writes)
        self.assertLess(bigger.stats.amat, default.stats.amat)
        self.assertLess(bigger.stats.total_cycles, default.stats.total_cycles)

    def test_the_table_signs_the_candidate_row_only(self) -> None:
        header, baseline, candidate = format_table(self.results).splitlines()
        self.assertIn("L3 miss", header)
        self.assertTrue(baseline.startswith("default"))
        self.assertTrue(candidate.startswith("l1-64k"))
        self.assertIn("12.50%", baseline)
        self.assertIn("6.25%", candidate)
        self.assertIn("175,104", baseline)
        self.assertIn("165,888", candidate)
        self.assertIn("-0.750", candidate)  # AMAT 13.5 - 14.25
        self.assertIn("-5.26%", candidate)  # 165,888 against 175,104
        self.assertTrue(baseline.rstrip().endswith("-"))  # the baseline is undelta'd


class TestLoadConfigs(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str, config: dict[str, Any], subdir: str = "") -> str:
        directory = os.path.join(self._tmp.name, subdir) if subdir else self._tmp.name
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, name)
        with open(path, "w") as handle:
            json.dump(config, handle)
        return path

    def test_labels_come_from_the_base_name(self) -> None:
        a = self.write("small.json", one_level(128))
        b = self.write("big.json", one_level(256))
        loaded = load_configs([a, b])
        self.assertEqual([label for label, _, _ in loaded], ["small", "big"])

    def test_default_is_the_built_in_hierarchy(self) -> None:
        loaded = load_configs(["default", self.write("big.json", one_level(256))])
        self.assertEqual(loaded[0][0], "default")
        self.assertEqual(loaded[0][2]["levels"][0]["size"], 32 * 1024)

    def test_clashing_base_names_fall_back_to_paths(self) -> None:
        a = self.write("cfg.json", one_level(128), subdir="a")
        b = self.write("cfg.json", one_level(256), subdir="b")
        loaded = load_configs([a, b])
        self.assertEqual([label for label, _, _ in loaded], [a, b])


class TestCompareCLI(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.trace = os.path.join(self._tmp.name, "seq.trace")
        write_trace(self.trace, sequential(buffer_bytes=8 * 1024, passes=2))
        self.small = self.write("small.json", one_level(128))
        self.big = self.write("big.json", one_level(256, associativity=4))

    def write(self, name: str, config: Any) -> str:
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as handle:
            if isinstance(config, str):
                handle.write(config)
            else:
                json.dump(config, handle)
        return path

    def run_main(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = int(exc.code) if exc.code is not None else 0
        return code, out.getvalue(), err.getvalue()

    def test_table_names_both_configs(self) -> None:
        code, out, _ = self.run_main([self.trace, "--config", self.small, "--config", self.big])
        self.assertEqual(code, 0)
        self.assertIn("baseline: small", out)
        self.assertIn("small", out)
        self.assertIn("big", out)
        self.assertIn("d cycles", out)

    def test_json_emits_one_stats_dict_per_config(self) -> None:
        code, out, _ = self.run_main(
            [self.trace, "--config", self.small, "--config", self.big, "--format", "json"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 2)
        self.assertEqual([entry["label"] for entry in payload], ["small", "big"])
        for entry in payload:
            self.assertEqual(entry["accesses"], 2048)
            self.assertIn("schema_version", entry)
            self.assertIn("levels", entry)

    def test_warmup_reduces_the_measured_accesses(self) -> None:
        code, out, _ = self.run_main(
            [
                self.trace,
                "--config",
                self.small,
                "--config",
                self.big,
                "--warmup",
                "1024",
                "--format",
                "json",
            ]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertTrue(all(entry["accesses"] == 1024 for entry in payload))

    def test_default_config_is_accepted(self) -> None:
        code, out, _ = self.run_main([self.trace, "--config", "default", "--config", self.big])
        self.assertEqual(code, 0)
        self.assertIn("baseline: default", out)
        self.assertIn("L3 miss", out)

    def test_fewer_than_two_configs_is_rejected(self) -> None:
        for argv in ([self.trace], [self.trace, "--config", self.small]):
            with self.subTest(argv=argv):
                code, _, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn("at least two --config", err)

    def test_missing_config_file(self) -> None:
        code, _, err = self.run_main(
            [
                self.trace,
                "--config",
                self.small,
                "--config",
                os.path.join(self._tmp.name, "no.json"),
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("cannot load config", err)

    def test_malformed_config_json(self) -> None:
        bad = self.write("bad.json", "{not json")
        code, _, err = self.run_main([self.trace, "--config", self.small, "--config", bad])
        self.assertEqual(code, 2)
        self.assertIn("cannot load config", err)

    def test_invalid_config_is_reported_without_traceback(self) -> None:
        bad = self.write("bad2.json", {"memory_access_time": 1, "levels": [{"name": "L1"}]})
        code, _, err = self.run_main([self.trace, "--config", self.small, "--config", bad])
        self.assertEqual(code, 2)
        self.assertIn("levels[0] (L1): missing required key(s)", err)

    def test_missing_trace(self) -> None:
        code, _, err = self.run_main(
            [
                os.path.join(self._tmp.name, "nope.trace"),
                "--config",
                self.small,
                "--config",
                self.big,
            ]
        )
        self.assertEqual(code, 2)
        self.assertIn("trace not found", err)

    def test_empty_trace(self) -> None:
        path = os.path.join(self._tmp.name, "empty.trace")
        with open(path, "w") as handle:
            handle.write("# nothing\n")
        code, _, err = self.run_main([path, "--config", self.small, "--config", self.big])
        self.assertEqual(code, 2)
        self.assertIn("no accesses", err)

    def test_warmup_past_the_end_is_rejected(self) -> None:
        code, _, err = self.run_main(
            [self.trace, "--config", self.small, "--config", self.big, "--warmup", "99999"]
        )
        self.assertEqual(code, 2)
        self.assertIn("leaves nothing to measure", err)


if __name__ == "__main__":
    unittest.main(verbosity=2)
