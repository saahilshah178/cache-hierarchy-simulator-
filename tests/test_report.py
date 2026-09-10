"""Tests for the statistics snapshot, the text report, and JSON output."""

from __future__ import annotations

import json
import unittest

from cachesim import tracecmd
from cachesim.cache import Cache
from cachesim.hierarchy import Hierarchy, Level
from cachesim.plot import format_bytes
from cachesim.report import _pct, build_report, format_report, format_size
from cachesim.stats import SCHEMA_VERSION, HierarchyStats


def small_run() -> Hierarchy:
    """A three-access run whose every number can be checked by hand."""
    l1 = Cache("L1", 256, 64, 1)
    l2 = Cache("L2", 1024, 64, 2)
    h = Hierarchy([Level(l1, 4), Level(l2, 12)], memory_access_time=100)
    h.access(0x0)  # miss, miss, DRAM: 116 cycles
    h.access(0x0, is_write=True)  # L1 hit: 4 cycles
    h.access(4 * 64)  # L1 conflict miss, L2 miss, DRAM: 116; evicts dirty block 0 -> L2
    return h


class TestFormatting(unittest.TestCase):
    def test_pct(self) -> None:
        self.assertEqual(_pct(0, 0), "   n/a ")
        self.assertEqual(_pct(1, 3), " 33.33%")
        self.assertEqual(_pct(3, 3), "100.00%")

    def test_size_str(self) -> None:
        self.assertEqual(format_size(0), "0 B")
        self.assertEqual(format_size(1023), "1023 B")
        self.assertEqual(format_size(1024), "1.0 KB")
        self.assertEqual(format_size(1536), "1.5 KB")
        self.assertEqual(format_size(1 << 20), "1.0 MB")
        self.assertEqual(format_size(3 << 19), "1.5 MB")

    def test_the_two_byte_conventions_differ_only_in_the_trailing_zero(self) -> None:
        """One implementation, one flag: the reports keep the .0, the tables drop it.

        There used to be three copies of this arithmetic, two of them
        byte-identical, so the same cache could print as 32.0 KB in `run`
        and 32 KB in `sweep` with nothing tying the two together.
        """
        for nbytes in (0, 64, 1023, 1024, 1536, 32768, 1 << 20, 3 << 19, 2 << 20):
            with self.subTest(nbytes=nbytes):
                self.assertEqual(format_size(nbytes), format_bytes(nbytes, decimals=1))
                self.assertEqual(format_size(nbytes).replace(".0 ", " "), format_bytes(nbytes))

    def test_trace_stats_formats_sizes_like_the_report(self) -> None:
        """The two used to hold byte-identical private copies of this."""
        stats = tracecmd.analyse_trace([(0, False)], block_size=1024)
        rendered = tracecmd.format_stats(stats)
        self.assertIn(f"footprint {format_size(1024)} at 1024 B blocks", rendered)
        self.assertIn(f"({format_size(stats.page_size)} pages)", rendered)


class TestStats(unittest.TestCase):
    def test_snapshot_matches_counters(self) -> None:
        h = small_run()
        s = h.stats()
        self.assertEqual((s.accesses, s.reads, s.writes), (3, 2, 1))
        self.assertEqual((s.dram_reads, s.dram_writes), (2, 0))
        self.assertEqual(s.total_cycles, 116 + 4 + 116)
        self.assertAlmostEqual(s.measured_amat, 236 / 3)
        self.assertAlmostEqual(s.amat, 4 + (2 / 3) * (12 + 1.0 * 100))
        l1, l2 = s.levels
        self.assertEqual((l1.hits, l1.misses, l1.write_hits), (1, 2, 1))
        self.assertEqual((l1.evictions, l1.writebacks), (1, 1))
        self.assertEqual((l2.writebacks_received, l2.writeback_allocations), (1, 0))
        self.assertEqual(l2.local_miss_rate, 1.0)
        self.assertAlmostEqual(l2.global_miss_rate, 2 / 3)
        assert l1.three_c is not None
        self.assertEqual((l1.three_c.compulsory, l1.three_c.conflict), (2, 0))

    def test_three_c_is_none_when_disabled(self) -> None:
        h = Hierarchy([Level(Cache("L1", 256, 64, 1, track_3c=False), 1)], 10)
        h.access(0)
        self.assertIsNone(h.stats().levels[0].three_c)
        self.assertNotIn("miss classification", build_report(h))

    def test_to_dict_is_json_serialisable_and_versioned(self) -> None:
        d = small_run().stats().to_dict()
        text = json.dumps(d)
        back = json.loads(text)
        self.assertEqual(back["schema_version"], SCHEMA_VERSION)
        self.assertEqual(back["levels"][0]["name"], "L1")
        self.assertEqual(back["levels"][1]["writebacks_received"], 1)
        self.assertEqual(back["dram_reads"], 2)

    def test_stats_is_a_frozen_snapshot(self) -> None:
        h = small_run()
        s = h.stats()
        h.access(0x0)
        self.assertEqual(s.accesses, 3)
        self.assertEqual(h.stats().accesses, 4)
        self.assertIsInstance(s, HierarchyStats)


class TestReport(unittest.TestCase):
    def test_report_contains_every_headline_number(self) -> None:
        h = small_run()
        text = build_report(h, trace_name="tiny")
        self.assertIn("(tiny)", text)
        self.assertIn("Total accesses :            3   (reads 2 / writes 1)", text)
        self.assertIn("DRAM reads     :            2", text)
        self.assertIn("DRAM writes    :            0", text)
        self.assertIn("--- L1: 256 B, 64 B blocks, 1-way, LRU, hit time 4 cyc ---", text)
        self.assertIn("local miss rate  :  66.67%", text)
        self.assertIn("global miss rate :  66.67%", text)
        self.assertIn("writebacks received :      1   (from L1; 0 allocated a line)", text)
        self.assertIn("AMAT (measured)         :   78.667 cycles   (236 cycles / 3 accesses)", text)

    def test_text_and_json_come_from_the_same_snapshot(self) -> None:
        h = small_run()
        s = h.stats()
        self.assertEqual(format_report(s), build_report(h))
        self.assertIn(f"{s.total_cycles:,} cycles", format_report(s))

    def test_untouched_hierarchy_reports_na(self) -> None:
        h = Hierarchy([Level(Cache("L1", 256, 64, 1), 1)], 10)
        text = build_report(h)
        self.assertIn("local miss rate  :    n/a", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
