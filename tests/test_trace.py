"""Tests for trace reading and writing, and for the trace-stats command."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import tempfile
import unittest
from typing import ClassVar

from cachesim import run_trace
from cachesim.cli import main
from cachesim.config import DEFAULT_CONFIG
from cachesim.hierarchy import Hierarchy
from cachesim.trace import detect_format, load_trace, open_trace, parse_trace, write_trace
from cachesim.tracecmd import TraceStats, analyse_trace
from cachesim.workloads import sequential


class TraceFileCase(unittest.TestCase):
    """Base class: writes trace text to a temporary file and parses it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self._tmp.name, name)

    def write(self, name: str, text: str) -> str:
        """Write ``text`` to ``name`` (gzipped if it ends in .gz); returns the path."""
        path = self.path(name)
        if name.endswith(".gz"):
            with gzip.open(path, "wt") as gz:
                gz.write(text)
        else:
            with open(path, "w") as f:
                f.write(text)
        return path

    def parse_file(self, name: str, text: str, fmt: str = "auto") -> list[tuple[int, bool]]:
        return list(parse_trace(self.write(name, text), fmt))


class TestTraceParsing(TraceFileCase):
    def parse(self, text: str) -> list[tuple[int, bool]]:
        return self.parse_file("t.trace", text)

    def test_valid_lines(self) -> None:
        got = self.parse("# comment\n\n0x10 R\n20 w\n0XFF W\n")
        self.assertEqual(got, [(0x10, False), (0x20, True), (0xFF, True)])

    def test_trailing_comments_and_whitespace(self) -> None:
        got = self.parse("  0x10   R   # first\n\t0x20\tW#second\n   # only a comment\n")
        self.assertEqual(got, [(0x10, False), (0x20, True)])

    def test_crlf_line_endings(self) -> None:
        got = self.parse("0x10 R\r\n0x20 W\r\n")
        self.assertEqual(got, [(0x10, False), (0x20, True)])

    def test_large_addresses(self) -> None:
        got = self.parse("0xffffffffffffffff R\n")
        self.assertEqual(got, [(2**64 - 1, False)])

    def test_bad_address_rejected(self) -> None:
        fullwidth_zero = "\uff10x10"  # non-ASCII digits are rejected too
        for bad in ("0xZZ", "-0x10", "+10", "0x1_0", "0x", "1.0", fullwidth_zero):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.parse(f"{bad} R\n")

    def test_bad_op_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.parse("0x10 READ\n")

    def test_missing_field_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.parse("0x10\n")

    def test_extra_field_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.parse("0x10 R 8\n")

    def test_error_names_file_and_line(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.parse("0x10 R\n0x20 R\nbogus\n")
        self.assertIn("t.trace:3", str(ctx.exception))


class TestDineroParsing(TraceFileCase):
    def test_labels_map_to_read_write_and_fetch(self) -> None:
        got = self.parse_file("t.din", "0 7fff0010\n1 7fff0018\n2 00400000\n")
        self.assertEqual(got, [(0x7FFF0010, False), (0x7FFF0018, True), (0x400000, False)])

    def test_blank_lines_ignored_and_0x_prefix_allowed(self) -> None:
        got = self.parse_file("t.din", "\n0 0x10\n\n   \n1 20\n")
        self.assertEqual(got, [(0x10, False), (0x20, True)])

    def test_escape_labels_rejected(self) -> None:
        for label in ("3", "4", "R"):
            with self.subTest(label=label), self.assertRaises(ValueError) as ctx:
                self.parse_file("t.din", f"{label} 7fff0010\n")
            self.assertIn("label must be 0 (read)", str(ctx.exception))

    def test_bad_address_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.parse_file("t.din", "0 7fff0010\n1 nothex\n")
        self.assertIn("t.din:2", str(ctx.exception))
        self.assertIn("bad hex address", str(ctx.exception))

    def test_wrong_field_count_rejected(self) -> None:
        for bad in ("0\n", "0 7fff0010 8\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError) as ctx:
                self.parse_file("t.din", bad)
            self.assertIn("expected 'LABEL ADDR'", str(ctx.exception))

    def test_native_comment_syntax_is_not_dinero_syntax(self) -> None:
        with self.assertRaises(ValueError):
            self.parse_file("t.din", "0 7fff0010 # not a dinero comment\n")


class TestLackeyParsing(TraceFileCase):
    #: A verbatim-shaped excerpt of `valgrind --tool=lackey --trace-mem=yes`.
    SAMPLE = (
        "==31976== Lackey, an example Valgrind tool\n"
        "==31976== Command: ./a.out\n"
        "==31976== \n"
        "I  0421f3d8,8\n"
        " L 0421f3e0,8\n"
        " S 0421f3e8,4\n"
        " M 0421f3f0,8\n"
        "==31976== \n"
        "==31976== counts for all instructions: 12345\n"
    )

    def test_sample_output(self) -> None:
        got = self.parse_file("t.lackey", self.SAMPLE)
        self.assertEqual(
            got,
            [
                (0x0421F3D8, False),  # I: instruction fetch -> read
                (0x0421F3E0, False),  # L: load -> read
                (0x0421F3E8, True),  # S: store -> write
                (0x0421F3F0, True),  # M: modify -> one write
            ],
        )

    def test_size_field_is_ignored(self) -> None:
        got = self.parse_file("t.lackey", " L 1000,1\n L 1000,64\n")
        self.assertEqual(got, [(0x1000, False), (0x1000, False)])

    def test_unrecognised_lines_are_skipped_not_rejected(self) -> None:
        text = "hello from the traced program\n L zzz,8\n S 40,8 extra\n L 40,8\n\n"
        self.assertEqual(self.parse_file("t.lackey", text), [(0x40, False)])

    def test_vg_extension_selects_lackey(self) -> None:
        self.assertEqual(self.parse_file("t.vg", " S 80,8\n"), [(0x80, True)])


class TestFormatSelection(TraceFileCase):
    def test_detect_format_by_extension(self) -> None:
        cases = {
            "x.trace": "native",
            "x": "native",
            "x.txt": "native",
            "x.din": "dinero",
            "x.DIN": "dinero",
            "x.lackey": "lackey",
            "x.vg": "lackey",
            "x.trace.gz": "native",
            "x.din.gz": "dinero",
            "dir.din/x.trace": "native",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(detect_format(path), expected)

    def test_explicit_format_overrides_the_extension(self) -> None:
        # A Dinero trace that happens to be named .trace.
        got = self.parse_file("t.trace", "0 10\n1 20\n", fmt="dinero")
        self.assertEqual(got, [(0x10, False), (0x20, True)])

    def test_unknown_format_raises_before_reading(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            open_trace(self.path("missing.trace"), fmt="pin")
        self.assertIn("unknown trace format 'pin'", str(ctx.exception))

    def test_gzip_is_transparent(self) -> None:
        self.assertEqual(
            self.parse_file("t.trace.gz", "0x10 R\n0x20 W\n"), [(0x10, False), (0x20, True)]
        )
        self.assertEqual(self.parse_file("t.din.gz", "0 10\n1 20\n"), [(0x10, False), (0x20, True)])

    def test_load_trace_returns_a_list(self) -> None:
        got = load_trace(self.write("t.trace", "0x10 R\n0x20 W\n"))
        self.assertEqual(got, [(0x10, False), (0x20, True)])


class TestTraceWriting(TraceFileCase):
    #: A reference stream with both operations and a wide address range.
    ACCESSES: ClassVar[list[tuple[int, bool]]] = [
        (0x0, False),
        (0x400000, True),
        (0xDEADBEEF, False),
        (0x7FFFFFFFFFFF, True),
    ]

    def test_native_output_is_byte_for_byte_stable(self) -> None:
        # The golden trace digests depend on this exact rendering.
        path = self.path("t.trace")
        self.assertEqual(write_trace(path, [(0x10, "R"), (0x20, "W")]), 2)
        with open(path) as f:
            self.assertEqual(f.read(), "0x00000010 R\n0x00000020 W\n")

    def test_dinero_output(self) -> None:
        path = self.path("t.din")
        write_trace(path, [(0x10, "R"), (0x20, "W")], fmt="dinero")
        with open(path) as f:
            self.assertEqual(f.read(), "0 00000010\n1 00000020\n")

    def test_lackey_output(self) -> None:
        path = self.path("t.lackey")
        write_trace(path, [(0x10, "R"), (0x20, "W")], fmt="lackey")
        with open(path) as f:
            self.assertEqual(f.read(), " L 00000010,8\n S 00000020,8\n")

    def test_bool_and_string_operations_are_both_accepted(self) -> None:
        strings = self.path("s.trace")
        bools = self.path("b.trace")
        write_trace(strings, [(0x10, "r"), (0x20, "W")])
        write_trace(bools, [(0x10, False), (0x20, True)])
        with open(strings) as a, open(bools) as b:
            self.assertEqual(a.read(), b.read())

    def test_bad_operation_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_trace(self.path("t.trace"), [(0x10, "X")])

    def test_unknown_format_rejected(self) -> None:
        with self.assertRaises(ValueError):
            write_trace(self.path("t.trace"), [(0x10, "R")], fmt="pin")

    def test_round_trip_through_every_format(self) -> None:
        # native -> dinero -> lackey -> native must preserve the stream.
        stream = list(self.ACCESSES)
        for name, fmt in (
            ("a.trace", "native"),
            ("b.din", "dinero"),
            ("c.lackey", "lackey"),
            ("d.trace", "native"),
        ):
            path = self.path(name)
            self.assertEqual(write_trace(path, stream, fmt=fmt), len(self.ACCESSES))
            stream = load_trace(path)  # format detected from the extension
            self.assertEqual(stream, self.ACCESSES, f"lost by {fmt}")

    def test_round_trip_through_gzip(self) -> None:
        path = self.path("t.din.gz")
        write_trace(path, self.ACCESSES, fmt="dinero")
        self.assertEqual(load_trace(path), self.ACCESSES)

    def test_written_workload_matches_the_generator(self) -> None:
        path = self.path("seq.trace")
        write_trace(path, sequential(buffer_bytes=1024, passes=2))
        expected = [(addr, op == "W") for addr, op in sequential(buffer_bytes=1024, passes=2)]
        self.assertEqual(load_trace(path), expected)


class TestSimulatingForeignFormats(TraceFileCase):
    """Every format reaches the simulator and produces the same run."""

    def accesses(self) -> list[tuple[int, str]]:
        return list(sequential(buffer_bytes=4096, passes=2))

    def test_run_trace_accepts_an_explicit_format(self) -> None:
        native = self.path("t.trace")
        misnamed = self.path("also.trace")  # a Dinero trace with a native name
        write_trace(native, self.accesses())
        write_trace(misnamed, self.accesses(), fmt="dinero")
        expected = run_trace(native).stats().to_dict()
        got = run_trace(misnamed, fmt="dinero").stats().to_dict()
        self.assertEqual(got, expected)

    def test_run_trace_detects_the_format_from_the_extension(self) -> None:
        native = self.path("t.trace")
        lackey = self.path("t.lackey.gz")
        write_trace(native, self.accesses())
        write_trace(lackey, self.accesses(), fmt="lackey")
        self.assertEqual(run_trace(lackey).stats().to_dict(), run_trace(native).stats().to_dict())

    def test_cli_trace_format_flag(self) -> None:
        misnamed = self.path("t.trace")
        write_trace(misnamed, self.accesses(), fmt="dinero")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["run", "--trace-format", "dinero", "--format", "json", misnamed])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["accesses"], 1024)

    def test_cli_rejects_an_unknown_trace_format(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run", "--trace-format", "pin", self.path("t.trace")])


class TestTraceStats(TraceFileCase):
    """`cachesim trace-stats`: properties of the address stream alone."""

    def stats_of(self, name: str, text: str, **kwargs: int) -> TraceStats:
        return analyse_trace(parse_trace(self.write(name, text)), **kwargs)

    def test_hand_checkable_trace(self) -> None:
        # Blocks 0 (twice, plus 0x08), 1 (0x40) and 64 (0x1000); pages 0 and 1.
        stats = self.stats_of("t.trace", "0x00 R\n0x08 W\n0x40 R\n0x1000 R\n0x00 R\n")
        self.assertEqual(stats.accesses, 5)
        self.assertEqual((stats.reads, stats.writes), (4, 1))
        self.assertEqual(stats.distinct_blocks, 3)
        self.assertEqual(stats.footprint_bytes, 192)
        self.assertEqual(stats.distinct_pages, 2)
        self.assertEqual((stats.min_address, stats.max_address), (0x00, 0x1000))
        self.assertEqual(stats.first_touches, 3)
        self.assertEqual(stats.compulsory_fraction, 0.6)

    def test_generated_trace(self) -> None:
        # 8 KB scanned twice by 8-byte words: 1024 words per pass, every
        # tenth access a write, 8192/64 = 128 blocks over two 4 KB pages.
        accesses = [(addr, op == "W") for addr, op in sequential(buffer_bytes=8 * 1024, passes=2)]
        stats = analyse_trace(accesses)
        self.assertEqual(stats.accesses, 2048)
        self.assertEqual((stats.reads, stats.writes), (1844, 204))
        self.assertEqual(stats.distinct_blocks, 128)
        self.assertEqual(stats.footprint_bytes, 8192)
        self.assertEqual(stats.distinct_pages, 2)
        self.assertEqual((stats.min_address, stats.max_address), (0x0010_0000, 0x0010_0000 + 8184))
        self.assertEqual(stats.compulsory_fraction, 0.0625)

    def test_block_size_changes_the_footprint_and_the_floor(self) -> None:
        accesses = [(addr, op == "W") for addr, op in sequential(buffer_bytes=8 * 1024, passes=2)]
        stats = analyse_trace(accesses, block_size=8)
        self.assertEqual(stats.distinct_blocks, 1024)  # one per word now
        self.assertEqual(stats.footprint_bytes, 8192)  # the same bytes, finer blocks
        self.assertEqual(stats.compulsory_fraction, 0.5)

    def test_first_touches_are_the_hierarchy_dram_reads(self) -> None:
        # A cold hierarchy that holds the whole footprint fetches exactly the
        # distinct blocks from DRAM: the compulsory floor is not just a bound.
        accesses = [(addr, op == "W") for addr, op in sequential(buffer_bytes=8 * 1024, passes=2)]
        hierarchy = Hierarchy.from_config(DEFAULT_CONFIG)
        for addr, is_write in accesses:
            hierarchy.access(addr, is_write)
        self.assertEqual(hierarchy.dram_reads, analyse_trace(accesses).distinct_blocks)

    def test_empty_trace_reports_no_address_range(self) -> None:
        stats = self.stats_of("t.trace", "# nothing here\n")
        self.assertEqual(stats.accesses, 0)
        self.assertIsNone(stats.min_address)
        self.assertIsNone(stats.max_address)
        self.assertEqual(stats.compulsory_fraction, 0.0)

    def test_rejects_a_non_positive_block_size(self) -> None:
        with self.assertRaises(ValueError):
            analyse_trace([(0, False)], block_size=0)

    def test_cli_text_output(self) -> None:
        path = self.path("t.trace")
        write_trace(path, sequential(buffer_bytes=8 * 1024, passes=2))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["trace-stats", path])
        self.assertEqual(code, 0)
        printed = out.getvalue()
        self.assertIn("accesses        :        2,048", printed)
        self.assertIn("distinct blocks :          128", printed)
        self.assertIn("footprint 8.0 KB at 64 B blocks", printed)
        self.assertIn("6.25% of accesses", printed)

    def test_cli_json_output_and_block_size_flag(self) -> None:
        path = self.path("t.trace")
        write_trace(path, sequential(buffer_bytes=8 * 1024, passes=2))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["trace-stats", "--block-size", "8", "--format", "json", path])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual(data["accesses"], 2048)
        self.assertEqual(data["block_size"], 8)
        self.assertEqual(data["distinct_blocks"], 1024)
        self.assertEqual(data["distinct_pages"], 2)

    def test_cli_reads_foreign_formats(self) -> None:
        path = self.path("t.din")
        write_trace(path, sequential(buffer_bytes=8 * 1024, passes=2), fmt="dinero")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["trace-stats", "--format", "json", path])
        self.assertEqual(json.loads(out.getvalue())["distinct_blocks"], 128)

    def test_cli_rejects_an_empty_trace(self) -> None:
        path = self.write("empty.trace", "# nothing\n")
        with self.assertRaises(SystemExit) as ctx:
            main(["trace-stats", path])
        self.assertIn("no accesses", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
