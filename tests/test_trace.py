"""Tests for trace parsing."""

from __future__ import annotations

import gzip
import os
import tempfile
import unittest

from cachesim.trace import detect_format, load_trace, open_trace, parse_trace


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
