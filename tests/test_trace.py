"""Tests for trace parsing."""

from __future__ import annotations

import os
import tempfile
import unittest

from cachesim.trace import parse_trace


class TestTraceParsing(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def parse(self, text: str) -> list[tuple[int, bool]]:
        path = os.path.join(self._tmp.name, "t.trace")
        with open(path, "w") as f:
            f.write(text)
        return list(parse_trace(path))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
