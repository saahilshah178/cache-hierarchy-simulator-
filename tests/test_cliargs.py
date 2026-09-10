"""Tests for the argument types and trace loading shared by the commands.

These helpers used to be copied into each analysis module, so the same
input could be accepted by one command and rejected by another. The tests
here pin the grammar and the error paths once, and the last class asserts
that all four commands really do reach this module rather than keeping a
private copy of it.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import tempfile
import unittest

from cachesim import cliargs, compare, setpressure, stackdist, sweep
from cachesim.cliargs import (
    load_trace_or_error,
    non_negative_int,
    parse_size,
    positive_int,
    read_trace,
)
from cachesim.stackdist import block_stream


class TestIntegerTypes(unittest.TestCase):
    def test_positive_int_accepts_one_and_up(self) -> None:
        self.assertEqual(positive_int("1"), 1)
        self.assertEqual(positive_int("512"), 512)

    def test_positive_int_rejects_zero_and_below(self) -> None:
        for bad in ("0", "-1"):
            with self.subTest(text=bad), self.assertRaises(argparse.ArgumentTypeError):
                positive_int(bad)

    def test_non_negative_int_accepts_zero(self) -> None:
        """Zero is meaningful for a hit time or a warm-up length."""
        self.assertEqual(non_negative_int("0"), 0)
        self.assertEqual(non_negative_int("40"), 40)

    def test_non_negative_int_rejects_negatives(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            non_negative_int("-1")

    def test_junk_raises_value_error_for_argparse_to_wrap(self) -> None:
        """A non-numeric string never reaches the range check."""
        for convert in (positive_int, non_negative_int):
            with self.subTest(convert=convert.__name__), self.assertRaises(ValueError):
                convert("eight")


class TestParseSize(unittest.TestCase):
    def test_suffixes_are_binary(self) -> None:
        self.assertEqual(parse_size("512"), 512)
        self.assertEqual(parse_size("4k"), 4096)
        self.assertEqual(parse_size("4KB"), 4096)
        self.assertEqual(parse_size("1M"), 1 << 20)
        self.assertEqual(parse_size("2mb"), 2 << 20)

    def test_surrounding_space_is_ignored(self) -> None:
        self.assertEqual(parse_size("  32k "), 32768)

    def test_rejects_junk(self) -> None:
        for bad in ("", "big", "0", "-4k", "4g"):
            with self.subTest(text=bad), self.assertRaises(argparse.ArgumentTypeError):
                parse_size(bad)


class TestTraceLoading(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.parser = argparse.ArgumentParser(prog="cachesim test")

    def write(self, name: str, text: str) -> str:
        path = os.path.join(self._tmp.name, name)
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def error(self, path: str) -> str:
        """Load ``path``, expecting exit status 2, and return what was printed."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            load_trace_or_error(self.parser, path)
        self.assertEqual(caught.exception.code, 2)
        return stderr.getvalue()

    def test_a_good_trace_comes_back_as_pairs(self) -> None:
        path = self.write("ok.trace", "0x40 R\n0x80 W\n# comment\n")
        self.assertEqual(load_trace_or_error(self.parser, path), [(0x40, False), (0x80, True)])

    def test_missing_file_points_at_gen_traces(self) -> None:
        message = self.error(os.path.join(self._tmp.name, "absent.trace"))
        self.assertIn("trace not found", message)
        self.assertIn("cachesim gen-traces", message)

    def test_a_trace_of_only_comments_is_empty_not_silent(self) -> None:
        self.assertIn("no accesses in", self.error(self.write("blank.trace", "# nothing\n\n")))

    def test_a_malformed_line_is_a_usage_error(self) -> None:
        """The parser's own message survives: it names the file and the line."""
        message = self.error(self.write("bad.trace", "0x40 R\n0x80 X\n"))
        self.assertIn("bad.trace:2", message)
        self.assertIn("op must be R or W", message)

    def test_read_trace_builds_whatever_the_command_needs(self) -> None:
        """The block-stream form never materialises the addresses."""
        path = self.write("blocks.trace", "0x00 R\n0x08 R\n0x40 W\n")
        blocks = read_trace(self.parser, path, lambda acc: block_stream(acc, 64))
        self.assertEqual(blocks, [0, 0, 1])

    def test_read_trace_reports_a_failure_inside_build(self) -> None:
        """A ValueError raised by ``build`` is a usage error, not a traceback."""
        path = self.write("blocks2.trace", "0x00 R\n")
        with self.assertRaises(SystemExit) as caught:
            read_trace(self.parser, path, lambda acc: block_stream(acc, 0))
        self.assertEqual(caught.exception.code, 2)


class TestCommandsShareTheHelpers(unittest.TestCase):
    """No command may keep a private copy that can drift from this one.

    Four modules parsing "4k" four different ways is exactly how one
    command ends up accepting a capacity another rejects. Identity, not
    equality: these must be the same function objects.
    """

    def test_every_module_imports_rather_than_redefines(self) -> None:
        for module, names in (
            (sweep, ("positive_int", "non_negative_int", "parse_size", "load_trace_or_error")),
            (stackdist, ("positive_int", "read_trace")),
            (compare, ("non_negative_int", "load_trace_or_error")),
            (
                setpressure,
                ("positive_int", "non_negative_int", "parse_size", "load_trace_or_error"),
            ),
        ):
            for name in names:
                with self.subTest(module=module.__name__, name=name):
                    self.assertIs(getattr(module, name), getattr(cliargs, name))

    def test_no_module_defines_a_private_lookalike(self) -> None:
        for module in (sweep, stackdist, compare, setpressure):
            for name in ("_positive_int", "_non_negative_int", "_parse_size", "_load_trace"):
                with self.subTest(module=module.__name__, name=name):
                    self.assertFalse(hasattr(module, name))
