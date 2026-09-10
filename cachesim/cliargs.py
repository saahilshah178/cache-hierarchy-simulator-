"""Argument types and trace loading shared by the analysis subcommands.

``sweep``, ``mrc``, ``compare`` and ``sets`` all take the same kinds of
argument -- a count that must be positive, a capacity written as ``4k``, a
trace file that may not exist -- and each had grown its own copy of the
same four helpers. Copies drift: one accepts ``4KB`` and another does not,
one says "trace not found" and another lets a FileNotFoundError escape as a
traceback. Defining them once makes the four commands agree by
construction.

The functions here belong to the command line, not to the simulator. They
raise ``argparse.ArgumentTypeError`` (which argparse turns into a usage
error and exit status 2) or call ``parser.error`` directly, so nothing in
this module is appropriate to call from library code -- ``cachesim.config``
is where a library caller's input is validated.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from typing import TypeVar

from cachesim.trace import parse_trace

T = TypeVar("T")


# -- argparse ``type=`` hooks --------------------------------------------------


def positive_int(text: str) -> int:
    """An integer greater than zero: ways, window lengths, block sizes."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {text}")
    return value


def non_negative_int(text: str) -> int:
    """An integer of zero or more: latencies and warm-up lengths, where 0 is meaningful."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be a non-negative integer, got {text}")
    return value


def parse_size(text: str) -> int:
    """``'4k'`` -> 4096, ``'2M'`` -> 2097152, ``'512'`` -> 512.

    A bare number is bytes; a ``k``/``K`` or ``m``/``M`` suffix (with an
    optional trailing ``b``/``B``) multiplies by 1024 or 1024*1024. The
    multipliers are binary because cache capacities are: a "32 KB" cache
    holds 32768 bytes, never 32000.
    """
    cleaned = text.strip().lower().removesuffix("b")
    scale = 1
    if cleaned.endswith("k"):
        scale, cleaned = 1 << 10, cleaned[:-1]
    elif cleaned.endswith("m"):
        scale, cleaned = 1 << 20, cleaned[:-1]
    try:
        value = int(cleaned) * scale
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (try 4096, 4k or 1M)") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"size must be positive, got {text!r}")
    return value


# -- trace loading -------------------------------------------------------------


def read_trace(
    parser: argparse.ArgumentParser,
    path: str,
    build: Callable[[Iterator[tuple[int, bool]]], list[T]],
) -> list[T]:
    """Consume the trace at ``path`` with ``build``, reporting failures as CLI errors.

    ``build`` receives the ``(address, is_write)`` iterator and returns
    whatever form the command wants -- the accesses themselves, or the
    block-number stream a profiler needs. It is called INSIDE the try, and
    given the iterator rather than a list, so a command that only needs
    block numbers never materialises the addresses as well.

    A missing file, an unreadable one, a malformed line, or a trace that
    parses to nothing all exit with status 2 and a one-line message; none
    of them reaches the user as a traceback.
    """
    try:
        result = build(parse_trace(path))
    except FileNotFoundError:
        parser.error(f"trace not found: {path} (run `cachesim gen-traces` to create the samples)")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not result:
        parser.error(f"no accesses in {path}")
    return result


def load_trace_or_error(parser: argparse.ArgumentParser, path: str) -> list[tuple[int, bool]]:
    """The whole trace in memory as ``(address, is_write)`` pairs.

    Commands that replay a trace several times -- once per sweep point,
    once per configuration -- must hold it, because a generator can only be
    walked once and re-parsing the file per point would dominate the run.

    Named for what distinguishes it from ``cachesim.trace.load_trace``,
    which is the library function this one wraps: this one takes a parser
    and turns a bad path into a usage error, so a reader moving between the
    command modules and the simulator cannot mistake one call for the other.
    """
    return read_trace(parser, path, list)


# -- writing output ------------------------------------------------------------


def write_or_error(parser: argparse.ArgumentParser, path: str, write: Callable[[str], T]) -> T:
    """Run ``write(path)``, reporting an unwritable path as a CLI error.

    Every command reports a bad *input* path as one line and exit status 2.
    An unwritable *output* path -- a --csv under a directory that cannot be
    created, a --plot into a read-only tree -- used to reach the user as a
    traceback, and after the table had already been printed, so a failed run
    looked like a successful one with a stack dump stapled to it.

    The writer takes the path rather than being a bare thunk so that the
    message can name the file the caller asked for.
    """
    try:
        return write(path)
    except OSError as exc:
        parser.error(f"cannot write {path}: {exc}")
