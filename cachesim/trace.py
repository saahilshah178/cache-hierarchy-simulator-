"""Trace file parsing.

Three input formats are understood. All of them are decoded to the same
thing: a stream of ``(byte address, is_write)`` pairs. Access size is not
part of that stream -- every reference is modelled as touching only the
block that contains its first byte, so a reference that straddles a block
boundary is charged one block, not two. Formats that carry a size field
(Dinero, Lackey) therefore ignore it.

native (any other extension)
    One access per line, ``ADDR R|W``::

        0x00400000 R
        0x00400008 R          # trailing comments are allowed
        0x7fff0010 W

    * first field:  the byte address as a non-negative hexadecimal integer
      (``0x`` prefix optional, no sign, no separators).
    * second field: ``R`` (read/load) or ``W`` (write/store), case-insensitive.
    * blank lines and everything after ``#`` are ignored.
    * anything else is a hard error naming the file and line.

dinero (``.din``)
    The trace format of Dinero IV (J. Edler and M. D. Hill, "Dinero IV:
    Trace-Driven Uniprocessor Cache Simulator", University of
    Wisconsin-Madison, 1998), one reference per line, ``LABEL ADDR``::

        0 7fff0010
        1 7fff0018
        2 00400000

    Label ``0`` is a read, ``1`` a write, and ``2`` an instruction fetch,
    which is decoded as a read because this simulator models one unified
    cache hierarchy rather than split instruction and data caches. Dinero's
    escape labels (``3`` and ``4``) carry no memory reference and are
    rejected rather than silently dropped. Blank lines are ignored;
    anything else is a hard error naming the file and line.

lackey (``.lackey``, ``.vg``)
    The output of Valgrind's Lackey tool run with ``--trace-mem=yes``
    (N. Nethercote and J. Seward, "Valgrind: A Framework for Heavyweight
    Dynamic Binary Instrumentation", PLDI 2007)::

        ==31976== Command: ./a.out
        I  0421f3d8,8
         L 0421f3e0,8
         S 0421f3e8,8
         M 0421f3f0,8

    ``I`` (instruction fetch) and ``L`` (load) decode to reads, ``S``
    (store) to a write. ``M`` (modify: a load and a store of the same
    address in one instruction) is decoded as a single write, which is
    exact for a write-allocate, write-back hierarchy: the load part of a
    read-modify-write brings the same block in that the store then
    dirties. The size field after the comma is ignored (see above).

    Lackey's records are interleaved with Valgrind's own ``==pid==``
    commentary and with whatever the traced program prints, so a line that
    does not match the reference pattern is skipped instead of raising.
    A corrupt reference line is skipped for the same reason; use the native
    format when strict validation matters.

Any path ending in ``.gz`` is compressed and decompressed transparently,
on both read and write; the extension that selects the format is the one
before ``.gz`` (``sequential.din.gz`` is a Dinero trace).

``write_trace`` emits any of the three formats, so a trace can be
converted by reading it in one and writing it in another.
"""

from __future__ import annotations

import gzip
import os
import re
from collections.abc import Callable, Iterable, Iterator
from typing import TextIO

#: Trace formats, in addition to ``"auto"`` (choose by file extension).
FORMATS = ("native", "dinero", "lackey")

#: File extension -> format, applied after any ``.gz`` suffix is stripped.
_EXTENSIONS = {".din": "dinero", ".lackey": "lackey", ".vg": "lackey"}

_HEX = re.compile(r"(?:0[xX])?[0-9A-Fa-f]+")
_LACKEY_LINE = re.compile(r"([ILSM])[ \t]+((?:0[xX])?[0-9A-Fa-f]+),[0-9]+")

#: Dinero IV reference labels: instruction fetches count as reads.
_DINERO_LABELS = {"0": False, "1": True, "2": False}
#: Lackey record kinds; a modify is decoded as one write (see module docstring).
_LACKEY_KINDS = {"I": False, "L": False, "S": True, "M": True}


def detect_format(path: str) -> str:
    """Choose a trace format from ``path``'s extension.

    ``.din`` is Dinero, ``.lackey`` and ``.vg`` are Lackey, everything else
    is the native format. A ``.gz`` suffix is stripped first, so the choice
    is made on the extension that names the format. Matching is
    case-insensitive; the file is not opened.
    """
    name = path.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return _EXTENSIONS.get(os.path.splitext(name)[1], "native")


def _open_text(path: str) -> TextIO:
    """Open a trace for reading, decompressing it if the name ends in ``.gz``."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _parse_native(path: str) -> Iterator[tuple[int, bool]]:
    """Yield ``(address, is_write)`` from a native-format trace."""
    with _open_text(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'ADDR R|W', got {line!r}")
            addr_str, op = parts
            if not _HEX.fullmatch(addr_str):
                raise ValueError(f"{path}:{lineno}: bad hex address {addr_str!r}")
            op = op.upper()
            if op not in ("R", "W"):
                raise ValueError(f"{path}:{lineno}: op must be R or W, got {op!r}")
            yield int(addr_str, 16), op == "W"


def _parse_dinero(path: str) -> Iterator[tuple[int, bool]]:
    """Yield ``(address, is_write)`` from a Dinero IV ``.din`` trace."""
    with _open_text(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'LABEL ADDR', got {line!r}")
            label, addr_str = parts
            if label not in _DINERO_LABELS:
                raise ValueError(
                    f"{path}:{lineno}: label must be 0 (read), 1 (write) or "
                    f"2 (instruction fetch), got {label!r}"
                )
            if not _HEX.fullmatch(addr_str):
                raise ValueError(f"{path}:{lineno}: bad hex address {addr_str!r}")
            yield int(addr_str, 16), _DINERO_LABELS[label]


def _parse_lackey(path: str) -> Iterator[tuple[int, bool]]:
    """Yield ``(address, is_write)`` from Valgrind Lackey ``--trace-mem=yes`` output.

    Lines that do not match the reference pattern are skipped, never
    rejected: the format is a log that Valgrind interleaves with its own
    commentary.
    """
    with _open_text(path) as f:
        for raw in f:
            match = _LACKEY_LINE.fullmatch(raw.strip())
            if match is None:
                continue
            yield int(match.group(2), 16), _LACKEY_KINDS[match.group(1)]


_READERS: dict[str, Callable[[str], Iterator[tuple[int, bool]]]] = {
    "native": _parse_native,
    "dinero": _parse_dinero,
    "lackey": _parse_lackey,
}


def open_trace(path: str, fmt: str = "auto") -> Iterator[tuple[int, bool]]:
    """Stream ``(address, is_write)`` pairs from a trace file.

    ``fmt`` is one of ``FORMATS`` or ``"auto"`` (the default), which picks
    the format from the extension via ``detect_format``. An unknown ``fmt``
    raises ``ValueError`` immediately; parse errors surface as ``ValueError``
    naming the file and line while the returned iterator is consumed.

    The file stays open until the iterator is exhausted or dropped.
    """
    if fmt == "auto":
        fmt = detect_format(path)
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"unknown trace format {fmt!r}; choose from auto, {', '.join(FORMATS)}")
    return reader(path)


def parse_trace(path: str, fmt: str = "auto") -> Iterator[tuple[int, bool]]:
    """Yield ``(address, is_write)`` tuples from a trace file.

    A thin alias for ``open_trace``; see it and the module docstring for
    the formats and their error behaviour.
    """
    return open_trace(path, fmt)


def load_trace(path: str, fmt: str = "auto") -> list[tuple[int, bool]]:
    """Read a whole trace into a list of ``(address, is_write)`` tuples.

    Materialising a trace costs roughly 100 bytes per access in CPython
    (a list slot, a tuple, and the address object), so a 10-million-access
    trace needs about 1 GB. Prefer ``open_trace`` when one pass is enough;
    use this when the same trace is replayed many times, as ``cachesim
    sweep`` does, and re-parsing would dominate the run.
    """
    return list(open_trace(path, fmt))


# -- writing ------------------------------------------------------------------

#: Bytes reported in the size field of the formats that have one. The
#: simulator ignores access size on input, so it is a constant on output.
WRITE_SIZE = 8


def _is_write(op: str | bool) -> bool:
    """Normalise an access's operation field to a bool.

    Accepts both shapes in circulation: the ``"R"``/``"W"`` strings the
    workload generators emit and the ``is_write`` bools the readers yield.
    """
    if isinstance(op, bool):
        return op
    upper = op.upper()
    if upper not in ("R", "W"):
        raise ValueError(f"op must be 'R', 'W', or a bool, got {op!r}")
    return upper == "W"


def _format_native(addr: int, is_write: bool) -> str:
    return f"0x{addr:08x} {'W' if is_write else 'R'}\n"


def _format_dinero(addr: int, is_write: bool) -> str:
    return f"{1 if is_write else 0} {addr:08x}\n"


def _format_lackey(addr: int, is_write: bool) -> str:
    return f" {'S' if is_write else 'L'} {addr:08x},{WRITE_SIZE}\n"


_WRITERS: dict[str, Callable[[int, bool], str]] = {
    "native": _format_native,
    "dinero": _format_dinero,
    "lackey": _format_lackey,
}


def write_trace(path: str, accesses: Iterable[tuple[int, str | bool]], fmt: str = "native") -> int:
    """Write ``accesses`` to ``path`` in ``fmt``; returns the number written.

    Each access is ``(address, op)``, where ``op`` is ``"R"``/``"W"`` (what
    the workload generators emit) or an ``is_write`` bool (what the readers
    yield). ``path`` ending in ``.gz`` is compressed.

    Only the read/write distinction survives a round trip: the writers emit
    Dinero label 0/1 and Lackey ``L``/``S`` records, never an instruction
    fetch or a modify, and the size field is the constant ``WRITE_SIZE``.
    Reading any written file back therefore reproduces the same
    ``(address, is_write)`` sequence in every format.
    """
    encode = _WRITERS.get(fmt)
    if encode is None:
        raise ValueError(f"unknown trace format {fmt!r}; choose from {', '.join(FORMATS)}")
    if path.lower().endswith(".gz"):
        with gzip.open(path, "wt") as gz:
            return _emit(gz, accesses, encode)
    with open(path, "w") as f:
        return _emit(f, accesses, encode)


def _emit(
    f: TextIO,
    accesses: Iterable[tuple[int, str | bool]],
    encode: Callable[[int, bool], str],
) -> int:
    """Write every access through ``encode``; returns the number written."""
    count = 0
    for addr, op in accesses:
        f.write(encode(addr, _is_write(op)))
        count += 1
    return count
