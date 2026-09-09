"""Trace file parsing.

Native trace format (one access per line)::

    0x00400000 R
    0x00400008 R          # trailing comments are allowed
    0x7fff0010 W

* first field:  the byte address as a non-negative hexadecimal integer
  (``0x`` prefix optional, no sign, no separators).
* second field: ``R`` (read/load) or ``W`` (write/store), case-insensitive.
* blank lines and everything after ``#`` are ignored.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_HEX = re.compile(r"(?:0[xX])?[0-9A-Fa-f]+")


def parse_trace(path: str) -> Iterator[tuple[int, bool]]:
    """Yield ``(address, is_write)`` tuples from a trace file.

    Malformed lines raise ``ValueError`` naming the file and line number.
    """
    with open(path) as f:
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
