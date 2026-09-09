"""Trace file parsing.

Native trace format (one access per line)::

    0x00400000 R
    0x00400008 R
    0x7fff0010 W

* first field:  the byte address in hex ("0x" prefix optional).
* second field: R (read/load) or W (write/store), case-insensitive.
* blank lines and lines starting with '#' are ignored.
"""

from __future__ import annotations

from collections.abc import Iterator


def parse_trace(path: str) -> Iterator[tuple[int, bool]]:
    """Yield (address, is_write) tuples from a trace file.

    Malformed lines raise ``ValueError`` with the offending line number.
    """
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: expected 'ADDR R|W', got {line!r}")
            addr_str, op = parts
            try:
                addr = int(addr_str, 16)
            except ValueError:
                raise ValueError(f"{path}:{lineno}: bad hex address {addr_str!r}") from None
            op = op.upper()
            if op not in ("R", "W"):
                raise ValueError(f"{path}:{lineno}: op must be R or W, got {op!r}")
            yield addr, op == "W"
