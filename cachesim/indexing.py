"""Set-index functions: how a block number chooses its set.

The classic mapping takes the low-order bits of the block number
(``block % num_sets``). It is free in hardware -- the index bits are already
in the address -- but it makes the set index a function of one narrow field,
so any two blocks whose distance is a multiple of ``num_sets`` collide. That
is exactly the distance produced by walking an array with a power-of-two
stride, which is why a column walk of a 64x64 matrix of doubles, or four
streams placed 1 MB apart, can pile a whole working set into one set while
the rest of the cache sits idle.

A hashed (or "XOR-folded") index spreads the same blocks by mixing the
high-order bits into the index: split the block number into
``log2(num_sets)``-bit chunks and XOR them together. Two blocks that differ
only in high-order bits now differ in their index, so a power-of-two stride
no longer aliases, while the mapping stays a cheap tree of XOR gates and
remains a function of the block number alone (no state, no extra latency in
the tag path). The trade is that address locality no longer implies set
locality, and reasoning about which blocks conflict becomes harder.

See Gonzalez, Valero, Topham and Parcerisa, "Eliminating cache conflict
misses through XOR-based placement functions", ICS 1997, and Seznec, "A case
for two-way skewed-associative caches", ISCA 1993, for the hardware
argument; production last-level caches hash their index (and their slice
selection) for the same reason.

Every index function is built from the set count alone and returns a plain
callable ``block -> set index``:

    index = make_index("xor", num_sets=128)
    index(4096)

Block numbers are non-negative: ``Cache.access`` and ``Hierarchy.access``
reject a negative address before a block number is formed, so a negative
argument can only come from a caller that builds one by hand and reaches
``probe``/``allocate``/``set_of`` directly. No index function rejects it,
and the two disagree about what it means -- ``modulo_index`` returns
``num_sets - 1`` for block -1 (Python's ``%`` takes the sign of the
divisor), while ``xor_index`` returns 0. Neither answer models anything;
do not depend on either.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

#: A function from block number to set index.
IndexFunction: TypeAlias = Callable[[int], int]


def modulo_index(num_sets: int) -> IndexFunction:
    """``block % num_sets``: the low-order bits of the block number.

    Defined for any positive set count, not only powers of two. A negative
    block is not rejected and not meaningful: Python's ``%`` takes the sign
    of the divisor, so block -1 maps to ``num_sets - 1``, not to 0.
    """
    if num_sets < 1:
        raise ValueError(f"num_sets must be positive, got {num_sets}")

    def index(block: int) -> int:
        return block % num_sets

    return index


def xor_index(num_sets: int) -> IndexFunction:
    """XOR-fold: split the block number into index-width chunks and XOR them.

    With ``num_sets = 2**k`` the block number is cut into k-bit fields and
    all of them are XOR-ed, so every bit of the block number reaches the
    index. Requires a power-of-two set count, because the fold is defined on
    bit-fields; a one-set cache is fully associative and always yields 0.

    The fold consumes chunks while the block is positive, so a negative
    block -- which cannot arise from an address, see the module docstring --
    is not rejected and folds to 0.
    """
    if num_sets < 1:
        raise ValueError(f"num_sets must be positive, got {num_sets}")
    if num_sets & (num_sets - 1):
        raise ValueError(f"xor indexing requires a power-of-two set count, got {num_sets}")
    if num_sets == 1:

        def only_set(block: int) -> int:
            return 0

        return only_set

    bits = num_sets.bit_length() - 1
    mask = num_sets - 1

    def index(block: int) -> int:
        folded = 0
        while block > 0:
            folded ^= block & mask
            block >>= bits
        return folded

    return index


#: Registry of selectable index functions, keyed by the name used in configs.
INDEX_FUNCTIONS: dict[str, Callable[[int], IndexFunction]] = {
    "modulo": modulo_index,
    "xor": xor_index,
}

#: The mapping every cache uses unless told otherwise.
DEFAULT_INDEX = "modulo"


def make_index(name: str, num_sets: int) -> IndexFunction:
    """Build the named index function for a cache with ``num_sets`` sets.

    Raises ``ValueError`` for an unknown name or a set count the function
    cannot handle (XOR-folding needs a power-of-two set count).
    """
    try:
        factory = INDEX_FUNCTIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown index function {name!r}; choose from {sorted(INDEX_FUNCTIONS)}"
        ) from None
    return factory(num_sets)
