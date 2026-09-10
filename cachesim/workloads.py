"""Synthetic memory-access workloads.

Each generator yields ``(address, op)`` pairs, where ``op`` is ``"R"`` or
``"W"``, modelling one classic access pattern whose effect on a cache can
be predicted in closed form. Generators are pure and deterministic:
parameters have defaults, randomness is drawn from a ``random.Random``
seeded by an argument, and nothing depends on the environment, so a name
always produces the same trace byte for byte.

``WORKLOADS`` maps a name to a ``Workload`` record holding the generator
(already bound to its default arguments), a description of the pattern,
and the expectation: the closed form or limiting behaviour the pattern is
designed to exhibit against a cache. ``cachesim gen-traces --list`` prints
the registry; ``cachesim gen-traces NAME...`` writes any of it.

``SAMPLE_NAMES`` are the six workloads ``gen-traces`` writes by default and
the ones whose simulation results are pinned by the golden regression
tests, so their generators must not change output once released.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial

# Re-exported (the `as` form marks it deliberate for mypy): a generator and
# the writer that serialises it are usually wanted together.
from cachesim.trace import write_trace as write_trace

WORD = 8  # 8-byte (double / pointer) accesses
BLOCK = 64  # cache-block size the patterns are designed around

Access = tuple[int, str]


def sequential(buffer_bytes: int = 256 * 1024, passes: int = 2) -> Iterator[Access]:
    """Scan a buffer word by word, ``passes`` times. Every 10th access is a write."""
    base = 0x0010_0000
    i = 0
    for _ in range(passes):
        for off in range(0, buffer_bytes, WORD):
            yield base + off, "W" if i % 10 == 9 else "R"
            i += 1


def random_access(
    region_bytes: int = 16 * 1024 * 1024, n: int = 60_000, seed: int = 1
) -> Iterator[Access]:
    """Uniformly random word-aligned accesses over a large region; 25% writes."""
    rng = random.Random(seed)
    base = 0x1000_0000
    for _ in range(n):
        addr = base + rng.randrange(region_bytes // WORD) * WORD
        yield addr, "W" if rng.random() < 0.25 else "R"


def _matmul_accesses(
    n: int,
    a_base: int,
    b_base: int,
    c_base: int,
    i0: int,
    i1: int,
    j0: int,
    j1: int,
    k0: int,
    k1: int,
) -> Iterator[Access]:
    """Emit the accesses for one (i,j,k) sub-block of C += A x B.

    Row-major layout: element [r][c] lives at base + (r*n + c) * WORD.
    Per (i,j,k) inner step: read A[i][k], read B[k][j]. C[i][j] is read once
    before the k loop and written once after it, matching accumulation into
    a register.
    """
    for i in range(i0, i1):
        for j in range(j0, j1):
            c_addr = c_base + (i * n + j) * WORD
            yield c_addr, "R"
            for k in range(k0, k1):
                yield a_base + (i * n + k) * WORD, "R"
                yield b_base + (k * n + j) * WORD, "R"
            yield c_addr, "W"


def matmul(n: int = 64, tile: int | None = None) -> Iterator[Access]:
    """Matrix multiply trace; ``tile=None`` is naive order, ``tile=T`` is TxT blocking."""
    a, b, c = 0x0020_0000, 0x0030_0000, 0x0040_0000
    if tile is None:
        yield from _matmul_accesses(n, a, b, c, 0, n, 0, n, 0, n)
        return
    for i0 in range(0, n, tile):
        for j0 in range(0, n, tile):
            for k0 in range(0, n, tile):
                yield from _matmul_accesses(n, a, b, c, i0, i0 + tile, j0, j0 + tile, k0, k0 + tile)


def conflict_streams(
    streams: int = 4, words_per_stream: int = 16_384, align: int = 0x0010_0000
) -> Iterator[Access]:
    """Read ``streams`` arrays in lockstep: a[0], b[0], c[0], d[0], a[1], ...

    The arrays are placed ``align`` bytes apart (1 MB = 16384 blocks), so
    element X of every array lands in the same set for any geometry whose
    set count divides 16384. Whether the accesses hit is then decided only
    by whether the set has enough ways to hold all ``streams`` hot blocks.
    """
    base = 0x3000_0000
    for i in range(words_per_stream):
        for s in range(streams):
            yield base + s * align + i * WORD, "R"


def pointer_chase(
    nodes: int = 16_384, node_bytes: int = BLOCK, hops: int = 60_000, seed: int = 2
) -> Iterator[Access]:
    """Walk a random-permutation linked list: each hop reads the next pointer.

    The nodes form one cycle covering ``nodes * node_bytes`` (1 MB) in
    shuffled order, so consecutive hops are far apart in memory. 60,000 hops
    over 16,384 nodes is ~3.7 laps; from the second lap on, every hop is a
    reuse that only a cache holding the whole list can serve.
    """
    rng = random.Random(seed)
    order = list(range(nodes))
    rng.shuffle(order)
    # next_of[order[i]] = order[i+1]: one big cycle through all nodes.
    next_of = {order[i]: order[(i + 1) % nodes] for i in range(nodes)}
    base = 0x2000_0000
    node = order[0]
    for _ in range(hops):
        yield base + node * node_bytes, "R"
        node = next_of[node]


def strided(
    stride_bytes: int = 512, count: int = 1024, passes: int = 4, base: int = 0x0400_0000
) -> Iterator[Access]:
    """Read ``count`` addresses ``stride_bytes`` apart, ``passes`` times over.

    Closed form, for ``stride_bytes >= block_size`` (so each access is a
    block of its own) and LRU:

    * every pass touches exactly ``count`` distinct blocks, a footprint of
      ``count * block_size`` bytes of cache;
    * if the footprint exceeds the capacity the pass is a cyclic sweep
      longer than the cache, LRU's worst case, and all ``count`` accesses of
      every pass miss;
    * if it fits, only the first pass misses (``count`` compulsory misses)
      and the remaining ``(passes - 1) * count`` accesses hit.

    A power-of-two stride shrinks the effective capacity further, because
    the blocks reach only ``num_sets / (stride_bytes / block_size)`` of the
    sets (D. H. Bailey, "Unfavorable Strides in Cache Memory Systems",
    Scientific Programming 4(2), 1995).
    """
    for _ in range(passes):
        for i in range(count):
            yield base + i * stride_bytes, "R"


def column_walk(
    rows: int = 256,
    cols: int = 256,
    row_stride_bytes: int | None = None,
    base: int = 0x0500_0000,
) -> Iterator[Access]:
    """Walk a row-major matrix of doubles down each column in turn.

    Element (i, j) lives at ``base + i*row_stride + j*8``; the loop nest is
    j outer, i inner, so consecutive accesses are one row stride apart.
    ``row_stride_bytes`` defaults to ``cols * 8``, the unpadded layout.

    A block holds 8 doubles, so 8 consecutive columns share one block per
    row: column j+1 is entirely reuse of the blocks column j fetched, and
    the compulsory floor is ``rows * ceil(cols / 8)`` blocks. Whether that
    reuse survives depends on the stride. With the unpadded power-of-two
    stride the ``rows`` blocks of a column land on only
    ``num_sets / (row_stride / block_size)`` sets and evict one another
    before the next column arrives; padding the stride by one block
    (``row_stride_bytes = cols*8 + 64``) makes the row-to-row distance an
    odd number of blocks, which is coprime with any power-of-two set count,
    so a column spreads over every set and the reuse is realised
    (D. H. Bailey, "Unfavorable Strides in Cache Memory Systems",
    Scientific Programming 4(2), 1995).
    """
    row_stride = cols * WORD if row_stride_bytes is None else row_stride_bytes
    for j in range(cols):
        for i in range(rows):
            yield base + i * row_stride + j * WORD, "R"


def stencil_2d(n: int = 96, passes: int = 2, base: int = 0x0600_0000) -> Iterator[Access]:
    """Five-point Jacobi stencil over an ``n`` x ``n`` grid of doubles.

    For each interior point (i, j) the pass reads in(i,j), in(i-1,j),
    in(i+1,j), in(i,j-1), in(i,j+1) and writes out(i,j): five reads and one
    write, ``6 * (n-2)**2`` accesses per pass. ``out`` is placed one block
    past the end of ``in`` so the two arrays are offset by an odd number of
    blocks and do not systematically alias.

    Closed form, for a cache that holds three grid rows of ``in`` plus one
    of ``out`` (``4 * n * 8`` bytes): the sweep of row i has already
    fetched rows i-1 and i, so the only new data is row i+1 of ``in`` and
    row i of ``out``, one block per 8 elements each. The miss rate tends to
    2 misses per 8 interior points, or 1 miss per 24 accesses; the
    compulsory floor is the blocks of the two arrays, ceil(n*n/8) for
    ``in`` plus those spanning the interior of ``out`` (2280 at n=96)
    (R. Rivera and C.-W. Tseng, "Tiling Optimizations for 3D Scientific
    Computations", SC 2000).
    """
    grid_bytes = n * n * WORD
    in_base = base
    out_base = base + grid_bytes + BLOCK
    for _ in range(passes):
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                centre = in_base + (i * n + j) * WORD
                yield centre, "R"
                yield in_base + ((i - 1) * n + j) * WORD, "R"
                yield in_base + ((i + 1) * n + j) * WORD, "R"
                yield centre - WORD, "R"
                yield centre + WORD, "R"
                yield out_base + (i * n + j) * WORD, "W"


def transpose(n: int = 128, base: int = 0x0700_0000) -> Iterator[Access]:
    """B = A-transpose for ``n`` x ``n`` doubles: read A by row, write B by column.

    Element (i, j) of A is read from ``base + (i*n + j)*8`` and written to
    B at ``b_base + (j*n + i)*8``, with B placed one block past the end of
    A. ``2 * n**2`` accesses.

    Closed form: the reads of A are sequential, so 1 in 8 misses. The
    writes of B stride ``n * 8`` bytes, one block each, so every write is a
    fresh block: with a power-of-two ``n`` those blocks alias onto
    ``num_sets / (n*8 / block_size)`` sets and a column of B is evicted long
    before the next column reuses it, giving 1 miss per write. Misses per
    element therefore tend to 1 + 1/8 = 1.125 until the cache holds a whole
    column of blocks of B (S. Chatterjee and S. Sen, "Cache-Efficient Matrix
    Transposition", HPCA 2000).
    """
    a_base = base
    b_base = base + n * n * WORD + BLOCK
    for i in range(n):
        for j in range(n):
            yield a_base + (i * n + j) * WORD, "R"
            yield b_base + (j * n + i) * WORD, "W"


def binary_search(
    n_elements: int = 65_536, queries: int = 5_000, seed: int = 3, base: int = 0x0800_0000
) -> Iterator[Access]:
    """Binary-search a sorted array of 8-byte keys for ``queries`` random targets.

    The array holds the keys 0..n_elements-1 in order, so searching for the
    key at index t probes the same positions a real search would (D. E.
    Knuth, "The Art of Computer Programming", Vol. 3, 2nd ed., 1998,
    section 6.2.1); each probe reads one 8-byte key and the search stops on
    the hit. Targets are drawn from a ``random.Random(seed)``, so the trace
    is fixed by the seed.

    Closed form: a query probes about log2(n_elements) positions, but the
    probes form a binary tree whose level d has only 2^d distinct
    positions, so the top levels are a handful of blocks that stay resident
    while the deep levels are effectively random over the array. A cache of
    C bytes holds the top log2(C / block_size) levels of that tree, leaving
    roughly log2(n_elements * 8 / C) probes per query to miss.
    """
    rng = random.Random(seed)
    for _ in range(queries):
        target = rng.randrange(n_elements)
        lo, hi = 0, n_elements - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            yield base + mid * WORD, "R"
            if mid == target:
                break
            if mid < target:
                lo = mid + 1
            else:
                hi = mid - 1


def hash_probe(
    table_bytes: int = 1024 * 1024,
    probes: int = 60_000,
    seed: int = 4,
    base: int = 0x0900_0000,
) -> Iterator[Access]:
    """Read uniformly random 8-byte slots of a ``table_bytes`` table.

    The access pattern of a lookup in a large hash table: the index is
    unpredictable and every slot is equally likely, so there is no locality
    beyond the block a probe lands in.

    Closed form: under the independent reference model with uniform
    probabilities, an LRU cache of C bytes holds a C/table_bytes fraction of
    the table and the steady-state miss rate is 1 - C/table_bytes
    (E. G. Coffman and P. J. Denning, "Operating Systems Theory",
    Prentice-Hall, 1973, chapter 6). The measured rate approaches it from
    above, because the cache starts empty.
    """
    rng = random.Random(seed)
    slots = table_bytes // WORD
    for _ in range(probes):
        yield base + rng.randrange(slots) * WORD, "R"


def cyclic(
    blocks: int = 1024, passes: int = 8, block_bytes: int = BLOCK, base: int = 0x0A00_0000
) -> Iterator[Access]:
    """Touch blocks 0..``blocks``-1 in order, ``passes`` times: LRU's worst case.

    Closed form, for a cache holding K blocks:

    * K >= blocks: only the first pass misses, ``blocks`` compulsory misses
      in ``blocks * passes`` accesses;
    * K < blocks: LRU evicts each block exactly one access before it is
      needed again, so every access of every pass misses, a 100% miss rate
      no matter how close K is to ``blocks``.

    The cliff between the two is the classic demonstration that LRU is not
    resistant to cyclic reuse patterns (L. A. Belady, "A Study of
    Replacement Algorithms for a Virtual-Storage Computer", IBM Systems
    Journal 5(2), 1966).
    """
    for _ in range(passes):
        for b in range(blocks):
            yield base + b * block_bytes, "R"


@dataclass(frozen=True)
class Workload:
    """A named access pattern, with the behaviour it is designed to show.

    name        : registry key, and the stem of the file ``gen-traces`` writes.
    description : what the pattern accesses, and with which parameters.
    generator   : the generator bound to its default arguments; calling it
                  yields the trace as ``(address, "R"|"W")`` pairs.
    expectation : the closed form or limiting behaviour to expect, stated
                  precisely enough to check against a simulation.
    """

    name: str
    description: str
    generator: Callable[[], Iterator[Access]]
    expectation: str


_WORKLOAD_LIST: tuple[Workload, ...] = (
    Workload(
        name="sequential",
        description=(
            "Linear scan of a 256 KB buffer by 8-byte words, twice, "
            "with every tenth access a store."
        ),
        generator=sequential,
        expectation=(
            "One miss per 64-byte block per pass that does not fit: the 32 KB L1 misses "
            "1 access in 8 (8 words per block), and the 256 KB L2 holds the buffer, so "
            "only the first pass reaches DRAM (4096 blocks)."
        ),
    ),
    Workload(
        name="random",
        description=(
            "Uniformly random 8-byte accesses over a 16 MB region, a quarter of them stores."
        ),
        generator=random_access,
        expectation=(
            "No reuse to speak of: a cache of C bytes settles at a miss rate of "
            "1 - C/16 MB, so a 2 MB last level lets 87.5% of accesses through to DRAM "
            "in steady state (91.2% measured from cold over 60,000 accesses)."
        ),
    ),
    Workload(
        name="matmul_naive",
        description=(
            "C = A x B for 64x64 matrices of doubles in i/j/k loop order, walking B by column."
        ),
        generator=partial(matmul, n=64),
        expectation=(
            "B's column walk strides 512 B, a power of two, so a column's 64 blocks "
            "alias onto 8 sets of the 4-way L1 and are evicted before the next column "
            "reuses them; DRAM traffic still falls to the compulsory floor of "
            "3 x 512 = 1536 blocks."
        ),
    ),
    Workload(
        name="matmul_blocked",
        description="The same 64x64 multiplication iterated in 16x16 tiles.",
        generator=partial(matmul, n=64, tile=16),
        expectation=(
            "Each 16x16 tile of A, B and C is 2 KB, so all three stay in the 32 KB L1 "
            "while they are reused: L1 misses drop from 27,912 to 3,572 for the same "
            "1536-block DRAM footprint."
        ),
    ),
    Workload(
        name="conflict",
        description="Four sequential streams read in lockstep, their bases 1 MB apart.",
        generator=conflict_streams,
        expectation=(
            "1 MB is 16,384 blocks, so element X of every stream maps to one set for any "
            "geometry with at most 16,384 sets: a 4-way L1 holds all four hot blocks and "
            "misses 1 access in 8, while 1 or 2 ways miss every access. No block is ever "
            "revisited, so every miss is compulsory and no lower level ever hits."
        ),
    ),
    Workload(
        name="pointer_chase",
        description=(
            "60,000 hops around a randomly shuffled 16,384-node linked list "
            "spanning 1 MB, one 64-byte node per hop."
        ),
        generator=pointer_chase,
        expectation=(
            "Hops are unpredictable and each node has its own block, so the 32 KB L1 and "
            "256 KB L2 never hit; the 2 MB L3 holds the whole list, so only the first lap "
            "misses there: 16,384 misses out of 60,000, the compulsory floor."
        ),
    ),
    Workload(
        name="strided",
        description="Reads 1024 addresses 512 B apart, four times over.",
        generator=strided,
        expectation=(
            "Each pass touches 1024 distinct blocks, a 64 KB footprint that both the "
            "256 KB L2 and the 2 MB L3 could hold; but a 512 B stride reaches only one "
            "set in eight, so L1 and L2 miss all 4096 accesses and only the 16-way L3 "
            "keeps the footprint, missing just the first pass (1024)."
        ),
    ),
    Workload(
        name="column_walk",
        description=("Column-major walk of a 256x256 row-major matrix of doubles, 2048 B rows."),
        generator=column_walk,
        expectation=(
            "Eight columns share one block per row, so the compulsory floor is 8192 "
            "blocks; unpadded, the power-of-two row stride confines a column to one set "
            "in 32 and all 65,536 accesses miss L1, while padding the row stride by one "
            "block (row_stride_bytes=2112) brings L1 misses down to the 8192 floor and "
            "the run from 4,489,216 to 1,507,328 cycles."
        ),
    ),
    Workload(
        name="stencil_2d",
        description="Two passes of a 5-point Jacobi stencil over a 96x96 grid of doubles.",
        generator=stencil_2d,
        expectation=(
            "Three grid rows stay resident, so the miss rate tends to 2 misses per 8 "
            "interior points, one for each array: 4560 L1 misses in 106,032 accesses "
            "(4.30%, against 1/24 = 4.17% asymptotically), and DRAM sees exactly the "
            "2280-block compulsory floor."
        ),
    ),
    Workload(
        name="transpose",
        description=(
            "B = A-transpose for 128x128 doubles: A read row-major, B written column-major."
        ),
        generator=transpose,
        expectation=(
            "The sequential reads miss 1 in 8 and every column-major write misses, so "
            "misses per element tend to 1.125: exactly 18,432 L1 misses for 16,384 "
            "elements, at the 4096-block DRAM floor."
        ),
    ),
    Workload(
        name="binary_search",
        description=(
            "5000 binary searches for uniformly random keys in a sorted 512 KB array "
            "of 65,536 8-byte keys."
        ),
        generator=binary_search,
        expectation=(
            "15.0 probes per query on average (74,982 accesses); the top levels of the "
            "search tree are a handful of blocks and stay cached, so the miss rate falls "
            "with cache size: 83.1% at the 32 KB L1, 48.7% at the 256 KB L2, 22.1% at "
            "the 2 MB L3."
        ),
    ),
    Workload(
        name="hash_probe",
        description="60,000 uniformly random 8-byte probes into a 1 MB table.",
        generator=hash_probe,
        expectation=(
            "The independent-reference model puts an LRU cache of C bytes at a "
            "steady-state miss rate of 1 - C/1 MB: the 32 KB L1 measures 96.89% against "
            "96.875% predicted, and the 2 MB L3 holds the table, converging on its "
            "compulsory floor (15,975 of the 16,384 blocks are ever probed)."
        ),
    ),
    Workload(
        name="cyclic",
        description="Eight passes over 1024 consecutive 64-byte blocks, one access each.",
        generator=cyclic,
        expectation=(
            "The 64 KB cycle exceeds the 32 KB L1, so LRU evicts every block one access "
            "before it is needed again and all 8192 accesses miss; the 256 KB L2 holds "
            "the cycle, so only the first pass misses there (1024, 1 in 8)."
        ),
    ),
)

#: Every synthetic workload, by name, in registry order.
WORKLOADS: dict[str, Workload] = {w.name: w for w in _WORKLOAD_LIST}

#: The workloads ``gen-traces`` writes when given no names. These six are
#: pinned by the golden regression tests: their output must not change.
SAMPLE_NAMES: tuple[str, ...] = (
    "sequential",
    "random",
    "matmul_naive",
    "matmul_blocked",
    "conflict",
    "pointer_chase",
)


def get_workload(name: str) -> Workload:
    """Look up a workload by name, or raise ``ValueError`` listing the names."""
    try:
        return WORKLOADS[name]
    except KeyError:
        raise ValueError(f"unknown workload {name!r}; choose from {', '.join(WORKLOADS)}") from None


def write_traces(
    out_dir: str, names: Iterable[str] | None = None, verbose: bool = True
) -> dict[str, int]:
    """Write one native-format trace per named workload into ``out_dir``.

    ``names`` defaults to ``SAMPLE_NAMES``. Returns name -> access count;
    each file is ``<out_dir>/<name>.trace``.
    """
    os.makedirs(out_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for name in names or SAMPLE_NAMES:
        workload = get_workload(name)
        path = os.path.join(out_dir, f"{name}.trace")
        counts[name] = write_trace(path, workload.generator())
        if verbose:
            print(f"  {name + '.trace':22s} {counts[name]:>9,} accesses")
    return counts
