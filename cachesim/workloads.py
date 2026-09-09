"""Synthetic memory-access workloads.

Each generator yields ``(address, op)`` pairs, where ``op`` is ``"R"`` or
``"W"``, modelling a classic access pattern. All generators are seeded, so
the traces they produce are reproducible.

sequential
    A linear scan over a buffer, repeated. Every 64-byte block is used 8 times
    (8-byte words), so misses only happen once per block per pass that does
    not fit in the cache.
random_access
    Uniformly random addresses over a 16 MB region: almost no reuse.
matmul (naive)
    C = A x B for 64x64 matrices of doubles in i/j/k loop order. Walking B by
    column strides 512 B per step; 512 is a power of two, so the column's
    blocks alias to a small group of sets within each column walk.
matmul (blocked)
    The same multiplication iterated in 16x16 tiles so each tile of A, B and
    C is reused while it is still cached.
conflict_streams
    Four sequential streams read in lockstep, with base addresses 1 MB apart.
    Address X of every stream maps to the same set for any geometry with at
    most 16384 sets, so hits require at least four ways.
pointer_chase
    Follows a randomly shuffled linked list through 1 MB, lapping it ~3.7
    times. Every hop is unpredictable, but the list is reused lap to lap, so
    a level that holds the whole megabyte serves the later laps.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Iterable, Iterator

WORD = 8          # 8-byte (double / pointer) accesses
BLOCK = 64        # cache-block size the patterns are designed around

Access = tuple[int, str]


def sequential(buffer_bytes: int = 256 * 1024, passes: int = 2) -> Iterator[Access]:
    """Scan a buffer word by word, ``passes`` times. Every 10th access is a write."""
    base = 0x0010_0000
    i = 0
    for _ in range(passes):
        for off in range(0, buffer_bytes, WORD):
            yield base + off, "W" if i % 10 == 9 else "R"
            i += 1


def random_access(region_bytes: int = 16 * 1024 * 1024, n: int = 60_000,
                  seed: int = 1) -> Iterator[Access]:
    """Uniformly random word-aligned accesses over a large region; 25% writes."""
    rng = random.Random(seed)
    base = 0x1000_0000
    for _ in range(n):
        addr = base + rng.randrange(region_bytes // WORD) * WORD
        yield addr, "W" if rng.random() < 0.25 else "R"


def _matmul_accesses(n: int, a_base: int, b_base: int, c_base: int,
                     i0: int, i1: int, j0: int, j1: int,
                     k0: int, k1: int) -> Iterator[Access]:
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
                yield from _matmul_accesses(
                    n, a, b, c,
                    i0, i0 + tile, j0, j0 + tile, k0, k0 + tile)


def conflict_streams(streams: int = 4, words_per_stream: int = 16_384,
                     align: int = 0x0010_0000) -> Iterator[Access]:
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


def pointer_chase(nodes: int = 16_384, node_bytes: int = BLOCK,
                  hops: int = 60_000, seed: int = 2) -> Iterator[Access]:
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


#: The sample traces shipped with the project: file name -> generator.
SAMPLE_TRACES: dict[str, Callable[[], Iterator[Access]]] = {
    "sequential": sequential,
    "random": random_access,
    "matmul_naive": lambda: matmul(n=64),
    "matmul_blocked": lambda: matmul(n=64, tile=16),
    "conflict": conflict_streams,
    "pointer_chase": pointer_chase,
}


def write_trace(path: str, accesses: Iterable[Access]) -> int:
    """Write an iterable of (addr, op) to ``path`` in the native format.

    Returns the number of accesses written.
    """
    count = 0
    with open(path, "w") as f:
        for addr, op in accesses:
            f.write(f"0x{addr:08x} {op}\n")
            count += 1
    return count


def write_sample_traces(out_dir: str, names: Iterable[str] | None = None,
                        verbose: bool = True) -> dict[str, int]:
    """Generate the sample traces into ``out_dir``; returns name -> access count."""
    os.makedirs(out_dir, exist_ok=True)
    counts: dict[str, int] = {}
    for name in names or SAMPLE_TRACES:
        path = os.path.join(out_dir, f"{name}.trace")
        counts[name] = write_trace(path, SAMPLE_TRACES[name]())
        if verbose:
            print(f"  {name + '.trace':22s} {counts[name]:>9,} accesses")
    return counts
