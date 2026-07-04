"""generate_traces.py — writes the sample memory-access traces.

Run from the repo root (or this directory)::

    python3 traces/generate_traces.py

Each trace is a stream of "0xADDR R|W" lines modelling a classic access
pattern. All generators are seeded, so the files are reproducible.

The patterns:

  sequential.trace     — a linear scan over a buffer, repeated. The friendliest
                         possible pattern: every 64 B block is used 8 times
                         (8-byte words), so misses only happen once per block.
  random.trace         — uniformly random addresses over a 16 MB region. The
                         cruelest pattern: almost no reuse, caches barely help.
  matmul_naive.trace   — C = A x B (accumulating into C) for 64x64 matrices
                         of doubles in the naive
                         i/j/k loop order. Walking B by COLUMN strides 512 B
                         per step, and 512 is a power of two, so B's accesses
                         hammer a handful of cache sets -> conflict misses.
  matmul_blocked.trace — the SAME multiplication, same matrices, same total
                         work, but iterated in 16x16 tiles so each tile of A,
                         B, C is reused while it is still cached. Compare its
                         miss rate against matmul_naive to see why loop order
                         matters.
  conflict.trace       — four sequential streams read in lockstep, with their
                         base addresses aligned 1 MB apart. 1 MB is a multiple
                         of every cache size worth simulating, so at any
                         moment the four hot blocks all map to the SAME set.
                         Below 4-way they evict each other endlessly (~100%
                         miss); at 4-way they coexist and the miss rate
                         collapses to one compulsory miss per block. This is
                         the cleanest possible "miss rate vs associativity"
                         knee, which is why sweep.py uses it by default.
  pointer_chase.trace  — follows a randomly-shuffled linked list through 1 MB,
                         lapping it ~4 times. Every hop lands somewhere
                         unpredictable, and (unlike random.trace) there IS
                         reuse from lap to lap — but only a cache big enough
                         to hold the whole list (here, the L3) can exploit it.
"""

import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
WORD = 8          # we model 8-byte (double / pointer) accesses
BLOCK = 64        # cache-block size the patterns are designed around


def _write(name, accesses):
    """Write an iterable of (addr, op) to `name` and report its size."""
    path = os.path.join(HERE, name)
    count = 0
    with open(path, "w") as f:
        for addr, op in accesses:
            f.write(f"0x{addr:08x} {op}\n")
            count += 1
    print(f"  {name:22s} {count:>9,} accesses")


def sequential(buffer_bytes=256 * 1024, passes=2):
    """Scan a buffer word by word, `passes` times. ~10% writes."""
    base = 0x0010_0000
    i = 0
    for _ in range(passes):
        for off in range(0, buffer_bytes, WORD):
            yield base + off, "W" if i % 10 == 9 else "R"
            i += 1


def random_access(region_bytes=16 * 1024 * 1024, n=60_000, seed=1):
    """Uniformly random word-aligned accesses over a large region. ~25% writes."""
    rng = random.Random(seed)
    base = 0x1000_0000
    for _ in range(n):
        addr = base + rng.randrange(region_bytes // WORD) * WORD
        yield addr, "W" if rng.random() < 0.25 else "R"


def _matmul_accesses(n, a_base, b_base, c_base, i0, i1, j0, j1, k0, k1):
    """Emit the accesses for one (i,j,k) sub-block of C += A x B.

    Row-major layout: element [r][c] lives at base + (r*n + c) * WORD.
    Per (i,j,k) inner step: read A[i][k], read B[k][j], read+write C[i][j].
    (We emit C's read/write once per k-run, matching `c = C[i][j]; ...` code.)
    """
    for i in range(i0, i1):
        for j in range(j0, j1):
            c_addr = c_base + (i * n + j) * WORD
            yield c_addr, "R"
            for k in range(k0, k1):
                yield a_base + (i * n + k) * WORD, "R"
                yield b_base + (k * n + j) * WORD, "R"
            yield c_addr, "W"


def matmul(n=64, tile=None):
    """Matrix multiply trace; tile=None -> naive order, tile=T -> TxT blocking."""
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


def conflict_streams(streams=4, words_per_stream=16_384, align=0x0010_0000):
    """Read `streams` arrays in lockstep: a[0], b[0], c[0], d[0], a[1], ...

    The arrays are placed `align` (1 MB) apart, and 1 MB is a multiple of
    every power-of-two cache size up to 1 MB — so address X in every array
    lands in the same set no matter the cache's size or associativity. The
    only thing that decides hit vs miss is whether the set has enough WAYS
    to hold all `streams` hot blocks at once.
    """
    base = 0x3000_0000
    for i in range(words_per_stream):
        for s in range(streams):
            yield base + s * align + i * WORD, "R"


def pointer_chase(nodes=16_384, node_bytes=BLOCK, hops=60_000, seed=2):
    """Walk a random-permutation linked list: each hop reads the next pointer.

    The nodes form one giant cycle covering nodes*node_bytes = 1 MB, in
    shuffled order, so consecutive hops are far apart in memory. 60,000 hops
    over 16,384 nodes means ~3.7 laps: the second lap onward is pure reuse
    that only a cache holding the whole megabyte can turn into hits.
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


def main():
    print("generating traces:")
    _write("sequential.trace", sequential())
    _write("random.trace", random_access())
    _write("matmul_naive.trace", matmul(n=64))
    _write("matmul_blocked.trace", matmul(n=64, tile=16))
    _write("conflict.trace", conflict_streams())
    _write("pointer_chase.trace", pointer_chase())


if __name__ == "__main__":
    main()
