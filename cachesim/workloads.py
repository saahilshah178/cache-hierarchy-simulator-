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
