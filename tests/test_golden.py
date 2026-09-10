"""Golden regressions for the six sample workloads.

Every number below was produced by this code and is pinned so that a change
in a generator, the trace format, the parser, or the simulator shows up as a
failing test rather than as different documentation. They are the figures the
project documentation quotes, and the whole path is covered end to end:
generator -> native trace file -> parser -> ``Hierarchy.from_config(DEFAULT_CONFIG)``,
which is exactly what ``cachesim run traces/<name>.trace`` does.

Regenerating the traces takes a few seconds, so the class honours
``CACHESIM_SKIP_SLOW`` (any value but ``0`` or empty skips it, as in
``CACHESIM_SKIP_SLOW=1 pytest``). It runs by default.

To re-pin after an intended change, run::

    python -m cachesim gen-traces --out-dir traces
    shasum -a 256 traces/*.trace
    python -m cachesim run traces/<name>.trace --format json
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from dataclasses import dataclass

from cachesim import run_trace
from cachesim.workloads import SAMPLE_NAMES, write_traces
from tests.helpers import SKIP_SLOW


@dataclass(frozen=True)
class Golden:
    """The pinned result of one workload on the default hierarchy.

    ``l1``/``l2``/``l3`` are ``(hits, misses)`` at that level; ``sha256`` is
    the digest of the native trace file ``gen-traces`` writes.
    """

    accesses: int
    reads: int
    writes: int
    l1: tuple[int, int]
    l2: tuple[int, int]
    l3: tuple[int, int]
    dram_reads: int
    dram_writes: int
    total_cycles: int
    sha256: str


#: name -> pinned result. See the module docstring for how to re-pin.
GOLDEN: dict[str, Golden] = {
    "sequential": Golden(
        accesses=65_536,
        reads=58_983,
        writes=6553,
        l1=(57_344, 8192),
        l2=(4096, 4096),
        l3=(0, 4096),
        dram_reads=4096,
        dram_writes=0,
        total_cycles=933_888,
        sha256="14c2703c518caa071d5d208bf395d6b6785c1d4e1e990637041ffc905e591a8d",
    ),
    "random": Golden(
        accesses=60_000,
        reads=44_832,
        writes=15_168,
        l1=(107, 59_893),
        l2=(786, 59_107),
        l3=(4368, 54_739),
        dram_reads=54_739,
        dram_writes=5823,
        total_cycles=8_796_896,
        sha256="a698bb268b5c5957823b10c80bd74a0958585b74f32dc3d8cb6c8e227cfc53f4",
    ),
    "matmul_naive": Golden(
        accesses=532_480,
        reads=528_384,
        writes=4096,
        l1=(504_568, 27_912),
        l2=(26_376, 1536),
        l3=(0, 1536),
        dram_reads=1536,
        dram_writes=0,
        total_cycles=2_679_904,
        sha256="2b0474dc8031159bec82ba24635d3b67d7a3606dfd812b2f1f5bdb1ccd7fac70",
    ),
    "matmul_blocked": Golden(
        accesses=557_056,
        reads=540_672,
        writes=16_384,
        l1=(553_484, 3572),
        l2=(2036, 1536),
        l3=(0, 1536),
        dram_reads=1536,
        dram_writes=0,
        total_cycles=2_486_128,
        sha256="c52a61b56520ecaa4c8885a36a6143bd5519ae19319ca5b5631b836808897b97",
    ),
    "conflict": Golden(
        accesses=65_536,
        reads=65_536,
        writes=0,
        l1=(57_344, 8192),
        l2=(0, 8192),
        l3=(0, 8192),
        dram_reads=8192,
        dram_writes=0,
        total_cycles=1_507_328,
        sha256="5f7948a5f11710e10f83533de06465f52d4744a074020945b8b7424dd7e043ea",
    ),
    "pointer_chase": Golden(
        accesses=60_000,
        reads=60_000,
        writes=0,
        l1=(0, 60_000),
        l2=(0, 60_000),
        l3=(43_616, 16_384),
        dram_reads=16_384,
        dram_writes=0,
        total_cycles=4_998_400,
        sha256="6c80ba174397ee9c859f0fb4fc0a054e512c2b975686364aa20a94db4aa7c179",
    ),
}


@unittest.skipIf(SKIP_SLOW, "CACHESIM_SKIP_SLOW is set")
class TestGolden(unittest.TestCase):
    """The sample traces and their simulation results must not drift."""

    tmp: tempfile.TemporaryDirectory[str]
    paths: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        write_traces(cls.tmp.name, SAMPLE_NAMES, verbose=False)
        cls.paths = {name: os.path.join(cls.tmp.name, f"{name}.trace") for name in SAMPLE_NAMES}

    def test_every_sample_is_pinned(self) -> None:
        self.assertEqual(sorted(GOLDEN), sorted(SAMPLE_NAMES))

    def test_trace_files_are_byte_for_byte_unchanged(self) -> None:
        for name, golden in GOLDEN.items():
            with self.subTest(name=name), open(self.paths[name], "rb") as f:
                self.assertEqual(hashlib.sha256(f.read()).hexdigest(), golden.sha256)

    def test_simulation_results_are_unchanged(self) -> None:
        for name, golden in GOLDEN.items():
            with self.subTest(name=name):
                stats = run_trace(self.paths[name]).stats()
                l1, l2, l3 = stats.levels
                self.assertEqual(
                    Golden(
                        accesses=stats.accesses,
                        reads=stats.reads,
                        writes=stats.writes,
                        l1=(l1.hits, l1.misses),
                        l2=(l2.hits, l2.misses),
                        l3=(l3.hits, l3.misses),
                        dram_reads=stats.dram_reads,
                        dram_writes=stats.dram_writes,
                        total_cycles=stats.total_cycles,
                        sha256=golden.sha256,
                    ),
                    golden,
                )

    def test_accesses_and_cycles_are_self_consistent(self) -> None:
        # A cross-check on the pinned numbers themselves: every access is a
        # read or a write, every access is a hit or a miss at L1, and the
        # cycle count is the one the level latencies imply.
        for name, g in GOLDEN.items():
            with self.subTest(name=name):
                self.assertEqual(g.reads + g.writes, g.accesses)
                self.assertEqual(g.l1[0] + g.l1[1], g.accesses)
                self.assertEqual(g.l2[0] + g.l2[1], g.l1[1])  # L2 sees L1's misses
                self.assertEqual(g.l3[0] + g.l3[1], g.l2[1])
                self.assertEqual(g.l3[1], g.dram_reads)  # and DRAM sees L3's
                expected_cycles = 4 * g.accesses + 12 * g.l1[1] + 40 * g.l2[1] + 100 * g.dram_reads
                self.assertEqual(g.total_cycles, expected_cycles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
