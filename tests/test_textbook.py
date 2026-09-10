"""Textbook known answers.

Every case here is a published example with a published result, so the
numbers are checks on the simulator rather than on the test author. Each
test names its source.

Page replacement and cache replacement are the same problem at different
granularities, so the page-replacement examples run on a fully associative
cache of C blocks: one set, C ways, one block per "page".
"""

from __future__ import annotations

import random
import unittest

from cachesim.cache import Cache
from cachesim.opt import opt_misses, run_opt

#: Page size stand-in; block numbers then equal the page numbers.
BLOCK = 16

# Silberschatz, Galvin and Gagne, "Operating System Concepts", 10th ed.,
# 2018, sec. 10.4. The reference string used for the FIFO, LRU and optimal
# examples in that section.
SILBERSCHATZ = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]

# Belady, Nelson and Shedler, "An anomaly in space-time characteristics of
# certain programs running in a paging machine", CACM 12(6), 1969. The
# standard string for which FIFO gets worse with more frames.
BELADY_ANOMALY = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]


def faults(policy: str, references: list[int], frames: int) -> int:
    """Misses of a fully associative cache of ``frames`` blocks."""
    if policy == "opt":
        accesses = [(page * BLOCK, False) for page in references]
        return run_opt(accesses, frames * BLOCK, BLOCK, frames, track_3c=False).misses
    cache = Cache("FA", frames * BLOCK, BLOCK, frames, policy=policy, track_3c=False)
    for page in references:
        cache.access(page * BLOCK)
    return cache.misses


def cyclic(blocks: int, passes: int) -> list[int]:
    """``passes`` sequential scans of blocks 0..blocks-1."""
    return [block for _ in range(passes) for block in range(blocks)]


class TestSilberschatzReferenceString(unittest.TestCase):
    """Operating System Concepts, 10th ed., sec. 10.4, three frames."""

    def test_fifo_lru_and_optimal_page_fault_counts(self) -> None:
        self.assertEqual(faults("fifo", SILBERSCHATZ, 3), 15)  # sec. 10.4.2
        self.assertEqual(faults("lru", SILBERSCHATZ, 3), 12)  # sec. 10.4.4
        self.assertEqual(faults("opt", SILBERSCHATZ, 3), 9)  # sec. 10.4.3
        self.assertEqual(opt_misses(SILBERSCHATZ, 3), 9)  # and the offline bound

    def test_lru_is_not_always_better_than_fifo(self) -> None:
        # Two frames on the same string: LRU takes 17 faults and FIFO 15.
        # Being closer to OPT on average does not make LRU better on every
        # input, which is the point of measuring instead of assuming.
        self.assertEqual(faults("fifo", SILBERSCHATZ, 2), 15)
        self.assertEqual(faults("lru", SILBERSCHATZ, 2), 17)
        self.assertEqual(faults("opt", SILBERSCHATZ, 2), 13)


class TestBeladysAnomaly(unittest.TestCase):
    """FIFO can fault more with more memory (Belady, Nelson, Shedler, 1969)."""

    def test_fifo_faults_more_with_four_frames_than_with_three(self) -> None:
        self.assertEqual(faults("fifo", BELADY_ANOMALY, 3), 9)
        self.assertEqual(faults("fifo", BELADY_ANOMALY, 4), 10)

    def test_lru_and_opt_do_not_show_the_anomaly_on_that_string(self) -> None:
        # Both are stack algorithms: the contents at C frames are always a
        # subset of the contents at C+1, so more memory cannot cost misses.
        self.assertEqual(
            (faults("lru", BELADY_ANOMALY, 3), faults("lru", BELADY_ANOMALY, 4)), (10, 8)
        )
        self.assertEqual(
            (faults("opt", BELADY_ANOMALY, 3), faults("opt", BELADY_ANOMALY, 4)), (7, 6)
        )


class TestStackProperty(unittest.TestCase):
    """Mattson, Gecsei, Slutz and Traiger, "Evaluation techniques for storage
    hierarchies", IBM Systems Journal 9(2), 1970: LRU and OPT are stack
    algorithms, so their miss counts are non-increasing in capacity. FIFO is
    not, so it may increase."""

    def test_lru_and_opt_misses_never_increase_with_capacity(self) -> None:
        rng = random.Random(31)
        for trial in range(10):
            stream = [rng.randrange(14) for _ in range(500)]
            lru = [faults("lru", stream, k) for k in range(1, 11)]
            optimal = [opt_misses(stream, k) for k in range(1, 11)]
            for k in range(len(lru) - 1):
                self.assertLessEqual(lru[k + 1], lru[k], f"trial {trial}, {k + 1} -> {k + 2}")
                self.assertLessEqual(optimal[k + 1], optimal[k], f"trial {trial}")
            self.assertEqual(optimal[-1], min(optimal))

    def test_fifo_violates_the_stack_property(self) -> None:
        counts = [faults("fifo", BELADY_ANOMALY, k) for k in range(1, 6)]
        self.assertEqual(counts, [12, 12, 9, 10, 5])
        self.assertGreater(counts[3], counts[2])  # 4 frames worse than 3


class TestCyclicScan(unittest.TestCase):
    """A loop over one more block than fits: the LRU worst case."""

    def test_lru_misses_every_access_after_the_first_pass(self) -> None:
        for ways in (3, 4, 8):
            passes = 10
            references = cyclic(ways + 1, passes)
            self.assertEqual(faults("lru", references, ways), len(references))
            # FIFO is just as bad here: with no hits to reorder anything,
            # the two policies make the same choices.
            self.assertEqual(faults("fifo", references, ways), len(references))

    def test_opt_misses_follow_the_measured_closed_form(self) -> None:
        # The tempting closed form -- "N - C = 1 miss per pass after the
        # first" -- is not quite right. Measured, OPT misses once per pass
        # except every C-th pass, which costs two, because the block it
        # sacrifices walks backwards through the loop and wraps:
        #     misses(P) = (C + 1) + (P - 1) + floor((P - 1) / C)
        # (The same count MRU gets, which is why MRU is the cheap stand-in
        # for Belady on cyclic patterns.)
        for ways in (2, 3, 4, 8):
            for passes in (1, 2, 5, 9, 10, 17):
                references = cyclic(ways + 1, passes)
                predicted = (ways + 1) + (passes - 1) + (passes - 1) // ways
                self.assertEqual(
                    opt_misses(references, ways), predicted, f"C={ways} passes={passes}"
                )
                self.assertEqual(
                    faults("opt", references, ways), predicted, f"C={ways} passes={passes}"
                )
                self.assertEqual(
                    faults("mru", references, ways), predicted, f"C={ways} passes={passes}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
