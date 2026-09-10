"""Known-answer tests for set-index functions."""

from __future__ import annotations

import unittest
from collections.abc import Iterable

from cachesim.cache import Cache
from cachesim.config import ConfigError, parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.indexing import INDEX_FUNCTIONS, make_index, modulo_index, xor_index
from cachesim.workloads import conflict_streams, matmul


def run_single_level(accesses: list[tuple[int, bool]], size: int, assoc: int, index: str) -> Cache:
    """Replay a stream of (address, is_write) through one cache level."""
    cache = Cache("L1", size, 64, assoc, index=index)
    for addr, is_write in accesses:
        cache.access(addr, is_write)
    return cache


def as_accesses(pairs: Iterable[tuple[int, str]]) -> list[tuple[int, bool]]:
    """Turn a workload generator's (addr, "R"|"W") pairs into (addr, is_write)."""
    return [(addr, op == "W") for addr, op in pairs]


class TestIndexFunctions(unittest.TestCase):
    def test_modulo_index_is_the_low_order_bits(self) -> None:
        index = modulo_index(128)
        self.assertEqual([index(b) for b in (0, 1, 127, 128, 129, 16384)], [0, 1, 127, 0, 1, 0])

    def test_modulo_index_allows_a_non_power_of_two_set_count(self) -> None:
        index = modulo_index(3)
        self.assertEqual([index(b) for b in range(7)], [0, 1, 2, 0, 1, 2, 0])

    def test_xor_index_folds_the_block_number_into_index_width_chunks(self) -> None:
        # 8 sets: 3-bit chunks. Block 43 = 0b101_011, so the index is
        # 0b011 ^ 0b101 = 0b110 = 6, while modulo indexing gives 3.
        index = xor_index(8)
        self.assertEqual(index(43), 6)
        self.assertEqual(43 % 8, 3)
        # 1024 = 2**10 = 0b010_000_000_000: its only set bit sits in the
        # fourth chunk, which folds to 0b010 = 2 (modulo would give 0).
        self.assertEqual(index(1024), 2)
        self.assertEqual(1024 % 8, 0)
        self.assertEqual(index(0), 0)

    def test_xor_index_separates_blocks_a_power_of_two_apart(self) -> None:
        # The four conflict_streams bases are 16384 blocks apart, which is a
        # multiple of any power-of-two set count: modulo indexing puts them
        # all in one set, XOR-folding does not.
        modulo = modulo_index(128)
        xor = xor_index(128)
        bases = [12582912 + stream * 16384 for stream in range(4)]
        self.assertEqual({modulo(b) for b in bases}, {0})
        self.assertEqual(len({xor(b) for b in bases}), 4)

    def test_xor_index_requires_a_power_of_two_set_count(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            xor_index(3)
        self.assertIn("power-of-two", str(ctx.exception))

    def test_xor_index_on_a_fully_associative_cache_is_always_zero(self) -> None:
        index = xor_index(1)
        self.assertEqual([index(b) for b in (0, 1, 7, 1 << 40)], [0, 0, 0, 0])

    def test_the_two_functions_disagree_on_a_negative_block(self) -> None:
        # A negative block cannot come from an address (Cache.access and
        # Hierarchy.access reject a negative one), but nothing stops a caller
        # from handing probe/allocate/set_of a hand-made block. Neither answer
        # below is meaningful; this pins what the docstrings claim, so the two
        # cannot silently drift apart.
        self.assertEqual(modulo_index(128)(-1), 127)  # Python's % takes the
        self.assertEqual(modulo_index(4)(-5), 3)  # sign of the divisor
        self.assertEqual(xor_index(128)(-1), 0)  # the fold consumes nothing
        self.assertEqual(xor_index(4)(-5), 0)
        with self.assertRaises(ValueError):
            Cache("L1", 4096, 64, 1).access(-1)

    def test_make_index_rejects_an_unknown_name(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            make_index("crc", 128)
        self.assertIn("unknown index function", str(ctx.exception))
        self.assertEqual(sorted(INDEX_FUNCTIONS), ["modulo", "xor"])


class TestCacheIndexing(unittest.TestCase):
    def test_a_cache_defaults_to_modulo_indexing(self) -> None:
        c = Cache("L1", 256, 16, 2)
        self.assertEqual(c.index_name, "modulo")
        self.assertEqual([c.set_of(b) for b in (0, 8, 43)], [0, 0, 3])

    def test_a_cache_rejects_an_unknown_index_function(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Cache("L1", 256, 16, 2, index="crc")
        self.assertIn("L1: unknown index function 'crc'", str(ctx.exception))

    def test_a_cache_rejects_xor_indexing_with_a_non_power_of_two_set_count(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            Cache("L2", 3 * 16, 16, 1, index="xor")  # 3 sets
        self.assertIn("L2: xor indexing requires a power-of-two", str(ctx.exception))

    def test_every_primitive_agrees_on_where_a_block_lives(self) -> None:
        # 8 sets, 2 ways. Block 43 indexes to set 6 under XOR-folding and to
        # set 3 under modulo, so any primitive still using the plain modulo
        # would look in the wrong set and report the block as absent.
        c = Cache("L1", 8 * 2 * 16, 16, 2, index="xor")
        self.assertEqual(c.set_of(43), 6)
        self.assertFalse(c.access(43 * 16, is_write=True))  # miss, then fill
        self.assertEqual([(s, w) for s, w, b, _ in c.lines() if b == 43], [(6, 0)])
        self.assertTrue(c.contains(43))
        self.assertTrue(c.is_dirty(43))
        self.assertTrue(c.clean(43))
        self.assertFalse(c.is_dirty(43))
        self.assertTrue(c.mark_dirty(43))
        removed = c.invalidate(43)
        self.assertEqual(removed, (43, True))
        self.assertFalse(c.contains(43))
        self.assertTrue(c.access(43 * 16) is False)  # gone: misses again

    def test_the_shadow_cache_is_unaffected_by_the_index_function(self) -> None:
        # The 3-C shadow is fully associative, so it sees no sets at all:
        # compulsory and fully-associative-LRU miss counts must match.
        accesses = as_accesses(conflict_streams())
        modulo = run_single_level(accesses, 8 * 1024, 1, "modulo")
        xor = run_single_level(accesses, 8 * 1024, 1, "xor")
        self.assertEqual(modulo.compulsory_misses, xor.compulsory_misses)
        self.assertEqual(modulo.shadow_misses, xor.shadow_misses)


class TestIndexingOnWorkloads(unittest.TestCase):
    """Pinned miss counts on the sample workloads."""

    def test_xor_indexing_rescues_the_conflict_streams_workload(self) -> None:
        # Four streams 1 MB (16,384 blocks) apart, read in lockstep, through
        # a direct-mapped 8 KB cache (128 sets). Under modulo indexing all
        # four map to the same set and every access misses; XOR-folding
        # sends them to four different sets, leaving only the 8,192
        # compulsory misses (one per block, 12.5% of the 65,536 accesses,
        # since eight 8-byte words share a 64-byte block).
        accesses = as_accesses(conflict_streams())
        modulo = run_single_level(accesses, 8 * 1024, 1, "modulo")
        xor = run_single_level(accesses, 8 * 1024, 1, "xor")
        self.assertEqual(modulo.misses, 65536)
        self.assertEqual(modulo.accesses, 65536)
        self.assertEqual(xor.misses, 8192)
        self.assertEqual(xor.misses, xor.compulsory_misses)

    def test_xor_indexing_removes_most_matmul_conflict_misses(self) -> None:
        # Naive 64x64 matmul through 32 KB, 4-way (128 sets). The column
        # walk of B strides 512 B, a power of two, so the column's blocks
        # alias into few sets. XOR-folding cuts the miss count by 80%.
        accesses = as_accesses(matmul(n=64))
        modulo = run_single_level(accesses, 32 * 1024, 4, "modulo")
        xor = run_single_level(accesses, 32 * 1024, 4, "xor")
        self.assertEqual(modulo.misses, 27912)
        self.assertEqual(xor.misses, 5470)
        self.assertEqual(modulo.compulsory_misses, 1536)
        self.assertEqual(xor.compulsory_misses, 1536)


class TestIndexConfig(unittest.TestCase):
    def base_level(self, **overrides: object) -> dict[str, object]:
        level: dict[str, object] = {
            "name": "L1",
            "size": 8192,
            "block_size": 64,
            "associativity": 1,
            "hit_time": 4,
        }
        level.update(overrides)
        return level

    def test_the_index_key_defaults_to_modulo(self) -> None:
        spec = parse_config({"memory_access_time": 100, "levels": [self.base_level()]})
        self.assertEqual(spec.levels[0].index, "modulo")

    def test_the_index_key_reaches_the_cache(self) -> None:
        config = {"memory_access_time": 100, "levels": [self.base_level(index="XOR")]}
        h = Hierarchy.from_config(config)
        self.assertEqual(h.levels[0].cache.index_name, "xor")
        self.assertEqual(h.levels[0].cache.set_of(16384), xor_index(128)(16384))

    def test_an_unknown_index_is_a_path_qualified_config_error(self) -> None:
        config = {"memory_access_time": 100, "levels": [self.base_level(index="crc")]}
        with self.assertRaises(ConfigError) as ctx:
            parse_config(config)
        self.assertIn("levels[0] (L1): unknown index function 'crc'", str(ctx.exception))

    def test_a_non_string_index_is_a_config_error(self) -> None:
        config = {"memory_access_time": 100, "levels": [self.base_level(index=1)]}
        with self.assertRaises(ConfigError) as ctx:
            parse_config(config)
        self.assertIn("'index' must be a string", str(ctx.exception))

    def test_xor_with_a_non_power_of_two_set_count_is_a_config_error(self) -> None:
        level = self.base_level(size=192, index="xor")  # 3 sets of 1 way
        with self.assertRaises(ConfigError) as ctx:
            Hierarchy.from_config({"memory_access_time": 100, "levels": [level]})
        self.assertIn("power-of-two set count", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
