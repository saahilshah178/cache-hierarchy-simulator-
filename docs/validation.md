# Validation

A simulator is only as useful as the confidence that its counters describe the model it
claims to model. `cachesim` establishes that confidence with eight independent
mechanisms: an independently written reference model, differential comparison against
it, property-based laws, a runtime invariant checker, published textbook and hand-traced
policy answers, hand-worked closed forms, pinned golden results, and an exact
cross-check against the stack-distance profiler. A defect would have to escape all
eight, in the same form, for a wrong number to reach a report.

This page states what each mechanism proves, names the test that enforces it, and gives
the command that reproduces the evidence. Every figure below was produced by running the
tool in this working tree.

All commands assume the sample traces exist:

```bash
cachesim gen-traces --out-dir traces
```

## Contents

| Layer | Establishes | Module | Tests | Subtests | Time |
| --- | --- | --- | ---: | ---: | ---: |
| [Reference model](#reference-model) | the oracle is itself correct | `tests/test_reference.py` | 28 | - | 0.08 s |
| [Differential](#differential-testing) | two implementations agree access by access | `tests/test_differential.py` | 10 | 163 | 1.24 s |
| [Property-based](#property-based-tests) | identities hold for arbitrary inputs | `tests/test_properties.py` | 16 | - | 1.58 s |
| [Invariants](#invariants-and---check) | a finished run is self-consistent | `tests/test_invariants.py` | 35 | 6 | 0.57 s |
| [Textbook](#textbook-known-answers) | published examples reproduce | `tests/test_textbook.py` | 8 | - | 0.12 s |
| [Policy traces](#policy-hand-traces) | each policy makes the documented choice | `tests/test_policies.py` | 29 | 20 | 0.17 s |
| [Closed form](#closed-form-derivations) | every reported number is derivable | `tests/test_closed_form.py` | 26 | 14 | 0.36 s |
| [Block size](#the-block-size-closed-form) | the transfer term is exact | `tests/test_timing.py` | 22 | 13 | 2.74 s |
| [Golden](#golden-regressions) | results do not drift | `tests/test_golden.py` | 4 | 18 | 2.59 s |
| [Stack distance](#stack-distance-cross-check) | profiler and simulator agree exactly | `tests/test_stackdist.py` | 39 | 34 | 1.55 s |

```bash
python -m pytest tests/test_reference.py
python -m pytest tests/test_differential.py
python -m pytest tests/test_properties.py
python -m pytest tests/test_invariants.py
python -m pytest tests/test_textbook.py
python -m pytest tests/test_policies.py
python -m pytest tests/test_closed_form.py
python -m pytest tests/test_timing.py
python -m pytest tests/test_golden.py
python -m pytest tests/test_stackdist.py
```

## Reference model

`cachesim/reference.py` is a second implementation of the same semantics, written from
the documented behaviour in `cachesim/cache.py` and `cachesim/hierarchy.py` rather than
from their code. It exists only as a test oracle, and it shares no algorithmic code with
the model it checks: the only import from the package is the `HierarchySpec` parameter
dataclass, which carries numbers, not behaviour.

The two implementations differ in every data structure that could hide a shared mistake:

| Aspect | `cachesim.cache` / `cachesim.hierarchy` | `cachesim.reference` |
| --- | --- | --- |
| Set representation | parallel `[set][way]` lists of block numbers, dirty bits and prefetch bits | one list of `RefLine(block, dirty, last_used, filled_at)` records per set |
| Victim choice | `ReplacementPolicy` per-set state, driven by `on_hit`, `on_fill` and `on_invalidate` hooks | materialise the set's timestamps into a list and sort it |
| LRU / FIFO | a recency order maintained by the policy | smallest `last_used` / smallest `filled_at` |
| Three-C shadow cache | `OrderedDict` of block numbers, oldest first | plain list scanned linearly, least recently used first |
| Cost per reference | O(associativity) | O(num_blocks), dominated by the shadow scan |

The reference is correspondingly slower, which is one reason it is confined to the test
suite. On 50,000 mixed read/write references through a three-level hierarchy
(2 KB / 4-way, 16 KB / 8-way, 64 KB / 16-way) the reference takes 0.757 s against the real
model's 0.263 s, a factor of 2.9, and the gap widens as the shadow cache grows:

```python
import random, time
from cachesim.config import parse_config
from cachesim.hierarchy import Hierarchy
from cachesim.reference import RefHierarchy

spec = parse_config(
    {
        "memory_access_time": 100,
        "levels": [
            {"name": "L1", "size": 2048, "block_size": 64, "associativity": 4, "hit_time": 4},
            {"name": "L2", "size": 16384, "block_size": 64, "associativity": 8, "hit_time": 12},
            {"name": "L3", "size": 65536, "block_size": 64, "associativity": 16, "hit_time": 40},
        ],
    }
)
rng = random.Random(1)
stream = [(rng.randrange(1 << 18) & ~7, rng.random() < 0.3) for _ in range(50_000)]

for name, factory in (("real", Hierarchy.from_spec), ("reference", RefHierarchy.from_spec)):
    h = factory(spec)
    start = time.perf_counter()
    for addr, is_write in stream:
        h.access(addr, is_write)
    print(f"{name:10s} {time.perf_counter() - start:7.3f} s")
```

```text
real         0.263 s
reference    0.757 s
```

The random replacement policy is the one place where agreement requires more than
matching semantics. `RefCache` draws from a `random.Random(rng_seed)` created once per
cache and consumes exactly one value per eviction, which is what
`cachesim.policies.RandomPolicy` does, so identically seeded runs make identical
replacement decisions and can be compared like the deterministic policies.

Because the oracle is only useful if it is independently correct, `tests/test_reference.py`
pins it against hand-worked scenarios rather than against `cachesim.cache`: LRU and FIFO
diverging on the same four-access stream
(`tests/test_reference.py::TestRefReplacement::test_fifo_ignores_the_hit_and_evicts_the_oldest_fill`),
the exact way index chosen by each RNG draw
(`TestRefRandomPolicy::test_victim_way_is_one_randrange_draw_per_eviction`), the three-C
labels of a five-access stream on a two-block direct-mapped cache
(`TestRefThreeC::test_compulsory_conflict_and_capacity_labels`), an anti-conflict hit
driving aggregate conflict misses negative
(`TestRefThreeC::test_anti_conflict_hit_makes_aggregate_conflict_negative`), and the
write-back chain and flush accounting through two and three levels
(`TestRefHierarchy::test_writeback_chain_through_three_levels`,
`TestRefHierarchy::test_flush_writes_every_dirty_line_once`).

## Differential testing

`tests/test_differential.py` drives both implementations with the same seeded streams
through the same `HierarchySpec` and compares two things (W. M. McKeeman, "Differential
Testing for Software", *Digital Technical Journal* 10(1), 1998).

**Access-by-access outcomes.** After every access, the outcome tuple is the cycle count
the access returned and the cumulative per-level miss vector. `DifferentialCase.run_both`
records both sequences and `assert_same_outcomes` fails at the first index at which they
differ, naming the access, its address, whether it was a read or a write, and both
outcome tuples. A divergence is therefore reported at the access that caused it rather
than surfacing as a difference in totals thousands of accesses later.

**Every counter.** `assert_same_counters` compares 20 per-level counters, 6 hierarchy
counters, the derived rates, and the residency of every level:

| Group | Compared |
| --- | --- |
| Per level (`LEVEL_COUNTERS`) | `hits`, `misses`, `read_hits`, `read_misses`, `write_hits`, `write_misses`, `fills`, `evictions`, `invalidations`, `writebacks`, `writebacks_received`, `writeback_allocations`, `compulsory_misses`, `capacity_misses`, `conflict_misses`, `shadow_misses`, `anti_conflict_hits`, `accesses`, `capacity_misses_aggregate`, `conflict_misses_aggregate` |
| Hierarchy (`HIERARCHY_COUNTERS`) | `accesses`, `reads`, `writes`, `dram_reads`, `dram_writes`, `total_time` |
| Derived | `memory_accesses`, `amat()`, `measured_amat()`, per-level `miss_rate`, `global_miss_rate(i)` |
| State | the sorted `(block, dirty)` set of every resident line, at every level |
| After `flush()` | `writebacks`, `writebacks_received`, `writeback_allocations`, `dram_writes`, and that no dirty line remains in either model |

### Geometries covered

Blocks are 64 bytes unless noted, so a level's block count is `size / 64` and its set
count is `size / (64 * associativity)`. The set counts are chosen to include
non-powers-of-two, which exercise the `block % num_sets` mapping that a bit-field
implementation would get wrong.

| Group | Name | Shape | Sets |
| --- | --- | --- | ---: |
| Single level | `direct_mapped_256B` | 256 B, 1-way | 4 |
| | `fully_associative_512B` | 512 B, 8-way | 1 |
| | `three_sets_2way` | 384 B, 2-way | 3 |
| | `five_sets_direct` | 320 B, 1-way | 5 |
| | `seven_sets_4way` | 1792 B, 4-way | 7 |
| | `4KB_4way` | 4 KB, 4-way | 16 |
| | `64KB_8way` | 64 KB, 8-way | 128 |
| | `small_blocks_8B` | 256 B, 2-way, 8 B blocks | 16 |
| Two level | `tiny_inclusive_shapes` | 256 B/1-way over 1 KB/2-way | 4, 8 |
| | `l2_no_bigger_than_l1` | 1 KB/4-way over 1 KB/1-way | 4, 16 |
| | `non_power_of_two_sets` | 384 B/2-way over 1280 B/4-way | 3, 5 |
| | `fully_associative_l1` | 512 B/8-way over 8 KB/8-way | 1, 16 |
| Three level | `desktop_shapes` | 2 KB/4-way, 16 KB/8-way, 64 KB/16-way | 8, 32, 64 |
| | `narrowing_associativity` | 1 KB/8-way, 2 KB/2-way, 4 KB/1-way | 2, 16, 64 |
| | `non_power_of_two_everywhere` | 384 B/2-way, 1152 B/3-way, 5760 B/5-way | 3, 6, 18 |

`l2_no_bigger_than_l1` is load-bearing rather than decorative: an L2 that is no larger
than L1 and less associative drops lines L1 still holds dirty, which is the only way to
reach the write-back allocation path.
`TestTwoLevel::test_writeback_allocation_path` asserts
`l2.writeback_allocations > 0`, so the case cannot pass without exercising it.

### Streams covered

| Name | Shape |
| --- | --- |
| `tight_2KB` | 1500 uniform word-aligned references over 2 KB, 30% stores |
| `medium_64KB` | 1500 uniform references over 64 KB, 30% stores |
| `sparse_4MB` | 1000 uniform references over 4 MB, 30% stores |
| `read_only_tight` | 1200 uniform references over 4 KB, no stores |
| `write_heavy` | 1200 uniform references over 8 KB, 90% stores |
| `cyclic_130_blocks` | 1600 references walking 130 distinct blocks in a fixed cycle |
| `clustered` | 1200 references, 70% into a hot 1 KB region and 30% into a cold 8 MB region |

The footprints run from almost pure reuse (2 KB) to almost pure compulsory misses (4 MB).
The cyclic stream is the classic LRU worst case and keeps the eviction path busy at every
level; the clustered stream lets cold references evict the hot working set, so the hot
references return as capacity and conflict misses rather than compulsory ones.

All three policies (`lru`, `fifo`, `random`) run against every geometry in
`TestSingleLevel::test_every_geometry_and_policy`,
`TestTwoLevel::test_every_geometry_and_policy` and
`TestThreeLevel::test_every_geometry_and_policy`. `TestPrimitives` compares `probe`,
`allocate` and `invalidate` directly over 2000 randomised operations for five
`(ways, sets)` shapes and all three policies, because `Hierarchy` never calls
`invalidate`: without it, the write-back counted when `invalidate` removes a dirty line
would be unreachable from a differential test.
`TestResetAndReuse::test_reset_stats_midstream_matches` covers a warm-up prefix followed
by `reset_stats()`.

Every case also asserts that it exercised what it claims. `test_every_geometry_and_policy`
fails if no configuration caused an eviction;
`test_dirty_evictions_reach_dram_identically` requires `dram_writes > 0`;
`test_writeback_chain_reaches_dram_identically` requires the last level to have written
back; and `test_reset_stats_midstream_matches` requires misses after the reset to exceed
compulsory misses. A geometry that degenerates into "everything hits" fails rather than
passing vacuously.

### Mutation experiments

The commit that introduced the differential suite records the bugs it was checked
against. Seven were injected into the simulator one at a time, and each one makes
`tests/test_differential.py` fail:

1. FIFO refreshing its timestamp on a hit.
2. `allocate` preferring the highest free way instead of the lowest.
3. The shadow cache being updated before the miss is classified.
4. A store dirtying every level rather than L1 only.
5. An uncounted write-back allocation.
6. An uncounted DRAM read on a store.
7. An uncounted write-back when `invalidate` removes a dirty line.

The property-based suite was checked the same way, and the result marks the division of
labour between the two techniques. Reordering the shadow update ahead of the miss
classification, dropping the `writeback_allocations` counter, and failing to count a DRAM
read on a store each make `tests/test_properties.py` fail. Swapping FIFO for LRU does not,
and should not: both satisfy every identity the property suite states. The differential
suite does catch it.

## Property-based tests

Known-answer tests pin one scenario to one number and differential tests compare two
implementations; neither states what is true of *every* input. `tests/test_properties.py`
does, and lets hypothesis search for a counterexample (K. Claessen and J. Hughes,
"QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs", ICFP 2000).

Each test runs `max_examples=100` with deadlines disabled, over generated geometries
(block sizes 8, 16, 32 and 64 B; 1-8 ways; 1-16 sets; all three policies; one to three
levels) and generated streams. Streams are drawn over a narrow address range and then
walked one to four times, because hypothesis on its own favours short lists of distinct
values, which barely touch the replacement path.

| Identity | Formula | Test |
| --- | --- | --- |
| Access algebra | `accesses == len(stream)`, `reads + writes == accesses` | `TestCounterAlgebra::test_hits_and_misses_partition_the_accesses` |
| Hit/miss partition | `hits + misses == accesses` at every level | same |
| Read/write partition | `read_hits + write_hits == hits`, `read_misses + write_misses == misses` | same |
| Miss bound | `0 <= misses <= accesses` | same |
| Fill accounting | `fills == misses + writeback_allocations`, `evictions <= fills` | same |
| L1 sees everything | `levels[0].accesses == hierarchy.accesses` | `TestCounterAlgebra::test_levels_see_exactly_the_misses_of_the_level_above` |
| Level chaining | `levels[i+1].accesses == levels[i].misses` | same |
| DRAM chaining | `dram_reads == levels[-1].misses` | same |
| Cycle accounting | `total_time == sum(hit_time * accesses) + memory_access_time * dram_reads` | `TestCounterAlgebra::test_total_cycles_equal_the_probe_and_memory_charges` |
| AMAT consistency | `amat() == measured_amat()` | same |
| Write-back conservation | `levels[i+1].writebacks_received == levels[i].writebacks` | `TestWriteBackConservation::test_each_level_receives_exactly_what_the_level_above_wrote` |
| DRAM write conservation | `dram_writes == levels[-1].writebacks` | same |
| Allocation bound | `writeback_allocations <= writebacks_received` | same |
| Conservation under flush | both conservation laws hold after `flush()`, and no dirty line remains | `TestWriteBackConservation::test_flush_preserves_the_conservation_law` |
| Per-reference three C | `compulsory + capacity + conflict == misses` | `TestThreeCDecomposition::test_both_taxonomies_sum_to_the_misses` |
| Aggregate three C | `compulsory + capacity_aggregate + conflict_aggregate == misses` | same |
| Taxonomy bridge | `capacity_aggregate == capacity + anti_conflict_hits` | `TestThreeCDecomposition::test_the_taxonomies_differ_by_the_anti_conflict_hits` |
| Taxonomy bridge | `conflict_aggregate == conflict - anti_conflict_hits` | same |
| Shadow decomposition | `shadow_misses == compulsory + capacity + anti_conflict_hits` | same |
| Compulsory floor | `compulsory_misses` equals the number of distinct blocks the stream touches, at *every* level | `TestThreeCDecomposition::test_compulsory_misses_count_the_distinct_blocks_at_every_level` |
| Fully-associative LRU | `conflict == anti_conflict_hits == conflict_aggregate == 0`, `shadow_misses == misses` | `TestThreeCDecomposition::test_a_fully_associative_lru_cache_has_no_conflict_misses` |
| LRU inclusion property | the contents of a `C`-block cache are a subset of a `C+k`-block cache's after every access | `TestLRUStackProperty::test_contents_are_nested_at_every_access` |
| LRU stack property | `misses(C)` is non-increasing in `C`, and `misses(C) >= distinct blocks` | `TestLRUStackProperty::test_misses_are_non_increasing_in_capacity` |
| Determinism | same spec plus same stream gives the same counters, random replacement included | `TestDeterminismAndReset::test_the_same_spec_and_stream_give_the_same_counters` |
| Reset semantics | `reset_stats()` zeroes every counter and changes no line | `TestDeterminismAndReset::test_reset_stats_zeroes_counters_and_keeps_contents` |
| Flush idempotence | a second `flush()` returns 0, L1 residency is unchanged, no level shrinks | `TestDeterminismAndReset::test_flush_cleans_every_line_and_is_idempotent` |
| One copy per block | a block occupies at most one way, and every valid line is found by `contains()` | `TestPrimitiveInvariants::test_a_block_has_at_most_one_copy_and_is_always_findable` |
| Write-back bound | `writebacks <= evictions + invalidations`, `writebacks + dirty_resident <= stores` | `TestPrimitiveInvariants::test_writebacks_never_exceed_dirty_evictions` |

The LRU stack property is R. L. Mattson, J. Gecsei, D. R. Slutz and I. L. Traiger,
"Evaluation techniques for storage hierarchies", *IBM Systems Journal* 9(2), 1970; the
three-C aggregate taxonomy is M. D. Hill and A. J. Smith, "Evaluating Associativity in
CPU Caches", *IEEE Transactions on Computers* 38(12), 1989.

The write-back conservation cases use a deliberately shrinking hierarchy (1 KB/4-way over
1 KB/1-way over 2 KB/2-way) and a write-heavy stream of at least 40 references: a lower
level only has to allocate on a write-back when it has already dropped a line the level
above still holds dirty, and nothing else reaches that code.

hypothesis is a development dependency (`pip install 'cachesim[dev]'`). Without it the
module raises `unittest.SkipTest` with a message naming the missing extra, so the suite
still runs on a bare install.

## Invariants and `--check`

`cachesim/invariants.py` states the identities a *finished* simulation must satisfy.
`check_invariants(hierarchy)` returns the list of violations, empty when the model is
consistent; `check_hierarchy(hierarchy)` returns the same list plus the number of checks
made and the reason anything was skipped. A violation means a counter and the state it
describes have drifted apart, which is a defect in `cachesim`, not in the trace.

### The identities

**Counter algebra, always checked.** Per hierarchy: `reads + writes == accesses` and
`dram_reads <= accesses`. Per level: `hits + misses == accesses`,
`read_hits + write_hits == hits`, `read_misses + write_misses == misses`,
`misses <= accesses`, `evictions <= fills`, and
`writeback_allocations <= writebacks_received`. `writebacks` is deliberately not bounded
by `evictions + invalidations`, because `flush()` writes a resident line out without
removing it.

**Three-C decomposition, checked when the level sets `track_3c`.**

```text
compulsory + capacity + conflict                     == misses
compulsory + capacity_aggregate + conflict_aggregate == misses
capacity_aggregate                                   == capacity + anti_conflict_hits
conflict_aggregate                                   == conflict - anti_conflict_hits
shadow_misses                                        == compulsory + capacity
                                                        + anti_conflict_hits
```

The two taxonomies differ by exactly the anti-conflict hits: references that hit the real
cache but would have missed its fully-associative LRU twin.

**Traffic flow, skipped when extra traffic is modelled.**

```text
levels[0].accesses   == hierarchy.accesses
levels[i+1].accesses == levels[i].misses
dram_reads           == levels[-1].misses
fills                == misses + writeback_allocations     (at every level)
```

All four assume demand fetching only, into non-inclusive non-exclusive, write-back,
write-allocate levels with no prefetcher and no victim cache. A prefetcher fills without a
miss; an exclusive level is filled from the eviction above it rather than from its own
miss; a no-write-allocate level declines write misses. Each such feature makes the check
step aside with a reason instead of firing spuriously.

**Write-back conservation, skipped when modified data takes another route.**

```text
levels[i].writebacks_received == levels[i-1].writebacks
dram_writes                   == levels[-1].writebacks
```

Dirty data is never duplicated and never dropped. Write-backs caused by `flush()` are
included, because `flush` increments both sides. Calling `Cache.invalidate` from outside
the hierarchy breaks the identity, since the removed dirty line is counted as written back
with nothing below to receive it; the violation message names the invalidation count when
that is what happened
(`tests/test_invariants.py::TestCorruptedCounters::test_an_invalidation_outside_the_hierarchy_is_explained`).

**Timing, skipped when the latency model is not constant.**

```text
total_cycles == sum(hit_time * accesses) + memory_access_time * dram_reads
amat()       == measured_amat()
```

This holds exactly for the constant-latency model with no transfer term. A per-level
transfer time (block size over bus width) or a non-constant DRAM model adds terms this sum
does not have.

**Structure, always checked.** A block has at most one copy in a cache; a level holds no
more lines than it has blocks; every line sits in the set its block maps to; every valid
line is found by `contains()`; `is_dirty()` agrees with the line's dirty bit. Lines parked
in a victim buffer are reported with set index `-1` and are exempt from the set-placement
check.

**Dirty lines after a flush, only with `after_flush=True`.** No level holds a dirty line.
This cannot be checked unconditionally: a hierarchy that has merely run a trace is
expected to hold dirty lines, and `reset_stats` zeroes the counters that would otherwise
explain them.

### When a check is skipped

The checker probes optional attributes with `getattr` rather than assuming they are
absent, so a later feature relaxes the affected identity instead of producing a false
alarm. Two guard functions decide:

| Guard | Condition | Skips |
| --- | --- | --- |
| `_traffic_features` | `level.inclusion != "nine"` | write-back conservation, traffic flow, timing |
| | `level.write_policy != "write-back"` | same |
| | `level.write_allocate is not True` | same |
| | `level.prefetcher is not None` | same |
| | `level.cache.victim is not None` | same |
| `_transfer_features` | `hierarchy.memory_access_time is None` | timing |
| | `level.bus_width is not None` | timing |
| (in `check_hierarchy`) | `hierarchy.accesses == 0` | timing |

`tests/test_invariants.py::TestOptionalFeatureGuards` pins each guard to the identities it
relaxes and no others: a bus width silences only the timing identities
(`test_a_bus_width_on_a_level_skips_the_timing_identities`), a non-constant memory model
does the same (`test_a_non_constant_memory_model_skips_the_timing_identities`), a
prefetcher silences all three (`test_a_prefetcher_skips_the_traffic_flow_identities`), and
setting the optional attributes to their default values relaxes nothing
(`test_default_feature_values_do_not_relax_anything`).

### `cachesim run --check`

The `run` subcommand sweeps the checker after the simulation, prints its diagnostics to
stderr so that `--format json` stays pipeable, and exits 1 listing any violation. On the
default three-level hierarchy the sweep makes 62 checks:

```bash
cachesim run --check traces/sequential.trace > /dev/null
```

```text
invariants: 62 checks passed
```

A configuration that turns a feature on reports what stepped aside and why, so a silent
pass cannot be mistaken for a pass that quietly checked nothing. The stride-prefetch
configuration disables all three feature-dependent groups and drops the sweep to 50
checks:

```bash
cachesim run --check --config configs/prefetch_stride.json traces/sequential.trace > /dev/null
```

```text
invariants: skipped write-back conservation (L1: prefetcher is 'next-line'; L2: prefetcher is 'stride')
invariants: skipped traffic-flow identities (L1: prefetcher is 'next-line'; L2: prefetcher is 'stride')
invariants: skipped timing identities (L1: prefetcher is 'next-line'; L2: prefetcher is 'stride')
invariants: 50 checks passed
```

The row-buffer DRAM configuration touches only the latency model and the L3 bus, so the
traffic identities still apply and 60 checks are made:

```bash
cachesim run --check --config configs/dram_row_buffer.json traces/sequential.trace > /dev/null
```

```text
invariants: skipped timing identities (memory model is not a constant latency; L3: bus_width is set)
invariants: 60 checks passed
```

The clean exit, the JSON output staying parseable with diagnostics on stderr, the absence
of any invariant output without the flag, and the non-zero exit with the violations listed
are covered by `tests/test_invariants.py::TestRunCheckFlag`
(`test_check_passes_on_a_normal_run`, `test_check_keeps_json_output_parseable`,
`test_without_the_flag_nothing_is_checked`,
`test_a_violation_exits_non_zero_and_lists_it`).

### Checking that the checker checks

Two things have to be true of a checker: it stays quiet on a correct run, and it speaks up
when something is wrong. The first is covered by running real workloads through several
configurations (`TestCleanRuns`). For the second,
`tests/test_invariants.py::TestCorruptedCounters` and `TestCorruptedState` corrupt one
counter, or one line of cache state, at a time and assert both that a violation is
reported and that its message names the identity that broke.

| Corruption | Identity it must break | Test |
| --- | --- | --- |
| A stray hit at L1 | `read_hits + write_hits != hits` | `TestCorruptedCounters::test_a_stray_hit_breaks_the_counter_algebra` |
| A stray miss | the level chain | `test_a_stray_miss_breaks_the_level_chain` |
| A lost compulsory miss | the three-C sum | `test_a_lost_compulsory_miss_breaks_the_three_c_sum` |
| A stray anti-conflict hit | the taxonomy bridge | `test_a_stray_anti_conflict_hit_breaks_the_taxonomy_bridge` |
| A lost shadow miss | the shadow decomposition | `test_a_lost_shadow_miss_is_detected` |
| A write-back nobody received | write-back conservation | `test_a_writeback_that_nobody_received` |
| A DRAM write no level produced | DRAM write conservation | `test_a_dram_write_that_no_level_produced` |
| A wrong cycle count | the timing identity | `test_a_wrong_cycle_count_is_detected` |
| An uncounted fill | `fills != misses + writeback_allocations` | `test_an_uncounted_fill_is_detected` |
| A DRAM read no level missed | `dram_reads != levels[-1].misses` | `test_a_dram_read_that_no_level_missed` |
| Reads and writes not partitioning accesses | the hierarchy algebra | `test_reads_and_writes_must_partition_the_accesses` |
| A block duplicated into a second way | one copy per block | `TestCorruptedState::test_a_block_held_in_two_ways_is_detected` |
| A line moved to the wrong set | set placement | `test_a_line_in_the_wrong_set_is_detected` |
| A dirty line after a flush | the `after_flush` identity | `test_a_dirty_line_after_a_flush_is_detected` |
| An empty way reported as valid | `contains()` agreement | `test_an_empty_way_reported_as_valid_is_detected` |

`TestCleanRuns::test_the_sweep_actually_checks_something` additionally requires that a
clean run makes more than 40 checks and skips nothing, so the suite cannot pass by
checking nothing at all.

## Textbook known answers

Every case in `tests/test_textbook.py` is a published example with a published result, so
the numbers check the simulator rather than the test author. Page replacement and cache
replacement are the same problem at different granularities, so the page-replacement
examples run on a fully associative cache of *C* blocks: one set, *C* ways, one block per
page.

The reference strings are

```text
Silberschatz:    7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1
Belady anomaly:  1 2 3 4 1 2 5 1 2 3 4 5
```

| Scenario | Expected | Source | Test |
| --- | --- | --- | --- |
| Silberschatz string, 3 frames, FIFO | 15 faults | Silberschatz, Galvin and Gagne, "Operating System Concepts", 10th ed., Wiley, 2018, sec. 10.4.2 | `TestSilberschatzReferenceString::test_fifo_lru_and_optimal_page_fault_counts` |
| Silberschatz string, 3 frames, LRU | 12 faults | ibid., sec. 10.4.4 | same |
| Silberschatz string, 3 frames, OPT | 9 faults | ibid., sec. 10.4.3 | same |
| Silberschatz string, 3 frames, `opt_misses` | 9 faults | offline bound, same section | same |
| Silberschatz string, 2 frames | FIFO 15, LRU 17, OPT 13 | measured; LRU is not uniformly better than FIFO | `TestSilberschatzReferenceString::test_lru_is_not_always_better_than_fifo` |
| Belady string, FIFO, 3 then 4 frames | 9 then 10 faults | Belady, Nelson and Shedler, "An anomaly in space-time characteristics of certain programs running in a paging machine", *CACM* 12(6), 1969 | `TestBeladysAnomaly::test_fifo_faults_more_with_four_frames_than_with_three` |
| Belady string, LRU, 3 then 4 frames | 10 then 8 faults | stack algorithm, no anomaly | `TestBeladysAnomaly::test_lru_and_opt_do_not_show_the_anomaly_on_that_string` |
| Belady string, OPT, 3 then 4 frames | 7 then 6 faults | stack algorithm, no anomaly | same |
| Belady string, FIFO, 1 to 5 frames | 12, 12, 9, 10, 5 | FIFO is not a stack algorithm | `TestStackProperty::test_fifo_violates_the_stack_property` |
| LRU and OPT over capacities 1..10, 10 random streams of 500 references | miss counts non-increasing in capacity | Mattson, Gecsei, Slutz and Traiger, "Evaluation techniques for storage hierarchies", *IBM Systems Journal* 9(2), 1970 | `TestStackProperty::test_lru_and_opt_misses_never_increase_with_capacity` |
| Cyclic scan of C+1 distinct blocks over a C-block fully-associative cache, 10 passes, C = 3, 4, 8 | LRU and FIFO miss every reference | the standard LRU worst case | `TestCyclicScan::test_lru_misses_every_access_after_the_first_pass` |
| The same scan, OPT and MRU, C = 2, 3, 4, 8 and P = 1, 2, 5, 9, 10, 17 passes | `misses(P) = (C+1) + (P-1) + floor((P-1)/C)` | measured closed form; OPT is Belady, "A study of replacement algorithms for a virtual-storage computer", *IBM Systems Journal* 5(2), 1966 | `TestCyclicScan::test_opt_misses_follow_the_measured_closed_form` |

The OPT closed form is worth stating precisely because the tempting version is wrong.
"N - C = one miss per pass after the first" does not hold: OPT misses once per pass except
every *C*-th pass, which costs two, because the block it sacrifices walks backwards through
the loop and wraps. MRU gets the same count, which is why MRU is the cheap stand-in for
Belady's MIN on cyclic patterns.

```bash
python -m pytest tests/test_textbook.py
```

```text
8 passed in 0.12s
```

## Policy hand traces

`tests/test_policies.py` pins each replacement policy to the choice its definition
requires, on sequences short enough to trace by hand. Unless noted, the cache is a single
set, so every block competes with every other.

| Scenario | Expected | Source | Test |
| --- | --- | --- | --- |
| LRU, 2 ways, `a b a x` | `b` evicted, `a` survives | recency order | `TestReplacementPolicies::test_lru_keeps_the_recently_hit_block` |
| FIFO, 2 ways, `a b a x` | `a` evicted despite the hit | insertion order | `TestReplacementPolicies::test_fifo_evicts_the_oldest_fill_even_if_hot` |
| tree-PLRU, 4 ways, `a b c d a e` | PLRU evicts `c`, LRU evicts `b`; both 1 hit / 5 misses | Hennessy and Patterson, "Computer Architecture: A Quantitative Approach", 6th ed., Morgan Kaufmann, 2017, sec. B.1 and 2.3 | `TestTreePLRU::test_plru_and_lru_disagree_on_the_classic_four_way_sequence` |
| tree-PLRU, cold 4-way set filled in order | first victim is way 0 | all bits clear means "victim to the left" | `TestTreePLRU::test_plru_fills_ways_in_tree_order_from_the_reset_state` |
| tree-PLRU, 2 ways, 400 random references | identical misses and residency to LRU | one bit records the full order of two ways | `TestTreePLRU::test_plru_is_exact_lru_at_two_ways` |
| tree-PLRU, associativity 3 | `ValueError` naming "power-of-two" | the tree must be complete | `TestTreePLRU::test_plru_rejects_a_non_power_of_two_associativity` |
| NRU, 2 ways, `a b a c` | `a` evicted, residency `{b, c}`, 1 hit / 3 misses | Silberschatz et al., 2018, sec. 10.4.5 | `TestNRU::test_nru_evicts_the_recently_used_block_when_every_bit_is_set` |
| NRU, 4 ways, five fills then a hit in way 1 | victim moves from way 1 to way 2 | lowest way with a clear bit | `TestNRU::test_nru_prefers_a_way_whose_bit_is_clear` |
| LFU, 2 ways, `a a a b c` | `b` evicted, residency `{a, c}`, 2 hits / 3 misses | Silberschatz et al., 2018, sec. 10.4.6 | `TestLFU::test_lfu_evicts_the_least_referenced_block` |
| LFU, 4 ways, every count 1 | victim is way 0 | ties break on the lowest way | `TestLFU::test_lfu_breaks_count_ties_with_the_lowest_way` |
| LFU, one block touched 5 times then 18 one-touch blocks | the hot block stays resident | counters do not age | `TestLFU::test_lfu_counters_do_not_age` |
| MRU, cyclic 5 blocks over 4 ways, 10 passes | LRU 50 misses, MRU 16 (`5, 1, 1, 1, 2, 1, 1, 1, 2, 1`) | the cyclic-scan counter-example | `TestMRU::test_mru_beats_lru_on_a_cyclic_scan_one_block_too_large` |
| MRU, C = 2, 3, 4, 8 and P = 1, 2, 5, 9, 10, 17 | `misses(P) = (C+1) + (P-1) + floor((P-1)/C)` | measured closed form | `TestMRU::test_mru_cyclic_scan_matches_its_closed_form` |
| SRRIP against a scan: 2 hot blocks touched twice, then 3 fresh blocks, 10 rounds, 4 ways | LRU 5 misses every round (50 total); SRRIP 5 then 3 (32 total), hot pair resident | Jaleel, Theobald, Steely and Emer, "High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP)", ISCA 2010 | `TestRRIP::test_srrip_survives_a_scan_that_flushes_lru` |
| BRRIP, cyclic 5 blocks over 4 ways, 10 passes | LRU 50, SRRIP 50, BRRIP 23 (`5` then `2` per pass) | ibid., bimodal insertion at RRPV 3 | `TestRRIP::test_brrip_keeps_a_thrashing_working_set_resident` |
| SRRIP, every way at RRPV 0 | victim is way 0 after three ageing rounds | ibid., ageing until a way reaches RRPV 3 | `TestRRIP::test_rrip_ages_a_set_until_a_victim_appears` |
| DRRIP set dueling on a thrashing workload | PSEL moves towards BRRIP | Qureshi, Jaleel, Patt, Steely and Emer, "Adaptive Insertion Policies for High Performance Caching", ISCA 2007 | `TestDRRIP::test_drrip_learns_to_use_brrip_on_a_thrashing_workload` |
| DRRIP when the working set fits | PSEL unchanged | ibid. | `TestDRRIP::test_drrip_leaves_psel_alone_when_the_working_set_fits` |

`TestReplacementPolicies::test_random_is_reproducible` pins the seeded RNG, and
`TestEveryPolicy` runs every registered policy at every level of a hierarchy and checks
that an invalidated way is refilled before any eviction.

## Closed-form derivations

`tests/test_closed_form.py` works each reported number out from the geometry and the
access pattern first, then asserts that the simulator produces exactly that. The
derivations live in the test code and its docstrings, so a number that changes has to be
argued with rather than re-recorded. The workload generators are called directly; no trace
files are involved.

The hierarchy throughout is the default: 32 KB / 4-way L1 at 4 cycles, 256 KB / 8-way L2
at 12 cycles, 2 MB / 16-way L3 at 40 cycles, 64-byte blocks, 100-cycle DRAM. An access
that reaches a level pays every level it probed on the way down, so the four possible
costs are 4, 16, 56 and 156 cycles.

### `sequential.trace`

The workload scans a 256 KB buffer twice, one 8-byte word at a time, with every tenth
access a store. The buffer starts at `0x00100000`, which is block 16,384.

Derived quantities: 32,768 words per pass, 65,536 accesses, 4,096 buffer blocks, 8 words
per block, 512 L1 blocks, 4,096 L2 blocks, 32,768 L3 blocks.

| Quantity | Derivation | Value | Test |
| --- | --- | --- | ---: |
| Accesses | `2 x 262144/8` | 65,536 | `TestSequentialClosedForm::test_the_trace_is_the_size_the_derivation_assumes` |
| Writes | indices 9, 19, ..., 65,529 | 6,553 | `test_read_and_write_counts` |
| Reads | `65536 - 6553` | 58,983 | same |
| L1 misses | buffer 4,096 blocks > L1 512 blocks, so both passes miss every block: `2 x 4096` | 8,192 | `test_l1_misses_once_per_block_per_pass` |
| L1 hits | `65536 - 8192` | 57,344 | same |
| L1 local miss rate | `8192 / 65536` | 12.50% | same |
| L1 write misses | a miss falls on an index that is a multiple of 8 (even); a store on an index congruent to 9 mod 10 (odd) | 0 | `test_no_store_ever_misses_l1` |
| L1 write hits | every store hits | 6,553 | same |
| L1 read hits | `57344 - 6553` | 50,791 | same |
| L1 fills | one per miss | 8,192 | `test_l1_fills_and_evictions` |
| L1 evictions | `8192 - 512` (the first 512 fills find empty ways) | 7,680 | same |
| Blocks dirtied, pass 1 | stores repeat with period `lcm(8, 10) = 40` words = 5 blocks, four stores per window landing in four distinct blocks | 3,276 | `test_l1_writebacks` |
| Blocks dirtied, pass 2 | pass 2 starts at index 32,768 = 8 mod 40, shifting the window one block and catching the buffer's last block | 3,277 | same |
| L1 lines still dirty | pass-2 dirty blocks among the last 512 blocks of the buffer | 409 | same |
| L1 write-backs | `6553 - 409` | 6,144 | same |
| L2 accesses | L1's misses | 8,192 | `test_l2_holds_the_whole_buffer_exactly` |
| L2 misses / hits | L2 holds 4,096 blocks = the buffer exactly, base block 16,384 is a multiple of 512 sets, so 4,096 consecutive blocks put exactly 8 in each set: pass 1 misses all, pass 2 hits all | 4,096 / 4,096 | same |
| L2 evictions | the fit is exact | 0 | same |
| L2 write-backs received | every L1 write-back | 6,144 | `test_l2_receives_every_writeback_without_allocating` |
| L2 write-back allocations | the copy is always still resident | 0 | same |
| L3 accesses / misses / hits | L2's 4,096 compulsory misses, into 32,768 blocks | 4,096 / 4,096 / 0 | `test_l3_sees_each_block_once_and_never_hits` |
| DRAM reads / writes | one fetch per distinct block; nothing dirty leaves L3 | 4,096 / 0 | `test_dram_traffic` |
| L1 three C | pass 1 compulsory, pass 2 capacity (a fully-associative 512-block LRU twin would have evicted them too) | 4,096 / 4,096 / 0 | `test_three_c_classification` |
| L2 and L3 three C | every miss is a first touch | 4,096 / 0 / 0 | same |
| Total cycles | `57344 x 4 + 4096 x 16 + 4096 x 156 = 229376 + 65536 + 638976` | 933,888 | `test_total_cycles_and_amat` |
| AMAT | `933888 / 65536`, analytic and measured | 14.250 | same |
| Flush to DRAM | L1's 409 remaining dirty lines push down, leaving all 4,096 buffer blocks dirty in L2 | 4,096 | `test_flush_writes_the_whole_buffer_back` |
| L1 write-backs after flush | `6144 + 409` | 6,553 | same |

`test_the_report_prints_the_derived_numbers` asserts that the formatted report contains
those figures verbatim, and `test_the_run_satisfies_every_invariant` asserts
`check_invariants(h) == []`.

```bash
cachesim run --check traces/sequential.trace
```

Selected lines of the report that command prints (the invariant diagnostic goes to
stderr, the report to stdout):

```text
invariants: 62 checks passed
Total accesses :       65,536   (reads 58,983 / writes 6,553)
  hits        :       57,344   (reads 50,791 / writes 6,553)
  misses      :        8,192   (reads 8,192 / writes 0)
  local miss rate  :  12.50%   (misses / accesses that reached this level)
  evictions   :        7,680   (writebacks of dirty blocks: 6,144)
  writebacks received :  6,144   (from L1; 0 allocated a line)
AMAT (analytic formula) :   14.250 cycles
AMAT (measured)         :   14.250 cycles   (933,888 cycles / 65,536 accesses)
```

### `conflict.trace`

Four arrays 1 MB apart are read in lockstep, one 8-byte word from each per step, 16,384
words per array. 1 MB is 16,384 blocks, which is a multiple of every level's set count
(128, 512 and 2,048), so at every step the four streams' blocks land in one set at every
level.

| Quantity | Derivation | Value | Test |
| --- | --- | --- | ---: |
| Accesses | `4 x 16384` | 65,536 | `TestConflictClosedForm::test_l1_misses_are_one_in_eight_per_stream` |
| Reads / writes | read-only workload | 65,536 / 0 | same |
| Distinct blocks | `4 x 16384/8` | 8,192 | same |
| Stream alignment | `16384 % num_sets == 0` for 128, 512 and 2,048 | collides at every level | `test_the_streams_collide_in_every_level` |
| L1 misses | L1 is 4-way and exactly four blocks are hot, so the set holds all four: each block misses once and serves its other seven words | 8,192 | `test_l1_misses_are_one_in_eight_per_stream` |
| L1 hits | `65536 - 8192` | 57,344 | same |
| L1 local miss rate | `8192 / 65536` | 12.50% | same |
| L1 evictions | `8192 - 512` | 7,680 | same |
| L2 and L3 accesses / misses / hits | no block is ever revisited, so nothing below L1 hits | 8,192 / 8,192 / 0 | `test_lower_levels_never_hit` |
| L2 evictions | `8192 - 4096` | 4,096 | same |
| L3 evictions | L3 holds 32,768 blocks | 0 | same |
| Three C at every level | every miss is a first touch | 8,192 / 0 / 0 | `test_every_miss_is_compulsory_at_every_level` |
| DRAM reads / writes | one fetch per distinct block, nothing dirty | 8,192 / 0 | `test_dram_traffic_and_cycles` |
| Total cycles | `57344 x 4 + 8192 x 156 = 229376 + 1277952` | 1,507,328 | same |
| AMAT | `1507328 / 65536`, analytic and measured | 23.000 | same |

Despite the name, this workload produces no conflict misses on the default hierarchy: the
four hot blocks fit in a 4-way set. The conflict behaviour appears when associativity is
reduced, which is what the associativity sweep in [results.md](results.md) measures.

```bash
cachesim run --check traces/conflict.trace
```

Selected lines of the report, the per-level ones from the L1 section:

```text
invariants: 62 checks passed
Total accesses :       65,536   (reads 65,536 / writes 0)
  hits        :       57,344   (reads 57,344 / writes 0)
  misses      :        8,192   (reads 8,192 / writes 0)
  local miss rate  :  12.50%   (misses / accesses that reached this level)
  evictions   :        7,680   (writebacks of dirty blocks: 0)
AMAT (analytic formula) :   23.000 cycles
AMAT (measured)         :   23.000 cycles   (1,507,328 cycles / 65,536 accesses)
```

### Cyclic working sets

`TestCyclicWorkingSet` pins the standard demonstration that LRU is not optimal. A cycle of
`capacity + 1` blocks over a fully-associative LRU cache of `capacity` blocks evicts
precisely the block needed next, so nothing is ever reused: for capacities 1, 2, 4, 8, 16
and 64 over 20 laps the miss rate is exactly 1.0, with `capacity + 1` compulsory misses
and the rest capacity misses
(`test_capacity_plus_one_blocks_never_hit`). Cycling `capacity` blocks instead misses once
per block (`test_a_cycle_that_fits_misses_only_once_per_block`). FIFO thrashes identically,
because in a pure cycle there are no hits to reorder anything
(`test_fifo_thrashes_identically`). Random replacement does not: at capacity 8 over 200
laps of 9 blocks it takes 416 misses out of 1,800 references, a 23.1% miss rate against
LRU's 100%, and the exact figure also pins that the policy draws once per eviction and
never anywhere else (`test_random_replacement_beats_lru_on_this_pattern`). The same
thrashing needs only `associativity + 1` blocks mapping to one set, leaving the rest of
the cache empty (`test_set_associative_thrashing_needs_only_one_hot_set`).

### The block-size closed form

A wider block lowers the miss rate but takes longer to move. With `bus_width` set, a fill
costs `ceil(block_size / bus_width)` cycles on top of the latency, charged to the access
that caused it, so AMAT against block size need not fall monotonically.

For a linear scan the trade-off has a closed form. `sequential` walks 8-byte words, so a
*B*-byte block absorbs *B*/8 of them and the miss rate is exactly

```text
miss rate = 8 / B
```

at every block size. Through a single 32 KB 4-way level at hit time 4, DRAM 100 and
`bus_width` 16, AMAT is therefore

```text
AMAT(B) = 4 + (8/B) x (100 + B/16) = 4 + 800/B + 0.5
```

which falls monotonically for *any* bus width: a linear scan uses every byte it fetches,
so it has no interior minimum and can never show the block-size U the transfer term is
supposed to produce. The trade-off needs imperfect spatial locality, which is why the
table below carries a second workload. `matmul_naive` strides down the columns of its
second operand: its miss rate is still falling from 256 B to 512 B, but AMAT turns at
256 B because the 32-cycle transfer of a 512 B block costs more than the misses it saves.

| Block | `sequential` miss rate | `sequential` AMAT | `matmul_naive` miss rate | `matmul_naive` AMAT | Transfer |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 50.0000% | 54.500 | 7.1575% | 11.229 | 1 |
| 32 | 25.0000% | 29.500 | 5.8804% | 9.998 | 2 |
| 64 | 12.5000% | 17.000 | 5.2419% | 9.452 | 4 |
| 128 | 6.2500% | 10.750 | 4.9226% | 9.316 | 8 |
| 256 | 3.1250% | 7.625 | 4.0587% | 8.708 | 16 |
| 512 | 1.5625% | 6.062 | 3.9288% | 9.186 | 32 |

Each row is one run of each trace through a single-level configuration. The file below is
the 256 B row; the other rows change `block_size` and nothing else:

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 256, "associativity": 4,
     "policy": "lru", "hit_time": 4, "bus_width": 16}
  ]
}
```

```bash
cachesim run --config bus16.json --format json traces/sequential.trace
cachesim run --config bus16.json --format json traces/matmul_naive.trace
```

The miss rate column is `levels[0].local_miss_rate`, the AMAT column `measured_amat`, and
the transfer column `levels[0].transfer_cycles`. The `sequential` column is asserted
against both closed forms to ten decimal places at all six sizes by
`tests/test_timing.py::TestSampleTraceBlockSizeTable::test_the_sequential_column_is_the_closed_form_and_never_turns`;
the `matmul_naive` rows around the minimum are pinned by
`test_the_matmul_naive_column_bottoms_out_at_256_bytes`. Both assert
`amat() == measured_amat()` at every point, so the analytic formula and the simulated
cycles agree even with a transfer term.

The one documented gap between the two is also pinned. An exclusive level is filled by the
level above rather than from below, and a no-write-allocate level declines write misses
outright, so the analytic formula charges a transfer term the simulation never spends and
becomes an upper bound. `TestAmatApproximationWhenALevelDoesNotFill` establishes that a
NINE hierarchy on the same stream and bus matches to nine decimal places
(`test_a_nine_hierarchy_stays_exact_on_the_same_stream`), that without a bus width both
policies also match exactly (`test_the_gap_appears_only_with_a_bus_width`), that with a
bus the analytic figure sits strictly above the measured one and never below
(`test_analytic_amat_is_an_upper_bound_with_a_bus_width`), and that the whole gap is the
transfer term, since a bus width changes what an access is charged and never which lines
are resident (`test_the_whole_gap_sits_in_the_transfer_term`).

## Golden regressions

`tests/test_golden.py` pins the exact result of every sample workload on the default
hierarchy, covering the whole path end to end: generator, native trace file, parser, and
`Hierarchy.from_config(DEFAULT_CONFIG)`, which is what `cachesim run traces/<name>.trace`
does. A change in a generator, the trace format, the parser or the simulator shows up as a
failing test rather than as different documentation.

| Workload | Accesses | Reads | Writes | L1 hits / misses | L2 hits / misses | L3 hits / misses | DRAM reads | DRAM writes | Cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential` | 65,536 | 58,983 | 6,553 | 57,344 / 8,192 | 4,096 / 4,096 | 0 / 4,096 | 4,096 | 0 | 933,888 |
| `random` | 60,000 | 44,832 | 15,168 | 107 / 59,893 | 786 / 59,107 | 4,368 / 54,739 | 54,739 | 5,823 | 8,796,896 |
| `matmul_naive` | 532,480 | 528,384 | 4,096 | 504,568 / 27,912 | 26,376 / 1,536 | 0 / 1,536 | 1,536 | 0 | 2,679,904 |
| `matmul_blocked` | 557,056 | 540,672 | 16,384 | 553,484 / 3,572 | 2,036 / 1,536 | 0 / 1,536 | 1,536 | 0 | 2,486,128 |
| `conflict` | 65,536 | 65,536 | 0 | 57,344 / 8,192 | 0 / 8,192 | 0 / 8,192 | 8,192 | 0 | 1,507,328 |
| `pointer_chase` | 60,000 | 60,000 | 0 | 0 / 60,000 | 0 / 60,000 | 43,616 / 16,384 | 16,384 | 0 | 4,998,400 |

```bash
cachesim run --format json traces/sequential.trace
cachesim run --format json traces/random.trace
cachesim run --format json traces/matmul_naive.trace
cachesim run --format json traces/matmul_blocked.trace
cachesim run --format json traces/conflict.trace
cachesim run --format json traces/pointer_chase.trace
```

The trace files themselves are pinned byte for byte, so a change in the generator or the
writer is caught before the simulation results are even compared:

| Trace | SHA-256 of the native trace file |
| --- | --- |
| `sequential.trace` | `14c2703c518caa071d5d208bf395d6b6785c1d4e1e990637041ffc905e591a8d` |
| `random.trace` | `a698bb268b5c5957823b10c80bd74a0958585b74f32dc3d8cb6c8e227cfc53f4` |
| `matmul_naive.trace` | `2b0474dc8031159bec82ba24635d3b67d7a3606dfd812b2f1f5bdb1ccd7fac70` |
| `matmul_blocked.trace` | `c52a61b56520ecaa4c8885a36a6143bd5519ae19319ca5b5631b836808897b97` |
| `conflict.trace` | `5f7948a5f11710e10f83533de06465f52d4744a074020945b8b7424dd7e043ea` |
| `pointer_chase.trace` | `6c80ba174397ee9c859f0fb4fc0a054e512c2b975686364aa20a94db4aa7c179` |

```bash
shasum -a 256 traces/*.trace
```

Four tests enforce the pin. `TestGolden::test_every_sample_is_pinned` requires the pinned
set to equal `SAMPLE_NAMES`, so a new workload cannot be added without a golden entry.
`test_trace_files_are_byte_for_byte_unchanged` compares the digests.
`test_simulation_results_are_unchanged` compares the whole result record per workload.
`test_accesses_and_cycles_are_self_consistent` cross-checks the pinned numbers against each
other without running anything: `reads + writes == accesses`, `l1_hits + l1_misses ==
accesses`, `l2_hits + l2_misses == l1_misses`, `l3_hits + l3_misses == l2_misses`,
`l3_misses == dram_reads`, and

```text
cycles == 4 x accesses + 12 x l1_misses + 40 x l2_misses + 100 x dram_reads
```

The module regenerates the traces into a temporary directory, which takes a few seconds, so
it honours `CACHESIM_SKIP_SLOW` (any value but `0` or empty skips it). It runs by default.

## Stack-distance cross-check

`cachesim/stackdist.py` computes LRU stack distances with a Fenwick tree in O(N log N) and
derives the miss count at every capacity from one histogram, using the inclusion property
of Mattson, Gecsei, Slutz and Traiger (1970):

```text
misses(C) = (first touches) + (references with stack distance >= C)
```

The load-bearing tests are exact-equality cross-checks against the simulator. They are
mutual validation: the profiler and the simulator reach the same numbers by completely
different routes, so agreement at every capacity and every associativity is evidence for
both.

| Statement | Coverage | Test |
| --- | --- | --- |
| `miss_ratio_curve(blocks, C)` equals the misses of a simulated fully-associative LRU `Cache` of `C` blocks, exactly, at every `C` | 4,000 references over 64 blocks, capacities 1, 2, 3, 4, 5, 7, 8, 11, 16, 32, 63, 64, 65, 128 | `TestMissRatioCurveMatchesSimulation::test_random_stream` |
| the same, for every sample workload | all six sample workloads, shrunk by `small_samples()` so the cross-check stays quick, capacities 1 to 512 in powers of two | `TestMissRatioCurveMatchesSimulation::test_every_sample_workload` |
| `profile.misses(num_sets x ways)` equals `cache.shadow_misses` of a simulated set-associative LRU cache, and `profile.infinite` equals its `compulsory_misses` | `conflict_streams`, geometries (64, 1), (32, 2), (16, 4), (8, 8), (4, 16) | `TestMissRatioCurveMatchesSimulation::test_matches_shadow_misses_of_a_set_associative_cache` |
| `per_set_profile(blocks, S).misses(w)` equals a simulated `S`-set, `w`-way LRU cache, exactly | the same six shrunk workloads x 4, 16 and 64 sets x 1 to 16 ways | `TestPerSetProfile::test_all_associativities_match_simulation` |
| a one-set per-set profile reproduces the fully-associative profile histogram and infinite count | `matmul(n=16)` | `TestPerSetProfile::test_one_set_reproduces_the_fully_associative_profile` |
| per-set reference and distinct counts sum to the stream and to `profile.infinite` | `conflict_streams` over 32 sets | `TestPerSetProfile::test_per_set_counts_sum_to_the_stream` |
| four aliasing streams miss everything at 1 and 2 ways and drop to the compulsory floor from 4 ways on | `conflict_streams` over 64 sets | `TestPerSetProfile::test_conflict_streams_need_four_ways` |

The third row is the tightest statement of the relationship. `Cache.shadow_misses` is the
miss count of the fully-associative LRU twin the three-C classifier maintains internally,
and `profile.misses(C)` is the miss count the stack-distance histogram predicts for a
fully-associative LRU cache of the same capacity. They are computed by unrelated code and
must be equal.

Supporting tests pin the distances themselves. The stream `A B C A C A` (blocks 10, 11,
12) has distances `inf, inf, inf, 2, 1, 1` counted by hand
(`TestStackDistances::test_hand_checked_distances`); immediate reuse is distance 0
(`test_immediate_reuse_is_distance_zero`); a block referenced repeatedly between two
references to another still costs 1
(`test_repeated_block_counted_once`); and no distance can reach the number of distinct
blocks (`test_distance_is_bounded_by_distinct_blocks`). `TestProfileShape` checks that the
histogram plus the infinite count equals the reference count, that the miss curve is
non-increasing, and that it reaches the compulsory floor at `max_distance + 1`.

## Test suite facts

```bash
python -m pytest --co -q | tail -1
```

```text
tests/test_write_policy.py: 18
```

The project sets `addopts = "-q"` in `pyproject.toml`, so `--co -q` prints a per-module
count rather than a total. The total is available by overriding the option:

```bash
python -m pytest --co -q -o addopts="" | tail -1
```

```text
627 tests collected in 0.15s
```

The whole suite:

```bash
python -m pytest
```

```text
627 passed, 509 subtests passed in 20.25s
```

The golden module regenerates the sample traces into a temporary directory, which is the
single largest fixed cost. Skipping it saves about 2.7 s:

```bash
CACHESIM_SKIP_SLOW=1 python -m pytest
```

```text
623 passed, 4 skipped, 491 subtests passed in 17.57s
```

The slowest individual tests are the ones that regenerate traces or sweep a sample
workload. Wall times vary by a few per cent between runs; every figure in this section
comes from one sequence of runs on one machine.

```bash
python -m pytest --durations=8
```

| Duration | Test |
| ---: | --- |
| 2.16 s | `tests/test_golden.py::TestGolden::test_simulation_results_are_unchanged` |
| 1.96 s | `tests/test_workloads.py::TestHashProbe::test_seeded_miss_count_is_exact` |
| 1.89 s | `tests/test_workloads.py::TestHashProbe::test_steady_state_miss_rate_matches_one_minus_c_over_n` |
| 1.25 s | `tests/test_timing.py::TestSampleTraceBlockSizeTable::test_the_matmul_naive_column_bottoms_out_at_256_bytes` |
| 0.63 s | `tests/test_stackdist.py::TestPerSetProfile::test_all_associativities_match_simulation` |
| 0.60 s | `tests/test_indexing.py::TestIndexingOnWorkloads::test_xor_indexing_removes_most_matmul_conflict_misses` |
| 0.46 s | `tests/test_stackdist.py::TestMissRatioCurveMatchesSimulation::test_every_sample_workload` |
| 0.44 s | `tests/test_golden.py::TestGolden` class setup (regenerating the traces) |

### Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull request, in two
jobs.

**`test`** runs on `ubuntu-latest` against the five-version matrix 3.10, 3.11, 3.12, 3.13
and 3.14, with `fail-fast: false`, so one version failing does not hide the others. Every
leg installs `pip install -e ".[dev]"`, runs the full suite with `python -m pytest`, and
then smoke-tests every subcommand through the installed console script, including the
invariant checker:

```bash
cachesim gen-traces --out-dir traces sequential conflict
cachesim run --check traces/sequential.trace --config configs/default.json
cachesim run --format json --config configs/prefetch_stride.json traces/conflict.trace > /dev/null
cachesim trace-stats traces/conflict.trace
cachesim sweep --size 8192 traces/conflict.trace
cachesim policies --size 8192 traces/conflict.trace
cachesim mrc traces/conflict.trace --sets 64
cachesim compare traces/conflict.trace --config default --config configs/victim_cache.json
cachesim sets traces/conflict.trace --size 4k --assoc 1
```

**`lint`** runs once, on Python 3.13:

```bash
ruff check .
ruff format --check .
mypy
```

`mypy` is configured with `strict = true` and `warn_unreachable = true` over both
`cachesim` and `tests`, so the test suite is type-checked to the same standard as the
package.

---

Related pages: [model.md](model.md) for the semantics these tests pin,
[configuration.md](configuration.md) for the configuration fields the guards read,
[cli.md](cli.md) for `run --check` and the other subcommands,
[workloads.md](workloads.md) for the generators behind the traces, and
[results.md](results.md) for the measurements. Back to [../README.md](../README.md).
