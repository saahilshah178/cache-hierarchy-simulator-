# cachesim

`cachesim` is a trace-driven simulator for configurable multi-level set-associative cache
hierarchies. It replays memory references through an L1/L2/L3 chain backed by a main-memory
model and reports hits, misses, evictions, traffic and AMAT per level. The model covers ten
replacement policies plus Belady's OPT as an offline lower bound, modulo and XOR-folded set
indexing, inclusive / exclusive / NINE inclusion, write-back and write-through with or
without write-allocate, hardware prefetchers, victim caches, an open-page DRAM row-buffer
model, and write-back propagation between levels. On top of it come per-reference and
aggregate (Hill and Smith) three-C miss classification, Mattson stack-distance analysis
yielding the miss-ratio curve at every capacity from one pass, parameter sweeps and
configuration comparisons, and a verification suite built on an independent reference model.

[![CI](https://github.com/saahilshah178/cache-hierarchy-simulator-/actions/workflows/ci.yml/badge.svg)](https://github.com/saahilshah178/cache-hierarchy-simulator-/actions/workflows/ci.yml) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Features

Model

- Any number of levels, each with its own `size`, `block_size`, `associativity`, `hit_time`.
- Replacement (`policy`): `lru`, `fifo`, `random`, `plru` (tree-PLRU), `nru`, `lfu`, `mru`,
  `srrip`, `brrip`, `drrip`, and the offline `opt` (Belady's MIN).
- Set indexing (`index`): `modulo` low-order bits, or `xor` XOR-folded hashing.
- Inclusion per level (`inclusion`): `nine`, `inclusive` with back-invalidation, `exclusive`.
- Writes per level (`write_policy`, `write_allocate`): write-back or write-through,
  write-allocate or no-write-allocate.
- Prefetching (`prefetcher`) and a fully-associative victim buffer (`victim_cache`).
- Main memory (`memory`): constant latency or open-page row buffer, plus `bus_width` timing.

Analysis

- `cachesim run` — per-level counters, three-C classification, traffic, and analytic and
  measured AMAT, as text or JSON.
- `cachesim sweep` — size, associativity, block size or policy, or a size x associativity
  grid, over a single level, with CSV and plots.
- `cachesim policies` — every replacement policy on one trace against the OPT bound.
- `cachesim mrc` — LRU miss-ratio curve and reuse-distance histogram from one pass.
- `cachesim compare` — one trace through several configurations, tabulated with deltas.
- `cachesim sets` — cumulative hot-set histogram and windowed set oversubscription.
- `cachesim trace-stats` — footprint, distinct blocks and pages, compulsory-miss floor.
- `cachesim bench` — wall-clock throughput of the simulator itself, best of N timed
  passes, as text or JSON.

Verification

- An independent naive reimplementation (`cachesim.reference`) shares no algorithmic code
  with the model and is compared against it access by access.
- `cachesim run --check` evaluates the algebraic identities in `cachesim.invariants`.
- Property-based tests search for counterexamples to the counter algebra, the three-C
  decomposition, level chaining and the LRU stack property.
- Textbook known answers, hand-derived closed forms, and pinned golden results.

## Installation

Python 3.10 or newer. The core is standard library only; `matplotlib` renders the plots.

```bash
pip install -e .              # simulator and CLI
pip install -e ".[plots]"     # adds matplotlib for --plot / --plot-dir
pip install -e ".[dev]"       # adds pytest, hypothesis, ruff, mypy, matplotlib
```

## Quick start

Generate the sample traces (about 17 MB; they are not checked in) and simulate one:

```bash
cachesim gen-traces --out-dir traces
cachesim run traces/matmul_naive.trace
```

```text
========================================================================
CACHE HIERARCHY SIMULATION REPORT  (traces/matmul_naive.trace)
========================================================================
Total accesses :      532,480   (reads 528,384 / writes 4,096)
DRAM reads     :        1,536   (0.29% of all accesses missed every level)
DRAM writes    :            0   (dirty lines written back from L3)

--- L1: 32.0 KB, 64 B blocks, 4-way, LRU, hit time 4 cyc ---
  accesses    :      532,480
  hits        :      504,568   (reads 504,568 / writes 0)
  misses      :       27,912   (reads 23,816 / writes 4,096)
  local miss rate  :   5.24%   (misses / accesses that reached this level)
  global miss rate :   5.24%   (misses / all CPU accesses)
  evictions   :       27,400   (writebacks of dirty blocks: 4,088)
  traffic     :       1.7 MB in from below / 255.5 KB out below
  miss classification      per-reference    aggregate
    compulsory                     1,536        1,536
    capacity                       4,032       32,256
    conflict                      22,344       -5,880
    fully-associative LRU misses: 33,792   hits it would have missed: 28,224

[the L2 and L3 sections are omitted here]

========================================================================
DRAM traffic            : 96.0 KB read / 0 B written
AMAT (analytic formula) :    5.033 cycles
AMAT (measured)         :    5.033 cycles   (2,679,904 cycles / 532,480 accesses)
  loads                 :    4.948 cycles   (2,614,368 cycles / 528,384 loads)
  stores                :   16.000 cycles   (65,536 cycles / 4,096 stores)
========================================================================
```

`--format json` emits every counter behind that report, versioned by `schema_version`;
`--check` prints `invariants: 62 checks passed` for this run and exits non-zero on any
violation; `--warmup N` simulates the first N accesses, resets the statistics, and reports
only the remainder. The analysis subcommands follow, with `bench` for throughput:

```bash
cachesim run --format json traces/matmul_naive.trace
cachesim run --check traces/matmul_naive.trace
cachesim run --warmup 100000 traces/matmul_naive.trace
cachesim sweep --param size --assoc 4 traces/matmul_naive.trace
cachesim sweep --grid traces/matmul_naive.trace
cachesim policies --size 32768 --assoc 4 traces/matmul_naive.trace
cachesim mrc traces/matmul_naive.trace
cachesim compare traces/matmul_naive.trace --config default --config configs/inclusive.json
cachesim sets traces/matmul_naive.trace --window 512
cachesim trace-stats traces/matmul_naive.trace
cachesim bench traces/matmul_naive.trace
```

`bench --repeat 5` on `matmul_naive.trace` measured 2,004,954 accesses/s (499 ns/access,
best of 5) on the development machine; throughput depends on the machine and interpreter,
not on the simulated hierarchy.

Full option reference: [docs/cli.md](docs/cli.md).

## Results

### Loop order

![L1 miss rate against cache size for the naive and blocked matrix multiply](docs/figures/miss_rate_vs_size_matmul.png)

Both traces perform the same 64x64 double-precision matrix multiply and touch the same 1,536
distinct blocks, so their compulsory floor and DRAM read count are identical. The naive
i/j/k order walks B by column with a 512-byte stride and loses each column before the next
reuse; the 16x16 blocked order keeps a 2 KB tile of A, B and C resident. Sweeping a single
4-way level, blocked falls below 1% at 16 KB while naive stays above 50% until 32 KB.

| Trace (default hierarchy) | Accesses | L1 misses | L1 miss rate | L2 local miss rate | AMAT (cycles) |
|---------------------------|----------:|----------:|-------------:|-------------------:|--------------:|
| `matmul_naive.trace`      |   532,480 |    27,912 |        5.24% |              5.50% |         5.033 |
| `matmul_blocked.trace`    |   557,056 |     3,572 |        0.64% |             43.00% |         4.463 |

```bash
cachesim run traces/matmul_naive.trace
cachesim run traces/matmul_blocked.trace
cachesim sweep --param size --assoc 4 traces/matmul_naive.trace
cachesim sweep --param size --assoc 4 traces/matmul_blocked.trace
```

### Replacement policy against the OPT bound

![Misses per replacement policy against the OPT bound on the naive matrix multiply](docs/figures/policies_matmul_naive.png)

At 32 KB, 4-way, LRU sits 2.30x above per-set OPT on this trace, and `random` beats it by a
quarter. LRU's worst case is cyclic re-reference of a working set slightly larger than the
capacity, which is what a 512-byte column stride produces; `random` never commits to that
victim choice, and the RRIP policies avoid it by predicting a distant re-reference for newly
filled blocks. Per-set OPT is optimal given the set mapping, while the fully-associative
Belady bound of 2,184 misses (0.41%) shows what the mapping itself costs.

| Policy   |  Misses | Miss rate | x OPT |
|----------|--------:|----------:|------:|
| `opt`    |  12,141 |     2.28% |  1.00 |
| `random` |  19,872 |     3.73% |  1.64 |
| `srrip`  |  22,432 |     4.21% |  1.85 |
| `drrip`  |  23,367 |     4.39% |  1.92 |
| `nru`    |  27,388 |     5.14% |  2.26 |
| `plru`   |  27,854 |     5.23% |  2.29 |
| `lru`    |  27,912 |     5.24% |  2.30 |
| `fifo`   |  28,024 |     5.26% |  2.31 |
| `brrip`  |  52,149 |     9.79% |  4.30 |
| `mru`    | 120,217 |    22.58% |  9.90 |
| `lfu`    | 181,125 |    34.02% | 14.92 |

Changing the set-index function is the more effective response, because the misses come from
the mapping rather than from the victim choice. Setting `"index": "xor"` on L1 folds the
high-order block bits into the index, so a power-of-two stride no longer aliases.

| L1 `index` | L1 misses | L1 miss rate | Conflict (per-reference) | AMAT (cycles) |
|------------|----------:|-------------:|-------------------------:|--------------:|
| `modulo`   |    27,912 |        5.24% |                   22,344 |         5.033 |
| `xor`      |     5,470 |        1.03% |                        0 |         4.527 |

```bash
cachesim policies --size 32768 --assoc 4 traces/matmul_naive.trace

cat > /tmp/xor-l1.json <<'JSON'
{"memory_access_time": 100,
 "levels": [
   {"name": "L1", "size": 32768, "block_size": 64, "associativity": 4,
    "policy": "lru", "hit_time": 4, "index": "xor"},
   {"name": "L2", "size": 262144, "block_size": 64, "associativity": 8,
    "policy": "lru", "hit_time": 12},
   {"name": "L3", "size": 2097152, "block_size": 64, "associativity": 16,
    "policy": "lru", "hit_time": 40}]}
JSON
cachesim run traces/matmul_naive.trace
cachesim run --config /tmp/xor-l1.json traces/matmul_naive.trace
```

### Associativity and the sign of aggregate conflict

![Miss rate against associativity on the conflict trace](docs/figures/miss_rate_vs_associativity_conflict.png)

`conflict.trace` reads four sequential streams in lockstep with their bases 1 MB apart, so
the corresponding element of all four maps to one set for any geometry with at most 16,384
sets. At 8 KB a direct-mapped or 2-way cache holds fewer of those blocks than the streams
need and misses every access; four ways hold all four, and the miss rate drops to the
compulsory floor of 12.5%, one miss per 64-byte block of eight 8-byte references. Past the
knee the curve is flat and conflict misses are zero.

| Associativity | Miss rate | AMAT (cycles) | Compulsory | Capacity | Conflict |
|--------------:|----------:|--------------:|-----------:|---------:|---------:|
|             1 |   100.00% |        104.00 |      8,192 |        0 |   57,344 |
|             2 |   100.00% |        104.00 |      8,192 |        0 |   57,344 |
|             4 |    12.50% |         16.50 |      8,192 |        0 |        0 |
|             8 |    12.50% |         16.50 |      8,192 |        0 |        0 |
|            16 |    12.50% |         16.50 |      8,192 |        0 |        0 |

```bash
cachesim sweep --param associativity --size 8192 traces/conflict.trace
```

The aggregate conflict count is a subtraction of totals — misses minus the misses of a
fully-associative LRU cache of the same capacity — so it goes negative whenever the set
mapping beats fully-associative LRU, which is what the 32 KB L1 of the report above shows:
27,912 - 33,792 = -5,880. Splitting the column walk across 128 sets shortens the
re-reference cycle each set sees, so LRU keeps blocks the shadow cache had already evicted;
those 28,224 anti-conflict hits are exactly the gap between the two taxonomies, since
per-reference conflict (22,344) minus them gives the aggregate figure.

Block size against AMAT, inclusion policies, prefetching, the DRAM row buffer, victim
caches, set pressure and miss-ratio curves: [docs/results.md](docs/results.md).

## Configuration

A hierarchy is a JSON object with a memory model and a list of levels, the one nearest the
core first. This is `configs/default.json`, the built-in hierarchy used without `--config`:

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40}
  ]
}
```

`name`, `size`, `block_size`, `associativity` and `hit_time` are required per level. Every
other key is optional and defaults to the behaviour above:

- `policy` — replacement policy name (default `lru`).
- `index` — `modulo` or `xor` set-index function (default `modulo`).
- `inclusion` — relation to the level above: `nine` (default), `inclusive`, `exclusive`.
- `write_policy` — `write-back` or `write-through` (default `write-back`).
- `write_allocate` — whether a write miss fetches the block here (default `true`).
- `bus_width` — bytes/cycle of the fill link, adding `ceil(block_size / bus_width)` cycles.
- `prefetcher` — `none`, `next-line` or `stride` (default `none`).
- `victim_cache` — `{"entries": N}` for a fully-associative victim buffer (default `null`).
- `memory` / `memory_access_time` — top level: a constant latency in cycles, or a `memory`
  object selecting a `constant` or `row-buffer` DRAM model.

`configs/` holds a worked example of each feature. Full key reference, validation rules and
defaults: [docs/configuration.md](docs/configuration.md).

## Trace format

The native format is one access per line: a hexadecimal byte address, whitespace, then `R`
for a load or `W` for a store. Blank lines and everything after `#` are ignored.

```text
0x00400000 R
0x00400008 R
0x7fff0010 W
```

Dinero IV traces (`.din`) and Valgrind Lackey output (`.lackey`, `.vg`, from
`--trace-mem=yes`) are read directly, any path ending in `.gz` is decompressed
transparently, and `--trace-format` overrides the extension. The sample workloads and their
expected behaviour: [docs/workloads.md](docs/workloads.md).

## How it works

| Module | Responsibility |
|--------|----------------|
| `cache.py` | One level as sets x ways: `probe`, `allocate`, `invalidate`, dirty bits, victim buffer, three-C classification against a shadow fully-associative LRU cache. |
| `policies.py` | Replacement policies and the `POLICIES` registry: LRU, FIFO, random, tree-PLRU, NRU, LFU, MRU, SRRIP, BRRIP, DRRIP. |
| `indexing.py` | Set-index functions built from the set count: low-order `modulo` and XOR-folded `xor`. |
| `hierarchy.py` | Chains levels over a memory model: miss handling, write-back propagation, inclusion, write policies, prefetch, transfer time, timing and AMAT. |
| `prefetch.py` | Prefetchers watching demand references at a level: tagged next-line and per-region stride. |
| `dram.py` | Main memory below the last level: constant latency, or an open-page row buffer with banks. |
| `config.py` | The JSON schema as typed `HierarchySpec` / `CacheSpec` records, with defaults and errors naming the offending key. |
| `stats.py` | `HierarchyStats`, the counter snapshot both the text report and the JSON output are rendered from. |
| `report.py` | Renders a stats snapshot as the human-readable text report. |
| `trace.py` | Reading and writing traces in the native, Dinero IV and Lackey formats, gzip included. |
| `workloads.py` | Deterministic synthetic workload generators and the registry behind `gen-traces`. |
| `opt.py` | Belady's MIN: the fully-associative `opt_misses` bound and the per-set `OPTPolicy`. |
| `stackdist.py` | Mattson stack-distance profiling: miss counts at every capacity, and every associativity, from one pass. |
| `sweep.py` | Sweeps one parameter, or a size x associativity grid, over a single level; tables, CSV and plots. |
| `compare.py` | Replays one parsed trace through several configurations and tabulates the deltas. |
| `setpressure.py` | Per-set diagnostics: cumulative hot sets and windowed oversubscription of ways. |
| `invariants.py` | The identities a finished simulation must satisfy; the engine behind `run --check`. |
| `reference.py` | An independent, deliberately naive reimplementation used only as a test oracle. |
| `cli.py` | Argument parsing and subcommand dispatch; each subcommand registers itself. |

An access is a byte address and a read/write flag. The address is divided by the block size
to give a block number, which chooses a set through the level's index function. The
hierarchy probes top-down, charging each level its hit time and passing a miss to the level
below, so the accesses a level sees equal the miss count of the level above it; if every
level misses, the memory model supplies the block and charges the memory access time. The
block is then filled bottom-up into every level that missed, and a fill into a full set
evicts a victim chosen by that level's replacement policy. A dirty victim is written into
the level below — marking its copy dirty, or allocating the line dirty when the copy is gone
— and a dirty line leaving the last level becomes a DRAM write; that propagation is counted
but charged no cycles, as if absorbed by write buffers. A store dirties the line only in the
highest level that allocates for it. The full model, inclusion and write-policy variations
included: [docs/model.md](docs/model.md).

## Validation

680 tests (`pytest --co -q`) establish correctness from six directions.

- **Differential testing.** `cachesim.reference` is a second implementation written from the
  documented semantics, sharing no algorithmic code with the model. Both are driven with the
  same seeded streams and compared per access — the cycle count and the cumulative per-level
  miss vector after every reference — so a divergence is reported where it first appears.
- **Property-based testing.** The hypothesis library searches for counterexamples to laws
  that hold for any geometry and any stream: counter algebra, the three-C decomposition and
  its two taxonomies, compulsory misses equalling distinct blocks at every level, level
  chaining, the LRU stack property, and determinism.
- **Algebraic invariants.** `cachesim.invariants` checks counter algebra, traffic flow,
  write-back conservation, the timing identity `analytic AMAT == measured AMAT`, and
  structural consistency, each stepping aside with a reason when a feature makes it
  inapplicable. It is reachable from the CLI as `run --check`.
- **Textbook known answers.** Published reference strings with published fault counts,
  Belady's anomaly under FIFO, and the LRU stack property.
- **Closed forms.** Whole-trace results derived by hand from the geometry and the access
  pattern before being asserted, so a changed number has to be argued with.
- **Golden regressions and a stack-distance cross-check.** The six sample workloads are
  pinned end to end from generator to report, and the Mattson profiler's miss curve must
  equal simulation exactly at every capacity and associativity.

What each identity assumes: [docs/validation.md](docs/validation.md).

## Development

```bash
make install     # pip install -e ".[dev]"
make test        # pytest
make lint        # ruff check . && ruff format --check .
make typecheck   # mypy (strict)
make check       # lint + typecheck + test
make traces      # cachesim gen-traces --out-dir traces
```

CI runs the test suite on CPython 3.10, 3.11, 3.12, 3.13 and 3.14, then smoke-tests `run`,
`trace-stats`, `sweep`, `policies`, `mrc`, `compare`, `sets` and the invariant checker on
freshly generated traces. A separate job runs
`ruff check`, `ruff format --check` and `mypy` under `strict = true` with
`warn_unreachable`, over both `cachesim` and `tests`.

## References

- Belady, L. A., "A study of replacement algorithms for a virtual-storage computer", IBM
  Systems Journal 5(2), 1966.
- Belady, L. A., Nelson, R. A., Shedler, G. S., "An anomaly in space-time characteristics
  of certain programs running in a paging machine", Communications of the ACM 12(6), 1969.
- Mattson, R. L., Gecsei, J., Slutz, D. R., Traiger, I. L., "Evaluation techniques for
  storage hierarchies", IBM Systems Journal 9(2), 1970.
- Smith, A. J., "Cache Memories", ACM Computing Surveys 14(3), 1982.
- Baer, J.-L., Wang, W.-H., "On the Inclusion Properties for Multi-Level Cache
  Hierarchies", ISCA, 1988.
- Hill, M. D., Smith, A. J., "Evaluating Associativity in CPU Caches", IEEE Transactions on
  Computers 38(12), 1989.
- Jouppi, N. P., "Improving Direct-Mapped Cache Performance by the Addition of a Small
  Fully-Associative Cache and Prefetch Buffers", ISCA, 1990.
- Chen, T.-F., Baer, J.-L., "Effective Hardware-Based Data Prefetching for High-Performance
  Processors", IEEE Transactions on Computers 44(5), 1995.
- Gonzalez, A., Valero, M., Topham, N., Parcerisa, J. M., "Eliminating cache conflict
  misses through XOR-based placement functions", ICS, 1997.
- McKeeman, W. M., "Differential Testing for Software", Digital Technical Journal 10(1),
  1998.
- Edler, J., Hill, M. D., "Dinero IV: Trace-Driven Uniprocessor Cache Simulator",
  University of Wisconsin-Madison, 1998.
- Rixner, S., Dally, W. J., Kapasi, U. J., Mattson, P., Owens, J. D., "Memory Access
  Scheduling", ISCA, 2000.
- Nethercote, N., Seward, J., "Valgrind: A Framework for Heavyweight Dynamic Binary
  Instrumentation", PLDI, 2007.
- Qureshi, M. K., Jaleel, A., Patt, Y. N., Steely, S. C., Emer, J., "Adaptive Insertion
  Policies for High Performance Caching", ISCA, 2007.
- Jaleel, A., Theobald, K. B., Steely, S. C., Emer, J., "High Performance Cache Replacement
  Using Re-Reference Interval Prediction (RRIP)", ISCA, 2010.
- Hennessy, J. L., Patterson, D. A., "Computer Architecture: A Quantitative Approach", 6th
  ed., Morgan Kaufmann, 2017.
- Silberschatz, A., Galvin, P. B., Gagne, G., "Operating System Concepts", 10th ed., Wiley,
  2018.

## License

MIT. See [LICENSE](LICENSE).
