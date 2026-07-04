# Cache & Memory-Hierarchy Simulator

A trace-driven, configurable simulator for the CPU cache hierarchy, written in
plain Python. Feed it a stream of memory addresses, describe a hierarchy
(L1 → L2 → L3 → DRAM), and it reports hits, misses, miss rates, a breakdown of
*why* each miss happened, and AMAT — the one number that summarizes how fast
your memory system is.

No computer-architecture background is assumed; every term is explained below.

## Why caches exist (30 seconds)

A CPU can do arithmetic in under a nanosecond, but fetching data from main
memory (DRAM) takes ~100 nanoseconds. So CPUs keep small, fast stashes of
recently-used data close to the core — **caches** — arranged in layers:

| Level | Typical size | Typical latency | Kitchen analogy |
|-------|--------------|-----------------|-----------------|
| L1    | ~32 KB       | ~4 cycles       | the counter     |
| L2    | ~256 KB      | ~12 cycles      | the pantry      |
| L3    | ~2–32 MB     | ~40 cycles      | the basement    |
| DRAM  | gigabytes    | ~100+ cycles    | the grocery store |

Every access checks L1 first. Found it? That's a **hit** — fast. Not there?
A **miss** — go down a level and try again. The entire game of memory-system
design is maximizing hits, and this simulator lets you watch how the three
big design knobs change the score.

## Quick start

Requires Python 3.8+. Only `sweep.py`'s plots need a third-party package
(`pip install matplotlib`); everything else is stdlib.

```bash
python3 traces/generate_traces.py        # create the sample traces (~17 MB)
python3 simulator.py traces/sequential.trace   # run one trace, get a report
python3 sweep.py                         # the two payoff plots -> plots/
python3 tests.py                         # 22 known-answer tests, all stdlib
```

## The files

| File | What it is |
|------|-----------|
| `cache.py` | One cache level: size, block size, associativity, and replacement policy are all parameters. Also classifies misses into the "3 Cs". |
| `hierarchy.py` | Chains levels into L1 → L2 → L3 → DRAM, forwards misses down, and does the timing/AMAT accounting. |
| `simulator.py` | Reads a trace file and drives it through a hierarchy. `--config file.json` to change the hierarchy (see `example_config.json`). |
| `report.py` | Formats the statistics into the report you see. |
| `sweep.py` | Sweeps associativity (1→16-way) and cache size (1→64 KB), prints tables, and plots miss rate against each. |
| `traces/generate_traces.py` | Regenerates the sample traces (deterministic — same files every time). |
| `tests.py` | Hand-checkable unit tests: LRU vs FIFO victims, conflict thrash, 3-C classification, AMAT identity, trace parsing. |
| `example_config.json` | A hierarchy config you can copy and edit. |

## Trace format

One access per line: a hex byte-address, then `R` (read) or `W` (write).
Blank lines and `#` comments are ignored.

```
0x00400000 R
0x00400008 R
0x7fff0010 W
```

## Configuration parameters, in plain language

These appear in `example_config.json` / `simulator.DEFAULT_CONFIG` and as
`sweep.py` flags.

- **`size`** — total bytes the cache holds (32768 = 32 KB). Bigger caches hit
  more but are slower and burn more power, which is why L1 stays small even
  though transistors are cheap.

- **`block_size`** — caches move data in fixed-size chunks called blocks (or
  "lines"), typically 64 bytes. Miss on one byte and the cache pulls in the
  whole 64-byte block, betting you'll want the neighbors soon (you usually
  do — that's *spatial locality*, and it's why scanning an array is cheap:
  one miss buys the next 7 8-byte elements for free).

- **`associativity`** — how many places within the cache a given block is
  *allowed* to live. The cache is organized as `sets × ways`: an address is
  assigned to exactly one **set** (by simple arithmetic on the address), and
  may occupy any of the set's `associativity` **ways**.
  - `1` (**direct-mapped**): one possible spot per address. Fast and cheap to
    check, but two hot addresses assigned the same spot evict each other
    forever.
  - `4`/`8` (**set-associative**): the real-world compromise; a small group
    of spots per address.
  - ways = all blocks (**fully associative**): a block can go anywhere. No
    conflicts, but hardware must compare against every entry at once —
    too expensive except for tiny structures (e.g. TLBs).

- **`policy`** — when a set is full, who gets evicted?
  - `lru` — Least Recently Used: evict the block untouched the longest.
    Usually best; costs bookkeeping.
  - `fifo` — evict whatever was *loaded* longest ago, even if it's hot.
  - `random` — evict a random way. Nearly free in hardware, and surprisingly
    decent — on our naive-matmul trace it actually *beats* LRU (3.7% vs 5.2%
    miss rate), because LRU's worst case is exactly the cyclic re-scan
    pattern that strided loops produce, while random never commits to a
    pathological choice. Adding your own policy is a ~10-line subclass in
    `cache.py` (see `POLICIES`).

- **`hit_time`** — cycles charged to probe that level (whether it hits or
  misses; a miss pays the probe *and then* the levels below).

- **`memory_access_time`** — cycles for DRAM once everything has missed.

## Output metrics, in plain language

- **hits / misses** — per level. A miss at L1 becomes an access at L2, so
  L2's numbers only count traffic L1 couldn't serve.

- **local miss rate** — misses ÷ accesses *that reached this level*. L2's
  local miss rate often looks terrible (50%+) because L1 already filtered
  out all the easy hits — that's normal, not a bug.

- **global miss rate** — misses ÷ *all* CPU accesses. This is the fraction
  of your program's memory traffic that got past this level.

- **evictions / writebacks** — how many blocks were kicked out, and how many
  of those had been written to ("dirty") and had to be saved downward. The
  simulator uses write-back, write-allocate: writes are cached like reads and
  modified data is only pushed down on eviction.

- **the 3 Cs** — every miss is classified by *what would have prevented it*:
  - **compulsory** — the first-ever touch of that block. No cache of any
    size or shape could have hit. Only bigger blocks (prefetching) help.
  - **capacity** — a *fully-associative* cache of the same total size would
    also have missed: your working set is simply bigger than the cache.
    Only a bigger cache (or better locality in your code) helps.
  - **conflict** — the fully-associative twin *would have hit*, so the miss
    is purely from too many blocks fighting over one set. More ways help.

  The classification works by running a shadow fully-associative LRU cache
  of identical capacity next to each level and comparing outcomes.

- **AMAT (Average Memory Access Time)** — the headline number:

  ```
  AMAT = hit_time + miss_rate × miss_penalty
  ```

  where the miss penalty of each level is the AMAT of everything below it,
  so the formula nests: `L1_hit + m1 × (L2_hit + m2 × (L3_hit + m3 × DRAM))`.
  The report prints both this formula and the measured average
  (total simulated cycles ÷ accesses); they agree, which is a good self-check.
  Note the nesting is why an L1 miss-rate improvement is worth so much more
  than the same improvement at L3: L1's miss rate multiplies *everything*.

## The sample traces

| Trace | Pattern | What it teaches |
|-------|---------|-----------------|
| `sequential.trace` | linear scan of 256 KB, twice | The friendliest case: one miss per 64 B block, then 7 free hits. Pass 2 hits in L2 (256 KB) but not L1 (32 KB) — you can see the hierarchy working level by level. |
| `random.trace` | uniform random over 16 MB | The cruelest case: ~91% of accesses fall all the way to DRAM, AMAT ≈ 147 cycles vs ~14 for sequential. Same machine, 10× slower — *this* is why cache-friendly code matters. |
| `matmul_naive.trace` | C = A×B, 64×64 doubles, i-j-k order | Walks B by column: 512-byte stride, terrible locality. L1 miss rate ≈ 5.2%. |
| `matmul_blocked.trace` | the *same* multiplication in 16×16 tiles | Identical math, different order: L1 miss rate ≈ 0.6% — **8× fewer misses just from loop order**. The classic interview demo. |
| `conflict.trace` | 4 streams whose addresses alias to the same sets | Pure conflict misses: ~100% miss below 4-way, 12.5% at 4-way and beyond. The cleanest associativity knee. |
| `pointer_chase.trace` | a shuffled linked list in 1 MB, ~4 laps | Unpredictable jumps: L1/L2 miss ~100%, but L3 (2 MB) holds the whole list and serves ~73% of accesses. Reuse only pays if some level can hold the working set. |

## The two payoff plots (`python3 sweep.py`)

Both sweeps use a single cache level backed by DRAM so the knob you're
turning isn't blurred by other levels.

**Miss rate vs associativity** (on `conflict.trace`, size fixed at 8 KB):
100% at 1-way and 2-way, then a cliff to 12.5% at 4-way, then dead flat.
The flattening is a real engineering decision — past the knee, extra ways
cost power and latency and buy *nothing*. Check the 3-C columns in the
table: the conflict misses go to exactly zero at the knee, and the
compulsory misses that remain are untouchable by definition.

**Miss rate vs cache size** (on `matmul_naive.trace`, 4-way fixed):
flat ~51% from 1 KB to 16 KB, then a cliff to 5% at 32 KB and 0.3% at 64 KB.
Caches don't help gradually — they help *when the working set fits*. The
cliff sits exactly where one 32 KB matrix starts fitting.

**A worthwhile detour:** run `python3 sweep.py traces/matmul_naive.trace`
and watch associativity *fail* to help (and at 32 KB, actively hurt —
try `--size 32768`). The 64×64 matrix rows are 512 bytes — a power of two —
so B's column accesses keep landing on the same few sets no matter how the
cache is shaped: at fixed size, doubling the ways halves the sets, and the
aliasing follows you. This is a famous real-world trap (it's why HPC
programmers pad arrays to avoid power-of-two leading dimensions) and a good
reminder that the clean textbook curve assumes conflicts among a *few* hot
blocks, not a systematic stride.

## Extending the replacement policy

Everything is registered by name in `cache.POLICIES`:

```python
class MyPolicy(ReplacementPolicy):
    def on_hit(self, set_idx, way): ...   # an access hit this way
    def on_fill(self, set_idx, way): ...  # a new block landed in this way
    def victim(self, set_idx): ...        # set is full: pick a way to evict

POLICIES["mine"] = MyPolicy
```

`victim()` is only called when the set is completely full — empty ways are
always filled first, so policies never see invalid lines. A natural next
step is SRRIP/BRRIP from the RRIP family (Jaleel, Theobald, Steely &
Emer, ISCA 2010 — the line of work Moinuddin Qureshi's group builds on):
keep a 2-bit "re-reference prediction" counter per line instead of full
LRU order. It's ~20 lines here and it is *actual* modern-cache research.

## What "cycle-accurate" means here

Hit/miss/eviction behavior and access-time accounting are exact for the
model described: probe levels in order, charge each level's hit time,
allocate on miss at every level, write-back + write-allocate for stores.
Deliberately not modeled: pipelining/overlap of misses (MSHRs), DRAM row
buffers and banks, prefetchers, cache inclusion policies, coherence, and
write-buffer timing (writebacks are counted but not charged cycles). Each
of those is a fine extension project.
