# Simulation model

This page is the normative description of what `cachesim` simulates: how an address
becomes a block, set and tag; the order in which levels are probed and filled; the exact
rule of every replacement policy; how stores and modified data travel downwards; how
inclusion is enforced; how cycles are charged; and what the model deliberately leaves out.

Every rule below is a statement about the code, and each section names the module that
implements it. The normative descriptions live in the module docstrings; this page
collects them, states the rules in one place, and pins each one to a command whose output
is reproduced next to it.

All commands are run from the repository root. The sample traces are not in the
repository; generate them first:

```bash
cachesim gen-traces --out-dir traces
```

Related pages: [configuration.md](configuration.md) for the JSON schema,
[cli.md](cli.md) for the subcommands, [workloads.md](workloads.md) for the sample traces,
[validation.md](validation.md) for how the model is checked, [results.md](results.md) for
measured behaviour. Back to [../README.md](../README.md).

## Addresses and geometry

Implemented in `cachesim/cache.py` (`Cache`, `_validate_geometry`) and
`cachesim/indexing.py`.

A level is described by four numbers: `size` in bytes, `block_size` in bytes,
`associativity` in ways, and a set-index function. From those:

```text
num_blocks = size // block_size
num_sets   = size // (block_size * associativity)
block      = addr // block_size
set index  = index_fn(block)          # modulo: block % num_sets
tag        = block // num_sets        # modulo indexing only
```

The trace supplies byte addresses; the boundary of the model is `Cache.block_of`, and
everything inside works on block numbers. An access is modelled as touching only the
block that contains its first byte, so a reference straddling a block boundary is charged
one block, not two (`cachesim/trace.py`).

`associativity == 1` is a direct-mapped level, `associativity == num_blocks` a fully
associative one, anything between an N-way set-associative one.

### Set-index functions

Two mappings are selectable per level through the `index` key.

`modulo` (the default) is the low-order bits of the block number. It is free in hardware
because the index bits are already in the address, but it makes the set index a function
of one narrow field, so any two blocks whose distance is a multiple of `num_sets`
collide — exactly the distance produced by walking an array with a power-of-two stride.

`xor` splits the block number into `log2(num_sets)`-bit chunks and XORs them together, so
every bit of the block number reaches the index and a power-of-two stride no longer
aliases. It requires a power-of-two set count. The trade is that address locality no
longer implies set locality. See Gonzalez, Valero, Topham and Parcerisa, "Eliminating
cache conflict misses through XOR-based placement functions", ICS 1997, and Seznec, "A
case for two-way skewed-associative caches", ISCA 1993.

```bash
python - <<'PY'
from cachesim import Cache
from cachesim.indexing import make_index

c = Cache("L1", size=32768, block_size=64, associativity=4)
print("num_sets", c.num_sets, "num_blocks", c.num_blocks)
for addr in (0x1000, 0x3040, 0x9040):
    b = c.block_of(addr)
    print(hex(addr), "block", b, "set", c.set_of(b), "tag", c.tag_of(b))
mod = make_index("modulo", 128)
xor = make_index("xor", 128)
print("stride 8 KB, modulo:", [mod(0x2000 * i // 64) for i in range(6)])
print("stride 8 KB, xor   :", [xor(0x2000 * i // 64) for i in range(6)])
PY
```

```text
num_sets 128 num_blocks 512
0x1000 block 64 set 64 tag 0
0x3040 block 193 set 65 tag 1
0x9040 block 577 set 65 tag 4
stride 8 KB, modulo: [0, 0, 0, 0, 0, 0]
stride 8 KB, xor   : [0, 1, 2, 3, 4, 5]
```

`0x3040` and `0x9040` are 24 KB apart and share set 65 under modulo indexing; the tag is
what tells them apart. Six addresses 8 KB apart all land in set 0 under modulo indexing
and in six different sets under XOR-folding.

`tag_of` is only meaningful for modulo indexing. A hashed index is not a bit-field of the
block number, so hardware using one stores the whole block number, or a wider tag,
instead.

### Validation rules

`_validate_geometry` rejects a geometry that cannot be simulated, and the index function
rejects a set count it cannot handle. All three of `size`, `block_size` and
`associativity` must be positive integers (`bool` is not an integer here).

| Rule | Reason |
|------|--------|
| `block_size` is a power of two | the block number is a bit-field of the address |
| `size % (block_size * associativity) == 0` | the level must hold a whole number of sets |
| `xor` indexing requires a power-of-two `num_sets` | the fold is defined on bit-fields |
| `plru` requires a power-of-two associativity | the tree must be complete |

The set count itself need not be a power of two: `block % num_sets` is defined for any
positive count, although real hardware uses a power of two.

```bash
python - <<'PY'
from cachesim import Cache

for kwargs in (
    dict(size=32768, block_size=48, associativity=4),
    dict(size=32768, block_size=64, associativity=3),
    dict(size=192, block_size=64, associativity=1),
    dict(size=192, block_size=64, associativity=1, index="xor"),
    dict(size=384, block_size=64, associativity=3, policy="plru"),
):
    try:
        c = Cache("L1", **kwargs)
        print(f"{kwargs} -> {c.num_sets} sets")
    except ValueError as exc:
        print(f"{kwargs} -> ValueError: {exc}")
PY
```

```text
{'size': 32768, 'block_size': 48, 'associativity': 4} -> ValueError: L1: block_size must be a power of two, got 48
{'size': 32768, 'block_size': 64, 'associativity': 3} -> ValueError: L1: size 32768 must be a multiple of block_size*associativity (192)
{'size': 192, 'block_size': 64, 'associativity': 1} -> 3 sets
{'size': 192, 'block_size': 64, 'associativity': 1, 'index': 'xor'} -> ValueError: L1: xor indexing requires a power-of-two set count, got 3
{'size': 384, 'block_size': 64, 'associativity': 3, 'policy': 'plru'} -> ValueError: L1: plru requires a power-of-two associativity, got 3
```

A hierarchy adds two rules of its own (`cachesim/config.py`, `cachesim/hierarchy.py`):
every level must share one block size, because multi-block fills are not modelled, and
the first level must declare `inclusion` as `nine`, because it has no level above it.
Level names must be unique. Addresses must be non-negative: both `Cache.access` and
`Hierarchy.access` raise `ValueError` before a block number is formed. The index
functions do not defend against a negative block number handed straight to `probe` or
`set_of`, and they disagree about what it would mean (`modulo` returns `num_sets - 1` for
block -1, `xor` returns 0); neither answer models anything.

## Access semantics

Implemented in `cachesim/hierarchy.py` (`Hierarchy.access`) over the three primitives of
`cachesim/cache.py`: `probe` (look up, update statistics and replacement state, classify
a miss, do not fill), `allocate` (install a block, evicting a victim if the set is full),
and `invalidate` (remove a block on command).

One access is:

1. **Probe top-down.** Each level in turn is charged its hit time and probed. The walk
   stops at the first level that hits; if none does, main memory answers.
2. **Allocate bottom-up.** The block is installed into every level between the level that
   supplied it and the topmost level that this access fills, deepest first. This is
   allocate-on-miss: a level that missed gets the block as part of the same access.
3. **Handle each eviction** the fills caused (see
   [Write-back propagation](#write-back-propagation) and
   [Inclusion policies](#inclusion-policies)).
4. **Let the prefetchers observe** the reference (see [Prefetchers](#prefetchers)).

Within a set, `allocate` fills the lowest-numbered empty way before it asks the policy
for a victim, so a policy is never shown an invalid way. If the block is already present,
`allocate` is a no-op except that the dirty bit is OR-ed in.

Dirty bits implement write-back semantics: a write hit sets the bit, a fill sets it when
the store was taken at that level, and a dirty line leaving a level counts as a
write-back. `invalidate` counts an invalidation rather than an eviction; a dirty
invalidated line still counts as a write-back, because its data must be written down.

### What counts as an access

A level's `accesses` is `hits + misses`, and only a `probe` produces either. This
distinction is what makes each level's miss rate a statement about demand references, and
what keeps `levels[i+1].accesses == levels[i].misses` in a plain hierarchy.

| Event at a level | Counted as an access | Counters it moves |
|------------------|----------------------|-------------------|
| Demand load or store that reaches the level | yes | `hits`/`misses`, `read_*`/`write_*` |
| Hit served by the level's victim buffer | yes | `hits`, `victim_hits` |
| Store passed down by a no-write-allocate level above | yes | `hits`/`misses` |
| Fill of a block the access missed on | no | `fills`, `evictions` |
| Write-back arriving from the level above | no | `writebacks_received`, `writeback_allocations` |
| Lookup caused by a prefetch issued above | no | `prefetch_probes`, `prefetch_probe_hits` |
| Back-invalidation from an inclusive level below | no | `back_invalidations`, `invalidations` |
| Line pushed down into an exclusive level | no | `fills`, and `writebacks_received` if it was dirty |

A store that no level has taken yet is still the access itself, so it is charged each
level's hit time and counted as an access at each level it reaches, exactly as a load
miss is.

The chain is visible in any default run:

```bash
python - <<'PY'
from cachesim import Hierarchy, load_config, parse_trace

h = Hierarchy.from_config(load_config("configs/default.json"))
for addr, is_write in parse_trace("traces/matmul_naive.trace"):
    h.access(addr, is_write)
print("hierarchy accesses %7d" % h.accesses)
for level in h.levels:
    print("%-3s accesses      %7d   hits %7d   misses %6d" % (
        level.cache.name, level.cache.accesses, level.cache.hits, level.cache.misses))
print("DRAM reads         %7d" % h.dram_reads)
PY
```

```text
hierarchy accesses  532480
L1  accesses       532480   hits  504568   misses  27912
L2  accesses        27912   hits   26376   misses   1536
L3  accesses         1536   hits       0   misses   1536
DRAM reads            1536
```

L1 is probed once per CPU access, L2 once per L1 miss, L3 once per L2 miss, and DRAM once
per L3 miss. `cachesim run --check` verifies that chain along with the rest of the
simulator's internal identities, 62 of them on this configuration and trace; see
[validation.md](validation.md).

```bash
cachesim run traces/matmul_naive.trace --config configs/default.json --check
```

```text
invariants: 62 checks passed
```

The full report the same command prints is reproduced in [results.md](results.md).

## Replacement policies

Implemented in `cachesim/policies.py`, plus `cachesim/opt.py` for the offline optimum.
A policy is selected per level by the `policy` key and decides which way of a **full** set
to evict.

Every policy is driven by the same hooks, and each policy's rule is stated below in terms
of them:

| Hook | When `Cache` calls it |
|------|-----------------------|
| `on_probe(set_idx, block)` | a lookup starts, hit or miss not yet known; only for policies that set `sees_references` |
| `on_hit(set_idx, way, block)` | the lookup hit this way |
| `on_fill(set_idx, way, block)` | `block` was installed into this way |
| `on_invalidate(set_idx, way)` | the way was emptied on command |
| `victim(set_idx)` | every way of the set holds valid data; return the way to evict |

`on_fill` also fires for a write-back arriving from above, which is not a lookup;
`on_probe` is the hook that counts references, and only the offline policy needs it.

All policies except `random`, `brrip` and `drrip` are deterministic. Those three draw from
the level's `random.Random(rng_seed)`, so a run is reproducible for a given seed.

The same reference sequence run against every policy shows how the rules differ. Four
blocks fill a 4-way set, `a` is re-referenced, and a fifth block arrives:

```bash
python - <<'PY'
from cachesim import POLICIES, Cache

BLOCK = 64
NAMES = "abcdefgh"
REFS = "abcdae"


def evicted_by(policy):
    c = Cache("S", BLOCK * 4, BLOCK, 4, policy=policy, track_3c=False)
    for ch in REFS[:-1]:
        c.access(NAMES.index(ch) * BLOCK)
    before = {block for _, _, block, _ in c.lines()}
    c.access(NAMES.index(REFS[-1]) * BLOCK)
    after = {block for _, _, block, _ in c.lines()}
    gone = before - after
    return NAMES[gone.pop()] if gone else "-"


print("reference sequence:", ", ".join(REFS))
for policy in sorted(POLICIES):
    if policy == "opt":
        continue
    print("%-8s evicts %s" % (policy, evicted_by(policy)))
PY
```

```text
reference sequence: a, b, c, d, a, e
brrip    evicts b
drrip    evicts b
fifo     evicts a
lfu      evicts b
lru      evicts b
mru      evicts a
nru      evicts a
plru     evicts c
random   evicts d
srrip    evicts b
```

Two behavioural contrasts separate the families. The first is a **scan**: two hot blocks
re-referenced every round while three fresh one-touch blocks stream past. The second is a
**cyclic** walk over one more block than the set can hold, which is the LRU worst case.

```bash
python - <<'PY'
from cachesim import Cache, run_opt

BLOCK = 64


def misses(policy, refs, ways):
    """Misses of one fully-associative set of `ways` blocks."""
    if policy == "opt":
        return run_opt([(b * BLOCK, False) for b in refs],
                       ways * BLOCK, BLOCK, ways, track_3c=False).misses
    cache = Cache("S", ways * BLOCK, BLOCK, ways, policy=policy, track_3c=False)
    for b in refs:
        cache.access(b * BLOCK)
    return cache.misses


# scan: two hot blocks re-referenced every round, three fresh one-touch blocks.
scan = []
for r in range(10):
    scan += [0, 1, 0, 1, 2 + 3 * r, 3 + 3 * r, 4 + 3 * r]
# cyclic: five blocks walked round and round through four ways.
cyclic = [b for _ in range(10) for b in range(5)]

print("%-8s %8s %8s" % ("policy", "scan", "cyclic"))
for policy in ("lru", "fifo", "plru", "nru", "lfu", "srrip", "brrip", "drrip", "mru", "opt"):
    print("%-8s %8d %8d" % (policy, misses(policy, scan, 4), misses(policy, cyclic, 4)))
PY
```

```text
policy       scan   cyclic
lru            50       50
fifo           50       50
plru           49       49
nru            44       50
lfu            32       23
srrip          32       50
brrip          32       23
drrip          32       50
mru            63       16
opt            32       16
```

The two streams are 70 and 50 references long. LRU's only hits on the scan are the 20
immediate repeats, because the three fresh blocks of each round flush both hot blocks out
of the set; on the cyclic walk it misses all 50, since it evicts precisely the block
about to be referenced. `drrip` matches `srrip` here because a single set has no room for
a duel; see [DRRIP](#drrip) below.

### LRU

`lru`. Each set keeps its ways in a recency list. A hit or a fill moves the way to the
most-recent end; `victim` returns the way at the least-recent end. Exact LRU, with the
full order per set. It is a stack algorithm, so its miss count never increases with
capacity (Mattson, Gecsei, Slutz and Traiger, "Evaluation techniques for storage
hierarchies", IBM Systems Journal 9(2), 1970). This is the default policy and the policy
of the shadow cache used for miss classification.

### FIFO

`fifo`. The same recency list, but only `on_fill` reorders it: hits do not refresh a
block's position, so the victim is the way filled longest ago. FIFO is not a stack
algorithm and can miss more with more capacity — Belady's anomaly (Belady, Nelson and
Shedler, "An anomaly in space-time characteristics of certain programs running in a
paging machine", CACM 12(6), 1969).

### Random

`random`. `victim` returns `rng.randrange(num_ways)`; no bookkeeping, no hooks used. One
value is drawn per eviction from the level's seeded generator, so a given `rng_seed`
reproduces a run exactly. Random is a useful control: it has no model of the reference
stream to be wrong about, so a workload on which it beats LRU is one where recency is
actively misleading.

### Tree-PLRU

`plru`. `num_ways - 1` bits per set, arranged as a complete binary tree stored as a heap:
node 0 is the root and the children of node `n` are `2n+1` (left) and `2n+2` (right). A
bit points at the half of its subtree holding the victim — 0 means left, 1 means right.

* `on_hit` and `on_fill` walk root to leaf along the path to that way and set every bit
  on the path to point **away** from it.
* `victim` walks root to leaf following the bits.
* `on_invalidate` aims the path **toward** the emptied way.

Exact LRU for two ways and an approximation beyond: one bit per node cannot record a
set's full order, so the victim is only guaranteed to lie in the less recently used half
at every level. In the sequence above it evicts `c` where true LRU evicts `b`. It costs
`num_ways - 1` bits per set instead of the `num_ways * log2(num_ways)` of true LRU, which
is why real L1 and L2 caches implement it (Hennessy and Patterson, "Computer
Architecture: A Quantitative Approach", 6th ed., 2017, sections B.1 and 2.3). Requires a
power-of-two associativity.

### NRU

`nru`. One reference bit per way. A fill or a hit sets the bit; `on_invalidate` clears
it. `victim` returns the lowest-numbered way whose bit is clear; if every bit is set, all
of them are cleared and way 0 is evicted, which is the choice a second pass would make
anyway. The coarsest useful recency approximation — one bit per way against LRU's full
order — and what large, highly associative last-level caches and TLBs use, sometimes
under the name clock or second chance (Silberschatz, Galvin and Gagne, "Operating System
Concepts", 10th ed., 2018, section 10.4.5, "LRU-Approximation Page Replacement").

### LFU

`lfu`. One counter per way: a fill sets it to 1 (the reference that caused the fill), a
hit adds one, an invalidation zeroes it. `victim` returns the lowest count, ties to the
lowest way number. The counters do not age, so a block that was hot early keeps its score
for as long as it stays resident. That is the classic weakness of plain LFU, which real
designs fix by halving counters periodically; nothing here ages, so `lfu` is a faithful
model of the textbook policy, cache pollution included (Silberschatz, Galvin and Gagne,
10th ed., 2018, section 10.4.6, "Counting-Based Page Replacement"). On `matmul_naive` it
is the worst policy in the registry by a wide margin.

### MRU

`mru`. The recency list again, with `victim` returning the **most** recently used way. It
is not a hardware candidate; it is the adversarial contrast to LRU. On a cyclic walk
whose working set is one block larger than the set, LRU misses on every access while MRU
sacrifices the same way repeatedly and keeps the rest resident — 16 misses against 50 in
the table above, matching OPT exactly on that pattern.

### SRRIP

`srrip`. Static re-reference interval prediction (Jaleel, Theobald, Steely and Emer,
"High Performance Cache Replacement Using Re-Reference Interval Prediction (RRIP)",
ISCA 2010). Each way carries a 2-bit re-reference prediction value (RRPV): 0 means "will
be re-referenced in the near future", 3 means "in the distant future", so 3 marks the
preferred victim.

* `on_fill` inserts at RRPV 2 ("long"), not 0.
* `on_hit` promotes to RRPV 0 (hit priority, the paper's SRRIP-HP).
* `on_invalidate` sets RRPV 3.
* `victim` scans for the first way at 3; if there is none, every RRPV in the set is
  incremented and the scan repeats, so at most three increments are ever needed.

Inserting at 2 makes the policy scan-resistant: a burst of one-touch blocks cycles
through the same ways at 2 → 3 while blocks that have been hit sit at 0 and survive. In
the scan above it takes 32 misses against LRU's 50, and it matches the offline optimum on
that stream.

### BRRIP

`brrip`. Bimodal RRIP: `on_fill` inserts at RRPV 3 (the next victim) except with
probability 1/32, when it inserts at 2. The occasional long insertion is what lets the
resident set change over time. A block inserted at 3 is evicted before the blocks already
resident, so a thrashing working set keeps most of its blocks instead of cycling all of
them out — 23 misses against LRU's 50 on the cyclic walk. The 1/32 draw comes from the
level's seeded generator.

### DRRIP

`drrip`. Dynamic RRIP: run SRRIP and BRRIP against each other and follow the winner, by
set dueling (Qureshi, Jaleel, Patt, Steely and Emer, "Adaptive Insertion Policies for
High Performance Caching", ISCA 2007; reused for RRIP by Jaleel et al., ISCA 2010).

A few sets always insert the SRRIP way, a few always the BRRIP way, and a saturating
counter records which group is missing more; every other set (the followers) uses
whichever insertion policy is winning. Leader sets are chosen by set index, so no state
is needed:

| Set count | Leaders |
|-----------|---------|
| `num_sets >= 64` | `set_idx % 64 == 0` leads SRRIP, `set_idx % 64 == 32` leads BRRIP |
| `2 <= num_sets < 64` | set 0 leads SRRIP, set `num_sets // 2` leads BRRIP |
| `num_sets == 1` | no leaders; PSEL never moves and the policy degenerates to SRRIP |

PSEL is a 10-bit saturating counter, incremented on a fill into an SRRIP leader set and
decremented on a fill into a BRRIP leader set (a fill is a miss: the line was not
resident, and a write-back allocation counts too). High means SRRIP is missing more, so
followers use BRRIP once the top bit is set. It starts at 511, one below the midpoint, so
that before any evidence has accumulated the followers run SRRIP. `role_of(set_idx)` and
`following()` expose the state.

```bash
python - <<'PY'
from cachesim import Cache

BLOCK, SETS, WAYS = 64, 64, 4

# Every set is asked to hold five blocks in four ways, twenty times over.
refs = [(b * SETS + s) * BLOCK
        for _ in range(20) for s in range(SETS) for b in range(5)]

for policy in ("srrip", "brrip", "drrip"):
    cache = Cache("L1", BLOCK * SETS * WAYS, BLOCK, WAYS, policy=policy, track_3c=False)
    for addr in refs:
        cache.access(addr)
    tail = ""
    if policy == "drrip":
        tail = " (PSEL %d, followers on %s)" % (cache.policy.psel, cache.policy.following())
    print("%-6s %6d misses of %d%s" % (policy, cache.misses, len(refs), tail))
PY
```

```text
srrip    6400 misses of 6400
brrip    2752 misses of 6400
drrip    2810 misses of 6400 (PSEL 568, followers on brrip)
```

DRRIP lands within 2% of the better of the two, and PSEL ends above the threshold with
the followers on BRRIP. The 58-miss gap to pure BRRIP is roughly what the duel itself
costs: a BRRIP set takes 43.0 misses here (2,752 over 64 sets) while the one SRRIP leader
set takes all 100, and that leader has to keep missing for PSEL to stay informed.

### OPT

`opt`, implemented in `cachesim/opt.py` rather than `policies.py` because it is offline.
Belady's MIN evicts the block whose next reference is farthest in the future, which
minimises misses for a fully associative cache with demand fetching (Belady, "A study of
replacement algorithms for a virtual-storage computer", IBM Systems Journal 5(2), 1966).
No implementable policy can do better, so it is the lower bound every real policy is
measured against.

Two things are provided.

`opt_misses(blocks, capacity)` computes the miss count of a **fully associative** OPT
cache from a block-number stream, in O(N log N): next-use indices are precomputed in one
backward pass, and the resident set is kept in a max-heap keyed by next use with lazy
deletion. A block never referenced again is given next use `len(blocks)`, which sorts as
the farthest possible future. Ties can only occur between blocks that are never
referenced again, so the count does not depend on how they are broken.

`OPTPolicy` is the **set-associative** Belady policy: within each set it evicts the way
whose block is referenced farthest in the future. It sets `sees_references`, so `Cache`
calls `on_probe` for every lookup. It must be told the future before the run:

* `preload(blocks)` supplies the exact sequence of block references the level will see,
  one entry per probe, and resets the cursor.
* `on_probe` advances the cursor, **verifies** that `stream[cursor] == block`, and
  refreshes that block's next-use time. A stream that does not match the simulation —
  wrong level, wrong trace, wrong order — raises `RuntimeError` immediately rather than
  silently producing a wrong bound. Running out of stream raises as well.
* `victim` evicts the way with the largest next-use time, never-referenced-again first,
  ties to the lowest way.
* Without a `preload`, the first probe and `victim` raise `RuntimeError` naming
  `preload`, `run_opt` and `simulate_with_opt`.

Blocks that arrive without being referenced — write-backs pushed down from above, which
allocate a line but are not lookups — are absent from the preloaded stream and therefore
count as never referenced again, which is correct: no future demand reference in that
stream asks this level for them.

Per-set Belady is optimal **given the set mapping**, not globally optimal. The true lower
bound for a cache of that capacity is what `opt_misses` computes for a fully associative
one, and `cachesim policies` prints both:

```bash
cachesim policies traces/matmul_naive.trace --size 32768 --assoc 4
```

```text
traces/matmul_naive.trace: 532,480 accesses through one level of 32 KB, 4-way, 64 B blocks (128 sets)
policy       misses  miss rate     AMAT   x OPT
opt          12,141     2.28%     6.28    1.00
random       19,872     3.73%     7.73    1.64
srrip        22,432     4.21%     8.21    1.85
drrip        23,367     4.39%     8.39    1.92
nru          27,388     5.14%     9.14    2.26
plru         27,854     5.23%     9.23    2.29
lru          27,912     5.24%     9.24    2.30
fifo         28,024     5.26%     9.26    2.31
brrip        52,149     9.79%    13.79    4.30
mru         120,217    22.58%    26.58    9.90
lfu         181,125    34.02%    38.02   14.92

fully associative Belady bound: 2,184 misses (0.41%)  -- the set mapping costs everything above this
```

**One level only.** `simulate_with_opt(trace_accesses, config)` drives a hierarchy in
which exactly one level asks for `opt`. Two or more such levels is a `ConfigError` from
`parse_config` and from `simulate_with_opt`; none at all is a `ConfigError` from
`simulate_with_opt`, which names `Hierarchy.from_config` as the alternative. Two passes
are needed, because the stream OPT needs is not the trace: an access reaches level L only
if every level above it missed.

* **Pass 1** runs the hierarchy with LRU substituted at level L and records the block of
  every probe L receives.
* **Pass 2** rebuilds the hierarchy, preloads that stream into the OPT policy, and
  replays the trace.

In a non-inclusive, non-exclusive hierarchy the recorded stream is exactly the stream the
OPT run sees, not an approximation, because whether an access probes L is decided entirely
by the levels above L; L's replacement policy changes only what L answers, which changes
what L+1 sees, never what L-1 does. Write-backs arriving at L from L-1 are the one thing
that reaches L without being a reference, and they are produced by L-1, so they replay
identically too.

That is also why a second OPT level is refused: OPT at L changes L's misses, hence the
stream L+1 sees, so a stream recorded for L+1 while L ran LRU would be wrong. Two OPT
levels would need a fixed point, not two passes.

Pass 1 rebuilds the OPT level with its geometry, index function and hit time but with
default inclusion, write and bus settings. Declaring `inclusion: inclusive` on the OPT
level therefore makes the two passes disagree about what L-1 holds, and the divergence is
caught by `on_probe`'s cursor check rather than being absorbed silently:

```bash
python - <<'PY'
from cachesim import parse_trace, simulate_with_opt

accesses = list(parse_trace("traces/random.trace"))[:20000]
config = {
    "memory_access_time": 100,
    "levels": [
        {"name": "L1", "size": 2048, "block_size": 64, "associativity": 4, "hit_time": 4},
        {"name": "L2", "size": 4096, "block_size": 64, "associativity": 4,
         "hit_time": 12, "policy": "opt", "inclusion": "inclusive"},
    ],
}
try:
    h = simulate_with_opt(accesses, config)
    print("completed, L2 misses", h.levels[1].cache.misses)
except RuntimeError as exc:
    print("RuntimeError:", str(exc)[:200])
PY
```

```text
RuntimeError: policy 'opt': the preloaded reference stream does not match the simulation at reference 1791: expected block 4454988, got block 4256891
```

Combine `opt` with a non-default inclusion policy on the same level only if that error is
acceptable; the combination is not supported.

```bash
python - <<'PY'
from cachesim import Hierarchy, parse_trace, simulate_with_opt

accesses = list(parse_trace("traces/matmul_naive.trace"))[:100000]
config = {
    "memory_access_time": 100,
    "levels": [
        {"name": "L1", "size": 8192, "block_size": 64, "associativity": 2, "hit_time": 4},
        {"name": "L2", "size": 32768, "block_size": 64, "associativity": 4, "hit_time": 12},
    ],
}
lru = Hierarchy.from_config(config)
for addr, is_write in accesses:
    lru.access(addr, is_write)

config["levels"][1]["policy"] = "opt"
opt = simulate_with_opt(accesses, config)

print("%-6s %10s %10s" % ("level", "LRU", "OPT at L2"))
for i, name in enumerate(("L1", "L2")):
    print("%-6s %10d %10d" % (name, lru.levels[i].cache.misses, opt.levels[i].cache.misses))
print("%-6s %10.3f %10.3f" % ("AMAT", lru.measured_amat(), opt.measured_amat()))
PY
```

```text
level         LRU  OPT at L2
L1          51041      51041
L2           5628       2586
AMAT       15.753     12.711
```

L1's miss count is identical in both passes, as the two-pass argument requires; only L2
and everything below it change.

## Write policies

Implemented in `cachesim/hierarchy.py` (`Hierarchy.access`, `Level`). Two orthogonal keys
per level, following Hennessy and Patterson, "Computer Architecture: A Quantitative
Approach", 6th ed., 2017, Appendix B.1.

`write_policy` decides what happens **after** a store is applied at a level:

* `write-back` (default) marks the line dirty and tells the level below only when the
  line is evicted.
* `write-through` applies the store here **and** sends a duplicate down as a store, which
  the level below handles under its own policy. A write-through level therefore never
  holds a dirty line, and a store that passes the last level is a DRAM write. The
  duplicates are untimed and counted in `write_throughs`.

`write_allocate` decides what a write **miss** does:

* `true` (default) fetches the line into this level, so the store is applied here.
* `false` leaves this level untouched and passes the store to the level below, which sees
  a store rather than a line fill. Counted in `write_bypasses`. Loads are unaffected.

The write intent travels down the probe walk until some level takes the store. That level
applies it (marking the line dirty unless it is write-through), and every level below
sees the remainder of the access as a plain line fill. If no level takes it, the store
becomes a DRAM write counted in `dram_demand_writes`, and the block is never fetched.

**What is charged.** A store still looking for a level to take it is the access itself,
so it is charged each level's hit time, and the memory access time if it gets that far,
exactly as a load miss is. Traffic generated *behind* an already-satisfied access —
write-back evictions, write-through duplicates, prefetches — is counted but charged no
time, as if absorbed by write buffers. That is what keeps every level's access count
equal to the miss count of the level above, and hence keeps analytic and measured AMAT
identical.

The four combinations, measured on the same trace through the same two-level hierarchy:

```bash
python - <<'PY'
from cachesim import Hierarchy, parse_trace

accesses = list(parse_trace("traces/random.trace"))
head = ("write_policy", "allocate", "wr hits", "wr misses", "thru", "bypass", "L1 wb", "DRAM wr")
print("%-14s %8s %8s %9s %6s %6s %6s %7s" % head)
for policy in ("write-back", "write-through"):
    for allocate in (True, False):
        level = {"name": "L1", "size": 32768, "block_size": 64, "associativity": 4,
                 "hit_time": 4, "write_policy": policy, "write_allocate": allocate}
        below = {"name": "L2", "size": 262144, "block_size": 64, "associativity": 8,
                 "hit_time": 12}
        h = Hierarchy.from_config({"memory_access_time": 100, "levels": [level, below]})
        for addr, is_write in accesses:
            h.access(addr, is_write)
        c = h.levels[0]
        print("%-14s %8s %8d %9d %6d %6d %6d %7d" % (
            policy, allocate, c.cache.write_hits, c.cache.write_misses,
            c.write_throughs, c.write_bypasses, c.cache.writebacks, h.dram_writes))
PY
```

```text
write_policy   allocate  wr hits wr misses   thru bypass  L1 wb DRAM wr
write-back         True       24     15144      0      0  15027   14115
write-back        False       27     15141      0  15141     26   14115
write-through      True       24     15144  15168      0      0   14115
write-through     False       27     15141     27  15141      0   14115
```

Reading the rows: with `write-back` and `write_allocate`, every store is absorbed at L1
and leaves later as one of 15,027 write-backs. With `write_allocate: false`, 15,141 write
misses are passed straight down, so only the 27 write hits ever dirty a line at L1 and it
writes back 26. With `write-through`, L1 holds no dirty data at all and forwards one
duplicate per store it applied: 15,168 with allocation (24 hits plus 15,144 misses), 27
without. The write-miss count differs by three between the allocating and non-allocating
rows because allocation changes which lines are resident.

On this trace and geometry, DRAM writes are 14,115 in all four rows: the route modified
data takes through the hierarchy changes, the amount of it that reaches memory does not.

## Write-back propagation

Implemented in `cachesim/hierarchy.py` (`_write_back`, `_handle_eviction`,
`_write_to_memory`, `flush`).

`_write_back(level_index, block)` is the single downward path for modified data, whether
it came from a write-back eviction above, from a write-through duplicate, or from a store
no level above allocated for. The receiving level applies its own policy:

* **write-back**: mark the resident copy dirty, or, if the level no longer holds the
  block, allocate it dirty (`writeback_allocations`). An eviction caused by that
  allocation is handled recursively. The data stops here.
* **write-through**: the level never holds dirty data, so it allocates a clean line — if
  it allocates on write misses at all — and the data continues to the next level.
* **no-write-allocate on a miss**: nothing is installed and the data continues down.

Data that runs past the last level becomes one DRAM write. Nothing on this path is
charged simulated time.

A level that allocates for data arriving from above installs the block **without**
fetching it from below. The model tracks block presence rather than bytes, so the
partial-line read real hardware would issue has no effect on any counter it keeps.

`_handle_eviction` decides where a line leaving a level goes: into the next level down if
that level is exclusive (clean or dirty), otherwise down as a write-back if it is dirty,
and nowhere at all if it is clean. If the evicting level is inclusive it then
back-invalidates the block out of every level above it.

**Dirty-data conservation.** Modified data is never duplicated and never dropped: every
line written out of one level is accounted for by the level below it, and everything
leaving the last level reaches DRAM. In a plain write-back, write-allocate, NINE
hierarchy that is the pair of identities `writebacks_received[i] == writebacks[i-1]` and
`dram_writes == writebacks[-1]`, which `cachesim run --check` verifies
(`cachesim/invariants.py`).

`flush()` writes every dirty line back to DRAM and cleans it, top-down, leaving the lines
resident, and returns the number of DRAM writes it performed. It increments both sides of
the identity, so conservation survives it. Use it at the end of a run to account for
modified data still sitting in the caches.

```bash
python - <<'PY'
from cachesim import Hierarchy, load_config, parse_trace

h = Hierarchy.from_config(load_config("configs/default.json"))
for addr, is_write in parse_trace("traces/random.trace"):
    h.access(addr, is_write)


def show(when):
    row = [when]
    for level in h.levels:
        row.append("%s wb %d / recv %d" % (
            level.cache.name, level.cache.writebacks, level.cache.writebacks_received))
    row.append("DRAM writes %d" % h.dram_writes)
    print("  ".join(row))


show("after the trace ")
extra = h.flush()
show("after flush     ")
print("flush wrote %d dirty blocks to DRAM" % extra)
PY
```

```text
after the trace   L1 wb 15027 / recv 0  L2 wb 14115 / recv 15027  L3 wb 5823 / recv 14115  DRAM writes 5823
after flush       L1 wb 15161 / recv 0  L2 wb 15113 / recv 15161  L3 wb 14833 / recv 15113  DRAM writes 14833
flush wrote 9010 dirty blocks to DRAM
```

Both lines satisfy the identity exactly, and the 9,010 blocks the flush drained are the
modified data the trace left behind in the three levels.

The one way to break the identity from outside is to call `Cache.invalidate` directly:
the removed dirty line is counted as written back with nothing below to receive it. The
invariant checker's message points at the invalidation count when that is what happened.

## Inclusion policies

Implemented in `cachesim/hierarchy.py` (`_handle_eviction`, `_back_invalidate`,
`check_inclusion`). Terminology follows Baer and Wang, "On the Inclusion Properties for
Multi-Level Cache Hierarchies", ISCA 1988.

`inclusion` is declared on the **lower** level of a pair and names the relation it
maintains with the level above it. The first level must be `nine`, since it has no level
above.

### nine

Non-inclusive non-exclusive, the default. Fills install the block in every level that
missed and nothing is enforced afterwards, so a lower level may drop a block an upper
level still holds.

### inclusive

The level is a superset of the level above. Whenever it drops a block — replacement by a
demand fill, replacement by a write-back allocation, or an invalidation — the block is
**back-invalidated** out of every level above it, counted in `back_invalidations` at the
level whose line is removed.

The walk goes from the level closest to the evicting level upwards, so that if several
levels somehow hold dirty copies, the one closest to the core is written down last and
wins. A dirty copy found above is written to the level **below** the evicting level, or
to DRAM, because the evicting level is losing the block on this same eviction and cannot
hold the data.

Back-invalidation is what lets an inclusive last level filter coherence traffic on behalf
of the whole hierarchy, and it is also why an inclusive level that is not much larger
than the level above destroys upper-level hits. The mechanism is visible in four
accesses, with a 2-way L1 over a 2-way L2 of the same capacity:

```bash
python - <<'PY'
from cachesim import Hierarchy

NAMES = "ABCD"


def residency(inclusion):
    config = {
        "memory_access_time": 100,
        "levels": [
            {"name": "L1", "size": 128, "block_size": 64, "associativity": 2, "hit_time": 1},
            {"name": "L2", "size": 128, "block_size": 64, "associativity": 2,
             "hit_time": 10, "inclusion": inclusion},
        ],
    }
    h = Hierarchy.from_config(config)
    print("  %-8s %-7s %-5s %-5s %s" % ("access", "cycles", "L1", "L2", "back-inv"))
    for name in "ABAC":
        cycles = h.access(NAMES.index(name) * 64)
        held = ["".join(sorted(NAMES[b] for _, _, b, _ in lv.cache.lines())) for lv in h.levels]
        print("  %-8s %-7d %-5s %-5s %d" % (
            name, cycles, held[0] or "-", held[1] or "-", h.levels[0].back_invalidations))


for inclusion in ("nine", "inclusive"):
    print("L2 %s:" % inclusion)
    residency(inclusion)
PY
```

```text
L2 nine:
  access   cycles  L1    L2    back-inv
  A        111     A     A     0
  B        111     AB    AB    0
  A        1       AB    AB    0
  C        111     AC    BC    0
L2 inclusive:
  access   cycles  L1    L2    back-inv
  A        111     A     A     0
  B        111     AB    AB    0
  A        1       AB    AB    0
  C        111     BC    BC    1
```

The third access hits in L1 and never reaches L2, so it does not refresh L2's recency
order: at L2, `A` is still the least recently used block. When `C` is filled, L2 evicts
`A` — and, being inclusive, back-invalidates it out of L1, which was holding a block the
program had just used. NINE keeps it.

How much this costs depends on the capacity ratio between the two levels. On a 32 KB L1
over a 64 KB L2, a ratio of two, it is measurable:

```bash
python - <<'PY'
from cachesim import Hierarchy, parse_trace

accesses = list(parse_trace("traces/random.trace"))
print("%-10s %9s %9s %9s %10s %9s" % (
    "inclusion", "L1 miss", "L2 miss", "back-inv", "DRAM rd", "AMAT"))
for inclusion in ("nine", "inclusive", "exclusive"):
    l1 = {"name": "L1", "size": 32768, "block_size": 64, "associativity": 4, "hit_time": 4}
    l2 = {"name": "L2", "size": 65536, "block_size": 64, "associativity": 4,
          "hit_time": 12, "inclusion": inclusion}
    h = Hierarchy.from_config({"memory_access_time": 100, "levels": [l1, l2]})
    for addr, is_write in accesses:
        h.access(addr, is_write)
    h.check_inclusion()
    print("%-10s %9d %9d %9d %10d %9.3f" % (
        inclusion, h.levels[0].cache.misses, h.levels[1].cache.misses,
        h.levels[0].back_invalidations, h.dram_reads, h.measured_amat()))
PY
```

```text
inclusion    L1 miss   L2 miss  back-inv    DRAM rd      AMAT
nine           59893     59769         0      59769   115.594
inclusive      59893     59775      3766      59775   115.604
exclusive      59893     59644         0      59644   115.385
```

The inclusion penalty is a function of that ratio, not of the workload alone. Widen the
ratio to the 8x steps of `configs/inclusive.json` (32 KB, 256 KB, 2 MB) and inclusion
becomes free: the report's `back-invalidated` line, which is printed only when the counter
is non-zero, does not appear at all.

```bash
cachesim run traces/random.trace --config configs/inclusive.json | grep -c back-invalidated
```

```text
0
```

### exclusive

The level holds only blocks the level above does not, acting as a victim cache for it:

* A demand fetch passing through the level does not fill it; only the level above is
  filled. The exception is a store the exclusive level takes itself.
* A hit here moves the block **up**: it is invalidated here with `count_writeback=False`,
  carrying its dirty bit with it, and allocated in the level above. This is the one case
  in which a dirty line leaves a cache without being written towards memory.
* Every line the level above evicts, clean **or** dirty, is allocated here, replacing the
  dirty-only write-back across that boundary. The insertion may evict a line here, which
  follows this level's own rule: written back if dirty, or passed on to the next
  exclusive level.

Total capacity is the sum of the two levels rather than the larger of them, at the cost
of moving every victim across the boundary. The move-up is visible in four accesses
through a 1-way L1 over a 2-way L2:

```bash
python - <<'PY'
from cachesim import Hierarchy

NAMES = "ABCD"


def residency(inclusion):
    config = {
        "memory_access_time": 100,
        "levels": [
            {"name": "L1", "size": 64, "block_size": 64, "associativity": 1, "hit_time": 1},
            {"name": "L2", "size": 128, "block_size": 64, "associativity": 2,
             "hit_time": 10, "inclusion": inclusion},
        ],
    }
    h = Hierarchy.from_config(config)
    print("  %-9s %-6s %-6s %-6s" % ("access", "cycles", "L1", "L2"))
    for name in "ABCA":
        cycles = h.access(NAMES.index(name) * 64)
        held = []
        for level in h.levels:
            held.append("".join(sorted(NAMES[b] for _, _, b, _ in level.cache.lines())))
        print("  %-9s %-6d %-6s %-6s" % (name, cycles, held[0] or "-", held[1] or "-"))


for inclusion in ("nine", "exclusive"):
    print("L2 %s:" % inclusion)
    residency(inclusion)
PY
```

```text
L2 nine:
  access    cycles L1     L2    
  A         111    A      A     
  B         111    B      AB    
  C         111    C      BC    
  A         111    A      AC    
L2 exclusive:
  access    cycles L1     L2    
  A         111    A      -     
  B         111    B      A     
  C         111    C      AB    
  A         11     A      BC    
```

Under NINE the first three accesses put each block in both levels, L2 evicts `A` to make
room for `C`, and the fourth access goes all the way to DRAM for 111 cycles. Under
exclusion the two levels never hold the same block, so `A` is still in L2 when it is
asked for again: the access costs 11 cycles, `A` moves up into L1, and L1's victim `C`
takes its place in L2.

### check_inclusion

`Hierarchy.check_inclusion()` asserts the invariant for every adjacent pair: an inclusive
level must hold every block the level above holds, and an exclusive level must share no
block with it. NINE pairs are unconstrained and always pass. It raises `AssertionError`
naming the offending blocks. It is intended for tests — run a random access stream
through a hierarchy and then call it — and is used that way throughout
`tests/test_inclusion.py`, in `tests/test_prefetch.py`, and in `tests/test_hierarchy.py`
against every file in `configs/`.

## Timing

Implemented in `cachesim/hierarchy.py` (`Hierarchy.access`, `amat`, `measured_amat`).

The timing model is serial and fixed-latency. There is no overlap between misses, no
queueing, and no bandwidth limit beyond the per-level transfer term below.

An access is charged:

* **the hit time of every level it probes**, hit or miss. A level that is not reached
  costs nothing.
* **the memory access time** if it missed every level, or if it is a store that no level
  allocated for.
* **the transfer time of every level it fills**, `ceil(block_size / bus_width)` cycles
  per fill, where `bus_width` is the width in bytes per cycle of the link carrying blocks
  from the level below **into** that level. The default, `null`, models an infinitely
  wide link and adds nothing.

Nothing else is charged. Write-back evictions, write-through duplicates and prefetches
are counted but untimed, on the assumption that write buffers and a non-blocking fetch
path absorb them.

The transfer term is a property of the level, computed once, and the report prints it in
the level header. `configs/dram_row_buffer.json` gives L3 a 16-byte bus and 64-byte
blocks:

```bash
cachesim run traces/sequential.trace --config configs/dram_row_buffer.json | grep "^--- L3"
```

```text
--- L3: 2.0 MB, 64 B blocks, 16-way, LRU, hit time 40 cyc, 16 B/cyc bus (4 cyc/block) ---
```

`Hierarchy.access` returns the cycles the access took, and the total is split into
`read_cycles` and `write_cycles`, from which `read_amat()` and `write_amat()` are the
measured averages per load and per store.

### AMAT

The analytic average memory access time nests: a level's miss penalty is the AMAT of the
rest of the hierarchy. With levels numbered from 0 and `transfer_i` the cycles a block
takes to cross the link into level i:

```text
AMAT         = hit_time_0 + miss_rate_0 * penalty_0
penalty_i    = hit_time_{i+1} + transfer_i + miss_rate_{i+1} * penalty_{i+1}
penalty_last = memory_access_time + transfer_last
```

So every level pays for its own fill however deep the data came from. `amat()` computes
it bottom-up: start at the memory access time and, for each level from the last upwards,
take `hit_time + miss_rate * (transfer + penalty_below)`. The miss rate used is the
**local** miss rate, misses over accesses that reached that level.

For the default three-level hierarchy with no bus widths the transfer terms vanish and
the formula is:

```text
AMAT = 4 + mr_L1 * (12 + mr_L2 * (40 + mr_L3 * 100))
```

```bash
python - <<'PY'
from cachesim import Hierarchy, load_config, parse_trace

h = Hierarchy.from_config(load_config("configs/default.json"))
for addr, is_write in parse_trace("traces/matmul_naive.trace"):
    h.access(addr, is_write)

mr = [level.cache.miss_rate for level in h.levels]
ht = [level.hit_time for level in h.levels]
by_hand = ht[0] + mr[0] * (ht[1] + mr[1] * (ht[2] + mr[2] * 100))
print("miss rates      %.8f %.8f %.8f" % tuple(mr))
print("by hand         %.6f" % by_hand)
print("amat()          %.6f" % h.amat())
print("measured_amat() %.6f" % h.measured_amat())
print("cycles          %d = 4*%d + 12*%d + 40*%d + 100*%d" % (
    h.total_time, h.levels[0].cache.accesses, h.levels[1].cache.accesses,
    h.levels[2].cache.accesses, h.dram_reads))
PY
```

```text
miss rates      0.05241887 0.05503009 1.00000000
by hand         5.032873
amat()          5.032873
measured_amat() 5.032873
cycles          2679904 = 4*532480 + 12*27912 + 40*1536 + 100*1536
```

### When analytic equals measured

`measured_amat()` is `total simulated cycles / accesses` and is the ground truth. The two
figures agree **exactly** for an allocate-on-miss NINE hierarchy, because each level is
then probed once per miss of the level above and filled once per miss of its own, which
is precisely what the formula assumes. The cycle decomposition printed above is the same
statement written as a sum.

Two features break the "filled once per miss" half of that while leaving the "probed once
per miss above" half intact, and both make the transfer term an **upper bound**:

* an **exclusive** level, which is filled by the level above's evictions rather than by
  its own misses;
* a **no-write-allocate** level, which declines write misses.

The hit-time and memory terms stay exact in both cases, so `amat()` is never an
underestimate. With `bus_width: null` — the default — there is no transfer term at all
and the identity is exact in every configuration:

```bash
python - <<'PY'
from cachesim import Hierarchy, parse_trace

accesses = list(parse_trace("traces/random.trace"))


def run(l1_extra, l2_extra, bus):
    l1 = {"name": "L1", "size": 2048, "block_size": 64, "associativity": 4, "hit_time": 4}
    l2 = {"name": "L2", "size": 8192, "block_size": 64, "associativity": 4, "hit_time": 12}
    l1.update(l1_extra)
    l2.update(l2_extra)
    if bus is not None:
        l1["bus_width"] = bus
        l2["bus_width"] = bus
    h = Hierarchy.from_config({"memory_access_time": 100, "levels": [l1, l2]})
    for addr, is_write in accesses:
        h.access(addr, is_write)
    return h


cases = (
    ("all NINE, write-allocate", {}, {}),
    ("L2 exclusive", {}, {"inclusion": "exclusive"}),
    ("L1 no-write-allocate", {"write_allocate": False}, {}),
)
print("%-26s %6s %12s %12s %9s" % ("configuration", "bus", "analytic", "measured", "gap"))
for label, l1_extra, l2_extra in cases:
    for bus in (None, 16):
        h = run(l1_extra, l2_extra, bus)
        print("%-26s %6s %12.6f %12.6f %9.6f" % (
            label, bus if bus else "null", h.amat(), h.measured_amat(),
            h.amat() - h.measured_amat()))
PY
```

```text
configuration                 bus     analytic     measured       gap
all NINE, write-allocate     null   115.963400   115.963400  0.000000
all NINE, write-allocate       16   123.961467   123.961467  0.000000
L2 exclusive                 null   115.953400   115.953400  0.000000
L2 exclusive                   16   123.951067   119.952867  3.998200
L1 no-write-allocate         null   115.964000   115.964000  0.000000
L1 no-write-allocate           16   123.962267   122.951133  1.011133
```

A memory model whose latency depends on the address stream is handled by taking the DRAM
term from the **measured** average latency of the charged accesses, since no single
number describes it in advance. The identity still holds; the analytic figure is then
analytic only in the miss rates.

### Load and store split

`read_amat()` and `write_amat()` are the measured averages over loads and over stores.
They can differ sharply.

```bash
cachesim run traces/matmul_naive.trace --config configs/default.json | grep -A2 "AMAT (measured)"
```

```text
AMAT (measured)         :    5.033 cycles   (2,679,904 cycles / 532,480 accesses)
  loads                 :    4.948 cycles   (2,614,368 cycles / 528,384 loads)
  stores                :   16.000 cycles   (65,536 cycles / 4,096 stores)
```

Every one of the 4,096 stores misses L1 and is satisfied at L2, which costs exactly
4 + 12 cycles and gives a store average with no variance at all; 95.5% of loads hit L1.
Note that L2 records those 4,096 accesses as **reads**: L1 took the store under its own
write-allocate policy, so below L1 the access is an ordinary line fill.

## Prefetchers

Implemented in `cachesim/prefetch.py`, driven from `cachesim/hierarchy.py`
(`_run_prefetchers`, `_prefetch_fill`).

A level may name a `prefetcher`, which watches the demand references that reach that
level and, for each one, may name a single block to fetch in advance (degree 1). A
prefetcher at level i sees a reference only if the access got that far, so an L1
prefetcher watches every reference while an L2 prefetcher watches only L1 misses. A
prediction the level already holds, or a negative block number, is dropped.

### next-line

Tagged next-line, or one-block-lookahead: on a demand miss for block b, or on the first
demand hit to a line that was itself prefetched, fetch b+1. The tag is what keeps a
stream running — without it the prefetcher would fetch one block ahead of the first miss
and then stop, because the subsequent references all hit. It is the cheapest prefetcher
there is, with no history and no table, and it captures sequential streams completely and
everything else not at all. Smith, "Cache Memories", ACM Computing Surveys 14(3), 1982,
section 2.4; the tagged variant is due to Gindele, "Buffer Block Prefetching Method", IBM
Technical Disclosure Bulletin 20(2), 1977.

### stride

Per-region stride detection with a 2-bit confidence counter. A direct-mapped, tagged
table of `entries` slots (256 by default) is indexed by `block >> region_shift`, which at
the default shift of 6 is a 4 KB region for 64-byte blocks. Each slot holds the region
tag, the last block referenced there, the last delta between consecutive references
there, and a 2-bit saturating confidence. A repeated delta raises the confidence; a
different delta replaces the recorded stride and drops the confidence to "seen once". At
confidence 2 of 3 the prefetcher predicts `block + stride`. A repeat of the same block
predicts nothing.

The reference-prediction table of Chen and Baer, "Effective Hardware-Based Data
Prefetching for High-Performance Processors", IEEE Transactions on Computers 44(5), 1995,
is indexed by program counter. An address trace carries no program counters, so there is
nothing to index by but the address region, and three limits follow. Two loops striding
through the same region interleave into one slot and cancel out, where a PC-indexed table
would track them separately. A stride wider than a region is never detected, because
consecutive references land in different slots, each of which sees only its first
reference; this is the common case for large strided scans, and it is why `region_shift`
is a parameter. And the table is direct-mapped and tagged, so aliasing regions evict each
other's history, exactly as the hardware structure does.

### What is counted, and what is not

Prefetch traffic is deliberately kept out of the demand counters. Two accounting rules:

* **Prefetch fills are untimed.** A prefetch is assumed to overlap with useful work and is
  charged no cycles. That is optimistic — a real prefetch competes for MSHRs, bus cycles
  and DRAM banks — which is why the model gets the *pollution* cost right (a prefetched
  line evicting a useful one) and cannot ask the *timeliness* question (did it arrive
  before the demand?) at all.
* **Prefetch lookups at lower levels are not accesses.** They are recorded as
  `prefetch_probes` and `prefetch_probe_hits`, and they leave the replacement state of
  those levels untouched. Every level's miss rate therefore remains a statement about
  demand references, and the AMAT identity stays exact.

A prefetched line carries a `prefetched` flag, set only at the issuing level. The first
demand hit on it clears the flag and counts a `prefetch_hit`; a line that leaves the level
with the flag still set counts a `prefetch_evicted_unused`. The two derived rates are:

```text
accuracy = prefetch_hits / prefetches_issued
coverage = prefetch_hits / (prefetch_hits + demand misses at the level)
```

Accuracy is how much of the extra traffic was worth fetching; coverage is the share of
the misses the level would otherwise have taken that the prefetcher turned into hits.
Both are 0.0 when the denominator is zero.

```bash
cachesim run traces/sequential.trace --config configs/prefetch_stride.json \
  | grep -E "^(Total|DRAM prefetch|---)|accesses    |  misses  |prefetch|accuracy"
```

```text
Total accesses :       65,536   (reads 58,983 / writes 6,553)
DRAM prefetches:        4,096   (speculative fetches that missed every level; untimed)
--- L1: 32.0 KB, 64 B blocks, 4-way, LRU, hit time 4 cyc, next-line prefetch ---
  accesses    :       65,536
  misses      :            2   (reads 2 / writes 0)
  prefetches issued   :  8,192   (useful 8,190, evicted unused 1)
    accuracy  :  99.98%   (useful / issued)    coverage :  99.98%   (useful / (useful + demand misses))
--- L2: 256.0 KB, 64 B blocks, 8-way, LRU, hit time 12 cyc, stride prefetch ---
  accesses    :            2
  misses      :            2   (reads 2 / writes 0)
  prefetches issued   :      0   (useful 0, evicted unused 0)
    accuracy  :   0.00%   (useful / issued)    coverage :   0.00%   (useful / (useful + demand misses))
  prefetch lookups    :  8,192   (from a level above; 4,088 found the block here)
--- L3: 2.0 MB, 64 B blocks, 16-way, LRU, hit time 40 cyc ---
  accesses    :            2
  misses      :            1   (reads 1 / writes 0)
  prefetch lookups    :  4,104   (from a level above; 8 found the block here)
```

L2's `accesses` is 2, the number of L1 demand misses; the 8,192 lookups the L1 prefetcher
sent through it are counted separately as `prefetch lookups`, of which 4,088 found the
block already there. The same run on `traces/random.trace` is the opposite case: 59,874
prefetches issued at 0.10% accuracy, 59,558 of them evicted before any demand reference
touched them. That is the pollution cost the model does represent.

Note that the three-C classification is computed against a shadow cache fed only by
demand references, so at a level with an effective prefetcher the classification becomes
hard to read: on the run above, L1's shadow still misses 8,192 times against 2 real
misses, and the aggregate conflict figure goes to -8,190. See
[Miss classification](#miss-classification).

## DRAM

Implemented in `cachesim/dram.py`. Below the last cache level sits a `MemoryModel`, which
answers one block-granular read or write and returns the cycles it took.

### ConstantMemory

Every access costs `latency` cycles, whatever the address. This is the
`memory_access_time` shorthand and the default. It makes AMAT a closed-form function of
the miss rates alone, which is exactly why it hides the difference between a sequential
and a random miss stream. When the latency is constant, the access path adds it inline
and never calls the model.

### RowBufferMemory

An open-page DRAM: each bank holds one activated row in its sense amplifiers, and an
access that finds its row already open is far cheaper than one that has to activate a new
row. Modelled after the open-page policy described by Rixner, Dally, Kapasi, Mattson and
Owens, "Memory Access Scheduling", ISCA 2000, and by Jacob, Ng and Wang, "Memory Systems:
Cache, DRAM, Disk", Morgan Kaufmann, 2007, chapter 13.

```text
row  = block_addr // row_size
bank = row % banks
```

Consecutive rows land in different banks, so a scan long enough to leave one row finds
the next in a bank of its own. An access whose row is already open costs `row_hit`
cycles; otherwise it costs `row_miss` and that row becomes the open one for its bank.
There is no separate precharge penalty for the row it displaced: `row_miss` stands for
precharge plus activate plus column access together. All banks start closed, so the first
access to each is a row miss. `row_miss` must be at least `row_hit`. Reads and writes
cost the same.

Every access consults the row buffer and updates it, but only accesses the hierarchy
actually charges — demand fetches, and stores no cache level allocated for — enter
`average_latency()`, which is the DRAM term the analytic AMAT uses. Write-backs and
prefetches occupy a bank and shift the open rows, which is what makes them interfere with
demand traffic, but they are drained off the critical path and are not charged.

```bash
cachesim run traces/sequential.trace --config configs/dram_row_buffer.json \
  | grep -E "DRAM model|row buffer|average latency|^AMAT"
```

```text
DRAM model              : row-buffer (row_size 8192, banks 8, row_hit 40, row_miss 100)
  row buffer            :  99.22% hits   (4,064 hits / 32 misses)
  average latency       :   40.469 cycles   (measured; the analytic AMAT below uses it)
AMAT (analytic formula) :   10.779 cycles
AMAT (measured)         :   10.779 cycles   (706,432 cycles / 65,536 accesses)
```

```bash
cachesim run traces/conflict.trace --config configs/dram_row_buffer.json \
  | grep -E "DRAM model|row buffer|average latency|^AMAT"
```

```text
DRAM model              : row-buffer (row_size 8192, banks 8, row_hit 40, row_miss 100)
  row buffer            :   0.00% hits   (0 hits / 8,192 misses)
  average latency       :  100.000 cycles   (measured; the analytic AMAT below uses it)
AMAT (analytic formula) :   23.500 cycles
AMAT (measured)         :   23.500 cycles   (1,540,096 cycles / 65,536 accesses)
```

The two runs have the same 100-cycle worst case and differ by a factor of 2.5 in what a
miss actually costs. `conflict.trace` walks four streams placed 1 MB apart; 1 MB is 128
rows, and 128 % 8 banks is 0, so all four streams land in the same bank and evict each
other's open row on every access.

## Victim cache

Implemented in `cachesim/cache.py` (`VictimBuffer`, and the buffer-aware paths of
`Cache`). Jouppi, "Improving Direct-Mapped Cache Performance by the Addition of a Small
Fully-Associative Cache and Prefetch Buffers", ISCA 1990.

`victim_cache` gives a level a small fully-associative LRU buffer beside its tag array.
The buffer is part of the level:

* Every line the array replaces is pushed into the buffer instead of leaving the level.
  What `allocate` returns as the eviction is the line the **buffer** pushed out, if any,
  so a dirty line moving from array to buffer is not a write-back.
* `probe` consults the buffer after the array. A block found there counts as a **hit at
  the level** and as a `victim_hit`, and is swapped back into the array immediately. The
  line that swap displaces goes into `pending_eviction`, which the hierarchy drains with
  `take_pending_eviction()` after every probe.
* `contains`, `is_dirty`, `mark_dirty`, `clean`, `invalidate`, `is_prefetched`,
  `clear_prefetched` and `lines()` all see the buffer. `lines()` yields buffered lines
  with a set index and way of -1, since they are as resident as any other.

Counting a victim-buffer hit as a hit at the level is deliberate: the buffer is probed
with the tag array and charged the level's hit time, so the access does not descend, the
AMAT identity survives, and the level's miss rate is the array-plus-buffer miss rate that
Jouppi reports.

A handful of entries removes the conflict misses of a low-associativity cache without
widening every set. On the `conflict` workload through a 32 KB direct-mapped level, the
threshold is exact:

```bash
python - <<'PY'
from cachesim import Cache, parse_trace

accesses = list(parse_trace("traces/conflict.trace"))
print("%8s %10s %12s %10s" % ("entries", "misses", "victim hits", "miss rate"))
for entries in (None, 1, 2, 3, 4, 8):
    c = Cache("L1", 32768, 64, 1, track_3c=False, victim_entries=entries)
    for addr, is_write in accesses:
        c.access(addr, is_write)
    print("%8s %10d %12d %9.2f%%" % (
        entries or 0, c.misses, c.victim_hits, 100 * c.miss_rate))
PY
```

```text
 entries     misses  victim hits  miss rate
       0      65536            0    100.00%
       1      65536            0    100.00%
       2      65536            0    100.00%
       3       8192        57344     12.50%
       4       8192        57344     12.50%
       8       8192        57344     12.50%
```

The workload rotates four streams through one set, so a three-entry buffer is the first
that can hold the three displaced blocks; below that it removes nothing at all. At three
entries the miss count drops to 8,192, which is the compulsory-miss floor a cache of any
size would also hit.

## Miss classification

Implemented in `cachesim/cache.py` (`_classify_miss`, `_update_shadow`, and the
`*_aggregate` properties). Enabled per level by `track_3c`, which defaults to true.

Two taxonomies are kept, because they differ. Both are computed by running a **shadow
fully-associative LRU cache of identical capacity** (`num_blocks` entries) alongside the
real one, fed the same demand references, plus a set of every block ever referenced.

### Per-reference classification

Each miss is labelled as it happens:

| Label | Rule |
|-------|------|
| compulsory | the block has never been referenced before |
| conflict | the shadow cache still holds the block, so the miss is due to the set mapping |
| capacity | the shadow cache had already evicted the block |

The three counters sum to `misses` by construction.

### Aggregate classification

The decomposition of Hill and Smith, "Evaluating Associativity in CPU Caches", IEEE
Transactions on Computers 38(12), 1989, also used by Hennessy and Patterson, is defined
on totals rather than on individual references:

```text
compulsory (aggregate) = compulsory
capacity   (aggregate) = shadow_misses - compulsory
conflict   (aggregate) = misses - shadow_misses
```

These also sum to `misses`.

### Where the two differ

They differ by exactly the `anti_conflict_hits`: references that **hit** the
set-associative cache while **missing** the shadow cache. Each such reference adds one to
per-reference conflict and subtracts one from per-reference capacity, relative to the
aggregate figures:

```text
capacity_aggregate = capacity + anti_conflict_hits
conflict_aggregate = conflict - anti_conflict_hits
shadow_misses      = compulsory + capacity + anti_conflict_hits
```

`cachesim run --check` verifies all three identities.

The consequence is that **aggregate conflict can be negative**, when the set mapping
happens to beat fully-associative LRU. That is not a bug; it is the aggregate
decomposition reporting that a fully associative cache of the same capacity would have
missed more. It is common on strided workloads: on `matmul_naive` through the default
hierarchy, L1 takes 27,912 misses where the shadow takes 33,792, so aggregate conflict is
-5,880 while per-reference conflict is 22,344, and the report prints both side by side.

### A four-access example

Two sets of one way each, so the level holds two blocks. Blocks 0 and 1 are referenced,
then block 3 evicts block 1 from set 1, and block 0, untouched in set 0 all along, is
referenced again.

```bash
python - <<'PY'
from cachesim import Cache

c = Cache("L1", size=128, block_size=64, associativity=1)   # 2 sets, 1 way
for addr in (0, 64, 192, 0):                                # blocks 0, 1, 3, 0
    print(addr, c.block_of(addr), c.set_of(c.block_of(addr)), "hit" if c.access(addr) else "miss")
print("hits", c.hits, "misses", c.misses)
print("per-reference : compulsory", c.compulsory_misses,
      "capacity", c.capacity_misses, "conflict", c.conflict_misses)
print("aggregate     : compulsory", c.compulsory_misses,
      "capacity", c.capacity_misses_aggregate, "conflict", c.conflict_misses_aggregate)
print("shadow misses", c.shadow_misses, "anti-conflict hits", c.anti_conflict_hits)
PY
```

```text
0 0 0 miss
64 1 1 miss
192 3 1 miss
0 0 0 hit
hits 1 misses 3
per-reference : compulsory 3 capacity 0 conflict 0
aggregate     : compulsory 3 capacity 1 conflict -1
shadow misses 4 anti-conflict hits 1
```

The two-block shadow cache holds blocks 1 and 3 by the end, having evicted block 0 as its
least recently used entry, so the final reference is a shadow miss and a real hit — one
anti-conflict hit. Per-reference, all three misses are compulsory and nothing else
happened. Aggregate, the same three misses read as three compulsory plus one capacity
minus one conflict: the fully associative twin would have taken four misses on this
stream, one more than the direct-mapped cache did.

### The FIFO and random caveat

The shadow cache is always fully associative **LRU**, whatever policy the real level runs.
With a non-LRU policy the conflict bucket therefore absorbs replacement-policy divergence
as well as set-mapping collisions, and it can be non-zero even in a fully associative
cache, where the classical definition admits no conflict misses at all.

```bash
python - <<'PY'
from cachesim import Cache

BLOCK, WAYS = 64, 4
refs = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]   # Belady, Nelson and Shedler, 1969

print("%-8s %8s %8s %10s %8s %8s %8s" % (
    "policy", "misses", "shadow", "compulsory", "cap/ref", "con/ref", "con/agg"))
for policy in ("lru", "fifo", "mru"):
    cache = Cache("FA", BLOCK * WAYS, BLOCK, WAYS, policy=policy)   # one set: fully associative
    for b in refs:
        cache.access(b * BLOCK)
    print("%-8s %8d %8d %10d %8d %8d %8d" % (
        policy, cache.misses, cache.shadow_misses, cache.compulsory_misses,
        cache.capacity_misses, cache.conflict_misses, cache.conflict_misses_aggregate))
PY
```

```text
policy     misses   shadow compulsory  cap/ref  con/ref  con/agg
lru             8        8          5        3        0        0
fifo           10        8          5        3        2        2
mru             6        8          5        0        1       -2
```

The cache has one set, so no reference can collide with another in the mapping sense, yet
FIFO reports two conflict misses under both taxonomies. They are the two misses by which
FIFO falls short of LRU on this string. Read the conflict bucket for a `fifo`, `random`,
`nru`, `mru`, `lfu` or RRIP level as "misses attributable to the set mapping **or** to
the replacement choice", and compare against `cachesim policies` to separate the two.

Two further reading notes. The classification counts demand references only, so a level
with a prefetcher reports a shadow miss count that ignores the prefetches (see
[Prefetchers](#prefetchers)). And the shadow cache is fully associative, so its miss count
is unaffected by the index function: a 32 KB 4-way level on `matmul_naive` takes 27,912
misses under `modulo` and 5,470 under `xor`, but reports 1,536 compulsory misses and
33,792 shadow misses either way. The whole difference between the two mappings lands in
the conflict bucket.

## Warm-up and reset_stats

Implemented in `Hierarchy.reset_stats`, `Level.reset_stats`, `Cache.reset_stats`, and
`cachesim.run_trace`.

`cachesim run --warmup N` simulates the first N accesses, resets every counter, and then
counts the rest, so the reported figures describe steady-state behaviour rather than cold
caches.

A reset zeroes counters and keeps state. Specifically it keeps:

* cache contents and dirty bits;
* replacement state at every level;
* prefetcher prediction tables (a warm-up should leave the prefetcher trained);
* the DRAM model's open rows;
* the shadow fully-associative cache and, importantly, the set of blocks already seen, so
  a later reference to a block touched before the reset is **not** counted as compulsory.

```bash
cachesim run traces/matmul_naive.trace --warmup 200000 \
  | grep -E "^(Total|DRAM reads|---)|accesses    |  misses  |local miss|compulsory"
```

```text
Total accesses :      332,480   (reads 329,922 / writes 2,558)
DRAM reads     :          631   (0.19% of all accesses missed every level)
--- L1: 32.0 KB, 64 B blocks, 4-way, LRU, hit time 4 cyc ---
  accesses    :      332,480
  misses      :       17,138   (reads 14,580 / writes 2,558)
  local miss rate  :   5.15%   (misses / accesses that reached this level)
    compulsory                       631          631
--- L2: 256.0 KB, 64 B blocks, 8-way, LRU, hit time 12 cyc ---
  accesses    :       17,138
  misses      :          631   (reads 631 / writes 0)
  local miss rate  :   3.68%   (misses / accesses that reached this level)
    compulsory                       631          631
--- L3: 2.0 MB, 64 B blocks, 16-way, LRU, hit time 40 cyc ---
  accesses    :          631
  misses      :          631   (reads 631 / writes 0)
  local miss rate  : 100.00%   (misses / accesses that reached this level)
    compulsory                       631          631
```

332,480 is 532,480 minus the 200,000 warm-up accesses. L1's compulsory count falls from
1,536 to 631 because the blocks touched during the warm-up are remembered, and its local
miss rate falls from 5.24% to 5.15% because the caches start warm.

## What is not modelled

The model is functional — exact hit, miss and eviction behaviour — with a serial,
fixed-latency timing model layered on top. The following are deliberately absent, and a
result that depends on any of them is outside what this simulator can say.

**Overlap and MSHRs.** Misses are serial. There are no miss status holding registers, no
memory-level parallelism, no hit-under-miss or miss-under-miss. A workload whose misses a
real machine would overlap is charged for all of them in sequence, so absolute cycle
counts are pessimistic while the miss counts they are derived from are exact.

**Coherence.** One core, one hierarchy. There are no coherence states, no snoops, no
invalidations from other cores, no false sharing. Back-invalidation from an inclusive
level is a capacity mechanism here, not a coherence one.

**Split instruction and data caches.** The hierarchy is unified at every level. Trace
formats that distinguish an instruction fetch decode it as a read (`cachesim/trace.py`).

**Address translation.** No TLB, no page tables, no page walks, no distinction between
virtual and physical addresses. Addresses in the trace are used exactly as given, so a
virtually indexed cache and a physically indexed one are the same thing here.

**DRAM beyond the row buffer.** `RowBufferMemory` models one open row per bank and
nothing else: no refresh, no bus turnaround between reads and writes, no request
scheduling or reordering, no bank conflicts as a queueing effect (a bank never makes a
request wait, it only changes its latency), and nothing below block granularity.

**Write buffers.** They are assumed, not modelled. Write-backs, write-through duplicates
and prefetches are charged no cycles at all, which is the perfect-write-buffer
assumption; a real buffer of finite depth would eventually stall the access stream.

**Sector and sub-block lines.** A block is present or absent as a whole. There are no
valid bits per sub-block, no partial fills, and a level that allocates for data arriving
from above installs the block without the partial-line read real hardware would issue.

**Access sizes.** The trace is decoded to `(byte address, is_write)` pairs and nothing
else. A reference is modelled as touching only the block containing its first byte, so a
reference straddling a block boundary is charged one block rather than two, and formats
carrying a size field ignore it.

**Multi-block fills and mixed geometries.** Every level must use the same block size,
which is why `parse_config` rejects a hierarchy that mixes them.

**Banked or pipelined cache arrays.** A level's hit time is one number, charged whether
the probe hits or misses, with no port contention and no variation by way or set.

---

Back to [../README.md](../README.md) ·
[configuration.md](configuration.md) ·
[cli.md](cli.md) ·
[workloads.md](workloads.md) ·
[validation.md](validation.md) ·
[results.md](results.md)
