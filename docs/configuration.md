# Configuration reference

A hierarchy is described by a single JSON object: a list of cache levels, outermost
level first, and a model of main memory. The object is passed to `cachesim run
--config FILE` and `cachesim compare --config FILE`, or to `Hierarchy.from_config` in
library code. The analysis subcommands (`sweep`, `policies`, `mrc`, `sets`) build a
single level from flags instead and take no configuration file; see
[docs/cli.md](cli.md).

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4}
  ]
}
```

Five per-level keys are required (`name`, `size`, `block_size`, `associativity`,
`hit_time`); every other key has a default, and the defaults reproduce the model
described in [docs/model.md](model.md) with no inclusion enforcement, write-back
write-allocate caches, no prefetching and no victim buffer. A configuration written
before a feature existed therefore keeps its original behaviour.

Validation is exhaustive and refuses anything it does not recognise: unknown keys are
errors rather than being ignored, so a misspelled key can never be silently dropped.

## Top-level keys

| Key | Type | Default | Allowed values | Semantics |
|-----|------|---------|----------------|-----------|
| `levels` | array of objects | required | one or more level objects | The cache levels, L1 first. Level *i* is the parent of level *i+1*; the last level is backed by main memory. See [docs/model.md](model.md#the-hierarchy). |
| `memory_access_time` | integer | required unless `memory` is given | `>= 0` | Cycles to reach main memory, for the fixed-latency memory model. Shorthand for `"memory": N`. See [docs/model.md](model.md#main-memory). |
| `memory` | integer or object | required unless `memory_access_time` is given | an integer `>= 0`, or an object with `type` `"constant"` or `"row-buffer"` | The main-memory model. The integer form is a fixed latency; the object form selects and parameterises a model. |

`memory_access_time` and `memory` are mutually exclusive, so a configuration can never
state two different memory latencies. `HierarchySpec.to_dict` writes a constant memory
back as `memory_access_time` and any other model as `memory`, and the result parses
again unchanged.

### `memory` object, `"type": "constant"`

| Key | Type | Default | Allowed values | Semantics |
|-----|------|---------|----------------|-----------|
| `type` | string | `"constant"` | `"constant"` | Selects the fixed-latency model. |
| `latency` | integer | required | `>= 0` | Cycles every memory access costs, regardless of address. |

### `memory` object, `"type": "row-buffer"`

Models one open row per bank: an access to the row already open in its bank costs
`row_hit`, any other access costs `row_miss` and opens the new row. See
[docs/model.md](model.md#main-memory).

| Key | Type | Default | Allowed values | Semantics |
|-----|------|---------|----------------|-----------|
| `type` | string | required for this form | `"row-buffer"` | Selects the row-buffer model. |
| `row_size` | integer | `8192` | `>= 1` | Bytes per DRAM row. |
| `banks` | integer | `8` | `>= 1` | Independent banks, each with its own open row. The bank is the row number modulo `banks`. |
| `row_hit` | integer | `40` | `>= 0` | Cycles for an access to the open row. |
| `row_miss` | integer | `100` | `>= 0`, and `>= row_hit` | Cycles for an access that must open a different row. |

## Per-level keys

Levels are validated in order and identified in messages as `levels[0]`, `levels[1]`,
and so on, with the level's `name` appended when it is present.

| Key | Type | Default | Allowed values | Semantics |
|-----|------|---------|----------------|-----------|
| `name` | string | required | non-empty; unique across levels | Label used in the report, the JSON output and error messages. |
| `size` | integer | required | `>= 1`, and a multiple of `block_size * associativity` | Total capacity of the level in bytes. |
| `block_size` | integer | required | `>= 1`, a power of two, identical at every level | Bytes per block, the unit of transfer and of tag storage. See [docs/model.md](model.md#blocks-sets-and-ways). |
| `associativity` | integer | required | `>= 1` | Ways per set; `1` is direct-mapped. The set count is `size / (block_size * associativity)`. |
| `hit_time` | integer | required | `>= 0` | Cycles charged for probing this level, whether the probe hits or misses. See [docs/model.md](model.md#timing-and-amat). |
| `policy` | string | `"lru"` | `brrip`, `drrip`, `fifo`, `lfu`, `lru`, `mru`, `nru`, `opt`, `plru`, `random`, `srrip` | Replacement policy choosing the victim within a set. See [docs/model.md](model.md#replacement-policies). |
| `index` | string | `"modulo"` | `modulo`, `xor` | Set-index function. `modulo` takes the low-order bits of the block number; `xor` XOR-folds the whole block number and requires a power-of-two set count. See [docs/model.md](model.md#set-index-functions). |
| `inclusion` | string | `"nine"` | `nine`, `inclusive`, `exclusive` | This level's relation to the level **above** it, so `levels[0]` must leave it at `"nine"`. See [docs/model.md](model.md#inclusion-policies). |
| `write_policy` | string | `"write-back"` | `write-back`, `write-through` | What a store applied here does about the level below: mark the block dirty and defer (`write-back`), or forward the store immediately (`write-through`). See [docs/model.md](model.md#write-policies). |
| `write_allocate` | boolean | `true` | `true`, `false` | Whether a write miss fetches the block into this level (`true`) or passes the store down untouched (`false`). See [docs/model.md](model.md#write-policies). |
| `bus_width` | integer or null | `null` | `>= 1`, or `null` | Bytes per cycle of the link carrying blocks from below **into** this level; a fill costs `block_size / bus_width` extra cycles. `null` models an infinitely wide link and adds no transfer term. See [docs/model.md](model.md#timing-and-amat). |
| `prefetcher` | string | `"none"` | `none`, `next-line`, `stride` | Hardware prefetcher watching this level's accesses. `next-line` is tagged next-line; `stride` is a per-region stride detector. See [docs/model.md](model.md#prefetching). |
| `victim_cache` | integer, object or null | `null` | integer `>= 1`, `{"entries": N}` with `N >= 1`, or `null` | Entries in a fully-associative victim buffer beside this level's array, holding blocks it has just evicted. See [docs/model.md](model.md#victim-cache). |
| `track_3c` | boolean | `true` | `true`, `false` | Run the shadow fully-associative cache that classifies misses as compulsory, capacity or conflict. Turning it off removes the `three_c` block from the statistics and skips the shadow cache's work on every access. See [docs/model.md](model.md#miss-classification). |
| `rng_seed` | integer | `0` | any integer `>= -2^63` | Seed for the policies that use randomness (`random`, `brrip`, `drrip`), so a run with them is reproducible. |

### Notes on individual keys

`policy: "opt"` names Belady's MIN, an offline policy that has to be told the future.
At most one level may ask for it, because the reference stream a second such level
would see depends on what the first one evicts and cannot be recorded in advance. A
configuration naming `opt` parses, but `cachesim run` does not supply the reference
stream and the first access fails with `RuntimeError: policy 'opt' is offline and has
not been given the future`. Drive it with `cachesim.opt.simulate_with_opt`, or use
[`cachesim policies`](cli.md#policies), which reports OPT alongside every online
policy.

`victim_cache` accepts both `4` and `{"entries": 4}`; the object form exists so the
buffer can grow further keys later without changing the shorthand's meaning.

## Validation

`parse_config` checks the whole object and raises `ConfigError` (a subclass of
`ValueError`) on the first problem it finds. Geometry that is well-typed but not
constructible — a block size that is not a power of two, a capacity that does not
hold a whole number of sets, tree-PLRU with a non-power-of-two associativity, XOR
indexing with a non-power-of-two set count — is caught when the levels are built, and
is reported as `ConfigError` too.

Every per-level message is path-qualified: the prefix is `levels[i]`, with the level's
`name` in parentheses when the object has one.

```text
levels[1] (L2): unknown key(s) 'assoc'
levels[1] (L2): 'hit_time' must be >= 0, got -1
levels[1]: missing required key(s) 'name', 'block_size', 'associativity', 'hit_time'
```

The command line prints the same text with an `error:` prefix and exits 1:

```bash
cachesim run traces/sequential.trace --config bad.json
```

```text
error: invalid config: levels[0] (L2): unknown key(s) 'assoc'
```

### Top-level and memory rules

| Condition | Message |
|-----------|---------|
| Top level is not a JSON object | `expected a JSON object at top level, got list` |
| A file whose top-level value is not an object | `<path>: expected a JSON object at top level` |
| Unrecognised top-level key | `unknown top-level key(s) 'dram'` |
| `levels` absent | `missing required top-level key(s) 'levels'` |
| Neither memory key present | `missing required top-level key(s) 'memory_access_time' (or 'memory')` |
| Both memory keys present | `top level: give either 'memory_access_time' or 'memory', not both; 'memory_access_time': N is shorthand for 'memory': N` |
| `memory_access_time` not an integer | `top level: 'memory_access_time' must be an integer, got '100'` |
| `memory_access_time` negative | `top level: 'memory_access_time' must be >= 0, got -1` |
| `memory` integer form negative | `top level: 'memory' must be >= 0, got -1` |
| `memory` neither integer nor object | `'memory' must be an integer latency or an object, got str` |
| `memory.type` not a string | `memory: 'type' must be a string, got 3` |
| Unknown `memory.type` | `memory: unknown type 'hbm'; choose from constant, row-buffer` |
| Extra key on the constant model | `memory (constant): unknown key(s) 'banks'` |
| `latency` absent from the constant model | `memory (constant): missing required key 'latency'` |
| Extra key on the row-buffer model | `memory (row-buffer): unknown key(s) 'rows'` |
| Non-positive row-buffer geometry | `memory (row-buffer): 'row_size' must be >= 1, got 0` |
| `row_miss` below `row_hit` | `memory (row-buffer): 'row_miss' (20) must be at least 'row_hit' (40); an open row is never the slower case` |
| `levels` empty or not a list | `'levels' must be a non-empty list of cache levels` |

### Per-level rules

| Condition | Message |
|-----------|---------|
| A level is not an object | `levels[0]: expected an object, got int` |
| Unrecognised key | `levels[0] (L1): unknown key(s) 'assoc', 'ways'` |
| Required keys absent | `levels[0] (L2): missing required key(s) 'block_size', 'associativity', 'hit_time'` |
| `name` empty or not a string | `levels[0] (): 'name' must be a non-empty string, got ''` |
| Non-integer geometry (booleans included) | `levels[0] (L1): 'size' must be an integer, got '32768'` |
| Non-positive geometry | `levels[0] (L1): 'size' must be >= 1, got 0` |
| Negative `hit_time` | `levels[0] (L1): 'hit_time' must be >= 0, got -1` |
| `policy` not a string | `levels[0] (L1): 'policy' must be a string, got 3` |
| Unknown `policy` | `levels[0] (L1): unknown policy 'clairvoyant'; choose from brrip, drrip, fifo, lfu, lru, mru, nru, opt, plru, random, srrip` |
| `index` not a string | `levels[0] (L1): 'index' must be a string, got 1` |
| Unknown `index` | `levels[0] (L1): unknown index function 'hash'; choose from modulo, xor` |
| `inclusion` not a string | `levels[1] (L2): 'inclusion' must be a string, got 1` |
| Unknown `inclusion` | `levels[1] (L2): unknown inclusion policy 'strict'; choose from nine, inclusive, exclusive` |
| `write_policy` not a string | `levels[0] (L1): 'write_policy' must be a string, got 1` |
| Unknown `write_policy` | `levels[0] (L1): unknown write policy 'write-around'; choose from write-back, write-through` |
| `write_allocate` not a boolean | `levels[0] (L1): 'write_allocate' must be true or false, got 'no'` |
| `track_3c` not a boolean | `levels[0] (L1): 'track_3c' must be true or false, got 'yes'` |
| `rng_seed` not an integer | `levels[0] (L1): 'rng_seed' must be an integer, got None` |
| `bus_width` not an integer | `levels[0] (L1): 'bus_width' must be an integer, got '16'` |
| `bus_width` below 1 | `levels[0] (L1): 'bus_width' must be >= 1, got 0` |
| `prefetcher` not a string | `levels[0] (L1): 'prefetcher' must be a string, got 1` |
| Unknown `prefetcher` | `levels[0] (L1): unknown prefetcher 'markov'; choose from none, next-line, stride` |
| `victim_cache` not an integer | `levels[0] (L1): 'victim_cache' must be an integer, got '4'` |
| `victim_cache` below 1 | `levels[0] (L1): 'victim_cache' must be >= 1, got 0` |
| Extra key in the `victim_cache` object | `levels[0] (L1): 'victim_cache' has unknown key(s) 'assoc'` |
| `entries` absent from the `victim_cache` object | `levels[0] (L1): 'victim_cache' is missing required key 'entries'` |

### Cross-level and geometry rules

| Condition | Message |
|-----------|---------|
| Two levels share a name | `level names must be unique, got ['L1', 'L1']` |
| More than one level asks for `opt` | `policy 'opt' is only meaningful at one level, but L1, L2 all ask for it` |
| `inclusion` set on the first level | `levels[0] (L1): 'inclusion' describes a level's relation to the level above it, and the first level has none; got 'inclusive'` |
| Levels disagree about `block_size` | `all levels must share one block_size (multi-line fills are not modelled), got [64, 128]` |
| `block_size` not a power of two | `L1: block_size must be a power of two, got 48` |
| `size` not a whole number of sets | `L1: size 1000 must be a multiple of block_size*associativity (256)` |
| `plru` with a non-power-of-two associativity | `L1: plru requires a power-of-two associativity, got 3` |
| `xor` indexing with a non-power-of-two set count | `L1: xor indexing requires a power-of-two set count, got 3` |

The four geometry messages come from the level constructor rather than from the schema
check, which is why they are prefixed with the level's name alone and not with
`levels[i]`.

## Example configurations

The configurations in `configs/` are the ones the test suite parses and simulates on
every run. Each changes one thing about `configs/default.json`.

### `configs/default.json`

The built-in hierarchy: sizes, associativities and latencies typical of a desktop
core, with everything else at its default. Identical to the compiled-in
`DEFAULT_CONFIG` used when `--config` is omitted, and the baseline the other examples
vary.

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

### `configs/inclusive.json`

Makes L2 and L3 inclusive of the levels above them, so that a block dropped by either
is back-invalidated from every level above it. The cost is visible in the
`back_invalidations` counter and in L1 misses that the same hierarchy would not have
taken under NINE.

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12,
     "inclusion": "inclusive"},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40,
     "inclusion": "inclusive"}
  ]
}
```

### `configs/exclusive.json`

Makes L2 exclusive of L1: it is filled by L1's evictions rather than by fetches, and a
block that hits in it moves up instead of being copied, so the two levels together
hold their combined capacity rather than L2's alone.

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12,
     "inclusion": "exclusive"},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40}
  ]
}
```

### `configs/write_through_l1.json`

An L1 that is write-through and no-write-allocate, the pairing real hardware uses: no
L1 block is ever dirty, every store is forwarded to L2, and a write miss does not
disturb L1's contents. The effect shows up in the `write_throughs`, `writebacks` and
`writebacks_received` counters rather than in the miss rate.

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4,
     "write_policy": "write-through", "write_allocate": false},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40}
  ]
}
```

### `configs/prefetch_stride.json`

A tagged next-line prefetcher at L1 and a stride prefetcher at L2, the arrangement
that turns a sequential or strided workload's compulsory misses into prefetch hits.
Accuracy and coverage are reported per level.

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4,
     "prefetcher": "next-line"},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12,
     "prefetcher": "stride"},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40}
  ]
}
```

### `configs/victim_cache.json`

Jouppi's arrangement: a direct-mapped L1 with a 3-cycle hit time rather than the
4-cycle 4-way default, and a four-entry fully-associative victim buffer beside it to
catch the conflict misses direct mapping causes.

```json
{
  "memory_access_time": 100,
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 1, "policy": "lru", "hit_time": 3,
     "victim_cache": {"entries": 4}},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40}
  ]
}
```

### `configs/dram_row_buffer.json`

Replaces the fixed 100-cycle memory with an eight-bank row-buffer model, so that a
last-level miss to an already-open 8 KB row costs 40 cycles instead of 100, and adds a
16 B/cycle bus into L3 so that the four cycles a 64 B block takes to cross it are
charged as well.

```json
{
  "memory": {
    "type": "row-buffer",
    "row_size": 8192,
    "banks": 8,
    "row_hit": 40,
    "row_miss": 100
  },
  "levels": [
    {"name": "L1", "size": 32768, "block_size": 64,
     "associativity": 4, "policy": "lru", "hit_time": 4},
    {"name": "L2", "size": 262144, "block_size": 64,
     "associativity": 8, "policy": "lru", "hit_time": 12},
    {"name": "L3", "size": 2097152, "block_size": 64,
     "associativity": 16, "policy": "lru", "hit_time": 40,
     "bus_width": 16}
  ]
}
```

### The examples side by side

```bash
cachesim compare traces/sequential.trace \
  --config default \
  --config configs/inclusive.json \
  --config configs/exclusive.json \
  --config configs/write_through_l1.json \
  --config configs/prefetch_stride.json \
  --config configs/victim_cache.json \
  --config configs/dram_row_buffer.json
```

```text
trace: traces/sequential.trace (65,536 accesses measured of 65,536)
baseline: default

config             L1 miss   L2 miss   L3 miss    dram rd    dram wr     AMAT   d AMAT        cycles  d cycles
default            12.50%    50.00%   100.00%       4,096          0   14.250        -       933,888         -
inclusive          12.50%    50.00%   100.00%       4,096          0   14.250   +0.000       933,888   +0.00%
exclusive          12.50%    50.00%   100.00%       4,096          0   14.250   +0.000       933,888   +0.00%
write_through_l1   12.50%    50.00%   100.00%       4,096          0   14.250   +0.000       933,888   +0.00%
prefetch_stride     0.00%   100.00%    50.00%           1          0    4.003  -10.247       262,348  -71.91%
victim_cache       12.50%    50.00%   100.00%       4,096          0   13.250   -1.000       868,352   -7.02%
dram_row_buffer    12.50%    50.00%   100.00%       4,096          0   10.779   -3.471       706,432  -24.36%
```

`sequential.trace` walks a 256 KB buffer twice, so the whole footprint fits in the
256 KB L2 and the 2 MB L3 and neither of them ever evicts a block that a level above
still holds: under `configs/inclusive.json` every level reports
`back_invalidations: 0`, which is why the inclusive and exclusive rows are identical
to the baseline here. The write row is identical in these columns for a different
reason: `configs/write_through_l1.json` turns L1's 6,144 write-backs into 6,553
write-throughs received by L2 without changing any miss count or any charged cycle.
Both differences are visible in the per-level counters that `cachesim run --format
json` prints and this table does not; see [docs/results.md](results.md) for
configurations that separate them.

## See also

- [docs/model.md](model.md) — what each option means in the simulated model.
- [docs/cli.md](cli.md) — the commands that read these files.
- [docs/workloads.md](workloads.md) — the traces the examples above are run on.
- [README.md](../README.md) — project overview.
