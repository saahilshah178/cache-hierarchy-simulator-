# Command-line reference

Installing the package puts a `cachesim` console script on the path; `python -m
cachesim` runs the same entry point from a source checkout. Every example below was
produced with the sample traces, which are not in the repository and are written by
[`gen-traces`](#gen-traces):

```bash
cachesim gen-traces --out-dir traces
```

```text
cachesim [-h] [--version] <command> ...

    run          simulate one trace and print a report
    trace-stats  summarise a trace: footprint, pages, compulsory-miss floor
    sweep        sweep a cache parameter (or a size x associativity grid)
    policies     compare every replacement policy on one trace against OPT
    mrc          miss-ratio curve: misses at every capacity, from one pass
    compare      run one trace through several configs and tabulate the differences
    sets         per-set diagnostics: hot sets and windowed set pressure
    gen-traces   write synthetic workload traces
```

`--version` prints `cachesim 0.9.0`. A missing subcommand, an unknown subcommand or an
unknown flag is a usage error.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | The command completed. |
| 1 | A run-time failure reported as `error: <message>` on stderr: a trace that cannot be read or parsed, a trace with no accesses, a configuration that cannot be loaded or does not validate, or an invariant violation found by `run --check`. |
| 2 | A usage error reported by the argument parser, which also prints the command's usage line: an unknown flag, a flag value of the wrong type, and the argument checks the analysis commands make (a missing trace file, a geometry that is not a whole number of sets, fewer than two configurations to compare, mutually exclusive flags). |

`run` and `trace-stats` report a missing trace as an `error:` line and exit 1; `sweep`,
`policies`, `mrc`, `compare` and `sets` report it through the argument parser and exit
2, naming `gen-traces` in the message.

## run

Simulates one trace through one hierarchy and prints the report.

```text
cachesim run [-h] [--trace-format {auto,native,dinero,lackey}] [--config FILE]
             [--warmup N] [--format {text,json}] [--check]
             trace
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | required | Trace file to simulate. |
| `--trace-format` | `auto` | Format of the input trace. `auto` chooses by extension: `.din` is Dinero IV, `.lackey`/`.vg` is Valgrind Lackey, anything else is the native `ADDR R|W` format. A `.gz` suffix is decompressed first. |
| `--config FILE` | built-in L1/L2/L3 hierarchy | JSON hierarchy configuration; see [docs/configuration.md](configuration.md). The built-in default is `configs/default.json`. |
| `--warmup N` | `0` | Simulate the first N accesses, then reset every statistic before counting the rest. `0` reports the whole trace from cold caches. |
| `--format` | `text` | Output format of the report: `text` or `json`. This is the output format; `--trace-format` selects the input format. |
| `--check` | off | After the run, verify the simulator's internal invariants and exit non-zero listing any violation. |

The text report gives the access counts and DRAM traffic, then one block per level —
geometry, hits and misses split by reads and writes, local and global miss rates,
evictions and write-backs, bytes moved, and the three-C classification in both its
per-reference and aggregate forms — and closes with the analytic and measured AMAT,
split by loads and stores.

```bash
cachesim run traces/sequential.trace --config configs/default.json
```

```text
========================================================================
CACHE HIERARCHY SIMULATION REPORT  (traces/sequential.trace)
========================================================================
Total accesses :       65,536   (reads 58,983 / writes 6,553)
DRAM reads     :        4,096   (6.25% of all accesses missed every level)
DRAM writes    :            0   (dirty lines written back from L3)

--- L1: 32.0 KB, 64 B blocks, 4-way, LRU, hit time 4 cyc ---
  accesses    :       65,536
  hits        :       57,344   (reads 50,791 / writes 6,553)
  misses      :        8,192   (reads 8,192 / writes 0)
  local miss rate  :  12.50%   (misses / accesses that reached this level)
  global miss rate :  12.50%   (misses / all CPU accesses)
  evictions   :        7,680   (writebacks of dirty blocks: 6,144)
  traffic     :     512.0 KB in from below / 384.0 KB out below
  miss classification      per-reference    aggregate
    compulsory                     4,096        4,096
    capacity                       4,096        4,096
    conflict                           0            0
    fully-associative LRU misses: 8,192   hits it would have missed: 0

[the L2 and L3 blocks are elided here]

========================================================================
DRAM traffic            : 256.0 KB read / 0 B written
AMAT (analytic formula) :   14.250 cycles
AMAT (measured)         :   14.250 cycles   (933,888 cycles / 65,536 accesses)
  loads                 :   15.389 cycles   (907,676 cycles / 58,983 loads)
  stores                :    4.000 cycles   (26,212 cycles / 6,553 stores)
========================================================================
```

`--format json` prints the same figures as the object documented in
[JSON output](#json-output). `--check` writes its diagnostics to stderr so that the
JSON on stdout stays parseable, and reports which invariants it skipped for
configurations they do not apply to:

```bash
cachesim run traces/conflict.trace --config configs/inclusive.json --check > /dev/null
```

```text
invariants: skipped write-back conservation (L2: inclusion is 'inclusive'; L3: inclusion is 'inclusive')
invariants: skipped traffic-flow identities (L2: inclusion is 'inclusive'; L3: inclusion is 'inclusive')
invariants: skipped timing identities (L2: inclusion is 'inclusive'; L3: inclusion is 'inclusive')
invariants: 50 checks passed
```

Errors exit 1: `error: [Errno 2] No such file or directory: 'traces/nope.trace'` for a
missing trace, `error: cannot load config <path>: ...` for unreadable or non-JSON
configuration files, `error: invalid config: levels[0] (L2): unknown key(s) 'assoc'`
for one that does not validate, and `error: no accesses found in <trace> after the
warm-up` when `--warmup` consumes the whole trace. An invariant violation also exits
1, after the report has been printed.

## trace-stats

Summarises the address stream without simulating a cache, so the figures bound what
any cache can achieve on it.

```text
cachesim trace-stats [-h] [--block-size B]
                     [--trace-format {auto,native,dinero,lackey}]
                     [--format {text,json}]
                     trace
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | required | Trace file to summarise. |
| `--block-size B` | `64` | Block size the footprint and first-touch counts assume. |
| `--trace-format` | `auto` | Format of the input trace, chosen by extension by default. |
| `--format` | `text` | `text` or `json`. |

It prints the access counts split into reads and writes, the number of distinct blocks
and the footprint they occupy, the distinct 4 KB pages, the address range, and the
first-touch count as a fraction of all accesses — the compulsory-miss floor, the
lowest miss rate any cache can reach on this trace from cold.

```bash
cachesim trace-stats traces/matmul_naive.trace
```

```text
TRACE STATISTICS  (traces/matmul_naive.trace)
  accesses        :      532,480
  reads           :      528,384   ( 99.2%)
  writes          :        4,096   (  0.8%)
  distinct blocks :        1,536   (footprint 96.0 KB at 64 B blocks)
  distinct pages  :           24   (4.0 KB pages)
  address range   : 0x000000200000 .. 0x000000407ff8   (span 2,129,912 B)
  first touches   :        1,536   ( 0.29% of accesses: the compulsory-miss floor)
```

`--format json` prints one flat object with the keys `accesses`, `reads`, `writes`,
`block_size`, `distinct_blocks`, `footprint_bytes`, `page_size`, `distinct_pages`,
`min_address`, `max_address`, `first_touches` and `compulsory_fraction`. The two
address bounds are `null` for an empty trace.

An unreadable or malformed trace, a trace with no accesses, and a non-positive
`--block-size` all print `error: ...` and exit 1.

## sweep

Sweeps one cache parameter, or a grid of two. Every point is a single cache level
backed directly by memory, so no lower level absorbs the misses of the level being
varied.

```text
cachesim sweep [-h] [--param {size,associativity,block_size,policy}] [--values LIST]
               [--grid] [--ways LIST] [--size SIZE] [--assoc ASSOC]
               [--block-size BLOCK_SIZE]
               [--policy {brrip,drrip,fifo,lfu,lru,mru,nru,plru,random,srrip}]
               [--hit-time HIT_TIME] [--mem-time MEM_TIME] [--csv FILE]
               [--plot-dir PLOT_DIR] [--no-plot]
               [trace]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | `conflict.trace` for the associativity sweep, `matmul_naive.trace` for the size sweep | Trace to sweep. |
| `--param` | none (run both default sweeps) | Sweep this one parameter: `size`, `associativity`, `block_size` or `policy`. Requires a trace argument. |
| `--values LIST` | depends on the parameter | Comma-separated values for `--param`, or the size axis of `--grid`. Sizes accept a `k`/`M` suffix, as in `4k,16k,64k`. |
| `--grid` | off | Sweep size against associativity and print a grid of miss rates. Mutually exclusive with `--param`. |
| `--ways LIST` | `1,2,4,8,16` | Associativity axis for `--grid`; an error unless `--grid` is given. |
| `--size SIZE` | `8192` | Fixed size for the associativity sweep. |
| `--assoc ASSOC` | `4` | Fixed associativity for the size sweep. |
| `--block-size BLOCK_SIZE` | `64` | Block size in bytes. |
| `--policy` | `lru` | Replacement policy. The offline optimum is not offered here because a sweep cannot supply the future; use [`policies`](#policies). |
| `--hit-time HIT_TIME` | `4` | Cache hit time in cycles. |
| `--mem-time MEM_TIME` | `100` | Memory access time in cycles. |
| `--csv FILE` | none | Write every row to FILE as CSV. |
| `--plot-dir PLOT_DIR` | `plots` when matplotlib is installed | Directory for the plots. |
| `--no-plot` | off | Skip plotting entirely. |

Each sweep prints one row per value: miss rate, AMAT, the three-C counts and the
fully-associative LRU miss count at the same capacity. Plots are written when
matplotlib is available, or when `--plot-dir` is given explicitly, and each written
file is named on stdout; a missing matplotlib is reported as a note rather than as an
error.

```bash
cachesim sweep --param associativity traces/conflict.trace --no-plot
```

```text
trace: traces/conflict.trace (65,536 accesses)

=== associativity sweep (8 KB, 64 B blocks, LRU) ===
associativity  miss rate     AMAT   compulsory  capacity  conflict    FA-LRU
         1   100.00%   104.00        8,192         0    57,344     8,192
         2   100.00%   104.00        8,192         0    57,344     8,192
         4    12.50%    16.50        8,192         0         0     8,192
         8    12.50%    16.50        8,192         0         0     8,192
        16    12.50%    16.50        8,192         0         0     8,192
```

With no `--param` and no `--grid` it runs both default sweeps; with `--grid` it prints
the size-by-associativity matrix:

```bash
cachesim sweep --grid traces/conflict.trace --no-plot
```

```text
trace: traces/conflict.trace (65,536 accesses)

=== size x associativity grid (64 B blocks, LRU, miss rate) ===
      size     1-way     2-way     4-way     8-way    16-way
      1 KB  100.00%   100.00%    12.50%    12.50%    12.50%
      2 KB  100.00%   100.00%    12.50%    12.50%    12.50%
      4 KB  100.00%   100.00%    12.50%    12.50%    12.50%
      8 KB  100.00%   100.00%    12.50%    12.50%    12.50%
     16 KB  100.00%   100.00%    12.50%    12.50%    12.50%
     32 KB  100.00%   100.00%    12.50%    12.50%    12.50%
     64 KB  100.00%   100.00%    12.50%    12.50%    12.50%
```

A sweep in which more of the parameter took more misses is reported under the table
rather than suppressed, since that is a property of the trace and not an error:

```bash
cachesim sweep --param size --values 4k,8k,12k,16k,24k,32k traces/matmul_naive.trace --no-plot
```

```text
trace: traces/matmul_naive.trace (532,480 accesses)

=== size sweep (4-way, 64 B blocks, LRU) ===
      size  miss rate     AMAT   compulsory  capacity  conflict    FA-LRU
      4 KB    51.03%    55.03        1,536   297,984   -27,776   299,520
      8 KB    51.01%    55.01        1,536    32,256   237,840    33,792
     12 KB    50.99%    54.99        1,536    32,256   237,696    33,792
     16 KB    50.99%    54.99        1,536    32,256   237,704    33,792
     24 KB    50.92%    54.92        1,536    32,256   237,357    33,792
     32 KB     5.24%     9.24        1,536    32,256    -5,880    33,792
note: not monotonic -- 1 of 5 steps increased misses; best 32 KB (27,912), worst 4 KB (271,744); a larger size made this trace worse
```

The CSV written by `--csv` carries every counter, not only the miss rate, so a grid can
be re-analysed without re-running it. Its columns are `param`, `value`, `size`,
`block_size`, `associativity`, `policy`, `num_sets`, `accesses`, `hits`, `misses`,
`miss_rate`, `compulsory`, `capacity`, `conflict`, `shadow_misses`, `amat`.

Usage errors exit 2: `--grid` together with `--param`, `--ways` without `--grid`,
`--param` without a trace, a `--values` list that does not parse, a missing trace file,
and a `--size` that is not a multiple of `block_size * ways` for every point the sweep
would visit.

## policies

Runs one cache level through every replacement policy in the registry plus Belady's
OPT, at one geometry, and reports how far each is from the optimum on that trace.

```text
cachesim policies [-h] [--size SIZE] [--assoc ASSOC] [--block-size BLOCK_SIZE]
                  [--hit-time HIT_TIME] [--mem-time MEM_TIME] [--csv FILE]
                  trace
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | required | Trace file. |
| `--size SIZE` | `32768` | Cache size in bytes. |
| `--assoc ASSOC` | `4` | Associativity in ways. |
| `--block-size BLOCK_SIZE` | `64` | Block size in bytes. |
| `--hit-time HIT_TIME` | `4` | Cache hit time in cycles. |
| `--mem-time MEM_TIME` | `100` | Memory access time in cycles. |
| `--csv FILE` | none | Also write the table to FILE as CSV. |

The table is sorted best first and gives misses, miss rate, AMAT and the ratio to OPT.
Two bounds are reported: the `opt` row is Belady's rule applied within each set, the
best any policy could do with this set mapping, and the line below the table is the
fully-associative Belady bound, the best any cache of this capacity could do at all.
The distance between them is the price of the set mapping, which no replacement policy
can recover.

```bash
cachesim policies traces/conflict.trace --size 8192
```

```text
traces/conflict.trace: 65,536 accesses through one level of 8 KB, 4-way, 64 B blocks (32 sets)
policy       misses  miss rate     AMAT   x OPT
opt           8,192    12.50%    16.50    1.00
fifo          8,192    12.50%    16.50    1.00
lru           8,192    12.50%    16.50    1.00
nru           8,192    12.50%    16.50    1.00
plru          8,192    12.50%    16.50    1.00
srrip         8,192    12.50%    16.50    1.00
drrip        10,213    15.58%    19.58    1.25
random       16,534    25.23%    29.23    2.02
brrip        60,142    91.77%    95.77    7.34
lfu          64,640    98.63%   102.63    7.89
mru          64,640    98.63%   102.63    7.89

fully associative Belady bound: 8,192 misses (12.50%)  -- the set mapping costs everything above this
```

The CSV columns are `policy`, `accesses`, `misses`, `miss_rate`, `amat`, `gap_to_opt`.

A missing trace, an empty trace, and a `--size` that is not a multiple of
`--block-size * --assoc` are usage errors and exit 2.

## mrc

Builds the LRU stack-distance profile of a trace in one pass and reports the miss
count at every capacity from it. The profile is exact, not sampled; it models demand
references to a single block size under LRU with write-allocate, and no other feature
of the hierarchy.

```text
cachesim mrc [-h] [--block-size BLOCK_SIZE] [--sets N] [--max-capacity BLOCKS]
             [--csv FILE] [--plot FILE]
             trace
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | required | Trace file. |
| `--block-size BLOCK_SIZE` | `64` | Block size in bytes. |
| `--sets N` | none | Also profile a cache with N sets, giving misses for every associativity. |
| `--max-capacity BLOCKS` | the smallest power of two that reaches the compulsory floor | Largest capacity to tabulate, in blocks. |
| `--csv FILE` | none | Write the curve or curves to FILE as CSV. |
| `--plot FILE` | none | Write the miss-ratio curve to FILE. |

It prints the reference and distinct-block counts with the compulsory floor, the
fully-associative LRU miss curve, the working-set size (the smallest capacity within
one percentage point of the floor), a reuse-distance histogram, and — with `--sets` —
the set-associative curve with the Hill and Smith aggregate conflict count beside it.

```bash
cachesim mrc traces/conflict.trace --sets 64
```

```text
trace: traces/conflict.trace
references: 65,536   distinct 64 B blocks: 8,192   compulsory floor: 12.50%

fully-associative LRU miss curve
 capacity      size       misses  miss rate
        1      64 B       65,536   100.00%
        2     128 B       65,536   100.00%
        4     256 B        8,192    12.50%

working set: 4 blocks (256 B) -- the smallest capacity within 1 percentage point of the floor

reuse-distance histogram
    distance         refs    share
           0            0   0.00%
           1            0   0.00%
         2-3       57,344  87.50%  ########################################
    infinite        8,192  12.50%  ######

set-associative LRU miss curve, 64 sets
 ways  capacity      size       misses  miss rate   conflict
    1        64      4 KB       65,536   100.00%     57,344
    2       128      8 KB       65,536   100.00%     57,344
    4       256     16 KB        8,192    12.50%          0
conflict = misses minus the fully-associative misses at the same capacity (Hill-Smith aggregate)
```

The CSV columns are `sets`, `ways`, `capacity_blocks`, `capacity_bytes`, `misses`,
`miss_ratio`, `conflict`; the fully-associative rows and the set-associative rows share
the file. `--plot` needs matplotlib and reports its absence as a note rather than an
error.

A missing trace or a trace with no accesses is a usage error and exits 2.

## compare

Replays one trace through several hierarchy configurations and prints one aligned
table. The trace is parsed once and the same access list is used for every
configuration.

```text
cachesim compare [-h] [--config FILE] [--warmup N] [--format {text,json}] trace
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | required | Trace file. |
| `--config FILE` | none; at least two are required | A configuration to compare; repeat the flag for each one. `default` names the built-in hierarchy. The first is the baseline the deltas are measured against. |
| `--warmup N` | `0` | Simulate the first N accesses, then reset every statistic before counting the rest. |
| `--format` | `text` | `text` for the aligned table, `json` for a list of statistics objects. |

Rows are labelled with each file's base name, or with the full paths if two base names
collide. The columns are the per-level local miss rates, DRAM reads and writes, AMAT
and total cycles, with signed deltas from the baseline; negative is better. Level
columns are the union of level names in the order they first appear, and a
configuration without a given level shows `-`.

```bash
cachesim compare traces/conflict.trace --config default --config configs/victim_cache.json
```

```text
trace: traces/conflict.trace (65,536 accesses measured of 65,536)
baseline: default

config         L1 miss   L2 miss   L3 miss    dram rd    dram wr     AMAT   d AMAT        cycles  d cycles
default        12.50%   100.00%   100.00%       8,192          0   23.000        -     1,507,328         -
victim_cache   12.50%   100.00%   100.00%       8,192          0   22.000   -1.000     1,441,792   -4.35%
```

`--format json` prints a JSON list, one element per configuration: the object of
[JSON output](#json-output) with two extra keys first, `label` (the row label) and
`config` (the path as given, or `default`).

Fewer than two `--config` values, a configuration that cannot be loaded or does not
validate, a missing trace, and a `--warmup` that leaves nothing to measure are all
usage errors and exit 2.

## sets

Per-set diagnostics for one cache level: which sets take the misses over the whole
trace, and how crowded the sets are at any one time.

```text
cachesim sets [-h] [--size SIZE] [--block-size BLOCK_SIZE] [--assoc ASSOC]
              [--policy {brrip,drrip,fifo,lfu,lru,mru,nru,opt,plru,random,srrip}]
              [--window W] [--top N] [--format {text,json}]
              trace
```

| Flag | Default | Meaning |
|------|---------|---------|
| `trace` | required | Trace file. |
| `--size SIZE` | `32k` | Cache size; accepts a `k`/`M` suffix. |
| `--block-size BLOCK_SIZE` | `64` | Block size in bytes. |
| `--assoc ASSOC` | `4` | Ways per set. |
| `--policy` | `lru` | Replacement policy for the cumulative half of the report. |
| `--window W` | `512` | Accesses per pressure window. |
| `--top N` | `8` | List the N busiest sets; `0` omits the list. |
| `--format` | `text` | `text` or `json`. |

Two measurements are reported separately. The cumulative half simulates the level and
tallies misses by set index, giving the minimum, mean and maximum per set, the ratio of
maximum to mean, and the share of misses landing in the hottest 5% of sets beside the
share a uniform distribution would give those same sets. The windowed half needs no
simulation: it slices the trace into windows of W accesses and counts the distinct
blocks mapping to each set in each window, calling a (window, set) pair oversubscribed
when that count exceeds the associativity. The denominator is the touched pairs, those
where the set saw at least one block in that window.

```bash
cachesim sets traces/conflict.trace --size 4k --assoc 1
```

```text
trace: traces/conflict.trace (65,536 accesses)
geometry: 4 KB, 64 B blocks, 1-way, LRU -> 64 sets

cumulative misses per set
  misses                  65,536  of 65,536 accesses
  per set           min 1,024   mean 1,024.0   max 1,024   (max/mean 1.00)
  hottest 5% of sets  4 of 64 sets hold 6.25% of misses (uniform would be 6.25%)
  busiest sets      0 (1,024), 1 (1,024), 2 (1,024), 3 (1,024), 4 (1,024), 5 (1,024), 6 (1,024), 7 (1,024)

windowed set pressure (windows of 512 accesses)
  windows                    128
  touched pairs            2,048  (window, set) pairs in use
  oversubscribed           2,048  = 100.00% of touched pairs (distinct blocks > 1 way)
  windows affected           128  = 100.00% of windows
  distinct blocks per touched set: mean 4.00, max 4
```

`--format json` prints one object with `trace`, a `geometry` block (`size`,
`block_size`, `associativity`, `num_sets`, `policy`), a `hot_sets` block (`num_sets`,
`ways`, `accesses`, `misses`, `min`, `mean`, `max`, `spread`, `top_fraction`,
`top_sets`, `top_share`, `uniform_share`) and a `window_pressure` block (`window`,
`num_sets`, `ways`, `windows`, `touched_pairs`, `oversubscribed_pairs`,
`oversubscribed_fraction`, `windows_oversubscribed`, `windows_oversubscribed_fraction`,
`mean_distinct`, `max_distinct`).

A missing trace, an empty trace and a geometry that is not a whole number of sets are
usage errors and exit 2. `--policy opt` is accepted by the parser but cannot be run
here: the offline policy needs the level's whole reference stream in advance, and the
command fails with `RuntimeError: policy 'opt' is offline and has not been given the
future`, exiting 1. Use [`policies`](#policies) for the offline optimum.

## gen-traces

Writes the synthetic workload traces. Generation is seeded, so the same name always
produces the same file.

```text
cachesim gen-traces [-h] [--out-dir OUT_DIR] [--list] [NAME ...]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `NAME ...` | the six samples `sequential`, `random`, `matmul_naive`, `matmul_blocked`, `conflict`, `pointer_chase` | Workloads to write. |
| `--out-dir OUT_DIR` | `traces` | Output directory. |
| `--list` | off | Print every workload's name, description and expectation, and write nothing. |

```bash
cachesim gen-traces --out-dir traces
```

```text
generating traces in traces:
  sequential.trace          65,536 accesses
  random.trace              60,000 accesses
  matmul_naive.trace       532,480 accesses
  matmul_blocked.trace     557,056 accesses
  conflict.trace            65,536 accesses
  pointer_chase.trace       60,000 accesses
```

The registry holds more workloads than the six samples; `--list` prints all of them
with the behaviour each is built to demonstrate. See
[docs/workloads.md](workloads.md). An unknown name is a usage error and exits 2.

## bench

Times the simulator itself on one trace and reports throughput. This section describes
the interface; run `cachesim bench --help` for the flag defaults as shipped.

```text
cachesim bench [-h] [--config FILE] [--repeat N] [--format {text,json}] trace
```

| Flag | Meaning |
|------|---------|
| `trace` | Trace file to replay. |
| `--config FILE` | JSON hierarchy configuration to time; the built-in hierarchy is used when omitted. |
| `--repeat N` | Number of timed repetitions to run. |
| `--format` | `text` for the summary, `json` for the same figures as an object. |

It reports the best and median wall-clock time over the repetitions and the resulting
accesses per second. These are properties of the machine the command runs on and of
the Python interpreter in use, not of the simulated hierarchy: they say how fast the
simulator processes a trace, and no simulated cycle count, miss rate or AMAT depends on
them. Comparisons are only meaningful between runs on the same machine, and the best
time is the more stable of the two figures.

## JSON output

`run --format json` prints one object, and `compare --format json` prints a list of
them, each with `label` and `config` prepended. `schema_version` is the first key; it
is bumped whenever a field is renamed or its meaning changes, and is currently `2`.

### Top level: `HierarchyStats`

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | integer | Version of this schema; currently `2`. |
| `accesses` | integer | Accesses the hierarchy was given. |
| `reads` | integer | Of those, loads. |
| `writes` | integer | Of those, stores. |
| `dram_reads` | integer | Reads that reached main memory. |
| `dram_writes` | integer | Writes that reached main memory. |
| `dram_demand_writes` | integer | Of those writes, stores no level allocated for, as opposed to write-backs. |
| `dram_prefetch_reads` | integer | Of the DRAM reads, those issued by a prefetcher. |
| `dram_bytes_read` | integer | Bytes read from main memory. |
| `dram_bytes_written` | integer | Bytes written to main memory. |
| `memory_access_time` | integer or null | The fixed memory latency, or `null` when the memory model is not a constant one. |
| `memory` | object | The main-memory model and what it saw; see below. |
| `total_cycles` | integer | Simulated cycles over all accesses. |
| `read_cycles` | integer | Of those, spent on loads. |
| `write_cycles` | integer | Of those, spent on stores. |
| `amat` | float | AMAT from the analytic nested formula, in cycles. |
| `measured_amat` | float | `total_cycles / accesses`. |
| `read_amat` | float | Measured average cycles per load. |
| `write_amat` | float | Measured average cycles per store. |
| `levels` | array | One `LevelStats` object per level, outermost first. |

### `memory`: `MemoryStats`

| Field | Type | Meaning |
|-------|------|---------|
| `type` | string | Name of the memory model, `constant` or `row-buffer`. |
| `parameters` | object | The model's parameters, as configured. |
| `average_latency` | float | Mean latency over the accesses the hierarchy was charged for; the DRAM term the analytic AMAT uses. |
| `row_hits` | integer | Accesses that found the row already open; `0` for a constant memory. |
| `row_misses` | integer | Accesses that had to open a row; `0` for a constant memory. |
| `row_buffer_hit_rate` | float | `row_hits` over the two counts together. |

### `levels[]`: `LevelStats`

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | Level name from the configuration. |
| `size` | integer | Capacity in bytes. |
| `block_size` | integer | Bytes per block. |
| `associativity` | integer | Ways per set. |
| `num_sets` | integer | `size / (block_size * associativity)`. |
| `policy` | string | Replacement policy in force. |
| `index` | string | Set-index function, `modulo` or `xor`. |
| `hit_time` | integer | Cycles charged per probe of this level. |
| `inclusion` | string | Relation to the level above: `nine`, `inclusive` or `exclusive`. |
| `write_policy` | string | `write-back` or `write-through`. |
| `write_allocate` | boolean | Whether a write miss allocates here. |
| `bus_width` | integer or null | Bytes per cycle of the link from below, or `null` for no transfer term. |
| `transfer_cycles` | integer | Cycles a whole block occupies that link, `ceil(block_size / bus_width)`. |
| `prefetcher` | string | Prefetcher at this level: `none`, `next-line` or `stride`. |
| `victim_cache_entries` | integer or null | Entries in the victim buffer beside this level, or `null`. |
| `accesses` | integer | Probes this level received. |
| `hits` | integer | Of those, hits. |
| `misses` | integer | Of those, misses. |
| `read_hits` | integer | Hits by loads. |
| `read_misses` | integer | Misses by loads. |
| `write_hits` | integer | Hits by stores. |
| `write_misses` | integer | Misses by stores. |
| `local_miss_rate` | float | `misses / accesses`: misses per access that reached this level. |
| `global_miss_rate` | float | `misses / hierarchy accesses`: misses per CPU access. |
| `fills` | integer | Blocks installed in this level. |
| `evictions` | integer | Valid blocks replaced to make room. |
| `invalidations` | integer | Valid blocks removed without being replaced. |
| `writebacks` | integer | Dirty blocks written down on eviction, invalidation or flush. |
| `writebacks_received` | integer | Dirty blocks written into this level from above. |
| `writeback_allocations` | integer | Of those, the ones whose block was absent here and had to be allocated. |
| `back_invalidations` | integer | Blocks removed from this level because an inclusive level below evicted them. |
| `write_throughs` | integer | Stores duplicated to the level below because this level is write-through. |
| `write_bypasses` | integer | Write misses this level declined to allocate for, passing the store down. |
| `bytes_read_from_below` | integer | Bytes filled into this level, demand and prefetch. |
| `bytes_written_below` | integer | Bytes sent down: write-backs, victims handed to an exclusive level, and forwarded stores. A forwarded store is charged a whole block, so this is an upper bound; `write_throughs` and `write_bypasses` are the exact transaction counts. |
| `prefetches_issued` | integer | Prefetches this level's prefetcher issued. |
| `prefetch_hits` | integer | Prefetched blocks a later demand reference hit: useful prefetches. |
| `prefetch_evicted_unused` | integer | Prefetched blocks that left this level without a demand hit. |
| `prefetch_probes` | integer | Lookups here caused by a prefetch from a level above; counted separately so that miss rates stay demand-only. |
| `prefetch_probe_hits` | integer | Of those, the ones that found the block here. |
| `prefetch_accuracy` | float | `prefetch_hits / prefetches_issued`, or `0.0` if none were issued. |
| `prefetch_coverage` | float | `prefetch_hits / (prefetch_hits + misses)`: the share of the misses this level would otherwise have taken that the prefetcher removed. |
| `victim_hits` | integer | Hits satisfied from the victim buffer. |
| `three_c` | object or null | Miss classification, or `null` when `track_3c` is off for this level. |

### `three_c`: `ThreeCStats`

| Field | Type | Meaning |
|-------|------|---------|
| `compulsory` | integer | Misses of blocks never referenced before. |
| `capacity` | integer | Per-reference capacity misses: misses the fully-associative LRU twin also took. |
| `conflict` | integer | Per-reference conflict misses: misses the twin would have hit. |
| `capacity_aggregate` | integer | Aggregate capacity misses: `shadow_misses - compulsory`, the Hill and Smith definition. |
| `conflict_aggregate` | integer | Aggregate conflict misses: `misses - shadow_misses`. Negative when the set mapping beats fully-associative LRU. |
| `shadow_misses` | integer | Misses of the fully-associative LRU cache of the same capacity. |
| `anti_conflict_hits` | integer | Hits here that the fully-associative twin would have missed; the reason the two classifications differ. |

## Environment variables

`CACHESIM_SKIP_SLOW` controls the golden regression suite in `tests/test_golden.py`,
which regenerates all six sample traces, checks their SHA-256 digests and simulates
each one. Any value other than empty or `0` skips it:

```bash
CACHESIM_SKIP_SLOW=1 python -m pytest
```

The suite runs by default. Nothing else in the package reads the environment; the
workload generators are seeded by argument, so traces do not vary with it.

## Make targets

`make help` lists the targets. All of them except `traces` need the development
extras, installed by `make install` (`pip install -e ".[dev]"`).

| Target | What it runs |
|--------|--------------|
| `make install` | `pip install -e ".[dev]"`: the package with pytest, hypothesis, ruff, mypy and matplotlib. |
| `make test` | `python -m pytest`. |
| `make lint` | `ruff check .` and `ruff format --check .`. |
| `make format` | `ruff format .` and `ruff check --fix .`. |
| `make typecheck` | `mypy`, configured as strict over `cachesim` and `tests`. |
| `make check` | `lint`, `typecheck` and `test`, in that order. |
| `make traces` | `python -m cachesim gen-traces --out-dir traces`. |
| `make clean` | Remove build artefacts and the tool caches. |

`PYTHON` overrides the interpreter, as in `make test PYTHON=python3.12`.

## See also

- [docs/configuration.md](configuration.md) — the `--config` file format.
- [docs/model.md](model.md) — what the reported quantities mean.
- [docs/workloads.md](workloads.md) — the traces `gen-traces` writes.
- [docs/validation.md](validation.md) — the invariants `run --check` verifies.
- [README.md](../README.md) — project overview.
