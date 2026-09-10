# Workloads and traces

The simulator is trace-driven: it consumes a stream of memory references and knows nothing
about the program that produced them. This page documents the trace formats it reads and
writes, the registry of synthetic workloads it can generate, what those workloads measure
on the default hierarchy, and how to feed it a trace from a real program.

Related pages: [model.md](model.md) for what the simulator does with a reference,
[configuration.md](configuration.md) for the hierarchy JSON, [cli.md](cli.md) for the full
command reference, [results.md](results.md) for the parameter studies, and
[validation.md](validation.md) for the tests that pin these numbers.

All commands below are run from the repository root. Every figure on this page was
produced by the command printed beside it, on CPython 3.13.7.

## Trace formats

Three input formats are understood, and all three decode to the same thing: a stream of
`(byte address, is_write)` pairs. Access size is not part of that stream. Every reference
is modelled as touching only the block that contains its first byte, so a reference that
straddles a block boundary is charged one block, not two, and the size field that Dinero
and Lackey records carry is read past and discarded.

### Native format

One access per line, `ADDR R|W`:

```text
line    := [ record ] [ comment ] newline
record  := address blank+ op
address := [ "0x" | "0X" ] hexdigit+
op      := "R" | "W" | "r" | "w"
comment := "#" any*
```

* The address is a non-negative hexadecimal integer. The `0x` prefix is optional, and
  there is no width limit: `0xffffffffffffffff` parses as 2^64 - 1.
* The operation is `R` (read, load) or `W` (write, store), case-insensitive.
* Fields are separated by any run of blanks or tabs; leading and trailing whitespace is
  ignored, and CRLF line endings are accepted.
* Everything from `#` to end of line is a comment, and blank lines are skipped. A line
  that is only a comment contributes no access.
* Anything else is a hard error naming the file and the line number.

```text
# a native trace
0x00400000 R
0x00400008 R          # trailing comments are allowed
0x7fff0010 W
```

This is the format `cachesim gen-traces` writes, as `0x%08x R` or `0x%08x W`: the address
is zero-padded to eight hex digits, and widens beyond that when it does not fit.

#### Validation

The native reader rejects, rather than skips, anything it cannot parse. The exact
messages, with the file and line prefix elided:

| Input line   | Error                                     |
| ------------ | ----------------------------------------- |
| `0xZZ R`     | `bad hex address '0xZZ'`                  |
| `-0x10 R`    | `bad hex address '-0x10'`                 |
| `+10 R`      | `bad hex address '+10'`                   |
| `0x1_0 R`    | `bad hex address '0x1_0'`                 |
| `0x R`       | `bad hex address '0x'`                    |
| `1.0 R`      | `bad hex address '1.0'`                   |
| `０x10 R`    | `bad hex address '０x10'`                 |
| `0x10, R`    | `bad hex address '0x10,'`                 |
| `0x10 READ`  | `op must be R or W, got 'READ'`            |
| `0x10`       | `expected 'ADDR R\|W', got '0x10'`         |
| `0x10 R 8`   | `expected 'ADDR R\|W', got '0x10 R 8'`     |

Signs, underscore separators, non-ASCII digits (the seventh row begins with U+FF10
FULLWIDTH DIGIT ZERO), trailing punctuation, spelled-out operations, a missing field and
an extra field are all rejected. Python's `int(s, 16)` accepts four of them — `-0x10`,
`+10`, `0x1_0` and the fullwidth `０x10`, the last three as the value 16 — so the reader
checks each literal against the grammar as well and the accepted language is exactly the
one written above.

The error carries the file name and the line number, and reaches the command line as an
`error:` line with exit status 1:

```bash
printf '0x10 R\n0x20 R\nbogus\n' > /tmp/bad.trace
cachesim trace-stats /tmp/bad.trace
```

```text
error: /tmp/bad.trace:3: expected 'ADDR R|W', got 'bogus'
```

### Dinero IV (`.din`)

The trace format of Dinero IV, one reference per line, `LABEL ADDR`, with the address in
hexadecimal and the `0x` prefix optional:

```text
0 7fff0010
1 7fff0018
2 00400000
```

| Label | Dinero meaning    | Decoded as |
| ----- | ----------------- | ---------- |
| `0`   | read              | read       |
| `1`   | write             | write      |
| `2`   | instruction fetch | read       |
| `3`   | escape record     | rejected   |
| `4`   | escape record     | rejected   |

Instruction fetches are decoded as reads because this simulator models one unified cache
hierarchy rather than split instruction and data caches; a fetch and a load are the same
event to it. Dinero's escape labels `3` and `4` carry no memory reference, and are
rejected rather than silently dropped, so a trace that uses them is not quietly truncated
into a different workload.

Blank lines are ignored. Everything else is a hard error naming the file and line:

| Input line       | Error                                                                    |
| ---------------- | ------------------------------------------------------------------------ |
| `3 7fff0010`     | `label must be 0 (read), 1 (write) or 2 (instruction fetch), got '3'`    |
| `R 7fff0010`     | `label must be 0 (read), 1 (write) or 2 (instruction fetch), got 'R'`    |
| `0 nothex`       | `bad hex address 'nothex'`                                               |
| `0`              | `expected 'LABEL ADDR', got '0'`                                         |
| `0 7fff0010 8`   | `expected 'LABEL ADDR', got '0 7fff0010 8'`                              |
| `0 10 # c`       | `expected 'LABEL ADDR', got '0 10 # c'`                                  |

The native `#` comment syntax is not part of the Dinero format, so the last row is an
extra field, not a comment.

### Valgrind Lackey (`.lackey`, `.vg`)

The output of Valgrind's Lackey tool run with `--trace-mem=yes`. Lackey writes its
records into the same stream as Valgrind's own commentary and whatever the traced program
prints:

```text
==31976== Lackey, an example Valgrind tool
==31976== Command: ./app
==31976==
I  0421f3d8,8
 L 0421f3e0,8
 S 0421f3e8,4
 M 0421f3f0,8
hello from the traced program
==31976== counts for all instructions: 12345
```

| Record | Lackey meaning                    | Decoded as |
| ------ | --------------------------------- | ---------- |
| `I`    | instruction fetch                 | read       |
| `L`    | load                              | read       |
| `S`    | store                             | write      |
| `M`    | modify (load and store, one insn) | one write  |

A modify is decoded as a single write, which is exact for a write-back, write-allocate
hierarchy: the load half of a read-modify-write brings in the same block that the store
half then dirties, so charging both would double-count the reference. Under a
no-write-allocate or write-through level the two halves are not equivalent, and a trace
whose modifies matter should be captured in the native format instead.

The size field after the comma is ignored, as it is in every format.

Because the format is a log rather than a data file, the Lackey reader skips every line it
does not recognise instead of raising: the `==pid==` banners, the program's own output,
and a corrupt reference line such as ` L zzz,8` or ` S 40,8 extra` all pass without
comment. The excerpt above yields exactly four accesses:

```bash
cachesim trace-stats app.lackey
```

```text
TRACE STATISTICS  (app.lackey)
  accesses        :            4
  reads           :            2   ( 50.0%)
  writes          :            2   ( 50.0%)
  distinct blocks :            1   (footprint 64 B at 64 B blocks)
  distinct pages  :            1   (4.0 KB pages)
  address range   : 0x00000421f3d8 .. 0x00000421f3f0   (span 24 B)
  first touches   :            1   (25.00% of accesses: the compulsory-miss floor)
```

Silent skipping is the right behaviour for a log and the wrong behaviour for a data file.
Use the native format when strict validation matters: a truncated or garbled Lackey trace
simulates cleanly and reports a miss rate for whatever survived.

### Compressed traces

Any path ending in `.gz` is compressed, transparently, on both read and write. The
extension that selects the format is the one before `.gz`, so `sequential.din.gz` is a
gzipped Dinero trace and `sequential.trace.gz` a gzipped native one.

Traces compress by roughly five to one, because every line is a fixed-width hexadecimal
address over a small range of pages. The same 65,536-access `sequential` workload in all
four combinations:

```bash
python - <<'PY'
from cachesim.trace import open_trace, write_trace

write_trace("traces/sequential.trace.gz", open_trace("traces/sequential.trace"))
write_trace("traces/sequential.din",      open_trace("traces/sequential.trace"), fmt="dinero")
write_trace("traces/sequential.din.gz",   open_trace("traces/sequential.trace"), fmt="dinero")
PY
ls -l traces/sequential.*
```

| File                          | Bytes   | Detected format |
| ----------------------------- | ------: | --------------- |
| `traces/sequential.trace`     | 851,968 | native          |
| `traces/sequential.trace.gz`  | 156,919 | native          |
| `traces/sequential.din`       | 720,896 | dinero          |
| `traces/sequential.din.gz`    | 151,668 | dinero          |

### Choosing the format

Format selection is by file extension, applied after any `.gz` suffix is stripped and
case-insensitively:

| Extension                     | Format  |
| ----------------------------- | ------- |
| `.din`                        | dinero  |
| `.lackey`, `.vg`              | lackey  |
| anything else, or none at all | native  |

The file is not opened to make the choice, and the extension is taken from the last path
component only, so a file named `x.trace` inside a directory named `dir.din` is native. A
file with no extension at all is native.

`cachesim run` and `cachesim trace-stats` both take
`--trace-format {auto,native,dinero,lackey}`, which overrides the extension; `auto` is the
default. An unknown value is rejected before the file is opened. On `cachesim run` the
flag is named `--trace-format` and not `--format` because `--format {text,json}` already
selects the output format there.

The analysis subcommands that replay a trace many times — `sweep`, `compare`, `sets`,
`mrc`, `policies` — detect the format from the extension and have no override flag, so a
foreign-format trace must be named for what it is, or converted first.

### Writing and converting traces

`cachesim.trace.write_trace(path, accesses, fmt="native")` emits any of the three formats
and returns the number of accesses written. Each access is `(address, op)`, where `op` is
`"R"`/`"W"` (what the workload generators yield) or an `is_write` bool (what the readers
yield); `.gz` paths are compressed. Reading a trace in one format and writing it in
another is therefore a conversion:

```bash
python - <<'PY'
from cachesim.trace import open_trace, write_trace

n = write_trace("traces/sequential.din.gz", open_trace("traces/sequential.trace"), fmt="dinero")
print(f"{n} accesses written")
PY
```

```text
65536 accesses written
```

Only the read/write distinction survives a conversion. The writers emit Dinero labels `0`
and `1` and Lackey `L` and `S` records, never an instruction fetch or a modify, and the
size field is the constant 8. Because the simulator ignores both of those on input,
converting changes nothing it can observe:

```bash
cachesim run traces/sequential.trace  --format json > /tmp/native.json
cachesim run traces/sequential.din.gz --format json > /tmp/dinero.json
diff /tmp/native.json /tmp/dinero.json && echo identical
```

```text
identical
```

Note that the repository's `.gitignore` covers `traces/*.trace` and `traces/*.trace.gz`
only, so a converted `.din` or `.lackey` file written into `traces/` shows up as
untracked.

### Streaming versus materialising

`open_trace(path, fmt="auto")` returns an iterator and holds the file open until it is
exhausted or dropped; one pass over a trace of any length costs constant memory.
`load_trace(path, fmt="auto")` returns the whole trace as a list, which costs a list slot,
a tuple and an address object per access:

```bash
python - <<'PY'
import tracemalloc
from cachesim.trace import load_trace

tracemalloc.start()
before = tracemalloc.get_traced_memory()[0]
trace = load_trace("traces/matmul_naive.trace")
after = tracemalloc.get_traced_memory()[0]
print(f"{len(trace):,} accesses, {after - before:,} bytes, {(after - before) / len(trace):.1f} per access")
PY
```

```text
532,480 accesses, 51,544,267 bytes, 96.8 per access
```

At roughly 97 bytes per access, a 10-million-access trace needs about 1 GB. `cachesim run`
and `cachesim trace-stats` stream; `sweep`, `compare`, `sets`, `mrc` and `policies`
materialise, because they replay the same trace once per sweep point and re-parsing the
file would dominate the run.

## The workload registry

Thirteen synthetic workloads are built in. Each one is a generator that yields
`(address, op)` pairs modelling one classic access pattern whose effect on a cache can be
predicted in closed form. The generators are pure and deterministic: parameters have
defaults, randomness comes from a `random.Random` seeded by an argument, and nothing
depends on the environment, so a name always produces the same trace byte for byte.

The registry maps a name to the generator (already bound to its default arguments), a
description of the pattern, and the expectation — the closed form or limiting behaviour
the pattern is designed to exhibit. `cachesim gen-traces --list` prints all three and
writes nothing:

```bash
cachesim gen-traces --list
```

```text
sequential
  Linear scan of a 256 KB buffer by 8-byte words, twice, with every tenth
  access a store.
  expectation: One miss per 64-byte block per pass that does not fit: the 32
  KB L1 misses 1 access in 8 (8 words per block), and the 256 KB L2 holds the
  buffer, so only the first pass reaches DRAM (4096 blocks).
...
```

`cachesim gen-traces NAME...` writes one native-format trace per name into `--out-dir`
(default `traces`), as `<out-dir>/<name>.trace`. With no names it writes the six samples
`sequential`, `random`, `matmul_naive`, `matmul_blocked`, `conflict` and `pointer_chase`,
which are the traces the golden regression tests pin, so their generators must not change
output once released:

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

Those six occupy 17,427,904 bytes; all thirteen occupy 21,998,782 bytes
(`wc -c traces/*.trace`). Trace files are gitignored and are meant to be regenerated
rather than stored.

Throughout, a word is 8 bytes (a double or a pointer) and the patterns are designed around
64-byte blocks, so eight consecutive words share one block.

### sequential

```python
sequential(buffer_bytes=262144, passes=2)
```

Linear scan of a 256 KB buffer by 8-byte words, twice, with every tenth access a store.

Expectation: one miss per 64-byte block per pass that does not fit. Eight words share a
block, so a cache smaller than the buffer misses 1 access in 8; the 256 KB L2 holds the
buffer, so only the first pass reaches DRAM, 4096 blocks.

### random

```python
random_access(region_bytes=16777216, n=60000, seed=1)
```

Uniformly random 8-byte accesses over a 16 MB region, a quarter of them stores. Registered
under the name `random`.

Expectation: no reuse to speak of. Under the independent reference model a cache of C
bytes settles at a miss rate of 1 - C/16 MB, which puts a 2 MB last level at 87.5% in
steady state; from cold over 60,000 accesses the measured global rate is higher, because
the cache starts empty and 60,000 accesses over 262,144 blocks rarely revisit one.

### matmul_naive

```python
matmul(n=64)          # tile=None: naive i/j/k order
```

C = A x B for 64x64 matrices of doubles in i/j/k loop order, walking B by column.

Expectation: B's column walk strides 512 B, a power of two, so a column's 64 blocks alias
onto 8 sets of the 4-way L1 and are evicted before the next column reuses them. The three
32 KB matrices total 96 KB, which the 256 KB L2 holds, so DRAM traffic still falls to the
compulsory floor of 3 x 512 = 1536 blocks.

### matmul_blocked

```python
matmul(n=64, tile=16)
```

The same 64x64 multiplication iterated in 16x16 tiles.

Expectation: each 16x16 tile of A, B and C is 2 KB, so all three stay in the 32 KB L1
while they are reused. L1 misses fall by a factor of about eight for the same 1536-block
DRAM footprint; the tiled order costs more accesses in total, because C[i][j] is read
before and written after every k-tile rather than once for the whole k loop.

### conflict

```python
conflict_streams(streams=4, words_per_stream=16384, align=0x100000)
```

Four sequential streams read in lockstep, their bases 1 MB apart. Registered under the
name `conflict`.

Expectation: 1 MB is 16,384 blocks, so element X of every stream maps to the same set for
any geometry with at most 16,384 sets. Whether the accesses hit is then decided only by
associativity: a 4-way cache holds all four hot blocks and misses 1 access in 8, while 1
or 2 ways miss every access. No block is ever revisited, so every miss is compulsory and
no lower level ever hits. This is the trace the associativity sweep uses.

### pointer_chase

```python
pointer_chase(nodes=16384, node_bytes=64, hops=60000, seed=2)
```

60,000 hops around a randomly shuffled 16,384-node linked list spanning 1 MB, one 64-byte
node per hop.

Expectation: the nodes form one cycle in shuffled order, so consecutive hops are far apart
and each node has a block to itself. The 32 KB L1 and 256 KB L2 never hit; the 2 MB L3
holds the whole list, so only the first lap misses there — 16,384 misses out of 60,000,
the compulsory floor. 60,000 hops over 16,384 nodes is about 3.7 laps.

### strided

```python
strided(stride_bytes=512, count=1024, passes=4, base=0x04000000)
```

Reads 1024 addresses 512 B apart, four times over.

Expectation, for a stride at least as large as the block size (so each access is a block
of its own) and LRU: every pass touches exactly `count` distinct blocks, a footprint of
`count * block_size`. If the footprint exceeds the capacity, the pass is a cyclic sweep
longer than the cache — LRU's worst case — and all `count` accesses of every pass miss. If
it fits, only the first pass misses. A power-of-two stride shrinks the effective capacity
further, because the blocks reach only `num_sets / (stride_bytes / block_size)` of the
sets (Bailey 1995): the 64 KB footprint would fit in the 256 KB L2, but a 512 B stride
reaches one set in eight and only the 16-way L3 keeps it.

### column_walk

```python
column_walk(rows=256, cols=256, row_stride_bytes=None, base=0x05000000)
```

Column-major walk of a 256x256 row-major matrix of doubles, so consecutive accesses are
one row stride apart. `row_stride_bytes` defaults to `cols * 8`, the unpadded layout,
here 2048 B.

Expectation: eight consecutive columns share one block per row, so column j+1 is entirely
reuse of the blocks column j fetched and the compulsory floor is `rows * ceil(cols / 8)` =
8192 blocks. Whether that reuse survives depends on the stride. With the unpadded
power-of-two stride the 256 blocks of a column land on only
`num_sets / (row_stride / block_size)` sets and evict one another before the next column
arrives. Padding the stride by one block makes the row-to-row distance an odd number of
blocks, coprime with any power-of-two set count, so a column spreads over every set and
the reuse is realised (Bailey 1995):

```bash
python - <<'PY'
from cachesim.workloads import column_walk, write_trace
write_trace("traces/column_walk_padded.trace", column_walk(row_stride_bytes=2112))
PY
cachesim run traces/column_walk.trace
cachesim run traces/column_walk_padded.trace
```

| Row stride                 | L1 misses | L1 miss rate | Total cycles |  AMAT |
| -------------------------- | --------: | -----------: | -----------: | ----: |
| 2048 B (unpadded, default) |    65,536 |      100.00% |    4,489,216 | 68.50 |
| 2112 B (padded one block)  |     8,192 |       12.50% |    1,507,328 | 23.00 |

Padding costs 64 bytes per row and removes every miss above the compulsory floor.

### stencil_2d

```python
stencil_2d(n=96, passes=2, base=0x06000000)
```

Two passes of a five-point Jacobi stencil over a 96x96 grid of doubles. Each interior
point reads in(i,j), in(i-1,j), in(i+1,j), in(i,j-1), in(i,j+1) and writes out(i,j): five
reads and one write, `6 * (n-2)**2` accesses per pass. `out` is placed one block past the
end of `in`, so the two arrays are offset by an odd number of blocks and do not
systematically alias.

Expectation, for a cache that holds three grid rows of `in` plus one of `out`
(`4 * n * 8` bytes, 3 KB at n=96): the sweep of row i has already fetched rows i-1 and i,
so the only new data is row i+1 of `in` and row i of `out`, one block per 8 elements each.
The miss rate tends to 2 misses per 8 interior points, or 1 in 24 accesses (4.17%). The
compulsory floor is 2280 blocks (Rivera and Tseng 2000).

### transpose

```python
transpose(n=128, base=0x07000000)
```

B = A-transpose for 128x128 doubles: A read row-major, B written column-major, with B
placed one block past the end of A. `2 * n**2` accesses.

Expectation: the reads of A are sequential, so 1 in 8 misses. The writes of B stride
`n * 8` = 1024 B, one block each, so every write is a fresh block; with a power-of-two n
those blocks alias onto `num_sets / (n*8 / block_size)` sets and a column of B is evicted
long before the next column reuses it. Misses per element therefore tend to 1 + 1/8 =
1.125 until the cache holds a whole column of blocks of B (Chatterjee and Sen 2000).

### binary_search

```python
binary_search(n_elements=65536, queries=5000, seed=3, base=0x08000000)
```

5000 binary searches for uniformly random keys in a sorted 512 KB array of 65,536 8-byte
keys. The array holds the keys 0..n-1 in order, so searching for the key at index t probes
the same positions a real search would (Knuth 1998); each probe reads one key and the
search stops on the hit.

Expectation: a query probes about log2(n_elements) = 16 positions — 15.0 measured, 74,982
accesses over 5000 queries — but the probes form a binary tree whose level d has only 2^d
distinct positions. The top levels are a handful of blocks and stay resident while the
deep levels are effectively random over the array, so a cache of C bytes holds the top
log2(C / block_size) levels and leaves roughly log2(n_elements * 8 / C) probes per query
to miss. The miss rate therefore falls steadily with cache size rather than dropping off a
cliff.

### hash_probe

```python
hash_probe(table_bytes=1048576, probes=60000, seed=4, base=0x09000000)
```

60,000 uniformly random 8-byte probes into a 1 MB table: the access pattern of a lookup in
a large hash table, where the index is unpredictable and every slot is equally likely, so
there is no locality beyond the block a probe lands in.

Expectation: under the independent reference model with uniform probabilities, an LRU
cache of C bytes holds a C/table_bytes fraction of the table and the steady-state miss
rate is 1 - C/table_bytes (Coffman and Denning 1973). For the 32 KB L1 that is
1 - 32768/1048576 = 96.875%. The measured rate approaches it from above, because the cache
starts empty.

### cyclic

```python
cyclic(blocks=1024, passes=8, block_bytes=64, base=0x0a000000)
```

Eight passes over 1024 consecutive 64-byte blocks, one access each: LRU's worst case.

Expectation, for a cache holding K blocks: if K >= 1024, only the first pass misses, 1024
compulsory misses in 8192 accesses. If K < 1024, LRU evicts each block exactly one access
before it is needed again, so every access of every pass misses — a 100% miss rate no
matter how close K is to 1024. The cliff between the two is the classic demonstration that
LRU is not resistant to cyclic reuse (Belady 1966). The 64 KB cycle exceeds the 32 KB L1
and fits the 256 KB L2, so this trace sits on the wrong side of the cliff at L1 and the
right side at L2.

### Measured results

Every workload generated and run against the default hierarchy — L1 32 KB 4-way, hit time
4; L2 256 KB 8-way, hit time 12; L3 2 MB 16-way, hit time 40; all LRU, write-back,
write-allocate, NINE; DRAM 100 cycles (`configs/default.json`):

```bash
cachesim gen-traces --out-dir traces \
    sequential random matmul_naive matmul_blocked conflict pointer_chase \
    strided column_walk stencil_2d transpose binary_search hash_probe cyclic
for w in sequential random matmul_naive matmul_blocked conflict pointer_chase \
         strided column_walk stencil_2d transpose binary_search hash_probe cyclic; do
    cachesim run "traces/$w.trace" --format json > "/tmp/$w.json"
done
python - <<'PY'
import json
names = "sequential random matmul_naive matmul_blocked conflict pointer_chase " \
        "strided column_walk stencil_2d transpose binary_search hash_probe cyclic".split()
for name in names:
    d = json.load(open(f"/tmp/{name}.json"))
    l1, l2, l3 = d["levels"]
    print(f"{name:15s} {d['accesses']:>9,} "
          f"{100 * l1['local_miss_rate']:6.2f} {100 * l2['local_miss_rate']:6.2f} "
          f"{100 * l3['local_miss_rate']:6.2f} {d['dram_reads']:>7,} {d['amat']:8.2f}")
PY
```

Miss rates are local: misses at a level divided by the accesses that reached it. AMAT is
in cycles, and equals the measured cycles per access exactly for this hierarchy.

| Workload         | Accesses | L1 miss | L2 miss | L3 miss | DRAM reads | AMAT   |
| ---------------- | -------: | ------: | ------: | ------: | ---------: | -----: |
| `sequential`     |   65,536 |  12.50% |  50.00% | 100.00% |      4,096 |  14.25 |
| `random`         |   60,000 |  99.82% |  98.69% |  92.61% |     54,739 | 146.61 |
| `matmul_naive`   |  532,480 |   5.24% |   5.50% | 100.00% |      1,536 |   5.03 |
| `matmul_blocked` |  557,056 |   0.64% |  43.00% | 100.00% |      1,536 |   4.46 |
| `conflict`       |   65,536 |  12.50% | 100.00% | 100.00% |      8,192 |  23.00 |
| `pointer_chase`  |   60,000 | 100.00% | 100.00% |  27.31% |     16,384 |  83.31 |
| `strided`        |    4,096 | 100.00% | 100.00% |  25.00% |      1,024 |  81.00 |
| `column_walk`    |   65,536 | 100.00% | 100.00% |  12.50% |      8,192 |  68.50 |
| `stencil_2d`     |  106,032 |   4.30% |  50.00% | 100.00% |      2,280 |   7.53 |
| `transpose`      |   32,768 |  56.25% |  22.22% | 100.00% |      4,096 |  28.25 |
| `binary_search`  |   74,982 |  83.14% |  48.67% |  22.08% |      6,698 |  39.09 |
| `hash_probe`     |   60,000 |  96.89% |  78.54% |  34.99% |     15,975 |  72.69 |
| `cyclic`         |    8,192 | 100.00% |  12.50% | 100.00% |      1,024 |  33.50 |

DRAM writes are zero for every workload except `random`, which writes back 5,823 dirty
blocks; every other trace either is read-only or keeps its dirty blocks resident to the
end of the run.

Each measured figure lands where the closed form puts it:

* `sequential` L1 12.50% is exactly 1 miss per 8 words in a block, and L2 50.00% is the
  first of two passes.
* `conflict` L1 12.50% is the same 1-in-8, confirming that four ways hold four streams
  that all map to one set.
* `matmul_blocked` cuts L1 misses from 27,912 to 3,572 against `matmul_naive`, 5.24% to
  0.64%, for the identical 1536-block DRAM footprint.
* `stencil_2d` 4.30% approaches the asymptotic 1/24 = 4.17% from above: 2280 L1 misses per
  pass against the 2209 the asymptote predicts for 94x94 interior points, the difference
  being the grid edges. The 4560 misses classify as 2280 compulsory and 2280 capacity with
  no conflict — the 142.5 KB array pair cannot survive from one pass to the next in a
  32 KB L1.
* `transpose` misses 18,432 times on 16,384 elements: 1.125 per element, exactly the
  predicted 1 + 1/8.
* `hash_probe` L1 96.89% against the 96.875% the independent reference model predicts.
* `cyclic` L1 100.00% and L2 12.50% are the two sides of the LRU cliff in one run.
* `strided` and `column_walk` miss everything at L1 and L2 despite footprints of 64 KB and
  512 KB, because a power-of-two stride confines them to one set in eight and one in
  thirty-two.

DRAM reads equal the trace's distinct-block count for twelve of the thirteen workloads, so
no block is fetched from memory twice and the hierarchy is already at the compulsory floor.
The exception is `random`, whose 3.3 MB footprint does not fit in the 2 MB L3, so its
54,739 DRAM reads exceed the 53,671-block floor.

A 100% local miss rate at the L3 therefore reads as "every block that got this far was a
first touch", not as a failure: `sequential`, `matmul_naive`, `matmul_blocked`, `conflict`,
`stencil_2d`, `transpose` and `cyclic` all sit at the floor with an L3 that never hits.

## Trace statistics

`cachesim trace-stats` reports properties of the address stream alone, with no cache
involved, so its figures bound what any cache can achieve. The distinct-block count is the
compulsory-miss floor — no cache, however large or associative, can miss fewer times than
that on a cold start (Hill and Smith 1989) — and the first-touch fraction is that floor
expressed as a miss rate. The distinct 4 KB page count bounds TLB behaviour the same way.
Nothing is simulated, so it is fast enough to run before deciding which geometries are
worth sweeping.

```bash
for w in sequential random matmul_naive matmul_blocked conflict pointer_chase; do
    cachesim trace-stats "traces/$w.trace"
done
```

| Trace                    | Accesses | Reads   | Writes | Blocks | Footprint | 4 KB pages | Compulsory floor |
| ------------------------ | -------: | ------: | -----: | -----: | --------: | ---------: | ---------------: |
| `sequential.trace`       |   65,536 |  58,983 |  6,553 |  4,096 |  256.0 KB |         64 |            6.25% |
| `random.trace`           |   60,000 |  44,832 | 15,168 | 53,671 |    3.3 MB |      4,096 |           89.45% |
| `matmul_naive.trace`     |  532,480 | 528,384 |  4,096 |  1,536 |   96.0 KB |         24 |            0.29% |
| `matmul_blocked.trace`   |  557,056 | 540,672 | 16,384 |  1,536 |   96.0 KB |         24 |            0.28% |
| `conflict.trace`         |   65,536 |  65,536 |      0 |  8,192 |  512.0 KB |        128 |           12.50% |
| `pointer_chase.trace`    |   60,000 |  60,000 |      0 | 16,384 |    1.0 MB |        256 |           27.31% |

The footprint column is the block count times the 64-byte block size, and says which level
of the hierarchy could hold the whole working set: `matmul_naive` and `matmul_blocked` fit
inside the 256 KB L2 with room to spare, `pointer_chase` needs the 2 MB L3, and `random`
needs more than the L3 has. The two matmul traces have identical footprints and page
counts — blocking changes the order of the accesses, not the set of blocks they touch —
which is why their DRAM traffic is identical while their L1 miss rates differ by roughly a
factor of eight.

`--block-size B` recomputes the block count and footprint at another block size, and
`--format json` emits every field, including the address range, as a JSON document.

## Bringing your own trace

Any tool that can print one memory reference per line can feed the simulator. The native
format is two fields and needs no library.

### From Valgrind

Lackey is distributed with Valgrind and needs no build step. Its trace records go to
Valgrind's log stream, so send that to a file rather than mixing it with the program's own
output:

```bash
valgrind --tool=lackey --trace-mem=yes --log-file=app.lackey ./app
cachesim run app.lackey
```

The `.lackey` extension selects the reader; `--trace-format lackey` forces it for a file
named otherwise. Instruction fetches are part of Lackey's output and are decoded as reads,
which is what a unified hierarchy wants. There is no simulator option to drop them, so a
data-only study filters them out first — the reader ignores whatever the filter leaves
behind:

```bash
grep -v '^I' app.lackey > data.lackey
cachesim run data.lackey
```

Lackey traces are large — one record per memory reference of the whole process, startup
and dynamic linking included — so compress them, and discount the startup with a warm-up
count:

```bash
valgrind --tool=lackey --trace-mem=yes --log-file=app.lackey ./app
gzip app.lackey
cachesim run app.lackey.gz --warmup 1000000
```

`--warmup N` simulates the first N accesses and then resets every statistic before
counting the rest, so the caches are warm and the process's startup does not appear in the
report.

### From Intel Pin

The `pinatrace` example tool prints one line per access as
`INSTRUCTION_POINTER: R|W ADDRESS`, with comment lines around it. The third field is the
data address and the second is the operation, which is the native format with the fields
transposed:

```bash
awk '/: [RW] /{print $3, $2}' pinatrace.out > pin.trace
cachesim trace-stats pin.trace
```

The `/: [RW] /` guard drops the header and `#eof` lines. A four-record `pinatrace.out`:

```text
#
# Memory Access Trace Generated By Pin
#
0x00007f2b1c0a1b40: R 0x0000000000100000
0x00007f2b1c0a1b44: W 0x0000000000100008
0x00007f2b1c0a1b48: R 0x0000000000100040
0x00007f2b1c0a1b4c: R 0x0000000000100000
#eof
```

becomes a native trace:

```text
0x0000000000100000 R
0x0000000000100008 W
0x0000000000100040 R
0x0000000000100000 R
```

which `trace-stats` reads as two distinct blocks over one page:

```text
TRACE STATISTICS  (pin.trace)
  accesses        :            4
  reads           :            3   ( 75.0%)
  writes          :            1   ( 25.0%)
  distinct blocks :            2   (footprint 128 B at 64 B blocks)
  distinct pages  :            1   (4.0 KB pages)
  address range   : 0x000000100000 .. 0x000000100040   (span 64 B)
  first touches   :            2   (50.00% of accesses: the compulsory-miss floor)
```

The same one-liner adapts to any line-oriented log: pick the address field and the
operation field, print them in that order, and let the extension or `--trace-format`
select `native`. If the log distinguishes more operations than read and write, map them
down first — the simulator has only those two.

### Checking a converted trace

Run `cachesim trace-stats` on a new trace before simulating it. Three numbers catch most
conversion mistakes: the access count should match what the producing tool reported, the
address range should look like the address space the program used, and the read/write
split should match its behaviour. A trace that converted into all reads, or whose span is
a few hundred bytes, was almost certainly parsed with the wrong field order.

## References

* D. H. Bailey, "Unfavorable Strides in Cache Memory Systems", Scientific Programming
  4(2), 1995.
* L. A. Belady, "A Study of Replacement Algorithms for a Virtual-Storage Computer", IBM
  Systems Journal 5(2), 1966.
* S. Chatterjee and S. Sen, "Cache-Efficient Matrix Transposition", HPCA 2000.
* E. G. Coffman and P. J. Denning, "Operating Systems Theory", Prentice-Hall, 1973,
  chapter 6.
* J. Edler and M. D. Hill, "Dinero IV: Trace-Driven Uniprocessor Cache Simulator",
  University of Wisconsin-Madison, 1998.
* M. D. Hill and A. J. Smith, "Evaluating Associativity in CPU Caches", IEEE Transactions
  on Computers 38(12), 1989.
* D. E. Knuth, "The Art of Computer Programming", Vol. 3, 2nd ed., 1998, section 6.2.1.
* N. Nethercote and J. Seward, "Valgrind: A Framework for Heavyweight Dynamic Binary
  Instrumentation", PLDI 2007.
* G. Rivera and C.-W. Tseng, "Tiling Optimizations for 3D Scientific Computations", SC
  2000.

---

[Back to the README](../README.md) | [model.md](model.md) |
[configuration.md](configuration.md) | [cli.md](cli.md) | [validation.md](validation.md) |
[results.md](results.md)
