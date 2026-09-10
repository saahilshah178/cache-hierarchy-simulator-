"""Main-memory models sitting below the last cache level.

The hierarchy asks main memory for one block at a time and is told how
many cycles it took. Two models answer:

``ConstantMemory``
    Every access costs the same number of cycles. This is the original
    behaviour of ``memory_access_time`` and remains the default.

``RowBufferMemory``
    An open-page DRAM: each bank holds one activated row in its sense
    amplifiers, and an access that finds its row already open (a row
    buffer hit) is far cheaper than one that has to activate a new row.
    This is where the address stream stops being a set of independent
    misses and starts being a pattern: a sequential scan walks a row at a
    time and hits almost always, while random addresses activate a new
    row nearly every time, so two workloads with identical miss counts
    can differ by more than a factor of two in memory time.

    Modelled after the open-page (row-buffer-hit-first) policy described
    by Rixner, Dally, Kapasi, Mattson and Owens, "Memory Access
    Scheduling", ISCA 2000, and by Jacob, Ng and Wang, "Memory Systems:
    Cache, DRAM, Disk", Morgan Kaufmann 2007, chapter 13.

What is not modelled: refresh, bus turnaround between reads and writes,
request scheduling and reordering, bank conflicts as a queueing effect
(banks never make a request wait, they only change its latency), and
anything below block granularity. Reads and writes cost the same.

Timing convention: every access consults the row buffer and updates it,
but only accesses the hierarchy actually charges -- demand fetches, and
stores that no cache level allocated for -- enter ``average_latency()``.
Write-backs and prefetches occupy a bank and shift the open rows, which
is what makes them interfere with demand traffic, but they are assumed to
be drained off the critical path and so are not charged.
"""

from __future__ import annotations


class MemoryModel:
    """Base class: answers block-granular reads and writes with a latency.

    Subclasses maintain ``row_hits``/``row_misses`` if the notion applies
    to them and expose their configuration through ``parameters`` so the
    report and the JSON can print it without knowing the subclass.
    """

    #: Name used in configurations and in the statistics snapshot.
    name = "constant"

    def __init__(self) -> None:
        self.reads = 0
        self.writes = 0
        self.row_hits = 0
        self.row_misses = 0
        self.charged_accesses = 0
        self.charged_cycles = 0

    @property
    def parameters(self) -> dict[str, int]:
        """The model's configuration, for reports and JSON."""
        return {}

    @property
    def constant_latency(self) -> int | None:
        """The latency of every access, or None if it varies.

        A hierarchy uses this to skip the model call entirely on the
        common path when memory has nothing to remember.
        """
        return None

    def read(self, block_addr: int, charged: bool = True) -> int:
        """Fetch the block at byte address ``block_addr``; returns cycles."""
        raise NotImplementedError

    def write(self, block_addr: int, charged: bool = False) -> int:
        """Store the block at byte address ``block_addr``; returns cycles."""
        raise NotImplementedError

    def average_latency(self) -> float:
        """Mean cycles of the accesses the hierarchy charged for.

        This is the figure the analytic AMAT uses as its DRAM term, so
        that it remains comparable with the measured AMAT even when the
        latency is not constant.
        """
        raise NotImplementedError

    def reset_stats(self) -> None:
        """Zero the counters, keeping any state (such as open rows)."""
        self.reads = self.writes = 0
        self.row_hits = self.row_misses = 0
        self.charged_accesses = self.charged_cycles = 0

    @property
    def row_buffer_hit_rate(self) -> float:
        """Row buffer hits / accesses; 0.0 when the model has no rows."""
        total = self.row_hits + self.row_misses
        return self.row_hits / total if total else 0.0


class ConstantMemory(MemoryModel):
    """Every access costs ``latency`` cycles, whatever the address.

    The model the simulator has always used: it makes AMAT a closed-form
    function of the miss rates alone, which is exactly why it hides the
    difference between a sequential and a random miss stream.
    """

    name = "constant"

    def __init__(self, latency: int) -> None:
        super().__init__()
        if isinstance(latency, bool) or not isinstance(latency, int):
            raise ValueError(f"memory latency must be an integer, got {latency!r}")
        if latency < 0:
            raise ValueError(f"memory latency must be non-negative, got {latency}")
        self.latency = latency

    @property
    def parameters(self) -> dict[str, int]:
        return {"latency": self.latency}

    @property
    def constant_latency(self) -> int | None:
        return self.latency

    def read(self, block_addr: int, charged: bool = True) -> int:
        self.reads += 1
        if charged:
            self.charged_accesses += 1
            self.charged_cycles += self.latency
        return self.latency

    def write(self, block_addr: int, charged: bool = False) -> int:
        self.writes += 1
        if charged:
            self.charged_accesses += 1
            self.charged_cycles += self.latency
        return self.latency

    def average_latency(self) -> float:
        """Always the fixed latency, even before any access has happened,
        so the analytic AMAT is well defined on an untouched hierarchy."""
        return float(self.latency)


class RowBufferMemory(MemoryModel):
    """Open-page DRAM: one activated row per bank.

    A block's row is ``block_addr // row_size`` and its bank is
    ``row // banks``'s remainder, so consecutive rows land in different
    banks and a scan long enough to leave one row still finds the next in
    a bank of its own. An access whose row is already open costs
    ``row_hit`` cycles; otherwise it costs ``row_miss``, and that row
    becomes the open one for its bank -- there is no separate precharge
    penalty for the row it displaced, so ``row_miss`` stands for
    precharge plus activate plus column access together.

    All banks start closed, so the first access to each is a row miss.
    """

    name = "row-buffer"

    def __init__(
        self,
        row_size: int = 8192,
        banks: int = 8,
        row_hit: int = 40,
        row_miss: int = 100,
    ) -> None:
        super().__init__()
        for label, value in (
            ("row_size", row_size),
            ("banks", banks),
            ("row_hit", row_hit),
            ("row_miss", row_miss),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"row-buffer memory: {label} must be an integer, got {value!r}")
        if row_size < 1:
            raise ValueError(f"row-buffer memory: row_size must be positive, got {row_size}")
        if banks < 1:
            raise ValueError(f"row-buffer memory: banks must be positive, got {banks}")
        if row_hit < 0:
            raise ValueError(f"row-buffer memory: row_hit must be non-negative, got {row_hit}")
        if row_miss < row_hit:
            raise ValueError(
                f"row-buffer memory: row_miss ({row_miss}) must be at least "
                f"row_hit ({row_hit}); an open row is never the slower case"
            )
        self.row_size = row_size
        self.banks = banks
        self.row_hit = row_hit
        self.row_miss = row_miss
        #: The row currently activated in each bank; -1 means closed.
        self._open_rows = [-1] * banks

    @property
    def parameters(self) -> dict[str, int]:
        return {
            "row_size": self.row_size,
            "banks": self.banks,
            "row_hit": self.row_hit,
            "row_miss": self.row_miss,
        }

    def _service(self, block_addr: int, charged: bool) -> int:
        row = block_addr // self.row_size
        bank = row % self.banks
        if self._open_rows[bank] == row:
            self.row_hits += 1
            latency = self.row_hit
        else:
            self.row_misses += 1
            self._open_rows[bank] = row
            latency = self.row_miss
        if charged:
            self.charged_accesses += 1
            self.charged_cycles += latency
        return latency

    def read(self, block_addr: int, charged: bool = True) -> int:
        self.reads += 1
        return self._service(block_addr, charged)

    def write(self, block_addr: int, charged: bool = False) -> int:
        self.writes += 1
        return self._service(block_addr, charged)

    def average_latency(self) -> float:
        """Mean latency of the charged accesses, falling back to the
        closed-page cost before any have happened."""
        if not self.charged_accesses:
            return float(self.row_miss)
        return self.charged_cycles / self.charged_accesses

    def open_rows(self) -> list[int]:
        """The row currently activated in each bank; -1 for a closed bank."""
        return list(self._open_rows)
