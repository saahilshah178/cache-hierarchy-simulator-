"""Machine-readable results of a simulation.

``HierarchyStats`` is a plain dataclass snapshot of every counter a
``Hierarchy`` and its levels maintain, plus the derived rates. The text
report and the JSON output are both rendered from it, so they can never
disagree. ``to_dict`` produces the JSON form; ``SCHEMA_VERSION`` is bumped
whenever a field is renamed or its meaning changes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cachesim.hierarchy import Hierarchy

#: Bumped to 2 when the hierarchy gained inclusion policies, per-level write
#: policies, transfer-time and byte accounting, prefetchers, and the DRAM
#: row-buffer model; every one of those added fields to this snapshot.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ThreeCStats:
    """Miss classification for one level; see ``cachesim.cache``."""

    compulsory: int
    capacity: int
    conflict: int
    capacity_aggregate: int
    conflict_aggregate: int
    shadow_misses: int
    anti_conflict_hits: int


@dataclass(frozen=True)
class LevelStats:
    """Counters and derived rates for one cache level."""

    name: str
    size: int
    block_size: int
    associativity: int
    num_sets: int
    policy: str
    hit_time: int
    inclusion: str
    write_policy: str
    write_allocate: bool
    bus_width: int | None
    transfer_cycles: int
    prefetcher: str
    accesses: int
    hits: int
    misses: int
    read_hits: int
    read_misses: int
    write_hits: int
    write_misses: int
    local_miss_rate: float
    global_miss_rate: float
    fills: int
    evictions: int
    invalidations: int
    writebacks: int
    writebacks_received: int
    writeback_allocations: int
    back_invalidations: int
    write_throughs: int
    write_bypasses: int
    bytes_read_from_below: int
    bytes_written_below: int
    prefetches_issued: int
    prefetch_hits: int
    prefetch_evicted_unused: int
    prefetch_probes: int
    prefetch_probe_hits: int
    prefetch_accuracy: float
    prefetch_coverage: float
    three_c: ThreeCStats | None


@dataclass(frozen=True)
class MemoryStats:
    """The main-memory model and what it saw; see ``cachesim.dram``.

    ``average_latency`` is the mean over the accesses the hierarchy
    charged for -- demand fetches and stores no level allocated for -- and
    is the DRAM term the analytic AMAT uses. The row-buffer figures are
    zero for a constant-latency memory, which has no rows.
    """

    type: str
    parameters: dict[str, int]
    average_latency: float
    row_hits: int
    row_misses: int
    row_buffer_hit_rate: float


@dataclass(frozen=True)
class HierarchyStats:
    """Counters and derived metrics for a whole hierarchy."""

    accesses: int
    reads: int
    writes: int
    dram_reads: int
    dram_writes: int
    dram_demand_writes: int
    dram_prefetch_reads: int
    dram_bytes_read: int
    dram_bytes_written: int
    memory_access_time: int | None
    memory: MemoryStats
    total_cycles: int
    read_cycles: int
    write_cycles: int
    amat: float
    measured_amat: float
    read_amat: float
    write_amat: float
    levels: list[LevelStats] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict, with ``schema_version`` first."""
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def collect(hierarchy: Hierarchy) -> HierarchyStats:
    """Snapshot the statistics of ``hierarchy``."""
    h = hierarchy
    levels = []
    for i, level in enumerate(h.levels):
        c = level.cache
        three_c = None
        if c.track_3c:
            three_c = ThreeCStats(
                compulsory=c.compulsory_misses,
                capacity=c.capacity_misses,
                conflict=c.conflict_misses,
                capacity_aggregate=c.capacity_misses_aggregate,
                conflict_aggregate=c.conflict_misses_aggregate,
                shadow_misses=c.shadow_misses,
                anti_conflict_hits=c.anti_conflict_hits,
            )
        levels.append(
            LevelStats(
                name=c.name,
                size=c.size,
                block_size=c.block_size,
                associativity=c.associativity,
                num_sets=c.num_sets,
                policy=c.policy_name,
                hit_time=level.hit_time,
                inclusion=level.inclusion,
                write_policy=level.write_policy,
                write_allocate=level.write_allocate,
                bus_width=level.bus_width,
                transfer_cycles=level.transfer_cycles,
                prefetcher=level.prefetcher_name,
                accesses=c.accesses,
                hits=c.hits,
                misses=c.misses,
                read_hits=c.read_hits,
                read_misses=c.read_misses,
                write_hits=c.write_hits,
                write_misses=c.write_misses,
                local_miss_rate=c.miss_rate,
                global_miss_rate=h.global_miss_rate(i),
                fills=c.fills,
                evictions=c.evictions,
                invalidations=c.invalidations,
                writebacks=c.writebacks,
                writebacks_received=c.writebacks_received,
                writeback_allocations=c.writeback_allocations,
                back_invalidations=level.back_invalidations,
                write_throughs=level.write_throughs,
                write_bypasses=level.write_bypasses,
                bytes_read_from_below=level.bytes_read_from_below,
                bytes_written_below=level.bytes_written_below,
                prefetches_issued=level.prefetches_issued,
                prefetch_hits=level.prefetch_hits,
                prefetch_evicted_unused=level.prefetch_evicted_unused,
                prefetch_probes=level.prefetch_probes,
                prefetch_probe_hits=level.prefetch_probe_hits,
                prefetch_accuracy=level.prefetch_accuracy,
                prefetch_coverage=level.prefetch_coverage,
                three_c=three_c,
            )
        )
    return HierarchyStats(
        accesses=h.accesses,
        reads=h.reads,
        writes=h.writes,
        dram_reads=h.dram_reads,
        dram_writes=h.dram_writes,
        dram_demand_writes=h.dram_demand_writes,
        dram_prefetch_reads=h.dram_prefetch_reads,
        dram_bytes_read=h.dram_bytes_read,
        dram_bytes_written=h.dram_bytes_written,
        memory_access_time=h.memory_access_time,
        memory=MemoryStats(
            type=h.memory.name,
            parameters=h.memory.parameters,
            average_latency=h.memory.average_latency(),
            row_hits=h.memory.row_hits,
            row_misses=h.memory.row_misses,
            row_buffer_hit_rate=h.memory.row_buffer_hit_rate,
        ),
        total_cycles=h.total_time,
        read_cycles=h.read_cycles,
        write_cycles=h.write_cycles,
        amat=h.amat(),
        measured_amat=h.measured_amat(),
        read_amat=h.read_amat(),
        write_amat=h.write_amat(),
        levels=levels,
    )
