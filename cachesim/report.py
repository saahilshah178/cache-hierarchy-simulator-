"""Renders a ``HierarchyStats`` snapshot as a human-readable text report.

Everything here is presentation: per-level hit/miss tables, local and global
miss rates, the miss classification, and AMAT.
"""

from __future__ import annotations

from cachesim.hierarchy import Hierarchy
from cachesim.plot import format_bytes
from cachesim.stats import HierarchyStats

BAR = "=" * 72


def _pct(part: float, whole: float) -> str:
    """Format part/whole as a percentage string, safe against divide-by-zero."""
    return f"{100.0 * part / whole:6.2f}%" if whole else "   n/a "


def format_size(nbytes: int) -> str:
    """32768 -> '32.0 KB', 2097152 -> '2.0 MB'.

    The report's fixed-one-decimal convention, from the one implementation
    the whole package shares; ``plot.format_bytes`` without ``decimals`` is
    the same figure with a trailing zero dropped, which is what the
    analysis tables and the axis labels use.
    """
    return format_bytes(nbytes, decimals=1)


def format_report(stats: HierarchyStats, trace_name: str | None = None) -> str:
    """Return the full multi-line report for a statistics snapshot."""
    s = stats
    lines: list[str] = [BAR]
    title = "CACHE HIERARCHY SIMULATION REPORT"
    if trace_name:
        title += f"  ({trace_name})"
    lines.append(title)
    lines.append(BAR)
    lines.append(f"Total accesses : {s.accesses:>12,}   (reads {s.reads:,} / writes {s.writes:,})")
    lines.append(
        f"DRAM reads     : {s.dram_reads:>12,}"
        f"   ({_pct(s.dram_reads, s.accesses).strip()} of all accesses missed every level)"
    )
    if s.dram_demand_writes:
        detail = (
            f"(modified blocks leaving {s.levels[-1].name}; "
            f"{s.dram_demand_writes:,} were stores no level allocated for)"
        )
    else:
        detail = f"(dirty lines written back from {s.levels[-1].name})"
    lines.append(f"DRAM writes    : {s.dram_writes:>12,}   {detail}")
    if s.dram_prefetch_reads:
        lines.append(
            f"DRAM prefetches: {s.dram_prefetch_reads:>12,}"
            "   (speculative fetches that missed every level; untimed)"
        )
    lines.append("")

    for i, lv in enumerate(s.levels):
        # Only features that are switched on appear in the header, so a
        # default hierarchy reports exactly what it used to.
        extra = "" if lv.inclusion == "nine" else f", {lv.inclusion} of {s.levels[i - 1].name}"
        if lv.index != "modulo":
            extra += f", {lv.index} indexing"
        if lv.write_policy != "write-back":
            extra += f", {lv.write_policy}"
        if not lv.write_allocate:
            extra += ", no-write-allocate"
        if lv.bus_width is not None:
            extra += f", {lv.bus_width} B/cyc bus ({lv.transfer_cycles} cyc/block)"
        if lv.prefetcher != "none":
            extra += f", {lv.prefetcher} prefetch"
        if lv.victim_cache_entries:
            extra += f", {lv.victim_cache_entries}-entry victim cache"
        lines.append(
            f"--- {lv.name}: {format_size(lv.size)}, {lv.block_size} B blocks, "
            f"{lv.associativity}-way, {lv.policy.upper()}, hit time {lv.hit_time} cyc"
            f"{extra} ---"
        )
        lines.append(f"  accesses    : {lv.accesses:>12,}")
        lines.append(
            f"  hits        : {lv.hits:>12,}   (reads {lv.read_hits:,} / writes {lv.write_hits:,})"
        )
        lines.append(
            f"  misses      : {lv.misses:>12,}   "
            f"(reads {lv.read_misses:,} / writes {lv.write_misses:,})"
        )
        lines.append(
            f"  local miss rate  : {_pct(lv.misses, lv.accesses)}"
            "   (misses / accesses that reached this level)"
        )
        lines.append(
            f"  global miss rate : {100 * lv.global_miss_rate:6.2f}%   (misses / all CPU accesses)"
        )
        lines.append(
            f"  evictions   : {lv.evictions:>12,}   (writebacks of dirty blocks: {lv.writebacks:,})"
        )
        if i > 0:
            lines.append(
                f"  writebacks received : {lv.writebacks_received:>6,}"
                f"   (from {s.levels[i - 1].name}; {lv.writeback_allocations:,} allocated a line)"
            )
        if lv.back_invalidations:
            lines.append(
                f"  back-invalidated    : {lv.back_invalidations:>6,}"
                "   (lines dropped because an inclusive level below evicted the block)"
            )
        if lv.write_throughs:
            lines.append(
                f"  write-throughs      : {lv.write_throughs:>6,}"
                "   (stores duplicated to the level below)"
            )
        if lv.write_bypasses:
            lines.append(
                f"  write bypasses      : {lv.write_bypasses:>6,}"
                "   (write misses passed down instead of allocating)"
            )
        lines.append(
            f"  traffic     : {format_size(lv.bytes_read_from_below):>12} in from below"
            f" / {format_size(lv.bytes_written_below)} out below"
        )
        if lv.prefetcher != "none":
            lines.append(
                f"  prefetches issued   : {lv.prefetches_issued:>6,}"
                f"   (useful {lv.prefetch_hits:,},"
                f" evicted unused {lv.prefetch_evicted_unused:,})"
            )
            lines.append(
                f"    accuracy  : {100 * lv.prefetch_accuracy:6.2f}%   (useful / issued)"
                f"    coverage : {100 * lv.prefetch_coverage:6.2f}%"
                "   (useful / (useful + demand misses))"
            )
        if lv.victim_cache_entries:
            lines.append(
                f"  victim cache hits   : {lv.victim_hits:>6,}"
                "   (array misses the victim buffer served)"
            )
        if lv.prefetch_probes:
            lines.append(
                f"  prefetch lookups    : {lv.prefetch_probes:>6,}"
                f"   (from a level above; {lv.prefetch_probe_hits:,} found the block here)"
            )
        c3 = lv.three_c
        if c3 is not None and lv.misses:
            lines.append(f"  miss classification     {'per-reference':>14} {'aggregate':>12}")
            lines.append(f"    compulsory            {c3.compulsory:>14,} {c3.compulsory:>12,}")
            lines.append(
                f"    capacity              {c3.capacity:>14,} {c3.capacity_aggregate:>12,}"
            )
            lines.append(
                f"    conflict              {c3.conflict:>14,} {c3.conflict_aggregate:>12,}"
            )
            lines.append(
                f"    fully-associative LRU misses: {c3.shadow_misses:,}"
                f"   hits it would have missed: {c3.anti_conflict_hits:,}"
            )
        lines.append("")

    lines.append(BAR)
    lines.append(
        f"DRAM traffic            : {format_size(s.dram_bytes_read)} read"
        f" / {format_size(s.dram_bytes_written)} written"
    )
    m = s.memory
    if m.type != "constant":
        params = ", ".join(f"{k} {v}" for k, v in m.parameters.items())
        lines.append(f"DRAM model              : {m.type} ({params})")
        lines.append(
            f"  row buffer            : {100 * m.row_buffer_hit_rate:6.2f}% hits"
            f"   ({m.row_hits:,} hits / {m.row_misses:,} misses)"
        )
        lines.append(
            f"  average latency       : {m.average_latency:8.3f} cycles"
            "   (measured; the analytic AMAT below uses it)"
        )
    lines.append(f"AMAT (analytic formula) : {s.amat:8.3f} cycles")
    lines.append(
        f"AMAT (measured)         : {s.measured_amat:8.3f} cycles"
        f"   ({s.total_cycles:,} cycles / {s.accesses:,} accesses)"
    )
    lines.append(
        f"  loads                 : {s.read_amat:8.3f} cycles"
        f"   ({s.read_cycles:,} cycles / {s.reads:,} loads)"
    )
    lines.append(
        f"  stores                : {s.write_amat:8.3f} cycles"
        f"   ({s.write_cycles:,} cycles / {s.writes:,} stores)"
    )
    lines.append(BAR)
    return "\n".join(lines)


def build_report(hierarchy: Hierarchy, trace_name: str | None = None) -> str:
    """Return the full multi-line report string for a finished simulation."""
    return format_report(hierarchy.stats(), trace_name)


def print_report(hierarchy: Hierarchy, trace_name: str | None = None) -> None:
    print(build_report(hierarchy, trace_name))
