"""Turns a simulated Hierarchy into a human-readable report.

Everything here is presentation: per-level hit/miss tables, local and global
miss rates, the 3-C miss breakdown, and AMAT.
"""

from __future__ import annotations

from cachesim.hierarchy import Hierarchy


def _pct(part: float, whole: float) -> str:
    """Format part/whole as a percentage string, safe against divide-by-zero."""
    return f"{100.0 * part / whole:6.2f}%" if whole else "   n/a "


def _size_str(nbytes: int) -> str:
    """32768 -> '32.0 KB', 2097152 -> '2.0 MB'."""
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


def build_report(hierarchy: Hierarchy, trace_name: str | None = None) -> str:
    """Return the full multi-line report string for a finished simulation."""
    h = hierarchy
    lines: list[str] = []
    bar = "=" * 72

    lines.append(bar)
    title = "CACHE HIERARCHY SIMULATION REPORT"
    if trace_name:
        title += f"  ({trace_name})"
    lines.append(title)
    lines.append(bar)
    lines.append(f"Total accesses : {h.accesses:>12,}   (reads {h.reads:,} / writes {h.writes:,})")
    lines.append(
        f"DRAM reads     : {h.dram_reads:>12,}"
        f"   ({_pct(h.dram_reads, h.accesses).strip()} of all accesses missed every level)"
    )
    lines.append(
        f"DRAM writes    : {h.dram_writes:>12,}"
        f"   (dirty lines written back from {h.levels[-1].cache.name})"
    )
    lines.append("")

    # --- per-level detail ---------------------------------------------------
    for i, level in enumerate(h.levels):
        c = level.cache
        lines.append(
            f"--- {c.name}: {_size_str(c.size)}, "
            f"{c.block_size} B blocks, {c.associativity}-way, "
            f"{c.policy_name.upper()}, hit time {level.hit_time} cyc ---"
        )
        lines.append(f"  accesses    : {c.accesses:>12,}")
        lines.append(
            f"  hits        : {c.hits:>12,}   (reads {c.read_hits:,} / writes {c.write_hits:,})"
        )
        lines.append(
            f"  misses      : {c.misses:>12,}   "
            f"(reads {c.read_misses:,} / writes {c.write_misses:,})"
        )
        lines.append(
            f"  local miss rate  : {_pct(c.misses, c.accesses)}"
            "   (misses / accesses that reached this level)"
        )
        lines.append(
            f"  global miss rate : {100 * h.global_miss_rate(i):6.2f}%"
            "   (misses / all CPU accesses)"
        )
        lines.append(
            f"  evictions   : {c.evictions:>12,}   (writebacks of dirty blocks: {c.writebacks:,})"
        )
        if i > 0:
            lines.append(
                f"  writebacks received : {c.writebacks_received:>6,}"
                f"   (from {h.levels[i - 1].cache.name}; "
                f"{c.writeback_allocations:,} allocated a line)"
            )
        if c.track_3c and c.misses:
            lines.append(f"  miss classification     {'per-reference':>14} {'aggregate':>12}")
            lines.append(
                f"    compulsory            {c.compulsory_misses:>14,} {c.compulsory_misses:>12,}"
            )
            lines.append(
                f"    capacity              {c.capacity_misses:>14,}"
                f" {c.capacity_misses_aggregate:>12,}"
            )
            lines.append(
                f"    conflict              {c.conflict_misses:>14,}"
                f" {c.conflict_misses_aggregate:>12,}"
            )
            lines.append(
                f"    fully-associative LRU misses: {c.shadow_misses:,}"
                f"   hits it would have missed: {c.anti_conflict_hits:,}"
            )
        lines.append("")

    # --- headline numbers -----------------------------------------------------
    lines.append(bar)
    lines.append(f"AMAT (analytic formula) : {h.amat():8.3f} cycles")
    lines.append(
        f"AMAT (measured)         : {h.measured_amat():8.3f} cycles"
        f"   ({h.total_time:,} cycles / {h.accesses:,} accesses)"
    )
    lines.append(bar)
    return "\n".join(lines)


def print_report(hierarchy: Hierarchy, trace_name: str | None = None) -> None:
    print(build_report(hierarchy, trace_name))
