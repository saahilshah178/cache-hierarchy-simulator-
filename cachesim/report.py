"""Renders a ``HierarchyStats`` snapshot as a human-readable text report.

Everything here is presentation: per-level hit/miss tables, local and global
miss rates, the miss classification, and AMAT.
"""

from __future__ import annotations

from cachesim.hierarchy import Hierarchy
from cachesim.stats import HierarchyStats

BAR = "=" * 72


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
    lines.append(
        f"DRAM writes    : {s.dram_writes:>12,}"
        f"   (dirty lines written back from {s.levels[-1].name})"
    )
    lines.append("")

    for i, lv in enumerate(s.levels):
        lines.append(
            f"--- {lv.name}: {_size_str(lv.size)}, {lv.block_size} B blocks, "
            f"{lv.associativity}-way, {lv.policy.upper()}, hit time {lv.hit_time} cyc ---"
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
    lines.append(f"AMAT (analytic formula) : {s.amat:8.3f} cycles")
    lines.append(
        f"AMAT (measured)         : {s.measured_amat:8.3f} cycles"
        f"   ({s.total_cycles:,} cycles / {s.accesses:,} accesses)"
    )
    lines.append(BAR)
    return "\n".join(lines)


def build_report(hierarchy: Hierarchy, trace_name: str | None = None) -> str:
    """Return the full multi-line report string for a finished simulation."""
    return format_report(hierarchy.stats(), trace_name)


def print_report(hierarchy: Hierarchy, trace_name: str | None = None) -> None:
    print(build_report(hierarchy, trace_name))
