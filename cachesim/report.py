"""report.py — turns a simulated Hierarchy into a human-readable report.

Everything here is presentation: per-level hit/miss tables, local vs global
miss rates, the 3-C miss breakdown, and AMAT. See README.md for what each
number means.
"""


def _pct(part, whole):
    """Format part/whole as a percentage string, safe against divide-by-zero."""
    return f"{100.0 * part / whole:6.2f}%" if whole else "   n/a "


def _size_str(nbytes):
    """32768 -> '32.0 KB', 2097152 -> '2.0 MB'."""
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.1f} MB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.1f} KB"
    return f"{nbytes} B"


def build_report(hierarchy, trace_name=None):
    """Return the full multi-line report string for a finished simulation."""
    h = hierarchy
    lines = []
    bar = "=" * 72

    lines.append(bar)
    title = "CACHE HIERARCHY SIMULATION REPORT"
    if trace_name:
        title += f"  ({trace_name})"
    lines.append(title)
    lines.append(bar)
    lines.append(f"Total accesses : {h.accesses:>12,}   (reads {h.reads:,} / writes {h.writes:,})")
    lines.append(
        f"DRAM accesses  : {h.memory_accesses:>12,}"
        f"   ({_pct(h.memory_accesses, h.accesses).strip()} of all"
        " accesses fell through every cache)"
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
        if c.track_3c and c.misses:
            lines.append("  miss breakdown (the 3 Cs):")
            lines.append(
                f"    compulsory : {c.compulsory_misses:>12,}  "
                f"{_pct(c.compulsory_misses, c.misses)} of misses"
                "  (first-ever touch, unavoidable)"
            )
            lines.append(
                f"    capacity   : {c.capacity_misses:>12,}  "
                f"{_pct(c.capacity_misses, c.misses)} of misses"
                "  (working set bigger than the cache)"
            )
            lines.append(
                f"    conflict   : {c.conflict_misses:>12,}  "
                f"{_pct(c.conflict_misses, c.misses)} of misses"
                "  (too many blocks fighting over one set)"
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


def print_report(hierarchy, trace_name=None):
    print(build_report(hierarchy, trace_name))
