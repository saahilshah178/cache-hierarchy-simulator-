"""Self-checks: identities a finished simulation must satisfy.

``check_invariants(hierarchy)`` returns a list of human-readable violation
messages, empty when the model is consistent. It is a safety net for the
simulator itself, not a validation of the workload: a violation means a
counter and the state it describes have drifted apart, which is a bug in
``cachesim``, not in the trace.

``cachesim run --check`` calls it after a run and exits non-zero listing
whatever it found.

The identities, and the assumptions each one rests on:

Counter algebra (always)
    ``hits + misses == accesses`` at every level, with reads and writes
    partitioning both; ``misses <= accesses``.

Three-C decomposition (when the level tracks it)
    The per-reference labels sum to the misses and so do the Hill-Smith
    aggregates, and the two taxonomies differ by exactly the anti-conflict
    hits: ``capacity_aggregate == capacity + anti_conflict_hits`` and
    ``conflict_aggregate == conflict - anti_conflict_hits``, with
    ``shadow_misses == compulsory + capacity + anti_conflict_hits``.

Traffic flow (skipped when extra traffic is modelled)
    ``levels[0].accesses == hierarchy.accesses``,
    ``levels[i+1].accesses == levels[i].misses``,
    ``dram_reads == levels[-1].misses``, and
    ``fills == misses + writeback_allocations`` at every level. All four
    assume demand fetching only into non-inclusive, non-exclusive levels:
    a prefetcher fills without a miss, and an exclusive hierarchy fills a
    level from the eviction above it rather than from its own miss. Both
    are detected through optional attributes (``prefetcher`` on a level or
    on the hierarchy, ``inclusion`` on the hierarchy) so that the check
    steps aside instead of firing spuriously if such a feature is added.

Write-back conservation (always)
    ``levels[i].writebacks_received == levels[i-1].writebacks`` and
    ``dram_writes == levels[-1].writebacks``. Dirty data is never
    duplicated and never dropped: every line written out of one level is
    accounted for by the level below it, and everything leaving the last
    level reaches DRAM. Write-backs caused by ``flush()`` are included,
    because ``flush`` increments both sides. Calling ``Cache.invalidate``
    from outside the hierarchy breaks the identity, since the removed
    dirty line is counted as written back with nothing below to receive
    it; the message points at the invalidation count when that is what
    happened.

Timing (skipped when the latency model is not constant)
    ``total_cycles == sum(hit_time * accesses) + memory_access_time *
    dram_reads``, and consequently ``amat() == measured_amat()``. This
    holds exactly for the constant-latency model with no transfer term:
    an access pays each probed level's hit time once, plus the memory
    access time if it reached DRAM. A per-level transfer time (block size
    over bus width) or a non-constant DRAM model would add terms this sum
    does not have, so both checks are skipped when a level carries a
    non-null ``bus_width`` or the hierarchy carries a non-null
    ``dram_model``.

Structure (always)
    A block has at most one copy in a cache, every valid line is found by
    ``contains``, every line sits in the set its block maps to,
    ``is_dirty`` agrees with the line's dirty bit, and a level holds no
    more lines than it has blocks.

Dirty lines after a flush (only when ``after_flush=True``)
    No level holds a dirty line. This cannot be checked unconditionally:
    a hierarchy that has simply run a trace is expected to hold dirty
    lines, and ``reset_stats`` zeroes the counters that would otherwise
    explain them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from cachesim.cache import Cache
from cachesim.hierarchy import Hierarchy

#: Attributes that, if present and not None, mean a level or the hierarchy
#: charges time this module's constant-latency identity does not model.
_TRANSFER_LEVEL_ATTRS = ("bus_width", "transfer_time")
_TRANSFER_HIERARCHY_ATTRS = ("dram_model", "dram")

#: Attributes that, if present and not None, mean traffic reaches a level
#: other than through a demand miss of the level above it.
_TRAFFIC_LEVEL_ATTRS = ("prefetcher",)
_TRAFFIC_HIERARCHY_ATTRS = ("prefetcher",)

#: Inclusion policies for which the traffic-flow identities still hold.
_NINE_INCLUSION = (None, "nine", "non-inclusive")


@dataclass(frozen=True)
class CheckReport:
    """The outcome of one invariant sweep."""

    violations: tuple[str, ...]
    checked: int
    skipped: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when nothing was violated."""
        return not self.violations


def _present(obj: object, names: tuple[str, ...]) -> str | None:
    """The first of ``names`` that ``obj`` carries with a non-None value."""
    for name in names:
        if getattr(obj, name, None) is not None:
            return name
    return None


def _transfer_time_modelled(h: Hierarchy) -> str | None:
    """Why the constant-latency timing identity does not apply, if it does not."""
    found = _present(h, _TRANSFER_HIERARCHY_ATTRS)
    if found is not None:
        return f"hierarchy.{found} is set"
    for level in h.levels:
        for obj, label in ((level, "level"), (level.cache, "cache")):
            found = _present(obj, _TRANSFER_LEVEL_ATTRS)
            if found is not None:
                return f"{level.cache.name} {label}.{found} is set"
    return None


def _extra_traffic_modelled(h: Hierarchy) -> str | None:
    """Why the traffic-flow identities do not apply, if they do not."""
    found = _present(h, _TRAFFIC_HIERARCHY_ATTRS)
    if found is not None:
        return f"hierarchy.{found} is set"
    inclusion = getattr(h, "inclusion", None)
    if inclusion not in _NINE_INCLUSION:
        return f"hierarchy.inclusion is {inclusion!r}"
    for level in h.levels:
        for obj, label in ((level, "level"), (level.cache, "cache")):
            found = _present(obj, _TRAFFIC_LEVEL_ATTRS)
            if found is not None:
                return f"{level.cache.name} {label}.{found} is set"
    return None


class _Sweep:
    """Accumulates violations and counts the checks that were made."""

    def __init__(self) -> None:
        self.violations: list[str] = []
        self.checked = 0
        self.skipped: list[str] = []

    def equal(self, where: str, what: str, got: Any, want: Any, detail: str = "") -> None:
        self.checked += 1
        if got != want:
            suffix = f" ({detail})" if detail else ""
            self.violations.append(f"{where}: {what}: {got} != {want}{suffix}")

    def at_most(self, where: str, what: str, got: int, limit: int) -> None:
        self.checked += 1
        if got > limit:
            self.violations.append(f"{where}: {what}: {got} exceeds {limit}")

    def close(self, where: str, what: str, got: float, want: float) -> None:
        self.checked += 1
        if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9):
            self.violations.append(f"{where}: {what}: {got!r} != {want!r}")

    def truth(self, where: str, message: str, ok: bool) -> None:
        self.checked += 1
        if not ok:
            self.violations.append(f"{where}: {message}")

    def skip(self, what: str, reason: str) -> None:
        self.skipped.append(f"{what} ({reason})")


def check_hierarchy(hierarchy: Hierarchy, *, after_flush: bool = False) -> CheckReport:
    """Run every applicable invariant and report what held, failed, or was skipped.

    ``after_flush`` additionally requires that no level holds a dirty line,
    which is only true straight after ``Hierarchy.flush()``.
    """
    h = hierarchy
    s = _Sweep()
    levels = h.levels

    s.equal("hierarchy", "reads + writes != accesses", h.reads + h.writes, h.accesses)
    s.at_most("hierarchy", "dram_reads", h.dram_reads, h.accesses)

    for level in levels:
        _check_level_counters(s, level.cache)
        _check_level_structure(s, level.cache)
        if after_flush:
            dirty = sum(1 for *_, is_dirty in level.cache.lines() if is_dirty)
            s.equal(level.cache.name, "dirty lines after a flush", dirty, 0)

    _check_writeback_conservation(s, h)

    traffic_reason = _extra_traffic_modelled(h)
    if traffic_reason is None:
        _check_traffic_flow(s, h)
    else:
        s.skip("traffic-flow identities", traffic_reason)

    timing_reason = _transfer_time_modelled(h)
    if timing_reason is not None:
        s.skip("timing identities", timing_reason)
    elif h.accesses == 0:
        s.skip("timing identities", "no accesses were simulated")
    else:
        _check_timing(s, h)

    return CheckReport(tuple(s.violations), s.checked, tuple(s.skipped))


def check_invariants(hierarchy: Hierarchy, *, after_flush: bool = False) -> list[str]:
    """Return the invariant violations of ``hierarchy``; empty means consistent."""
    return list(check_hierarchy(hierarchy, after_flush=after_flush).violations)


# -- individual checks --------------------------------------------------------


def _check_level_counters(s: _Sweep, c: Cache) -> None:
    """Counter algebra and the three-C decomposition of one level."""
    name = c.name
    s.equal(name, "hits + misses != accesses", c.hits + c.misses, c.accesses)
    s.equal(name, "read_hits + write_hits != hits", c.read_hits + c.write_hits, c.hits)
    s.equal(name, "read_misses + write_misses != misses", c.read_misses + c.write_misses, c.misses)
    s.at_most(name, "misses", c.misses, c.accesses)
    s.at_most(name, "evictions", c.evictions, c.fills)
    # writebacks is not bounded by evictions + invalidations: flush() writes
    # a resident line out without removing it.
    s.at_most(name, "writeback_allocations", c.writeback_allocations, c.writebacks_received)

    if not c.track_3c:
        return
    s.equal(
        name,
        "compulsory + capacity + conflict != misses",
        c.compulsory_misses + c.capacity_misses + c.conflict_misses,
        c.misses,
    )
    s.equal(
        name,
        "compulsory + capacity_aggregate + conflict_aggregate != misses",
        c.compulsory_misses + c.capacity_misses_aggregate + c.conflict_misses_aggregate,
        c.misses,
    )
    s.equal(
        name,
        "capacity_aggregate != capacity + anti_conflict_hits",
        c.capacity_misses_aggregate,
        c.capacity_misses + c.anti_conflict_hits,
    )
    s.equal(
        name,
        "conflict_aggregate != conflict - anti_conflict_hits",
        c.conflict_misses_aggregate,
        c.conflict_misses - c.anti_conflict_hits,
    )
    s.equal(
        name,
        "shadow_misses != compulsory + capacity + anti_conflict_hits",
        c.shadow_misses,
        c.compulsory_misses + c.capacity_misses + c.anti_conflict_hits,
    )


def _check_level_structure(s: _Sweep, c: Cache) -> None:
    """Residency: one copy per block, in the right set, findable, dirty bits agree."""
    name = c.name
    lines = list(c.lines())
    blocks = [block for *_, block, _ in lines]
    s.truth(
        name,
        f"a block is held in more than one way ({len(blocks) - len(set(blocks))} duplicates)",
        len(blocks) == len(set(blocks)),
    )
    s.at_most(name, "resident lines", len(lines), c.num_blocks)
    misplaced = [block for set_idx, _, block, _ in lines if set_idx != block % c.num_sets]
    s.truth(name, f"lines sit in the wrong set: {misplaced[:4]}", not misplaced)
    unfindable = [block for *_, block, _ in lines if not c.contains(block)]
    s.truth(name, f"valid lines not found by contains(): {unfindable[:4]}", not unfindable)
    inconsistent = [block for *_, block, dirty in lines if c.is_dirty(block) != dirty]
    s.truth(name, f"is_dirty() disagrees with the line: {inconsistent[:4]}", not inconsistent)


def _check_writeback_conservation(s: _Sweep, h: Hierarchy) -> None:
    """Every line written out of a level is received by the level below it."""
    for i in range(1, len(h.levels)):
        above, below = h.levels[i - 1].cache, h.levels[i].cache
        detail = ""
        if above.writebacks != below.writebacks_received and above.invalidations:
            detail = (
                f"{above.name} recorded {above.invalidations} invalidation(s); "
                "a dirty line removed by Cache.invalidate outside the hierarchy "
                "is written back with nothing below to receive it"
            )
        s.equal(
            below.name,
            f"writebacks_received != {above.name}.writebacks",
            below.writebacks_received,
            above.writebacks,
            detail,
        )
    last = h.levels[-1].cache
    detail = ""
    if h.dram_writes != last.writebacks and last.invalidations:
        detail = f"{last.name} recorded {last.invalidations} invalidation(s)"
    s.equal(
        "hierarchy",
        f"dram_writes != {last.name}.writebacks",
        h.dram_writes,
        last.writebacks,
        detail,
    )


def _check_traffic_flow(s: _Sweep, h: Hierarchy) -> None:
    """Nothing but a demand miss makes the level below do work."""
    s.equal(
        h.levels[0].cache.name,
        "accesses != hierarchy.accesses",
        h.levels[0].cache.accesses,
        h.accesses,
    )
    for i in range(1, len(h.levels)):
        above, below = h.levels[i - 1].cache, h.levels[i].cache
        s.equal(below.name, f"accesses != {above.name}.misses", below.accesses, above.misses)
    last = h.levels[-1].cache
    s.equal("hierarchy", f"dram_reads != {last.name}.misses", h.dram_reads, last.misses)
    for level in h.levels:
        c = level.cache
        s.equal(
            c.name,
            "fills != misses + writeback_allocations",
            c.fills,
            c.misses + c.writeback_allocations,
        )


def _check_timing(s: _Sweep, h: Hierarchy) -> None:
    """Cycles are the probe charges plus the memory charges, and AMAT follows."""
    expected = sum(level.hit_time * level.cache.accesses for level in h.levels)
    expected += h.memory_access_time * h.dram_reads
    s.equal(
        "hierarchy",
        "total_cycles != sum(hit_time * accesses) + memory_access_time * dram_reads",
        h.total_time,
        expected,
    )
    s.close("hierarchy", "amat() != measured_amat()", h.amat(), h.measured_amat())
