"""Hierarchy configuration: typed specification, validation, and loading.

A configuration is a JSON object of the form::

    {
      "memory_access_time": 100,
      "levels": [
        {"name": "L1", "size": 32768, "block_size": 64,
         "associativity": 4, "hit_time": 4, "policy": "lru"},
        ...
      ]
    }

Optional per-level keys: ``policy`` (default ``"lru"``), ``index``
(``"modulo"`` or ``"xor"``, default ``"modulo"``), ``track_3c`` (default
true) and ``rng_seed`` (default 0).

``parse_config`` turns such a dict into a ``HierarchySpec`` and rejects
anything malformed with a ``ConfigError`` whose message names the offending
key (for example ``levels[1] (L2): unknown key 'assoc'``).

Every key beyond the five required ones has a default that reproduces the
original model exactly, so an old configuration keeps its old behaviour:

======================  =============  ==================================
key                     default        effect
======================  =============  ==================================
``policy``              ``"lru"``      replacement policy
``track_3c``            ``true``       run the 3-C shadow cache
``rng_seed``            ``0``          seed for the random policy
``inclusion``           ``"nine"``     relation to the level ABOVE
======================  =============  ==================================
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from cachesim.indexing import DEFAULT_INDEX, INDEX_FUNCTIONS
from cachesim.policies import POLICIES


class ConfigError(ValueError):
    """A configuration is structurally or semantically invalid."""


#: Inclusion policies a level may declare towards the level ABOVE it.
#:
#: nine
#:     Non-Inclusive Non-Exclusive: nothing is enforced between the two
#:     levels. A fill installs the block in every level that missed, and no
#:     level is told when another evicts. This is the default and the
#:     behaviour of the original model.
#: inclusive
#:     The level is a superset of the level above: whenever it drops a block
#:     the block is back-invalidated from every level above it.
#: exclusive
#:     The level holds only blocks that the level above does not: it is
#:     filled by the level above's evictions and emptied when the level
#:     above pulls a block back up (Jouppi's victim-cache arrangement,
#:     generalised to a whole level).
#:
#: The terminology follows Baer and Wang, "On the Inclusion Properties for
#: Multi-Level Cache Hierarchies", ISCA 1988.
INCLUSION_POLICIES = ("nine", "inclusive", "exclusive")


@dataclass(frozen=True)
class CacheSpec:
    """Parameters of one cache level.

    ``inclusion`` describes this level's relation to the level ABOVE it (see
    ``INCLUSION_POLICIES``), so the first level must leave it at ``"nine"``.
    """

    name: str
    size: int
    block_size: int
    associativity: int
    hit_time: int
    policy: str = "lru"
    track_3c: bool = True
    rng_seed: int = 0
    index: str = DEFAULT_INDEX
    inclusion: str = "nine"


@dataclass(frozen=True)
class HierarchySpec:
    """Parameters of a whole hierarchy: its levels, L1 first, and DRAM."""

    levels: tuple[CacheSpec, ...]
    memory_access_time: int

    def to_dict(self) -> dict[str, Any]:
        """The JSON-compatible dict form accepted by ``parse_config``."""
        return {
            "memory_access_time": self.memory_access_time,
            "levels": [asdict(level) for level in self.levels],
        }


#: The default hierarchy: sizes, latencies, and shapes typical of a modern
#: desktop core. Copy this into a .json file and edit to experiment.
DEFAULT_CONFIG: dict[str, Any] = {
    "memory_access_time": 100,  # cycles to reach DRAM
    "levels": [
        {
            "name": "L1",
            "size": 32 * 1024,
            "block_size": 64,
            "associativity": 4,
            "policy": "lru",
            "hit_time": 4,
        },
        {
            "name": "L2",
            "size": 256 * 1024,
            "block_size": 64,
            "associativity": 8,
            "policy": "lru",
            "hit_time": 12,
        },
        {
            "name": "L3",
            "size": 2 * 1024 * 1024,
            "block_size": 64,
            "associativity": 16,
            "policy": "lru",
            "hit_time": 40,
        },
    ],
}


def default_config() -> dict[str, Any]:
    """Return a fresh copy of ``DEFAULT_CONFIG`` that is safe to mutate."""
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: str) -> dict[str, Any]:
    """Read a hierarchy configuration from a JSON file (unvalidated dict)."""
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: expected a JSON object at top level")
    return config


# -- validation ----------------------------------------------------------------

_LEVEL_REQUIRED = ("name", "size", "block_size", "associativity", "hit_time")
_LEVEL_KEYS = frozenset(f.name for f in fields(CacheSpec))
_TOP_KEYS = frozenset({"levels", "memory_access_time"})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(where: str, key: str, value: Any, minimum: int) -> int:
    if not _is_int(value):
        raise ConfigError(f"{where}: {key!r} must be an integer, got {value!r}")
    if value < minimum:
        raise ConfigError(f"{where}: {key!r} must be >= {minimum}, got {value}")
    return int(value)


def _parse_level(where: str, spec: Any) -> CacheSpec:
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: expected an object, got {type(spec).__name__}")
    name = spec.get("name")
    if name is not None:
        where = f"{where} ({name})"
    unknown = sorted(set(spec) - _LEVEL_KEYS)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(map(repr, unknown))}")
    missing = [k for k in _LEVEL_REQUIRED if k not in spec]
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {', '.join(map(repr, missing))}")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: 'name' must be a non-empty string, got {name!r}")

    policy = spec.get("policy", "lru")
    if not isinstance(policy, str):
        raise ConfigError(f"{where}: 'policy' must be a string, got {policy!r}")
    policy = policy.lower()
    if policy not in POLICIES:
        raise ConfigError(
            f"{where}: unknown policy {policy!r}; choose from {', '.join(sorted(POLICIES))}"
        )
    track_3c = spec.get("track_3c", True)
    if not isinstance(track_3c, bool):
        raise ConfigError(f"{where}: 'track_3c' must be true or false, got {track_3c!r}")

    index = spec.get("index", DEFAULT_INDEX)
    if not isinstance(index, str):
        raise ConfigError(f"{where}: 'index' must be a string, got {index!r}")
    index = index.lower()
    if index not in INDEX_FUNCTIONS:
        raise ConfigError(
            f"{where}: unknown index function {index!r}; "
            f"choose from {', '.join(sorted(INDEX_FUNCTIONS))}"
        )
    inclusion = spec.get("inclusion", "nine")
    if not isinstance(inclusion, str):
        raise ConfigError(f"{where}: 'inclusion' must be a string, got {inclusion!r}")
    inclusion = inclusion.lower()
    if inclusion not in INCLUSION_POLICIES:
        raise ConfigError(
            f"{where}: unknown inclusion policy {inclusion!r}; "
            f"choose from {', '.join(INCLUSION_POLICIES)}"
        )

    return CacheSpec(
        name=name,
        size=_require_int(where, "size", spec["size"], 1),
        block_size=_require_int(where, "block_size", spec["block_size"], 1),
        associativity=_require_int(where, "associativity", spec["associativity"], 1),
        hit_time=_require_int(where, "hit_time", spec["hit_time"], 0),
        policy=policy,
        track_3c=track_3c,
        rng_seed=_require_int(where, "rng_seed", spec.get("rng_seed", 0), -(1 << 63)),
        index=index,
        inclusion=inclusion,
    )


def parse_config(config: Mapping[str, Any]) -> HierarchySpec:
    """Validate a configuration dict and return the typed ``HierarchySpec``.

    Raises ``ConfigError`` (a ``ValueError``) with a path-qualified message
    for unknown or missing keys, wrong types, non-positive geometry,
    unknown policies, duplicate level names, or levels with different
    block sizes.
    """
    if not isinstance(config, Mapping):
        raise ConfigError(f"expected a JSON object at top level, got {type(config).__name__}")
    unknown = sorted(set(config) - _TOP_KEYS)
    if unknown:
        raise ConfigError(f"unknown top-level key(s) {', '.join(map(repr, unknown))}")
    missing = [k for k in ("levels", "memory_access_time") if k not in config]
    if missing:
        raise ConfigError(f"missing required top-level key(s) {', '.join(map(repr, missing))}")
    memory_access_time = _require_int(
        "top level", "memory_access_time", config["memory_access_time"], 0
    )
    raw_levels = config["levels"]
    if not isinstance(raw_levels, list) or not raw_levels:
        raise ConfigError("'levels' must be a non-empty list of cache levels")

    levels = tuple(_parse_level(f"levels[{i}]", spec) for i, spec in enumerate(raw_levels))

    names = [level.name for level in levels]
    if len(set(names)) != len(names):
        raise ConfigError(f"level names must be unique, got {names}")
    offline = [level.name for level in levels if level.policy == "opt"]
    if len(offline) > 1:
        # OPT at one level changes what the level below it sees, so the
        # reference stream a second OPT level needs cannot be recorded in
        # advance; see cachesim.opt.simulate_with_opt.
        raise ConfigError(
            f"policy 'opt' is only meaningful at one level, but {', '.join(offline)} all ask for it"
        )
    if levels[0].inclusion != "nine":
        raise ConfigError(
            f"levels[0] ({levels[0].name}): 'inclusion' describes a level's relation to the "
            f"level above it, and the first level has none; got {levels[0].inclusion!r}"
        )
    block_sizes = {level.block_size for level in levels}
    if len(block_sizes) != 1:
        raise ConfigError(
            "all levels must share one block_size (multi-line fills are not modelled), "
            f"got {[level.block_size for level in levels]}"
        )
    return HierarchySpec(levels=levels, memory_access_time=memory_access_time)
