from __future__ import annotations

import random
import re
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rbtl_data import (
    DataBundle,
    parse_int_maybe,
    parse_tags,
    s_get,
    s_get_bool,
    s_get_float,
    s_get_int,
    s_get_list,
)
from rbtl_shared import (
    DIFF_IDX,
    DIFFICULTY_ORDER,
    eligible_enemy_unit,
    leadership_rank,
    max_leadership,
    minlead_from_entry,
    must_min_leadership,
    normalize_leadership_value,
    pick_scenario_type_id,
    pick_weighted_tag,
    threat_pool_from_settings,
)

# ============================================================
# RBTL DEV RULES — IO + Architecture Guardrails (REFERENCE)
# ============================================================
#
# [IO / DATA LOADING]
# 1) ONLY rbtl_data.py may touch the filesystem for game data:
#    - load_pipe_file(...)
#    - load_settings(...)
#    - open("data/*.txt")
#    - DATA_DIR / PROJECT_ROOT path construction for data files
#
# 2) rbtl_core.py and rbtl_cli.py must be IO-free for data.
#    They may:
#      - read DataBundle fields
#      - generate strings
#      - use random
#      - (optionally) write NOTHING to disk
#
# 3) All generators accept a DataBundle:
#      generate_encounter(data: DataBundle, inputs: dict) -> (filename, text)
#    No generator should reload enemy_units/clues/events/etc internally.
#
# 4) No “sneaky reloads” inside helpers:
#    - Helpers must not call load_pipe_file/load_settings/open() for data.
#    - Helpers must not build paths to /data and read files.
#
# [DEPENDENCY VISIBILITY]
# 5) If a function needs something, it takes it as an argument.
#    No hidden globals (DATA_DIR, PROJECT_ROOT) inside core logic.
#
# [TESTABILITY]
# 6) Core logic should be testable with an in-memory DataBundle.
#    (Unit tests can pass a fake DataBundle without touching disk.)
#
# [RELOAD STRATEGY]
# 7) Reloading data is a deliberate action:
#    - done once at startup, or
#    - done on explicit “refresh/restart” in main, not deep helpers.
#
# [SINGLE SOURCE OF TRUTH]
# 8) Settings are read once and passed around via DataBundle.
#    No partial settings reads in scattered modules.
#
# [OUTPUT IO]
# 9) Writing output files is allowed ONLY in the tiny _main_ script,
#    not in rbtl_core.py (preferred).
#
# [SEARCH / AUDIT]
# 10) To audit for violations, search the repo for:
#     - "load_pipe_file("
#     - "load_settings("
#     - "open("
#     - "DATA_DIR"
#     - "PROJECT_ROOT"
#     and ensure data IO only appears in rbtl_data.py and mains.
#
# RULE: Core modules do not read /data directly.
# If you need a dataset, add it to DataBundle in rbtl_data.py and pass it in.
# ============================================================
# rbtl_core.py
# ============================================================
# Rangers at the Borderlands — Core Scenario Generator (IO-FREE)
# - Consumes: DataBundle (from rbtl_data.load_data_bundle)
# - Produces: (suggested_filename, briefing_text)
# - No filesystem reads, no output writes, no PROJECT_ROOT/DATA_DIR globals
# ============================================================

# ============================================================
# #CONFIG — Constants, enums, weights
# ============================================================

RARITY_WEIGHTS = {"common": 60, "uncommon": 25, "rare": 10, "legendary": 5}

# BBEG never random (only forced by minlead / scenario type)
LEADERSHIP_WEIGHTS = [("None", 50), ("Lieutenant", 35), ("Boss", 15)]
STAT_KEYS = ["Move", "Fight", "Shoot", "Armor", "Will", "Health"]

COMPLETION_XP = 15
GOLD_PER_XP = 3


# ============================================================
# #UTIL — small helpers
# ============================================================

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def entry_weight(e: Dict[str, Any]) -> int:
    rarity = (e.get("rarity") or "common").strip().lower()
    return RARITY_WEIGHTS.get(rarity, 1)


def weighted_choice(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    weights = [max(1, entry_weight(e)) for e in entries]
    return random.choices(entries, weights=weights, k=1)[0]


def entry_pick_weight(e: Dict[str, Any], settings: Optional[Dict[str, str]] = None, prefix: str = "") -> float:
    """
    Generic picker weight.
    - If entry has weight=..., use it.
    - If settings has something like objective_weight.duel=..., multiply by it.
    """
    settings = settings or {}
    ident = (e.get('ID') or e.get('id') or '').strip()
    base = s_get_float(settings or {}, f"{prefix}_weight.{ident}", 1.0) if settings else 1.0

    raw_w = str(e.get("weight", "")).strip()
    w = 1.0
    if raw_w:
        try:
            w = float(raw_w)
        except Exception:
            w = 1.0
    if w <= 0:
        w = 1.0

    return max(0.0, w * base)


def weighted_choice_by_weight(entries: List[Dict[str, Any]], settings: Optional[Dict[str, str]] = None, prefix: str = "") -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    weights = [max(0.0001, entry_pick_weight(e, settings=settings, prefix=prefix)) for e in entries]
    return random.choices(entries, weights=weights, k=1)[0]


def roll_from_list_hard_threat(
    entries: List[Dict[str, Any]],
    required_threats: Set[str],
    exclude_names: Optional[Set[str]] = None
) -> Optional[Dict[str, Any]]:
    """Hard-match on entry['threats']."""
    exclude_names = exclude_names or set()
    req = {t.strip().lower() for t in (required_threats or set()) if t and t.strip().lower() != "none"}

    pool: List[Dict[str, Any]] = []
    for e in entries:
        nm = e.get("name", "")
        if nm in exclude_names:
            continue
        e_threats = {t.strip().lower() for t in (e.get("threats") or set())}
        if req and not e_threats.intersection(req):
            continue
        pool.append(e)

    return weighted_choice(pool)




def _find_entry_by_id(entries: List[Dict[str, Any]], entry_id: str) -> Optional[Dict[str, Any]]:
    entry_id = str(entry_id or "").strip()
    if not entry_id:
        return None
    for e in entries or []:
        if str(e.get("id", "")).strip() == entry_id:
            return e
    return None


def _campaign_biome_id_from_key(raw_key: str) -> str:
    key = (raw_key or "").strip()
    if not key:
        return ""
    parts = [p for p in key.split("-") if p != ""]
    if len(parts) < 8:
        return ""
    if parts[0] != "RBTL" or parts[1] != "CAMP":
        return ""
    return parts[3]


def find_scenario_type_by_id(scen_entries: List[Dict[str, Any]], st_id: str) -> Optional[Dict[str, Any]]:
    st_id = (st_id or "").strip()
    if not st_id:
        return None
    for e in scen_entries:
        if "scenariotype" in (e.get("tags", set()) or set()) and (e.get("ID") or "").strip() == st_id:
            return e
    return None


# ============================================================
# #UNITS — Eligibility + leader identification
# ============================================================

def unit_tier(e: Dict[str, Any]) -> str:
    return str(e.get("tier", "")).strip().lower()


def is_leader_unit(e: Dict[str, Any]) -> bool:
    if unit_tier(e) == "leader":
        return True
    return "leader" in (e.get("tags", set()) or set())  # legacy fallback


def is_nonleader_unit(e: Dict[str, Any]) -> bool:
    return eligible_enemy_unit(e) and not is_leader_unit(e)


# ============================================================
# #STATS — Stat parsing + modifiers
# ============================================================

def parse_statline(stat_str: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for chunk in (stat_str or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = None
        for i, ch in enumerate(chunk):
            if ch.isdigit() or ch in "+-":
                idx = i
                break
        if idx is None:
            continue
        key = chunk[:idx].strip()
        val = chunk[idx:].strip()
        out[key] = parse_int_maybe(val, 0)

    for k in STAT_KEYS:
        out.setdefault(k, 0)
    return out


def statline_to_string(stats: Dict[str, int]) -> str:
    def fmt(k: str, v: int) -> str:
        if k in ("Fight", "Shoot", "Will"):
            sign = "+" if v >= 0 else ""
            return f"{k}{sign}{v}"
        return f"{k}{v}"
    return ",".join(fmt(k, stats.get(k, 0)) for k in STAT_KEYS)


def apply_stat_mods(stats: Dict[str, int], mod_str: Optional[str]) -> None:
    if not mod_str:
        return
    for chunk in str(mod_str).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = None
        for i, ch in enumerate(chunk):
            if ch.isdigit() or ch in "+-":
                idx = i
                break
        if idx is None:
            continue
        key = chunk[:idx].strip()
        delta = parse_int_maybe(chunk[idx:].strip(), 0)
        stats[key] = stats.get(key, 0) + delta


def apply_difficulty_mods(stats: Dict[str, int], unit_tags: Set[str], difficulty: str, is_leader: bool) -> None:
    if difficulty in ("Easy", "Normal"):
        return

    if difficulty == "Hard":
        if "ranged" in unit_tags:
            stats["Shoot"] += 1
        else:
            stats["Fight"] += 1
        if is_leader:
            stats["Armor"] += 1
            stats["Health"] += 1

    elif difficulty == "Brutal":
        stats["Fight"] += 1
        stats["Shoot"] += 1
        stats["Armor"] += 1
        stats["Will"] += 1
        stats["Health"] += 1
        if is_leader:
            stats["Armor"] += 1
            stats["Health"] += 1


def apply_leadership_mods(stats: Dict[str, int], tier: str) -> None:
    if tier == "Lieutenant":
        stats["Fight"] += 1
        stats["Will"] += 1
        stats["Health"] += 2
    elif tier == "Boss":
        stats["Fight"] += 2
        stats["Armor"] += 1
        stats["Will"] += 1
        stats["Health"] += 4
    elif tier == "BBEG":
        stats["Fight"] += 3
        stats["Armor"] += 2
        stats["Will"] += 2
        stats["Health"] += 6


# ============================================================
# #TRAITS — Trait tiering and roll logic
# ============================================================

def infer_trait_tier(tags: Set[str]) -> Tuple[str, Optional[str]]:
    tiers = {"minor", "major", "legendary"}
    found = tiers.intersection(tags)
    if found:
        if "legendary" in found:
            return "legendary", None
        if "major" in found:
            return "major", None
        return "minor", None

    if "boss_only" in tags:
        return "major", "Trait tier inferred as major from boss_only"
    if "enemy_only" in tags:
        return "minor", "Trait tier inferred as minor from enemy_only"
    return "minor", "Trait tier inferred as minor (no tier tag found)"


def trait_allowed(trait_tags: Set[str], leader_tier: str, trait_tier: str) -> bool:
    if "legendary" in trait_tags and leader_tier != "BBEG":
        return False
    if "boss_only" in trait_tags and leader_tier not in ("Boss", "BBEG"):
        return False

    if leader_tier == "Lieutenant":
        return trait_tier == "minor"
    if leader_tier == "Boss":
        return trait_tier in ("minor", "major")
    if leader_tier == "BBEG":
        return trait_tier in ("minor", "major", "legendary")
    return False


def roll_leader_traits(traits: List[Dict[str, Any]], leader_tier: str, footnotes: List[str]) -> List[Dict[str, Any]]:
    if leader_tier == "Lieutenant":
        target = {"minor": 1, "major": 0, "legendary": 0}
    elif leader_tier == "Boss":
        target = {"minor": 1, "major": 1, "legendary": 0}
    elif leader_tier == "BBEG":
        target = {"minor": random.randint(0, 2), "major": random.randint(1, 2), "legendary": random.randint(1, 2)}
    else:
        return []

    tier_pools = {"minor": [], "major": [], "legendary": []}
    for t in traits:
        ttags = t.get("tags", set()) or set()
        tier, warn = infer_trait_tier(ttags)
        if warn:
            footnotes.append(f"WARNING: Trait '{t['name']}' — {warn}")
        if trait_allowed(ttags, leader_tier, tier):
            tier_pools[tier].append(t)

    picked: List[Dict[str, Any]] = []
    picked_names: Set[str] = set()

    for tier_name, cnt in target.items():
        for _ in range(cnt):
            pool = [t for t in tier_pools[tier_name] if t["name"] not in picked_names]
            if not pool:
                footnotes.append(f"WARNING: No eligible {tier_name} traits available for {leader_tier}.")
                break
            chosen = weighted_choice(pool)
            if not chosen:
                break
            picked.append(chosen)
            picked_names.add(chosen["name"])

    return picked


# ============================================================
# #COUNTS — Enemy totals, clue counts, event counts, battle size
# ============================================================

def roll_enemy_total(allies: int, difficulty: str, scenario_type_id: str, leadership_tier: str) -> int:
    base = allies

    if difficulty == "Easy":
        total = random.randint(base - 1, base)
    elif difficulty == "Normal":
        total = random.randint(base, base + 1)
    elif difficulty == "Hard":
        total = random.randint(base + 2, base + 4)
    else:
        total = random.randint(base + 4, base + 7)

    if (scenario_type_id or "").strip().lower() == "defensive":
        total += random.randint(2, 4)

    if leadership_tier == "Boss":
        total -= random.randint(1, 2)
    elif leadership_tier == "BBEG":
        total -= random.randint(2, 4)

    total = max(total, base - 1)
    return max(1, total)


def battle_size_label(allies: int, enemies: int, difficulty: str, leadership_tier: str) -> str:
    if difficulty == "Brutal" or leadership_tier == "BBEG" or enemies >= allies + 7:
        return "Legendary"
    if difficulty == "Hard" or leadership_tier == "Boss" or enemies >= allies + 4:
        return "Epic"
    return "Normal"


def roll_clue_count(allies: int, difficulty: str, leadership_tier: str) -> int:
    di = DIFF_IDX.get(difficulty, 1)
    base = 5 + (allies // 4) + di + (1 if leadership_tier != "None" else 0)
    return clamp(base, 5, 12)


def roll_event_count(allies: int, difficulty: str, leadership_tier: str, scenario_type_id: str) -> int:
    di = DIFF_IDX.get(difficulty, 1)
    defensive_bonus = 2 if (scenario_type_id or "").strip().lower() == "defensive" else 0
    base = 5 + (allies // 3) + (2 * di) + defensive_bonus + (2 if leadership_tier != "None" else 0)
    return clamp(base, 5, 20)


# ============================================================
# #ROSTER — Enemy type count + role balance
# ============================================================

def roll_type_count(players: int, *, max_types: int = 6) -> int:
    """
    Target enemy MINION type count (leader not included).
    Your gut:
      1p: 2-3
      2p: 3-4
      3p: 4-5
      4p+: 5-6 (cap)
    """
    players = max(1, int(players))
    base = 2 + min(players, 4)  # 1->3, 2->4, 3->5, 4+->6
    lo = max(2, base - 1)
    hi = min(max_types, base)
    return random.randint(lo, hi)


def _has_tag(u: Dict[str, Any], tag: str) -> bool:
    return tag in (u.get("tags", set()) or set())


def _role(u: Dict[str, Any]) -> str:
    if _has_tag(u, "support"):
        return "support"
    if _has_tag(u, "tank"):
        return "tank"
    return "combat"


def _is_large(u: Dict[str, Any]) -> bool:
    return _has_tag(u, "large")


def _is_animal(u: Dict[str, Any]) -> bool:
    return _has_tag(u, "animal")


def _type_pick_weight(u: Dict[str, Any], role_counts: Dict[str, int], caps: Dict[str, int]) -> float:
    """
    Base rarity weight * role multipliers * 'large' downweight.
    Soft-caps: if role already at cap, weight becomes tiny (still possible).
    """
    w = float(max(1, entry_weight(u)))

    r = _role(u)
    if r == "support":
        w *= 0.35
    elif r == "tank":
        w *= 0.60
    else:
        w *= 1.00

    if _is_animal(u):
        w *= 0.65

    if _is_large(u):
        w *= 0.30

    # mobility flavor (light touch)
    if _has_tag(u, "flying"):
        w *= 0.85
    if _has_tag(u, "mounted"):
        w *= 0.90
    if _has_tag(u, "fast"):
        w *= 1.05

    if r in caps and role_counts.get(r, 0) >= caps[r]:
        w *= 0.05
    if _is_animal(u) and role_counts.get("animal", 0) >= caps.get("animal", 999):
        w *= 0.05
    if _is_large(u) and role_counts.get("large", 0) >= caps.get("large", 999):
        w *= 0.05

    return max(0.0, w)


def _weighted_choice_by_float(pool: List[Dict[str, Any]], weights: List[float]) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    if not any(w > 0 for w in weights):
        return weighted_choice(pool)
    return random.choices(pool, weights=weights, k=1)[0]


def select_enemy_types_v2(
    enemy_units: List[Dict[str, Any]],
    threat_tags: Set[str],
    footnotes: List[str],
    *,
    players: int,
    max_types: int = 6
) -> List[Dict[str, Any]]:
    """
    Picks 2..6 non-leader minion TYPES.
    - Prefers starting with a combat type.
    - Soft-caps support/tank/animal/large types.
    - Downplays large types.
    """
    tags = [t for t in sorted(threat_tags) if t and t.lower() != "none"]
    if not tags:
        footnotes.append("WARNING: No threat tags provided; cannot select enemies.")
        return []

    def threat_match(u: Dict[str, Any]) -> bool:
        return bool(set(tags).intersection(u.get("threats", set()) or set()))

    pool_all = [u for u in enemy_units if is_nonleader_unit(u) and threat_match(u)]
    if not pool_all:
        footnotes.append(f"WARNING: No eligible non-leader units for threats {tags}.")
        return []

    target_types = roll_type_count(players, max_types=max_types)

    caps = {
        "support": 1 if players <= 2 else 2,
        "tank": 1 if players <= 1 else 2,
        "animal": 1 if players <= 3 else 2,
        "large": 1,
    }

    picked: List[Dict[str, Any]] = []
    picked_names: Set[str] = set()

    combat_first = [u for u in pool_all if _role(u) == "combat" and not _is_animal(u) and not _is_large(u)]
    first = weighted_choice(combat_first) or weighted_choice([u for u in pool_all if _role(u) == "combat"]) or weighted_choice(pool_all)
    if not first:
        return []
    picked.append(first)
    picked_names.add(first["name"])

    role_counts = {"support": 0, "tank": 0, "combat": 1}
    if _role(first) != "combat":
        role_counts[_role(first)] = 1
        role_counts["combat"] = 0
    if _is_animal(first):
        role_counts["animal"] = 1
    if _is_large(first):
        role_counts["large"] = 1

    while len(picked) < target_types:
        candidates = [u for u in pool_all if u["name"] not in picked_names]
        if not candidates:
            break

        weights = [_type_pick_weight(u, role_counts, caps) for u in candidates]
        chosen = _weighted_choice_by_float(candidates, weights)
        if not chosen:
            break

        picked.append(chosen)
        picked_names.add(chosen["name"])

        r = _role(chosen)
        role_counts[r] = role_counts.get(r, 0) + 1
        if _is_animal(chosen):
            role_counts["animal"] = role_counts.get("animal", 0) + 1
        if _is_large(chosen):
            role_counts["large"] = role_counts.get("large", 0) + 1

    return picked


def enforce_role_model_caps(
    enemy_types: List[Dict[str, Any]],
    enemy_counts: Dict[str, int],
    *,
    total_enemies: int,
    has_leader: bool,
    players: int,
    footnotes: List[str],
) -> Dict[str, int]:
    """
    Post-pass to keep SUPPORT/TANK/ANIMAL/LARGE model counts from dominating.
    Moves excess bodies onto a combat type (if one exists), otherwise the first type.
    """
    minions = max(0, total_enemies - (1 if has_leader else 0))
    if minions <= 0 or not enemy_types:
        return enemy_counts

    support_cap = max(1, round(minions * 0.20))
    tank_cap    = max(1, round(minions * 0.35))
    animal_cap  = max(1, round(minions * 0.25))
    large_cap   = max(1, round(minions * 0.15))

    name_to_unit = {u["name"]: u for u in enemy_types}

    def count_where(pred) -> int:
        return sum(enemy_counts.get(nm, 0) for nm, u in name_to_unit.items() if pred(u))

    def pick_sink_name() -> str:
        for u in enemy_types:
            if _role(u) == "combat" and not _is_animal(u) and not _is_large(u):
                return u["name"]
        for u in enemy_types:
            if _role(u) == "combat":
                return u["name"]
        return enemy_types[0]["name"]

    sink = pick_sink_name()

    def move_excess(pred, cap: int, label: str):
        nonlocal sink
        cur = count_where(pred)
        if cur <= cap:
            return

        excess = cur - cap
        role_names = [u["name"] for u in enemy_types if pred(u) and u["name"] != sink]
        role_names.sort(key=lambda nm: enemy_counts.get(nm, 0), reverse=True)

        moved = 0
        for nm in role_names:
            while excess > 0 and enemy_counts.get(nm, 0) > 0:
                enemy_counts[nm] -= 1
                enemy_counts[sink] = enemy_counts.get(sink, 0) + 1
                excess -= 1
                moved += 1
            if excess <= 0:
                break

        if moved > 0:
            footnotes.append(f"ROSTER: shifted {moved} {label} minions onto '{sink}' to prevent role spam.")

    move_excess(lambda u: _role(u) == "support", support_cap, "support")
    move_excess(lambda u: _role(u) == "tank", tank_cap, "tank")
    move_excess(lambda u: _is_animal(u), animal_cap, "animal")
    move_excess(lambda u: _is_large(u), large_cap, "large")

    return enemy_counts


# ============================================================
# #LEADERSHIP — selection + minimum requirements
# ============================================================

def choose_leadership_tier(requested: str, allies_total: int, scenario_min: str, footnotes: List[str]) -> str:
    tier = normalize_leadership_value(requested)

    if tier == "Random":
        tier = random.choices(
            [x[0] for x in LEADERSHIP_WEIGHTS],
            weights=[x[1] for x in LEADERSHIP_WEIGHTS],
            k=1
        )[0]

    # Large parties should almost always have leadership
    if allies_total > 6 and tier == "None":
        footnotes.append("REROLL: Leadership cannot be None when total allies > 6; upgraded to Lieutenant.")
        tier = "Lieutenant"

    # Respect scenario minlead (e.g. Final Showdown)
    if leadership_rank(tier) < leadership_rank(scenario_min):
        footnotes.append(f"MUST: leadership upgraded from {tier} to {scenario_min} (scenario minlead).")
        tier = scenario_min

    return tier


# ============================================================
# #SCENARIO TYPES — selection + hard objective rules
# ============================================================
def objectives_allowed_for_scenario(picked_scen: Dict[str, Any], objectives_all: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scenario row: allow_obj_IDs=eliminate,leader,ritual
    Objective rows: ID=leader, ID=ritual, etc.
    If allow_obj_IDs missing/blank, allow all objectives.
    """
    allow_raw = (picked_scen.get("allow_obj_IDs") or "").strip()
    if not allow_raw:
        return objectives_all[:]

    allowed_ids = {x.strip() for x in allow_raw.split(",") if x.strip()}
    out: List[Dict[str, Any]] = []
    for o in objectives_all:
        oid = (o.get("ID") or "").strip()
        if oid and oid in allowed_ids:
            out.append(o)
    return out


# ============================================================
# #OBJECTIVES — selection + must requirements
# ============================================================

def pick_objective_for_scenario(
    scen_entries: List[Dict[str, Any]],
    requested_name: str,
    scenario_type: Optional[Dict[str, Any]],
    settings: Dict[str, str],
    footnotes: List[str],
) -> Optional[Dict[str, Any]]:
    objectives_all = [e for e in scen_entries if "objective" in (e.get("tags", set()) or set())]
    if not objectives_all:
        footnotes.append("WARNING: No objective-tagged entries found in scenario_objectives.txt.")
        return None

    allowed = objectives_all
    if scenario_type:
        allowed = objectives_allowed_for_scenario(scenario_type, objectives_all)
        if not allowed:
            footnotes.append("WARNING: Scenario type has no allowed objectives; falling back to all objectives.")
            allowed = objectives_all

    if requested_name and requested_name.strip().lower() != "random":
        req = requested_name.strip().lower()
        for e in allowed:
            if e["name"].strip().lower() == req:
                return e
        footnotes.append(f"WARNING: Requested objective '{requested_name}' not allowed/found; rolling allowed objective.")

    return weighted_choice_by_weight(allowed, settings=settings, prefix="objective")


def objective_required_tags(obj: Optional[Dict[str, Any]]) -> Set[str]:
    if not obj:
        return set()
    t = set(obj.get("tags", set()) or set())
    t.discard("generic")
    t.discard("objective")
    t.discard("scenariotype")
    return t


def parse_objective_musts(obj: Optional[Dict[str, Any]]) -> List[str]:
    if not obj:
        return []

    musts: List[str] = []
    must_field = str(obj.get("must", "") or "").strip()
    if must_field:
        for m in must_field.split(","):
            m = m.strip()
            if m:
                musts.append(m)

    for d in obj.get("directives", []) or []:
        if d.lower().startswith("must-clue:"):
            musts.append(d[len("must-"):])  # "must-clue:..." -> "clue:..."

    return musts


def must_clue_requirements(obj: Optional[Dict[str, Any]], footnotes: List[str]) -> Dict[str, int]:
    reqs: Dict[str, int] = {}

    for m in parse_objective_musts(obj):
        parts = [p.strip() for p in m.split(":") if p.strip()]
        if len(parts) < 2:
            footnotes.append(f"WARNING: Unrecognized must directive '{m}'.")
            continue

        kind = parts[0].lower()
        if kind == "target":
            # Handled elsewhere (e.g., objective_requires_leader). Ignore here to avoid noisy warnings.
            continue
        if kind != "clue":
            footnotes.append(f"WARNING: must directive kind '{kind}' not implemented yet: '{m}'.")
            continue

        tag = parts[1]
        count = 1
        for p in parts[2:]:
            if p.lower().startswith("count="):
                count = parse_int_maybe(p.split("=", 1)[1], 1)

        if tag:
            reqs[tag] = reqs.get(tag, 0) + max(1, count)

    return reqs


# ============================================================
# #THREATS — settings filter + weighted selection
# ============================================================
def sanitize_threat_tags(
    threat_tags: Set[str],
    all_threats: List[str],
    settings: Dict[str, str],
    footnotes: List[str]
) -> Set[str]:
    """
    Applies enabled/disabled/strict behavior to threats.
    If strict and none remain, replace with one allowed threat (weighted).
    """
    strict = s_get_bool(settings, "strict_collections", False)

    enabled = set(s_get_list(settings, "enabled.threats")) | set(s_get_list(settings, "enable_threat"))
    disabled = set(s_get_list(settings, "disabled.threats")) | set(s_get_list(settings, "disable_threat"))

    filtered = {t for t in threat_tags if t and t.lower() != "none"}
    filtered = {t for t in filtered if t not in disabled}
    if enabled:
        # 'monsters' is a special pool used by lair mode; allow it even if it's not in enabled.threats.
        filtered = {t for t in filtered if (t in enabled) or (t == "monsters")}

    if filtered:
        return filtered

    if strict and threat_tags:
        weighted_pool = threat_pool_from_settings(all_threats, settings)
        picked = pick_weighted_tag(weighted_pool, exclude=set())
        if picked:
            footnotes.append(f"SETTINGS: strict_collections enforced; threats overridden to '{picked}'.")
            return {picked}
        footnotes.append("SETTINGS: strict_collections enforced but no allowed threats remain.")
        return set()

    if threat_tags and not filtered:
        footnotes.append("SETTINGS: threat filters removed all selected threats; proceeding with original threat selection.")
        return threat_tags

    return filtered


# ============================================================
# #LEADER SELECTION + minion distribution
# ============================================================

def select_leader_unit(enemy_units: List[Dict[str, Any]], threat_tags: Set[str], footnotes: List[str]) -> Optional[Dict[str, Any]]:
    """Pick a leader matching threats; fallback to threat=generic; last resort any leader."""
    leader_pool = [u for u in enemy_units if eligible_enemy_unit(u) and is_leader_unit(u)]

    leader = roll_from_list_hard_threat(leader_pool, set(threat_tags))
    if leader:
        return leader

    generic_pool = [u for u in leader_pool if "generic" in (u.get("threats", set()) or set())]
    if generic_pool:
        footnotes.append("FALLBACK: No threat-matching leader found; using threat=generic leader.")
        return weighted_choice(generic_pool)

    if leader_pool:
        footnotes.append("FALLBACK: No generic leader found; using any available leader (off-threat).")
        return weighted_choice(leader_pool)

    footnotes.append(f"WARNING: No leader-eligible units exist for threats {sorted(threat_tags)}.")
    return None


def distribute_minions(total_enemies: int, has_leader: bool, enemy_types: List[Dict[str, Any]]) -> Dict[str, int]:
    minions = max(0, total_enemies - (1 if has_leader else 0))
    if not enemy_types or minions == 0:
        return {}

    counts = {t["name"]: 0 for t in enemy_types}
    per = minions // len(enemy_types)
    rem = minions % len(enemy_types)

    for t in enemy_types:
        counts[t["name"]] = per

    types_shuffled = enemy_types[:]
    random.shuffle(types_shuffled)
    for i in range(rem):
        counts[types_shuffled[i]["name"]] += 1

    return counts


# ============================================================
# #TEXT — Article helper (a/an)
# ============================================================

def _needs_an(word: str) -> bool:
    w = (word or "").strip()
    if not w:
        return False

    if w.isupper():
        return w[0] in set("AEFHILMNORSX")

    wl = w.lower()
    for x in ("honest", "honor", "hour", "heir"):
        if wl.startswith(x):
            return True
    for x in ("uni", "use", "user", "euro", "one"):
        if wl.startswith(x):
            return False
    return wl[0] in "aeiou"


def fix_indefinite_articles(text: str) -> str:
    def repl(m: re.Match) -> str:
        art = m.group(1)
        word = m.group(2)
        use_an = _needs_an(word)
        if art[0].isupper():
            return ("An " if use_an else "A ") + word
        return ("an " if use_an else "a ") + word
    return re.sub(r"\b([Aa]n?)\s+([A-Za-z][A-Za-z'\-]*)\b", repl, text)


# ============================================================
# #TEMPLATES — [roll:...] parsing + filtering + resolving
# ============================================================

ROLL_RE = re.compile(r"\[roll:(?P<kind>[a-zA-Z_]+)(?P<filters>:[^\]]+)?\]")


def parse_roll_filters(filters: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not filters:
        return out
    parts = [p.strip() for p in filters.split(":") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def filter_entries_by_roll(e: Dict[str, Any], f: Dict[str, str]) -> bool:
    """
    Supported filters:
      - rarity=...
      - tag=comma list (intersects)
      - tier=...
      - threat=comma list (intersects)
    """
    if "rarity" in f:
        # allow comma-lists, e.g. rarity=common,uncommon
        wanted = parse_tags(f["rarity"])
        if wanted:
            rv = str(e.get("rarity", "")).strip().lower()
            if rv not in wanted:
                return False
        else:
            if str(e.get("rarity", "")).strip().lower() != f["rarity"].strip().lower():
                return False

    if "tier" in f:
        # allow comma-lists, e.g. tier=minion,elite
        wanted = parse_tags(f["tier"])
        tv = str(e.get("tier", "")).strip().lower()
        if wanted and tv not in wanted:
            return False

    if "tag" in f:
        wanted = parse_tags(f["tag"])
        if wanted and not wanted.intersection(e.get("tags", set()) or set()):
            return False

    if "threat" in f:
        wanted = parse_tags(f["threat"])
        if wanted and not wanted.intersection(e.get("threats", set()) or set()):
            return False

    return True


def resolve_roll_token(
    kind: str,
    filters: Dict[str, str],
    *,
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    footnotes: List[str]
) -> str:
    """
    Kinds: unit, animal, item, spell
    Aliases: herb/potion/armor/weapon -> item subsets
    count= / n= supported (best-effort unique)
    """
    k = (kind or "").strip().lower()
    f = dict(filters or {})

    count = parse_int_maybe(f.pop("count", None), 1)
    if count == 1:
        count = parse_int_maybe(f.pop("n", None), 1)
    count = max(1, count)

    def choose_many(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not pool:
            return []
        picks: List[Dict[str, Any]] = []
        used: Set[str] = set()
        for _ in range(count):
            uniq = [p for p in pool if p.get("name") not in used]
            chosen = weighted_choice(uniq if uniq else pool)
            if not chosen:
                break
            picks.append(chosen)
            used.add(chosen.get("name"))
        return picks

    # item aliases
    if k in ("herb", "potion", "armor"):
        if "tag" in f and f["tag"].strip():
            f["tag"] = f"{f['tag']},{k}"
        else:
            f["tag"] = k
        k = "item"

    if k == "weapon":
        pool = []
        for it in items:
            if not filter_entries_by_roll(it, f):
                continue
            it_tags = it.get("tags", set()) or set()
            if it_tags.intersection({"hand", "twohand", "ranged"}):
                pool.append(it)
        picks = choose_many(pool)
        if picks:
            return ", ".join(p["name"] for p in picks)
        footnotes.append(f"WARNING: roll:weapon found no matches for filters={filters}.")
        return "Unknown Weapon"

    if k == "animal":
        if "tag" in f and f["tag"].strip():
            f["tag"] = f"{f['tag']},animal"
        else:
            f["tag"] = "animal"
        pool = [u for u in enemy_units if eligible_enemy_unit(u) and filter_entries_by_roll(u, f)]
        picks = choose_many(pool)
        if picks:
            return ", ".join(p["name"] for p in picks)
    # Fallback: if a threat filter starves the pool, relax it (animal identity is already enforced via tag).
        if "threat" in f:
            f_relaxed = dict(f)
            f_relaxed.pop("threat", None)
            pool2 = [u for u in enemy_units if eligible_enemy_unit(u) and filter_entries_by_roll(u, f_relaxed)]
            picks2 = choose_many(pool2)
            if picks2:
                footnotes.append(f"WARNING: roll:animal had no matches for filters={filters}; relaxed threat constraint.")
                return ", ".join(p["name"] for p in picks2)
        footnotes.append(f"WARNING: roll:animal found no matches for filters={filters}.")
        return "Unknown Animal"

    if k == "unit":
        # Fix common authoring mistake: 'monsters' is a threat pool label, not a unit tag.
        if "tag" in f:
            tagset = parse_tags(f.get("tag") or "")
            if "monsters" in tagset:
                tagset.discard("monsters")
                if tagset:
                    f["tag"] = ",".join(sorted(tagset))
                else:
                    f.pop("tag", None)
                f.setdefault("threat", "monsters")

        pool = [u for u in enemy_units if eligible_enemy_unit(u) and filter_entries_by_roll(u, f)]
        picks = choose_many(pool)
        if picks:
            return ", ".join(p["name"] for p in picks)

        # Fallbacks (in order): relax tag, then relax threat.
        # This prevents "Unknown Unit" spam when a specific filter has no matches.
        if "tag" in f and "threat" in f:
            f_relaxed = dict(f)
            f_relaxed.pop("tag", None)
            pool2 = [u for u in enemy_units if eligible_enemy_unit(u) and filter_entries_by_roll(u, f_relaxed)]
            picks2 = choose_many(pool2)
            if picks2:
                footnotes.append(f"WARNING: roll:unit had no matches for filters={filters}; relaxed tag constraint.")
                return ", ".join(p["name"] for p in picks2)

        if "threat" in f:
            f_relaxed = dict(f)
            f_relaxed.pop("threat", None)
            pool3 = [u for u in enemy_units if eligible_enemy_unit(u) and filter_entries_by_roll(u, f_relaxed)]
            picks3 = choose_many(pool3)
            if picks3:
                footnotes.append(f"WARNING: roll:unit had no matches for filters={filters}; relaxed threat constraint.")
                return ", ".join(p["name"] for p in picks3)

        footnotes.append(f"WARNING: roll:unit found no matches for filters={filters}.")
        return "Unknown Unit"

    if k == "item":
        pool = [it for it in items if filter_entries_by_roll(it, f)]
        picks = choose_many(pool)
        if picks:
            return ", ".join(p["name"] for p in picks)
        footnotes.append(f"WARNING: roll:item found no matches for filters={filters}.")
        return "Unknown Item"

    if k == "spell":
        pool = [sp for sp in spells if filter_entries_by_roll(sp, f)]
        picks = choose_many(pool)
        if picks:
            return ", ".join(p["name"] for p in picks)
        footnotes.append(f"WARNING: roll:spell found no matches for filters={filters}.")
        return "Unknown Spell"

    footnotes.append(f"WARNING: Unknown roll kind '{kind}'.")
    return f"Unknown {kind.title()}"


def expand_templates(
    text: str,
    *,
    chosen_unit_names: List[str],
    allowed_threats: Set[str],
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    footnotes: List[str],
) -> str:
    """
    Expands:
      - legacy: [unit], [item]
      - roll:   [roll:unit:...], [roll:item:...], [roll:spell:...], [roll:animal:...]
    Default rule:
      - roll:unit / roll:animal get scoped to selected threats unless they specify threat=...
    """
    if not text:
        return text

    if "[unit]" in text:
        text = text.replace("[unit]", random.choice(chosen_unit_names) if chosen_unit_names else "Unknown Unit")
        if not chosen_unit_names:
            footnotes.append("WARNING: [unit] placeholder found but no chosen units available.")

    if "[item]" in text:
        text = text.replace("[item]", resolve_roll_token("item", {}, enemy_units=enemy_units, items=items, spells=spells, footnotes=footnotes))

    def _repl(m: re.Match) -> str:
        kind = (m.group("kind") or "").strip()
        f = parse_roll_filters(m.group("filters") or "")
        if kind.lower() in ("unit", "animal"):
            # Compatibility: treat tag=monster as threat=monsters (units generally use threat=monsters, not tag=monster).
            if "tag" in f:
                tagset = parse_tags(f.get("tag", ""))
                if "monster" in tagset:
                    tagset.discard("monster")
                    if tagset:
                        f["tag"] = ",".join(sorted(tagset))
                    else:
                        f.pop("tag", None)

                    # Ensure monsters is present in threat filter.
                    tset = parse_tags(f.get("threat", ""))
                    tset.add("monsters")
                    f["threat"] = ",".join(sorted(tset))

            # Lair rule: if the scenario has forced monsters-only threats, do not allow explicit non-monster threat filters
            # inside [roll:unit:...] tokens to "escape" that constraint.
            if allowed_threats == {"monsters"}:
                f["threat"] = "monsters"
            else:
                # Default rule: scope roll:unit / roll:animal to selected threats unless they specify threat=...
                if allowed_threats and "threat" not in f:
                    f["threat"] = ",".join(sorted(allowed_threats))

        return resolve_roll_token(kind, f, enemy_units=enemy_units, items=items, spells=spells, footnotes=footnotes)

    text = ROLL_RE.sub(_repl, text)
    text = fix_indefinite_articles(text)
    return text


# ============================================================
# #CLUES — Dice range assignment + picking with tag caps
# ============================================================

def chance_weight(ch: str) -> int:
    ch = (ch or "").strip().lower()
    if ch == "high":
        return 8
    if ch == "mid":
        return 6
    if ch == "low":
        return 4
    return 6


def assign_clue_ranges(clues_picked: List[Dict[str, Any]], die_sides: int = 20) -> List[Tuple[int, int, Dict[str, Any]]]:
    if not clues_picked:
        return []

    if len(clues_picked) > die_sides:
        die_sides = 100 if len(clues_picked) <= 100 else len(clues_picked)

    weights = [chance_weight(c.get("chance")) for c in clues_picked]
    total_w = sum(weights) or 1

    raw = [(w / total_w) * die_sides for w in weights]
    counts = [max(1, int(round(x))) for x in raw]

    def total_counts() -> int:
        return sum(counts)

    if total_counts() > die_sides:
        idxs = sorted(range(len(counts)), key=lambda i: (weights[i], counts[i]))
        guard = 0
        while total_counts() > die_sides and guard < 10000:
            for idx in idxs:
                if total_counts() <= die_sides:
                    break
                if counts[idx] > 1:
                    counts[idx] -= 1
            guard += 1

    if total_counts() < die_sides:
        idxs = sorted(range(len(counts)), key=lambda i: (weights[i], raw[i]), reverse=True)
        guard = 0
        while total_counts() < die_sides and guard < 10000:
            for idx in idxs:
                if total_counts() >= die_sides:
                    break
                counts[idx] += 1
            guard += 1

    ranges: List[Tuple[int, int, Dict[str, Any]]] = []
    cursor = 1
    for clue, cnt in zip(clues_picked, counts):
        start = cursor
        end = cursor + cnt - 1
        ranges.append((start, end, clue))
        cursor = end + 1

    return ranges


def can_pick_with_tag_caps(entry: Dict[str, Any], tag_counts: Dict[str, int], tag_caps: Dict[str, int]) -> bool:
    tags = entry.get("tags", set()) or set()
    for tag, cap in (tag_caps or {}).items():
        if tag in tags and tag_counts.get(tag, 0) >= cap:
            return False
    return True


def bump_tag_caps(entry: Dict[str, Any], tag_counts: Dict[str, int], tag_caps: Dict[str, int]) -> None:
    tags = entry.get("tags", set()) or set()
    for tag in (tag_caps or {}).keys():
        if tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1


def pick_required_clues_by_tag(
    *,
    clues: List[Dict[str, Any]],
    required: Dict[str, int],
    chosen_unit_names: List[str],
    allowed_threats: Set[str],
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    picked_names: Set[str],
    footnotes: List[str],
    tag_counts: Optional[Dict[str, int]] = None,
    tag_caps: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    For each required clue tag, pick N unique clues that include that tag.
    Applies tag caps (ex: weather max=2) and expands templates.
    """
    picked: List[Dict[str, Any]] = []
    tag_counts = tag_counts if tag_counts is not None else {}
    tag_caps = tag_caps if tag_caps is not None else {}

    for tag, cnt in required.items():
        for _ in range(cnt):
            pool = [
                c for c in clues
                if c["name"] not in picked_names
                and tag in (c.get("tags", set()) or set())
                and can_pick_with_tag_caps(c, tag_counts, tag_caps)
            ]
            if not pool:
                footnotes.append(f"WARNING: Missing required clue for tag '{tag}' (needed {cnt}).")
                break

            chosen = weighted_choice(pool)
            if not chosen:
                footnotes.append(f"WARNING: Failed to roll required clue for tag '{tag}'.")
                break

            ccopy = dict(chosen)
            text = ccopy.get("description", "")
            ccopy["description"] = expand_templates(
                text,
                chosen_unit_names=chosen_unit_names,
                allowed_threats=allowed_threats,
                enemy_units=enemy_units,
                items=items,
                spells=spells,
                footnotes=footnotes,
            )

            picked.append(ccopy)
            picked_names.add(ccopy["name"])
            bump_tag_caps(ccopy, tag_counts, tag_caps)

    return picked


def pick_weighted_clues(
    *,
    clues: List[Dict[str, Any]],
    count: int,
    obj_tags: Set[str],
    chosen_unit_names: List[str],
    allowed_threats: Set[str],
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    picked_names: Set[str],
    footnotes: List[str],
    tag_counts: Optional[Dict[str, int]] = None,
    tag_caps: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Fill remaining clue slots using generic + objective-tag-weighted logic (no duplicates), respecting tag caps."""
    picked: List[Dict[str, Any]] = []
    tag_counts = tag_counts if tag_counts is not None else {}
    tag_caps = tag_caps if tag_caps is not None else {}

    for _ in range(count):
        pool: List[Dict[str, Any]] = []
        for c in clues:
            if c["name"] in picked_names:
                continue
            if not can_pick_with_tag_caps(c, tag_counts, tag_caps):
                continue

            tags = c.get("tags", set()) or set()
            if "generic" in tags:
                pool.append(c)
            elif obj_tags and obj_tags.intersection(tags):
                pool.append(c)

        if not pool:
            footnotes.append("WARNING: Ran out of eligible clues before reaching target.")
            break

        chosen = weighted_choice(pool)
        if not chosen:
            break

        ccopy = dict(chosen)
        text = ccopy.get("description", "")
        ccopy["description"] = expand_templates(
            text,
            chosen_unit_names=chosen_unit_names,
            allowed_threats=allowed_threats,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            footnotes=footnotes,
        )

        picked.append(ccopy)
        picked_names.add(ccopy["name"])
        bump_tag_caps(ccopy, tag_counts, tag_caps)

    return picked


# ============================================================
# #EVENTS — Difficulty gating + finals last
# ============================================================

def event_allowed_by_difficulty(e: Dict[str, Any], difficulty: str) -> bool:
    min_d = (e.get("min_difficulty") or "").strip()
    max_d = (e.get("max_difficulty") or "").strip()
    severity = (e.get("severity") or "").strip().lower()

    if min_d and DIFF_IDX.get(difficulty, 0) < DIFF_IDX.get(min_d, 0):
        return False
    if max_d and DIFF_IDX.get(difficulty, 0) > DIFF_IDX.get(max_d, 99):
        return False
    if severity == "harsh" and difficulty in ("Easy", "Normal"):
        return False
    return True


def assign_event_cards(n: int) -> List[str]:
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    suits = ["Hearts", "Spades", "Diamonds", "Clubs"]
    cards: List[str] = []
    for suit in suits:
        for r in ranks:
            cards.append(f"{r} of {suit}")
            if len(cards) >= n:
                return cards
    return cards[:n]


def pick_weighted_events(
    *,
    events: List[Dict[str, Any]],
    count: int,
    obj_tags: Set[str],
    difficulty: str,
    chosen_unit_names: List[str],
    allowed_threats: Set[str],
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    picked_names: Set[str],
    footnotes: List[str]
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []

    finals_all = [e for e in events if "final" in (e.get("tags", set()) or set())]
    normals_all = [e for e in events if "final" not in (e.get("tags", set()) or set())]

    def build_pool(src: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []
        for e in src:
            if e["name"] in picked_names:
                continue
            if not event_allowed_by_difficulty(e, difficulty):
                continue
            tags = e.get("tags", set()) or set()
            if "generic" in tags:
                pool.append(e)
            elif obj_tags and obj_tags.intersection(tags):
                pool.append(e)
        return pool

    picked: List[Dict[str, Any]] = []

    final_pool = build_pool(finals_all)
    want_final = True if final_pool and count >= 1 else False
    normal_target = count - (1 if want_final else 0)

    for _ in range(normal_target):
        pool = build_pool(normals_all)
        if not pool:
            footnotes.append("WARNING: Ran out of eligible events before reaching target.")
            break

        chosen = weighted_choice(pool)
        if not chosen:
            break

        ecopy = dict(chosen)
        text = ecopy.get("description", "")
        ecopy["description"] = expand_templates(
            text,
            chosen_unit_names=chosen_unit_names,
            allowed_threats=allowed_threats,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            footnotes=footnotes,
        )
        picked.append(ecopy)
        picked_names.add(ecopy["name"])

    if want_final:
        final_pool = build_pool(finals_all)
        chosen = weighted_choice(final_pool)
        if chosen:
            ecopy = dict(chosen)
            text = ecopy.get("description", "")
            ecopy["description"] = expand_templates(
                text,
                chosen_unit_names=chosen_unit_names,
                allowed_threats=allowed_threats,
                enemy_units=enemy_units,
                items=items,
                spells=spells,
                footnotes=footnotes,
            )
            picked.append(ecopy)
            picked_names.add(ecopy["name"])

    return picked


# ============================================================
# #EVENTS — Rooms-based event deck builder (v1)
# ============================================================

def _event_weight_with_settings(e: Dict[str, Any], settings: Dict[str, str]) -> float:
    """Tag-weighted event selection for rooms-based decks (soft balancing)."""
    base = 1.0
    try:
        base = float(e.get("weight", 1.0) or 1.0)
    except Exception:
        base = 1.0
    tags = set(e.get("tags", set()) or set())

    mult = 1.0
    # Global weights: events.tag_weight.<tag>
    # Rooms-based overrides: rooms.events.tag_weight.<tag>
    for t in tags:
        mult *= s_get_float(settings, f"events.tag_weight.{t}", 1.0)
        mult *= s_get_float(settings, f"rooms.events.tag_weight.{t}", 1.0)

    w = base * mult
    # Clamp to keep random.choices stable even if someone cranks weights wildly.
    return max(0.0001, min(w, 10000.0))


def _weighted_choice_by(pool: List[Dict[str, Any]], weight_fn) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    weights = [max(0.0001, float(weight_fn(e) or 0.0)) for e in pool]
    return random.choices(pool, weights=weights, k=1)[0]


def build_rooms_based_event_deck(
    *,
    events: List[Dict[str, Any]],
    count: int,
    scenario_type_id: str,
    difficulty: str,
    chosen_unit_names: List[str],
    allowed_threats: Set[str],
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    settings: Dict[str, str],
    footnotes: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Rooms-based event decks are primarily built from tag=room_event entries.
    Unlike the legacy clue/event picker, we do NOT require 'generic' or objective-tag matching here.

    Blending behavior (best-effort):
      - pick unique room_event events first
      - if we still need more, optionally top up from generic events (excluding weather by default)

    Tag balancing:
      - use multiplicative weights from settings:
          events.tag_weight.<tag>   (global)
          rooms.events.tag_weight.<tag> (rooms-only override)
    """
    target = max(0, int(count or 0))
    if target <= 0:
        return [], {}

    exclude_tags = set(s_get_list(settings, "rooms.events.exclude_tags")) or {"weather"}
    include_generic = s_get_bool(settings, "rooms.events.include_generic", True)

    def eligible(e: Dict[str, Any]) -> bool:
        if not event_allowed_by_difficulty(e, difficulty):
            return False
        tags = set(e.get("tags", set()) or set())
        if exclude_tags and tags.intersection(exclude_tags):
            return False
        return True

    room_pool = [e for e in events if eligible(e) and ("room_event" in (e.get("tags", set()) or set()))]
    generic_pool: List[Dict[str, Any]] = []
    if include_generic:
        generic_pool = [
            e for e in events
            if eligible(e)
            and ("generic" in (e.get("tags", set()) or set()))
            and ("room_event" not in (e.get("tags", set()) or set()))
        ]

    
    # Special pressure-trigger events are NOT dealt into the base shuffled deck.
    # They are shown in a separate section and are injected later if/when Pressure reaches thresholds.
    stid = (scenario_type_id or "").strip().lower()

    # Global special cards (never random-dealt). We will display a mode-relevant subset.
    global_special_names: Dict[str, str] = {
        "alarm": "Alarm",
        "lockdown": "Lockdown",
        "villain_flees": "The Villain Flees",
        "lair_alert": "Lair Alert",
        "beast_stirs": "The Beast Stirs",
    }

    # Find specials by name (case-insensitive) from eligible room_event pool first, else from eligible generic.
    specials_all: Dict[str, Dict[str, Any]] = {}

    def _find_by_name(name: str) -> Optional[Dict[str, Any]]:
        low = (name or "").strip().lower()
        for pool in (room_pool, generic_pool):
            for e in pool:
                if (e.get("name","") or "").strip().lower() == low:
                    return e
        return None

    for key, nm in global_special_names.items():
        found = _find_by_name(nm)
        if found:
            specials_all[key] = found

    # Remove all found specials from selectable pools
    special_set = {(e.get("name","") or "").strip().lower() for e in specials_all.values()}
    if special_set:
        room_pool = [e for e in room_pool if (e.get("name","") or "").strip().lower() not in special_set]
        generic_pool = [e for e in generic_pool if (e.get("name","") or "").strip().lower() not in special_set]

    # Pressure-trigger events returned for separate display / later injection.
    pressure_cards: Dict[str, Dict[str, Any]] = {}

    # Mode-specific subset
    if stid == "lair":
        for k in ("lair_alert", "beast_stirs"):
            if k in specials_all:
                pressure_cards[k] = specials_all[k]
    else:
        for k in ("alarm", "lockdown", "villain_flees"):
            if k in specials_all:
                pressure_cards[k] = specials_all[k]

    # Also keep any generic threshold-tagged cards available (hunt/catastrophe, etc.) if present.
    for e in events:
        if not eligible(e):
            continue
        tags = set(e.get("tags", set()) or set())
        if "hunt" in tags and "hunt" not in pressure_cards:
            pressure_cards["hunt"] = e
        if "catastrophe" in tags and "catastrophe" not in pressure_cards:
            pressure_cards["catastrophe"] = e

    # Expand templates in pressure cards too (so [roll:unit:...] resolves)
    for k, e in list(pressure_cards.items()):
        ecopy = dict(e)
        ecopy["description"] = expand_templates(
            ecopy.get("description", ""),
            chosen_unit_names=chosen_unit_names,
            allowed_threats=allowed_threats,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            footnotes=footnotes,
        )
        # Sanitize any legacy instructions that talk about shuffling/placing cards into the deck.
        # Pressure trigger events are NOT part of the random deck; they are injected separately.
        desc = str(ecopy.get("description",""))
        desc = re.sub(r"\s*Then place this card[^.]*\.(\s*)", " ", desc, flags=re.IGNORECASE)
        desc = re.sub(r"\b(shuffle|place)\b[^.]*\bdeck\b[^.]*\.(\s*)", " ", desc, flags=re.IGNORECASE)
        ecopy["description"] = re.sub(r"\s{2,}", " ", desc).strip()
        pressure_cards[k] = ecopy

    picked: List[Dict[str, Any]] = []
    used: Set[str] = set()

    # Reserve a final card if available
    finals = [e for e in room_pool if "final" in (e.get("tags", set()) or set())]
    final_event = None
    if finals and target >= 1:
        final_event = _weighted_choice_by(finals, lambda x: _event_weight_with_settings(x, settings))

    normal_target = target - (1 if final_event else 0)

    def pick_from(pool: List[Dict[str, Any]], want: int) -> None:
        nonlocal picked, used
        for _ in range(want):
            candidates = [e for e in pool if e.get("name") not in used]
            if not candidates:
                break
            chosen = _weighted_choice_by(candidates, lambda x: _event_weight_with_settings(x, settings))
            if not chosen:
                break
            ecopy = dict(chosen)
            ecopy["description"] = expand_templates(
                ecopy.get("description", ""),
                chosen_unit_names=chosen_unit_names,
                allowed_threats=allowed_threats,
                enemy_units=enemy_units,
                items=items,
                spells=spells,
                footnotes=footnotes,
            )
            picked.append(ecopy)
            used.add(ecopy.get("name", ""))

    # Primary: room_event pool
    pick_from(room_pool, normal_target)

    # Top-up: generic pool if needed
    if len(picked) < normal_target and generic_pool:
        need = normal_target - len(picked)
        pick_from(generic_pool, need)

    if final_event:
        fe = dict(final_event)
        fe["description"] = expand_templates(
            fe.get("description", ""),
            chosen_unit_names=chosen_unit_names,
            allowed_threats=allowed_threats,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            footnotes=footnotes,
        )
        picked.append(fe)


    # ----------------------------
    # Coverage / balance repair pass
    # ----------------------------
    def _has_tag(ev: Dict[str, Any], tag: str) -> bool:
        return tag in (ev.get("tags", set()) or set())

    # Minimum tag targets
    min_loot = int(s_get_int(settings, "rooms.events.min.loot", 1))
    min_skill = int(s_get_int(settings, "rooms.events.min.skillcheck", 1))
    # Cap combat share (0-1); if unset, no cap
    max_combat_frac = s_get_float(settings, "rooms.events.max_frac.combat", 0.60)

    def _count_tag(tag: str) -> int:
        return sum(1 for e in picked if _has_tag(e, tag))

    # Candidates not yet used (raw pools, templates expanded on swap-in)
    def _remaining_candidates() -> List[Dict[str, Any]]:
        pools = []
        pools.extend(room_pool)
        pools.extend(generic_pool)
        return [e for e in pools if (e.get("name") not in used)]

    def _swap_in(tag_needed: str) -> bool:
        """Replace a non-needed card with one that has tag_needed."""
        candidates = [e for e in _remaining_candidates() if _has_tag(e, tag_needed)]
        if not candidates:
            return False
        chosen = _weighted_choice_by(candidates, lambda x: _event_weight_with_settings(x, settings))
        if not chosen:
            return False

        # choose a swap-out index: prefer swapping out something that is NOT tag_needed
        swap_idxs = [i for i, e in enumerate(picked) if not _has_tag(e, tag_needed) and "final" not in (e.get("tags", set()) or set())]
        if not swap_idxs:
            return False
        swap_i = random.choice(swap_idxs)

        ecopy = dict(chosen)
        ecopy["description"] = expand_templates(
            ecopy.get("description", ""),
            chosen_unit_names=chosen_unit_names,
            allowed_threats=allowed_threats,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            footnotes=footnotes,
        )

        used.discard(picked[swap_i].get("name", ""))
        picked[swap_i] = ecopy
        used.add(ecopy.get("name", ""))
        return True

    # Enforce minimums (best-effort)
    while _count_tag("loot") < min_loot:
        if not _swap_in("loot"):
            break
    while _count_tag("skillcheck") < min_skill:
        if not _swap_in("skillcheck"):
            break

    # Enforce combat cap (best-effort)
    if max_combat_frac is not None:
        try:
            max_combat = int(math.floor(max_combat_frac * float(target)))
            combat_now = _count_tag("combat")
            if combat_now > max_combat:
                # swap out some combat cards for non-combat
                for _ in range(combat_now - max_combat):
                    noncombat_candidates = [e for e in _remaining_candidates() if not _has_tag(e, "combat")]
                    if not noncombat_candidates:
                        break
                    chosen = _weighted_choice_by(noncombat_candidates, lambda x: _event_weight_with_settings(x, settings))
                    if not chosen:
                        break
                    swap_idxs = [i for i, e in enumerate(picked) if _has_tag(e, "combat") and "final" not in (e.get("tags", set()) or set())]
                    if not swap_idxs:
                        break
                    swap_i = random.choice(swap_idxs)
                    ecopy = dict(chosen)
                    ecopy["description"] = expand_templates(
                        ecopy.get("description", ""),
                        chosen_unit_names=chosen_unit_names,
                        allowed_threats=allowed_threats,
                        enemy_units=enemy_units,
                        items=items,
                        spells=spells,
                        footnotes=footnotes,
                    )
                    used.discard(picked[swap_i].get("name", ""))
                    picked[swap_i] = ecopy
                    used.add(ecopy.get("name", ""))
        except Exception:
            pass
    if len(picked) < target:
        footnotes.append(
            f"WARNING: Ran out of eligible events before reaching target (picked {len(picked)}/{target})."
        )

    footnotes.append(
        f"EVENT_DECK(rooms): room_pool={len(room_pool)}, generic_pool={len(generic_pool)}, picked={len(picked)} (target={target})."
    )

    return picked, pressure_cards

# ============================================================
# #DEBUG — Pool summary
# ============================================================

def pool_summary(enemy_units: List[Dict[str, Any]], threat_tags: Set[str]) -> Dict[str, Dict[str, int]]:
    minions = [u for u in enemy_units if is_nonleader_unit(u)]
    leaders = [u for u in enemy_units if eligible_enemy_unit(u) and is_leader_unit(u)]

    summary: Dict[str, Dict[str, int]] = {}
    for t in sorted(threat_tags):
        summary[t] = {
            "minions": sum(1 for u in minions if t in (u.get("threats", set()) or set())),
            "leaders": sum(1 for u in leaders if t in (u.get("threats", set()) or set())),
        }
    return summary


# ============================================================
# #BRIEFING — Output formatting
# ============================================================

# ============================================================
# ROOMS — Delve layout (rooms unique, transitions may repeat)
# ============================================================

_PRESSURE_RE = re.compile(r"\bPressure\s*=?\s*([+-]?\d+)\b", re.IGNORECASE)


def _csv_set(raw: str) -> Set[str]:
    return {x.strip().lower() for x in (raw or "").split(",") if x.strip()}


def _has_flag(e: Dict[str, Any], flag: str) -> bool:
    flags = _csv_set(e.get("flag", ""))
    return flag.lower() in flags


def _is_room_entry(e: Dict[str, Any]) -> bool:
    return str(e.get("id") or e.get("ID") or "").strip().upper().startswith("R")


def _is_transition_entry(e: Dict[str, Any]) -> bool:
    return str(e.get("id") or e.get("ID") or "").strip().upper().startswith("T")


def _room_allowed_for_threats(e: Dict[str, Any], threat_tags: Set[str]) -> bool:
    """
    Threat rules:
      - only=... is OR semantics (any match passes)
      - not=... blocks if any match
      - blank only/not means "no restriction"
    """
    threats = {t.strip().lower() for t in (threat_tags or set()) if t and t.strip().lower() != "none"}
    only_set = _csv_set(e.get("only", ""))
    not_set  = _csv_set(e.get("not", ""))

    if not_set and threats.intersection(not_set):
        return False
    if only_set and not threats.intersection(only_set):
        return False
    return True



def validate_room_graph(
    room_plan: List[Dict[str, Any]],
    *,
    objective_id: str,
    scenario_type_id: str,
    leadership_tier: str,
    leader_block: Optional[Dict[str, Any]],
    settings: Dict[str, str],
) -> List[str]:
    """Return a list of WARNING:/ERROR: strings describing structural issues."""
    msgs: List[str] = []
    if not room_plan:
        return msgs

    labels = [(chr(ord("A") + i) if i < 26 else f"Z+{i-25}") for i in range(len(room_plan))]
    edges = None
    try:
        edges = room_plan[0].get("_graph_edges")
    except Exception:
        edges = None

    if not edges:
        msgs.append("WARNING: No graph edges found; treating layout as linear.")
        edges = [(i, i+1) for i in range(len(room_plan)-1)]

    # Build adjacency
    adj = {i: set() for i in range(len(room_plan))}
    for a, b in edges:
        if isinstance(a, int) and isinstance(b, int) and 0 <= a < len(room_plan) and 0 <= b < len(room_plan):
            adj[a].add(b)
            adj[b].add(a)

    # Find entrance (assume first room) and objective/boss indices
    entrance_idx = 0
    objective_idx = None
    boss_idx = None
    for i, e in enumerate(room_plan):
        # Single source of truth: explicit markers set during plan build.
        if boss_idx is None and (e.get("_boss_room") or e.get("_is_boss_room")):
            boss_idx = i
        if objective_idx is None and (e.get("_objective_room") or e.get("_is_objective_room")):
            objective_idx = i

    if objective_idx is None:
        msgs.append("ERROR: Objective room missing (no _objective_room marker).")
    else:
        # BFS for reachability entrance -> objective
        q = [entrance_idx]
        seen = {entrance_idx}
        while q:
            cur = q.pop(0)
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if objective_idx not in seen:
            msgs.append(
                f"ERROR: No valid path from entrance {labels[entrance_idx]} to objective {labels[objective_idx]}."
            )

    # Composition checks: on the spine BEFORE objective (branches + transitions excluded)
    spine = [i for i, e in enumerate(room_plan) if not e.get("_branch")]
    if objective_idx is not None and objective_idx in spine:
        obj_pos = spine.index(objective_idx)
        before = spine[:obj_pos]
    else:
        before = spine[:-1]  # best-effort if objective is missing

    def _is_transition(i: int) -> bool:
        return (room_plan[i].get("kind") == "transition") or _is_transition_entry(room_plan[i])

    def _has_check_room(e: Dict[str, Any]) -> bool:
        if (e.get("check") or "").strip():
            return True
        tags = set(e.get("tags", set()) or set())
        return "check" in tags

    def _has_spawn_text(e: Dict[str, Any]) -> bool:
        for k in ("entering", "effect", "fail", "description"):
            v = e.get(k)
            if isinstance(v, str) and re.search(r"\bSpawn\b", v, flags=re.IGNORECASE):
                return True
        return False

    def _has_combat_room(e: Dict[str, Any]) -> bool:
        tags = set(e.get("tags", set()) or set())
        if "combat" in tags:
            return True
        # Many rooms encode combat implicitly via Spawn text.
        return _has_spawn_text(e)

    combat_ct = 0
    check_ct = 0
    for i in before:
        if _is_transition(i):
            continue
        e = room_plan[i]
        if _has_combat_room(e):
            combat_ct += 1
        if _has_check_room(e):
            check_ct += 1

    min_combat = int(s_get_int(settings, "rooms.min_combat_before_objective", 2))
    min_check  = int(s_get_int(settings, "rooms.min_check_before_objective", 1))
    if combat_ct < min_combat:
        msgs.append(f"ERROR: Pre-objective spine has only {combat_ct} combat rooms (needs ≥{min_combat}).")
    if check_ct < min_check:
        msgs.append(f"ERROR: Pre-objective spine has only {check_ct} check rooms (needs ≥{min_check}).")

    # Lair-specific: nests at branch ends, no loops expected
    stid = (scenario_type_id or "").strip().lower()
    if stid == "lair":
        for i, e in enumerate(room_plan):
            tags = set(e.get("tags", set()) or set())
            if "nest" in tags:
                if not e.get("_branch_end"):
                    msgs.append(f"ERROR: Nest room {labels[i]} is not marked as end-of-branch.")
                if not e.get("_branch"):
                    msgs.append(f"WARNING: Nest room {labels[i]} is not a branch room (expected side branch).")

        # crude cycle detection: edges > nodes-1 implies a loop somewhere
        unique_edges = {tuple(sorted((a,b))) for a,b in edges if isinstance(a,int) and isinstance(b,int)}
        if len(unique_edges) > (len(room_plan) - 1):
            msgs.append("WARNING: Lair graph contains extra edges (may form loops); lairs typically avoid loops.")

    # Boss/leader presence sanity
    if leader_block and leadership_tier.strip().lower() == "boss":
        if boss_idx is None:
            msgs.append("WARNING: Leader exists but no boss room was marked (_boss_room/flag=boss).")

    return msgs

def _pressure_delta_from_entry(e: Dict[str, Any]) -> int:
    """
    Preferred: pressure= / Pressure= fields.
    Fallback: scan entering/effect text for 'Pressure+X' / 'Pressure=+X' etc.
    """
    for k in ("pressure", "Pressure"):
        if k in e and str(e.get(k, "")).strip() != "":
            return parse_int_maybe(str(e.get(k)), 0)

    delta = 0
    for field in ("entering", "effect"):
        txt = str(e.get(field, "") or "")
        for m in _PRESSURE_RE.finditer(txt):
            delta += parse_int_maybe(m.group(1), 0)
    return delta


def _room_style_tag(e: Dict[str, Any]) -> str:
    if _is_transition_entry(e):
        return "Transition"
    return "Room"


def _rooms_target_count(players: int, difficulty: str, settings: Dict[str, str]) -> int:
    """
    v1.1 default (tweakable):
      rooms.count=... overrides everything (includes the entrance room)
      else: 5 + difficulty_index + ceil(players/2), clamped 5..10
    """
    override = s_get_int(settings, "rooms.count", 0)
    if override > 0:
        return clamp(override, 3, 15)

    di = DIFF_IDX.get(difficulty, 1)
    # players influence gently; allied combatants tend to matter less for exploration pacing
    base = 5 + di + ((max(1, players) + 1) // 2)
    return clamp(base, 5, 10)


def build_room_plan(
    rooms_entries: List[Dict[str, Any]],
    *,
    threat_tags: Set[str],
    players: int,
    difficulty: str,
    settings: Dict[str, str],
    chosen_unit_names: List[str],
    enemy_units: List[Dict[str, Any]],
    items: List[Dict[str, Any]],
    spells: List[Dict[str, Any]],
    footnotes: List[str],
    leadership_tier: Optional[str] = None,
    scenario_type_id: str = "",
    objective_id: str = "",
) -> List[Dict[str, Any]]:
    """Build a linear room/transition plan.

    Rules (v1):
    - Rooms do not repeat, except: transitions may repeat; rooms tagged 'nest' may repeat.
    - Rooms tagged 'lair' are LAIR-ONLY (excluded from non-lair room modes).
    - Entrance room (flag=entrance) is forced first when available.
    - Objective room is forced last:
        * Prefer rooms with obj_only matching objective_id
        * Fallback to flag=objective
    - Transitions are inserted between rooms with a configurable chance.
    - Mainline plan avoids transitions tagged 'deadend_risk' (to preserve guaranteed completion path).
    """

    # --------------------------------------------------------------------
    # Basic knobs
    # --------------------------------------------------------------------
    try:
        transition_chance = float(settings.get("rooms.transition_chance", "0.35"))
    except Exception:
        transition_chance = 0.35

    # Optional global multiplier for implicit single-unit rolls inside rooms.
    # Only applies when a room uses [roll:unit:...] without an explicit count.
    try:
        unit_mul = int(settings.get("rooms.unit_mul", "1"))
    except Exception:
        unit_mul = 1
    unit_mul = max(1, unit_mul)

    allow_deadend_risk = settings.get("rooms.allow_deadend_risk", "0").strip() in {"1", "true", "yes", "on"}

    stid = (scenario_type_id or "").strip().lower()
    is_lair_mode = stid == "lair"

    # Lair-only rooms: tag=lair means ONLY for lair mode.
    def _is_lair_only(e: Dict[str, Any]) -> bool:
        tags = set((e.get("tags") or []))
        return "lair" in tags

    # Repeatable rooms: transitions always, and rooms tagged nest.
    def _is_repeatable(e: Dict[str, Any]) -> bool:
        tags = set((e.get("tags") or []))
        return ("transition" in tags and "room" not in tags) or ("nest" in tags)

    # Eligibility wrapper: threat filters + lair-only filter
    def _eligible(e: Dict[str, Any]) -> bool:
        if not _room_allowed_for_threats(e, threat_tags):
            return False
        if _is_lair_only(e) and not is_lair_mode:
            return False
        return True

    # --------------------------------------------------------------------
    # Split entries
    # --------------------------------------------------------------------
    eligible = [e for e in rooms_entries if _eligible(e)]
    rooms = [e for e in eligible if not _is_transition_entry(e)]
    transitions_all = [e for e in eligible if _is_transition_entry(e)]

    transitions: List[Dict[str, Any]] = []
    for t in transitions_all:
        tags = set((t.get("tags") or []))
        if (not allow_deadend_risk) and ("deadend_risk" in tags):
            continue
        transitions.append(t)

    # --------------------------------------------------------------------
    # Pick entrance room (forced first)
    # --------------------------------------------------------------------
    entrance_room = None
    for r in rooms:
        if _has_flag(r, "entrance"):
            entrance_room = r
            break

    # --------------------------------------------------------------------
    # Pick objective room (rooms-based v1 priority)
    # --------------------------------------------------------------------
    obj_id = (objective_id or "").strip().lower()

    def _obj_only_matches(r: Dict[str, Any]) -> bool:
        raw = (r.get("obj_only") or "").strip().lower()
        if not raw or not obj_id:
            return False
        allowed = {x.strip().lower() for x in raw.split(",") if x.strip()}
        return obj_id in allowed

    def _is_objective_candidate(r: Dict[str, Any]) -> bool:
        """Return True if this room is eligible to serve as the objective room for obj_id.

        Important: if a room declares obj_only=..., it is ONLY eligible for those objectives.
        This prevents using rescue-only rooms for ritual/fetch, etc.
        """
        raw = (r.get("obj_only") or "").strip().lower()
        if raw:
            return _obj_only_matches(r)
        return _has_flag(r, "objective")

    # Gather candidates, weighted
    obj_candidates = [r for r in rooms if _is_objective_candidate(r)]

    # Lair rescue: never select a boss-flagged room as the damsel objective (must be separate).
    if is_lair_mode and obj_id == "rescue":
        _no_boss = [r for r in obj_candidates if not (_has_flag(r, "boss") or _has_flag(r, "last"))]
        # Mandatory: if we cannot find a non-boss candidate, force the generic fallback room
        # (prevents boss+objective overlap in lair rescue).
        obj_candidates = _no_boss

    objective_room = None
    if obj_candidates:
        objective_room = weighted_choice(obj_candidates)
    else:
        # Hard fallback (prevents "ritual objective but no ritual room")
        objective_room = {
            "id": "R_OBJ",
            "name": "Objective Chamber",
            "size": "medium",
            "description": "A purpose-built chamber tied to the current objective.",
            "entering": "",
            "effect": "",
            "check": "",
            "fail": "",
            "loot": "",
            "tag": "room,objective",
            "flag": "objective",
        }
        footnotes.append(f"WARNING: no objective room candidates found for obj_id={obj_id!r}; using generic Objective Chamber.")

    # Lair rescue rule: objective (damsel) must be separate from boss and not near entrance
    if is_lair_mode and obj_id == "rescue":
        # Mark objective_room so we can keep it away from entrance/boss later
        objective_room["_force_far_from_entrance"] = True
        objective_room["_force_not_adjacent_boss"] = True

    # --------------------------------------------------------------------
    # Leader/Boss room handling
    # --------------------------------------------------------------------
    # Rooms-based modes can have a "leader" (unit) without the objective being "leader".
    # In that case, we often want a distinct boss room (Leader Location) separate from the
    # objective room (Objective Location). Lair rescue is mandatory-separate.
    wants_leader = (leadership_tier or "").strip().lower() not in {"", "none"}
    objective_is_leader = obj_id == "leader"

    separate_chance = s_get_float(settings, "rooms.separate_leader_objective_chance", 0.65)
    must_separate = (is_lair_mode and obj_id == "rescue")
    should_separate = (must_separate or (wants_leader and (not objective_is_leader) and (random.random() < separate_chance)))

    def _is_boss_candidate(r: Dict[str, Any]) -> bool:
        return _has_flag(r, "boss") or _has_flag(r, "last")

    # If separation is desired, avoid boss-flagged rooms as the objective room.
    if should_separate and objective_room is not None and _is_boss_candidate(objective_room):
        obj_candidates_no_boss = [
            r for r in obj_candidates
            if (not _is_boss_candidate(r))
        ]
        if obj_candidates_no_boss:
            objective_room = weighted_choice(obj_candidates_no_boss)
            if is_lair_mode and obj_id == "rescue":
                objective_room["_force_far_from_entrance"] = True
                objective_room["_force_not_adjacent_boss"] = True
        else:
            footnotes.append("Note: Objective room candidates were all boss-flagged; allowing objective to share a boss room.")
            should_separate = False

    boss_room: Optional[Dict[str, Any]] = None
    if wants_leader:
        # If the objective IS the leader objective, treat the objective room as the boss room.
        if objective_is_leader:
            boss_room = objective_room
        else:
            boss_candidates = [r for r in rooms if ("leader" in _csv_set(str(r.get("obj_only", "") or "")))]
            if should_separate:
                boss_candidates = [r for r in boss_candidates if r.get("id") != objective_room.get("id")]
            if boss_candidates:
                boss_room = weighted_choice(boss_candidates)
            else:
                # Hard fallback boss room (prevents leader without a location)
                boss_room = {
                    "id": "R_BOSS",
                    "name": "Boss Lair",
                    "size": "large",
                    "description": "A fortified chamber where the enemy leader makes their stand.",
                    "entering": "",
                    "effect": "",
                    "check": "",
                    "fail": "",
                    "loot": "",
                    "tag": "room,boss",
                    "flag": "boss,last",
                }
                footnotes.append("WARNING: no boss-flagged rooms found; using generic Boss Lair for leader location.")


    # --------------------------------------------------------------------
    # Decide how many rooms total
    # --------------------------------------------------------------------
    count = _rooms_target_count(players, difficulty, settings)

    forced = [r for r in [entrance_room, objective_room, boss_room] if r is not None]
    forced_ids = {r.get("id") for r in forced if r.get("id")}

    # --------------------------------------------------------------------
    # Monster nests (Option B)
    # --------------------------------------------------------------------
    nests_to_place = 0
    nest_room_types: List[Dict[str, Any]] = []
    if is_lair_mode:
        base = 2 if players <= 2 else (3 if players <= 4 else 4)
        diff_add = 0
        diff_key = (difficulty or "").strip().lower()
        if diff_key == "hard":
            diff_add = 1
        elif diff_key == "brutal":
            diff_add = 2
        nests_to_place = base + diff_add

        nest_room_types = [r for r in rooms if "nest" in set((r.get("tags") or []))]
        if not nest_room_types:
            footnotes.append("Note: Lair mode is enabled but no rooms with tag=nest were found; no nests were placed.")
            nests_to_place = 0

    def _is_nest_room(r: Dict[str, Any]) -> bool:
        return "nest" in set((r.get("tags") or []))

    # --------------------------------------------------------------------
    # Build candidate pools for enforcing requirements
    # --------------------------------------------------------------------
    def _is_combat_room(r: Dict[str, Any]) -> bool:
        return "combat" in set((r.get("tags") or []))

    def _is_check_room(r: Dict[str, Any]) -> bool:
        if r.get("check"):
            return True
        return "check" in set((r.get("tags") or []))

    # Exclude forced rooms and (in lair) nest rooms from the unique pool
    pool_unique = [
        r for r in rooms
        if (r.get("id") not in forced_ids)
        and (not (is_lair_mode and _is_nest_room(r)))
    ]

    combat_pool = [r for r in pool_unique if _is_combat_room(r)]
    check_pool = [r for r in pool_unique if _is_check_room(r)]
    other_pool = [r for r in pool_unique if (r not in combat_pool and r not in check_pool)]

    forced_count = (1 if entrance_room is not None else 0) + (1 if objective_room is not None else 0)
    remaining_slots = max(0, count - forced_count - nests_to_place)

    picked_rooms: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()

    def _pick_from(pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        cand = [r for r in pool if (r.get("id") not in used_ids)]
        if not cand:
            return None
        r = weighted_choice(cand)
        if r.get("id"):
            used_ids.add(r["id"])
        return r

    # Enforce: >=2 combat + >=1 check
    if remaining_slots > 0:
        for _ in range(2):
            if len(picked_rooms) >= remaining_slots:
                break
            r = _pick_from(combat_pool)
            if r is not None:
                picked_rooms.append(r)
        if len(picked_rooms) < remaining_slots:
            r = _pick_from(check_pool)
            if r is not None:
                picked_rooms.append(r)

    while len(picked_rooms) < remaining_slots:
        r = _pick_from(pool_unique)
        if r is None:
            break
        picked_rooms.append(r)

    # --------------------------------------------------------------------
    # Insert nest instances (repeatable) for lairs
    # --------------------------------------------------------------------
    nest_instances: List[Dict[str, Any]] = []
    if nests_to_place > 0 and nest_room_types:
        for i in range(1, nests_to_place + 1):
            base_room = weighted_choice(nest_room_types)
            inst = copy.deepcopy(base_room)
            inst["_nest_instance"] = i
            inst["name"] = f'{inst.get("name", "Nest")} (Nest {i})'
            nest_instances.append(inst)

    # NOTE: nests are meant to feel like side objectives. We attach them as *branch endpoints*
    # later when building the graph, rather than forcing them onto the main spine.

    # --------------------------------------------------------------------
    # Assemble main spine order (single source of truth for objective/boss)
    # --------------------------------------------------------------------
    ordered_rooms: List[Dict[str, Any]] = []
    if entrance_room is not None:
        ordered_rooms.append(entrance_room)

    ordered_rooms.extend(picked_rooms)

    # Ensure boss room is on the spine and END-BIASED when a leader exists.
    # If a boss-flagged room was randomly picked early, we move it to the end so that
    # "objective before boss" doesn't accidentally place the objective near the entrance.
    if wants_leader and boss_room is not None and (boss_room is not objective_room):
        boss_id = boss_room.get("id")
        if boss_id:
            ordered_rooms = [r for r in ordered_rooms if r.get("id") != boss_id]
        ordered_rooms.append(boss_room)

    # Place objective room with separation rules.
    if objective_room is not None:
        obj_id_val = objective_room.get("id")
        if obj_id_val:
            ordered_rooms = [r for r in ordered_rooms if r.get("id") != obj_id_val]

        if objective_is_leader:
            ordered_rooms.append(objective_room)
        else:
            if wants_leader and boss_room is not None and boss_room is not objective_room:
                boss_id = boss_room.get("id")
                boss_idx = None
                if boss_id:
                    for i, r in enumerate(ordered_rooms):
                        if r.get("id") == boss_id:
                            boss_idx = i
                            break
                if boss_idx is None:
                    boss_idx = len(ordered_rooms)

                insert_at = max(1, boss_idx)  # default right before boss
                # Repair: keep objective after the required pre-objective rooms.
                min_floor = 1 + int(s_get_int(settings, "rooms.min_combat_before_objective", 2)) + int(s_get_int(settings, "rooms.min_check_before_objective", 1))
                insert_at = max(insert_at, min_floor)

                if is_lair_mode and obj_id == "rescue":
                    insert_at = max(insert_at, 3)  # deep
                    insert_at = min(insert_at, max(1, boss_idx - 2))  # not adjacent

                ordered_rooms.insert(insert_at, objective_room)
            else:
                ordered_rooms.append(objective_room)


    # --------------------------------------------------------------------
    # Repair pass: enforce pre-objective spine composition (rooms only)
    # --------------------------------------------------------------------
    min_combat = int(s_get_int(settings, "rooms.min_combat_before_objective", 2))
    min_check = int(s_get_int(settings, "rooms.min_check_before_objective", 1))

    def _room_text_blob(r: Dict[str, Any]) -> str:
        return " ".join(str(r.get(k, "") or "") for k in ("entering","description","effect","check","fail","loot"))

    def _is_check_room(r: Dict[str, Any]) -> bool:
        tags = set((r.get("tags") or []))
        if "check" in tags:
            return True
        return bool((r.get("check") or "").strip())

    def _is_combat_room(r: Dict[str, Any]) -> bool:
        tags = set((r.get("tags") or []))
        if "combat" in tags:
            return True
        blob = _room_text_blob(r)
        return "spawn" in blob.lower()

    def _count_preobjective(rooms_only: List[Dict[str, Any]]) -> Tuple[int,int]:
        if not objective_room or objective_room not in rooms_only:
            return (0,0)
        obj_idx = rooms_only.index(objective_room)
        pre = rooms_only[:obj_idx]
        # exclude entrance from pacing counts (it can be empty setup)
        if pre and entrance_room and pre[0] is entrance_room:
            pre = pre[1:]
        c = sum(1 for rr in pre if _is_combat_room(rr))
        k = sum(1 for rr in pre if _is_check_room(rr))
        return (c,k)

    def _pick_unused_room(predicate) -> Optional[Dict[str, Any]]:
        pool = []
        for rr in rooms:
            rid = rr.get("id")
            if not rid:
                continue
            if rid in used_ids:
                continue
            # never use the objective itself here; we'll insert it later
            if objective_room is not None and rid == objective_room.get("id"):
                continue
            if boss_room is not None and rid == boss_room.get("id"):
                continue
            # avoid lair-only rooms outside lair (already filtered) and avoid nests unless lair
            tags = set((rr.get("tags") or []))
            if ("nest" in tags) and (not is_lair_mode):
                continue
            if predicate(rr):
                pool.append(rr)
        return weighted_choice(pool) if pool else None

    # Try to insert missing pacing rooms directly before the objective (and before boss, if boss is last).
    # This makes the "guarantees" real instead of only warnings.
    max_repairs = 6
    for _ in range(max_repairs):
        combat_ct, check_ct = _count_preobjective(ordered_rooms)
        if combat_ct >= min_combat and check_ct >= min_check:
            break

        need_check = check_ct < min_check
        need_combat = combat_ct < min_combat

        insert_room = None
        if need_check:
            insert_room = _pick_unused_room(_is_check_room)
        if insert_room is None and need_combat:
            insert_room = _pick_unused_room(_is_combat_room)

        if insert_room is not None:
            # Insert immediately before objective
            obj_idx = ordered_rooms.index(objective_room) if objective_room in ordered_rooms else len(ordered_rooms)
            # Never insert after boss if boss is supposed to be last
            if boss_room is not None and boss_room in ordered_rooms:
                boss_idx = ordered_rooms.index(boss_room)
                obj_idx = min(obj_idx, boss_idx)
            obj_idx = max(1, obj_idx)  # keep entrance first
            ordered_rooms.insert(obj_idx, insert_room)
            if insert_room.get("id"):
                used_ids.add(insert_room["id"])
            continue

        # If we cannot insert missing room types (pool exhausted), push objective deeper.
        if objective_room in ordered_rooms:
            obj_idx = ordered_rooms.index(objective_room)
            # Find a later, swappable room (not boss)
            for j in range(obj_idx + 1, len(ordered_rooms)):
                if boss_room is not None and ordered_rooms[j] is boss_room:
                    continue
                ordered_rooms[obj_idx], ordered_rooms[j] = ordered_rooms[j], ordered_rooms[obj_idx]
                break
        else:
            break

    # Mark forced rooms as used so we don't duplicate them in branches.
    for _r in [entrance_room, boss_room, objective_room]:
        if _r is not None and _r.get("id"):
            used_ids.add(_r["id"])


    # --------------------------------------------------------------------
    # Insert transitions between rooms
    # --------------------------------------------------------------------
    plan: List[Dict[str, Any]] = []
    for i, room in enumerate(ordered_rooms):
        plan.append(room)
        if i == len(ordered_rooms) - 1:
            break
        if transitions and random.random() < transition_chance:
            t = weighted_choice(transitions)
            # Lair implicit roaming guard behavior on every transition
            if is_lair_mode:
                t2 = copy.deepcopy(t)
                entering = (t2.get("entering") or "").strip()
                # Choose roaming guard tier (default minion; chance for elite)
                elite_chance = s_get_float(settings, "rooms.lair.roaming_elite_chance", 0.25)
                tier = "elite" if random.random() < elite_chance else "minion"
                add = f"Roaming Guard: Spawn 1 [roll:unit:threat=monsters:tier={tier}]" 
                if add.lower() not in entering.lower():
                    entering = (entering.rstrip("; ") + "; " + add) if entering else add
                    t2["entering"] = entering
                plan.append(t2)
            else:
                plan.append(t)

    
    # --------------------------------------------------------------------
    # Branches (v1.1): make branches feel real, not like single-room "appendices".
    # - Branches are small chains (length 2-3 by default)
    # - Lair nests, when present, should be the *end* of a branch
    # - Delve/site: prefer some kind of loot/cache at the end of a branch
    # --------------------------------------------------------------------
    edges: List[Tuple[int, int]] = [(i, i + 1) for i in range(len(plan) - 1)]

    branch_chance = s_get_float(settings, "rooms.branch_chance", 0.75)
    branch_count = s_get_int(settings, "rooms.branch_count", 0)
    if branch_count <= 0:
        branch_count = 1 if players <= 2 else (2 if players <= 4 else 3)

    branch_min = s_get_int(settings, "rooms.branch_min", 3)
    if branch_min > 0:
        branch_count = max(branch_count, branch_min)

    # If nests exist, ensure we have enough branches to place them as endpoints.
    if is_lair_mode and nest_instances:
        branch_count = max(branch_count, len(nest_instances))

    # Difficulty scaling: easier adventures have smaller/shorter branches; harder adventures have deeper branches and more of them.
    diff_norm = (difficulty or "normal").strip().lower()
    diff_level = 2
    if diff_norm.startswith("easy"):
        diff_level = 1
    elif diff_norm.startswith("hard"):
        diff_level = 3
    elif diff_norm.startswith("brutal"):
        diff_level = 4

    # Scale number of branches by difficulty (clamped).
    # Easy: -1, Normal: +0, Hard: +1, Brutal: +2
    branch_count = max(0, branch_count + (diff_level - 2))
    branch_len_min = s_get_int(settings, "rooms.branch_length_min", 2)
    branch_len_max = s_get_int(settings, "rooms.branch_length_max", 3)
    branch_len_min = max(1, branch_len_min)
    branch_len_max = max(branch_len_min, branch_len_max)

    # Scale branch depth by difficulty.
    # - Easy: shorter branches
    # - Normal: baseline
    # - Hard/Brutal: longer branches
    if diff_level == 1:
        branch_len_min = max(1, branch_len_min - 1)
        branch_len_max = max(branch_len_min, branch_len_max - 1)
    elif diff_level == 3:
        branch_len_max = max(branch_len_min, branch_len_max + 1)
    elif diff_level >= 4:
        branch_len_min = max(1, branch_len_min + 1)
        branch_len_max = max(branch_len_min, branch_len_max + 2)

    # Candidate branch rooms: any eligible, non-repeatable room not already used in the spine.
    branch_candidates = [r for r in pool_unique if (r.get("id") not in used_ids)]

    def _is_loot_room(r: Dict[str, Any]) -> bool:
        tags = set((r.get("tags") or []))
        if "loot" in tags:
            return True
        return bool((r.get("loot") or "").strip())

    def _pick_branch_room(prefer_loot: bool = False) -> Optional[Dict[str, Any]]:
        nonlocal branch_candidates
        if not branch_candidates:
            return None
        pool = branch_candidates
        if prefer_loot:
            loot_pool = [r for r in branch_candidates if _is_loot_room(r)]
            if loot_pool:
                pool = loot_pool
        base_room = weighted_choice(pool)
        if not base_room:
            return None
        # remove to keep branches unique
        branch_candidates = [r for r in branch_candidates if (r is not base_room and r.get("id") != base_room.get("id"))]
        br = copy.deepcopy(base_room)
        if br.get("id"):
            used_ids.add(br["id"])
        return br

    attach_points = [i for i, e in enumerate(plan) if (not _is_transition_entry(e)) and i != len(plan) - 1]
    # Avoid attaching to the final spine room if possible (keeps branches feeling optional).
    if len(attach_points) > 1:
        attach_points = attach_points[:-1]

    force_branches = (stid in {"lair", "site", "delve"})
    if attach_points and (force_branches or random.random() < branch_chance):
        # We'll build up to branch_count branches, but stop if we run out of material.
        nest_queue = list(nest_instances) if (is_lair_mode and nest_instances) else []

        for b_idx in range(branch_count):
            attach = random.choice(attach_points)
            prev_idx = attach

            # Determine branch length and whether it should end with a nest/loot.
            target_len = random.randint(branch_len_min, branch_len_max)
            end_with_nest = bool(nest_queue)
            end_with_loot = (stid in {"site", "delve"})

            # If we are ending with a nest/loot room, reserve the last slot.
            interior_len = target_len - (1 if (end_with_nest or end_with_loot) else 0)
            interior_len = max(1, interior_len)

            # Optionally place a short transition as the "door" to the branch.
            if transitions and random.random() < s_get_float(settings, "rooms.branch_transition_chance", 0.6):
                t = weighted_choice(transitions)
                if t:
                    t2 = copy.deepcopy(t)
                    t2["_branch"] = True
                    t2["_branch_from"] = attach
                    plan.append(t2)
                    edges.append((prev_idx, len(plan) - 1))
                    prev_idx = len(plan) - 1

            # Interior rooms
            for _ in range(interior_len):
                room = _pick_branch_room(prefer_loot=False)
                if not room:
                    break
                room["_branch"] = True
                room["_branch_from"] = attach
                plan.append(room)
                edges.append((prev_idx, len(plan) - 1))
                prev_idx = len(plan) - 1

            # Endpoint room
            if end_with_nest and nest_queue:
                end_room = copy.deepcopy(nest_queue.pop(0))
                end_room["_branch"] = True
                end_room["_branch_from"] = attach
                end_room["_branch_end"] = True
                plan.append(end_room)
                edges.append((prev_idx, len(plan) - 1))
                prev_idx = len(plan) - 1
            elif end_with_loot:
                loot_room = _pick_branch_room(prefer_loot=True)
                if loot_room is None:
                    loot_room = {
                        "id": "R_CACHE",
                        "name": "Hidden Cache",
                        "size": "small",
                        "description": "A tucked-away stash of supplies and valuables.",
                        "entering": "",
                        "effect": "",
                        "check": "",
                        "fail": "",
                        "loot": "Roll 1 [roll:item]",
                        "tag": "room,loot",
                        "flag": "",
                    }
                loot_room["_branch"] = True
                loot_room["_branch_from"] = attach
                loot_room["_branch_end"] = True
                plan.append(loot_room)
                edges.append((prev_idx, len(plan) - 1))

    # --------------------------------------------------------------------
    # Loops / extra corridors (v1.2)
    # - Site/Delve: allow loops to feel like buildings/encampments (multiple routes).
    # - Lair: default is NO loops (feels more organic / monstery).
    #
    # Implementation: add a small number of extra edges:
    #   (a) connect some branch endpoints back into the spine (shortcuts)
    #   (b) connect two non-adjacent spine rooms (a side corridor)
    # --------------------------------------------------------------------
    allow_loops = (stid in {"site", "delve"})
    if allow_loops:
        # Base chance can be tuned in settings. Difficulty nudges it up/down.
        base_loop_chance = s_get_float(settings, "rooms.loop_chance", 0.30)
        loop_chance = base_loop_chance + (diff_level - 2) * 0.10  # easy -0.10, hard +0.10, brutal +0.20
        loop_chance = max(0.0, min(0.75, loop_chance))

        # Build a set for duplicate checking.
        edge_set = {tuple(sorted(e)) for e in edges}

        # Determine spine nodes (non-branch entries in their listed order).
        spine_nodes = [i for i, e in enumerate(plan) if not e.get("_branch")]
        spine_non_transition = [i for i in spine_nodes if not _is_transition_entry(plan[i])]

        # (a) Branch endpoint reconnects
        branch_endpoints = [i for i, e in enumerate(plan) if e.get("_branch_end")]
        for end_idx in branch_endpoints:
            if random.random() > loop_chance:
                continue
            # Attach point for this branch
            attach = plan[end_idx].get("_branch_from")
            if attach is None:
                continue
            # Prefer connecting forward along the spine (feels like a loop/corridor).
            candidates = [n for n in spine_non_transition if n > int(attach) + 1]
            if len(candidates) < 1:
                candidates = [n for n in spine_non_transition if n != int(attach)]
            if not candidates:
                continue
            target = random.choice(candidates)
            key = tuple(sorted((end_idx, target)))
            if key not in edge_set and end_idx != target:
                edges.append((end_idx, target))
                edge_set.add(key)

        # (b) Extra spine corridors
        # Number increases with difficulty: easy 0, normal 1, hard 1-2, brutal 2-3
        extra_min = 0 if diff_level == 1 else (1 if diff_level == 2 else 1)
        extra_max = 0 if diff_level == 1 else (1 if diff_level == 2 else (2 if diff_level == 3 else 3))
        extra_corridors = random.randint(extra_min, extra_max) if extra_max > 0 else 0

        # Attempt to add corridors without breaking readability (avoid immediate neighbors).
        attempts = 0
        while extra_corridors > 0 and attempts < 50 and len(spine_non_transition) >= 4:
            attempts += 1
            a = random.choice(spine_non_transition)
            b = random.choice(spine_non_transition)
            if a == b:
                continue
            if abs(a - b) <= 1:
                continue
            # Avoid directly connecting entrance to objective by accident; keep some mystery.
            if plan[a].get("_entrance_room") and plan[b].get("_objective_room"):
                continue
            key = tuple(sorted((a, b)))
            if key in edge_set:
                continue
            edges.append((a, b))
            edge_set.add(key)
            extra_corridors -= 1

    # Persist edges on the first entry for output rendering.
    if plan:
        plan[0]["_graph_edges"] = edges

    # --------------------------------------------------------------------
    # Normalize kinds for output (rooms vs transitions)
    # --------------------------------------------------------------------
    for entry in plan:
        entry.setdefault("kind", "transition" if _is_transition_entry(entry) else "room")

    # --------------------------------------------------------------------
    # Expand templates and apply implicit unit_mul
    # --------------------------------------------------------------------
    enemy_units_for_templates = enemy_units
    unit_threat_filter = threat_tags
    stid = (scenario_type_id or "").strip().lower()
    if stid == "lair":
        # Lair templates always draw from the global monsters pool (threat-independent).
        unit_threat_filter = {"monsters"}
        enemy_units_for_templates = [u for u in enemy_units_for_templates if "monsters" in (u.get("threats", set()) or set())]

    def _expand_field(val: str) -> str:
        if not val:
            return val

        # Apply unit_mul pre-processing (settings-driven, done here)
        if unit_mul > 1:
            def repl(m):
                inner = m.group(1)
                if re.search(r"(?:^|:)\s*(?:n|count)\s*=", inner):
                    return m.group(0)
                return f"[roll:unit:count={unit_mul}:{inner}]"
            val = re.sub(r"\[roll:unit:([^\]]+)\]", repl, val)

        # In lair mode, prevent leaders from spawning in incidental rolls unless explicitly requested.
        if stid == "lair":
            def _tier_guard(m):
                inner = m.group(1)
                if re.search(r"(?:^|:)\s*tier\s*=", inner):
                    return m.group(0)
                inner2 = inner.strip()
                return f"[roll:unit:tier=minion,elite:{inner2}]" if inner2 else "[roll:unit:tier=minion,elite]"
            val = re.sub(r"\[roll:unit:([^\]]+)\]", _tier_guard, val)

        # Clean item roll phrasing like 'Roll 1 Boots of Soft Tread' -> 'Boots of Soft Tread'
        m_item = re.search(r"\bRoll\s+1\s+([^.;\n]+)$", val.strip())
        if m_item:
            cand = m_item.group(1).strip()
            item_names = {it.get("name") for it in (items or []) if it.get("name")}
            if cand in item_names:
                val = re.sub(r"\bRoll\s+1\s+" + re.escape(cand) + r"\b", cand, val)

        return expand_templates(
            val,
            chosen_unit_names=chosen_unit_names,
            allowed_threats=unit_threat_filter,   # <-- add this if expand_templates supports it
            enemy_units=enemy_units_for_templates, # <-- use the right pool (see below)
            items=items,
            spells=spells,
            footnotes=footnotes,
        )


    for entry in plan:
        for k in ("description", "entering", "effect", "fail", "loot"):
            if isinstance(entry.get(k), str) and entry.get(k):
                entry[k] = _expand_field(entry[k])
    # --------------------------------------------------------------------
    # Pressure (minimal v1): baseline by difficulty + per-room deltas
    # --------------------------------------------------------------------
    base_map = {"Easy": 1, "Normal": 2, "Hard": 3, "Brutal": 4}
    start_pressure = base_map.get(difficulty, 2)
    start_pressure = s_get_int(settings, f"pressure.start.{(difficulty or '').strip().lower()}", start_pressure)

    thresholds = {
        "Alert": s_get_int(settings, "pressure.threshold.alert", 4),
        "Lockdown": s_get_int(settings, "pressure.threshold.lockdown", 7),
        "Hunt": s_get_int(settings, "pressure.threshold.hunt", 9),
        "Catastrophe": s_get_int(settings, "pressure.threshold.catastrophe", 11),
    }
    threshold_rules = {
        "Alert": "Trigger the Alert event (see Pressure Trigger Events). Increase Pressure as directed; do not shuffle into the deck.",
        "Lockdown": "Doors/transitions require unlock/check; forcing through adds noise (Pressure+1).",
        "Hunt": "Spawn 2 minions + 2 elites that sweep rooms toward the party.",
        "Catastrophe": "Lair: a big monster hunts you. Site: the villain attempts to escape.",
    }

    footnotes.append(
        f"PRESSURE: starts at {start_pressure} for {difficulty}. Thresholds: "
        + ", ".join([f"{k}≥{v}" for k, v in thresholds.items()])
        + "."
    )

    running = start_pressure
    hit: Set[str] = set()
    for idx, entry in enumerate(plan, start=1):
        delta = _pressure_delta_from_entry(entry)
        entry["pressure_delta"] = delta
        running += delta
        entry["pressure_running"] = running

        # annotate threshold reach points (first time only)
        for name, th in thresholds.items():
            if name in hit:
                continue
            if running >= th:
                hit.add(name)
                entry.setdefault("pressure_triggers", []).append(name)
                footnotes.append(
                    f"PRESSURE: reached {name} (≥{th}) at step {idx} ({entry.get('id','?')} — {entry.get('name','Unknown')})."
                )



    # --------------------------------------------------------------------
    # Mark boss room (where the leader is placed)
    # --------------------------------------------------------------------
    if wants_leader and boss_room is not None:
        boss_idx = None
        boss_id = boss_room.get("id")
        if boss_id:
            for i, e in enumerate(plan):
                if e.get("id") == boss_id:
                    boss_idx = i
                    break
        # Fallback: final non-branch, non-transition room.
        if boss_idx is None:
            for i in range(len(plan) - 1, -1, -1):
                e = plan[i]
                if e.get("_branch") or _is_transition_entry(e):
                    continue
                boss_idx = i
                break
        if boss_idx is not None:
            # Clear any accidental boss markers first
            for e in plan:
                if "_boss_room" in e:
                    e.pop("_boss_room", None)
                if "_is_boss_room" in e:
                    e.pop("_is_boss_room", None)
            plan[boss_idx]["_boss_room"] = True
            entering = (plan[boss_idx].get("entering") or "").strip()
            note = "Boss: Place the leader here."
            if note.lower() not in entering.lower():
                plan[boss_idx]["entering"] = (entering.rstrip("; ") + "; " + note) if entering else note



    # --------------------------------------------------------------------
    # Mark objective room (single source of truth for table output/validation)
    # --------------------------------------------------------------------
    for e in plan:
        if "_objective_room" in e:
            e.pop("_objective_room", None)
        if "_is_objective_room" in e:
            e.pop("_is_objective_room", None)

    if objective_room is not None:
        obj_idx = None
        obj_id_val = objective_room.get("id")
        if obj_id_val:
            for i, e in enumerate(plan):
                if e.get("id") == obj_id_val:
                    obj_idx = i
                    break
        if obj_idx is None:
            # Fallback: last non-branch, non-transition room.
            for i in range(len(plan) - 1, -1, -1):
                e = plan[i]
                if e.get("_branch") or _is_transition_entry(e):
                    continue
                obj_idx = i
                break
        if obj_idx is not None:
            plan[obj_idx]["_objective_room"] = True

    return plan





# --------------------------------------------------------------------
# Rooms-based helpers: extract spawned units from expanded room/event text
# --------------------------------------------------------------------
_SPAWN_RE = re.compile(r"\bSpawn\s+(\d+)\s+([^.;\n]+)", re.IGNORECASE)
_ROAM_RE  = re.compile(r"\bRoaming Guard:\s*Spawn\s+(\d+)\s+([^.;\n]+)", re.IGNORECASE)

# Patrol phrasing (common in room_event cards)
_PATROL_RE = re.compile(
    r"\bSpawn\s+(?:a|an)\s+patrol\s+(?:consisting\s+of|of)\s+(\d+)\s+([^.;\n]+)",
    re.IGNORECASE,
)

# "add 1 X to that patrol" phrasing
_ADD_RE = re.compile(
    r"\badd\s+(\d+)\s+([^.;\n]+?)\s+to\s+that\s+patrol\b",
    re.IGNORECASE,
)

# "spawn an additional patrol" (no unit name; ignore safely)
_EXTRA_PATROL_RE = re.compile(r"\bspawn\s+an?\s+additional\s+patrol\b", re.IGNORECASE)


def _clean_spawn_blob(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # remove parentheticals like "(Large)" or "(hidden)"
    s = re.sub(r"\([^)]*\)", "", s)
    # strip quotes
    s = s.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_spawn_names(blob: str) -> List[Tuple[int, str]]:
    """Split a spawn description into (count, name) pairs.

    Handles:
      - "2 Giant Snake and 2 Cyclops on opposite sides"
      - "1 Griffin (Large) in cover overlooking..."
      - "3 Gremlin at a random transition..."
      - already-counted blobs (caller may prefix the count)
    """
    out: List[Tuple[int, str]] = []
    s = _clean_spawn_blob(blob)
    if not s:
        return out

    # split on " and " (covers multi-spawn lines)
    parts = [p.strip() for p in re.split(r"\s+and\s+", s) if p.strip()]
    for p in parts:
        # cut off trailing location clauses / extra prose
        p = re.sub(
            r"(?:\s+(?:on|at|near|within|adjacent|in|behind|guarding|"
            r"overlooking|under|over|along|by|from|leading|into)\b.*)$",
            "",
            p,
            flags=re.IGNORECASE,
        ).strip(" .,:;—-")

        m = re.match(r"^(\d+)\s+(.+)$", p)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
        else:
            out.append((1, p.strip()))
    return out


def _extract_unit_counts_from_room_plan(room_plan: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract Spawn X <UnitName> patterns from room plan fields.

    Note: this reads expanded text (after templates are expanded). If the text still contains
    [roll:unit:...] or "Unknown Unit", those will be ignored here.
    """
    counts: Dict[str, int] = {}
    if not room_plan:
        return counts

    def _add(name: str, n: int):
        name = (name or "").strip()
        if not name:
            return
        # Skip unresolved templates and unknown placeholders
        if name.lower().startswith("[roll:") or name.lower().startswith("unknown unit"):
            return
        counts[name] = counts.get(name, 0) + int(n)

    for e in room_plan:
        for k in ("entering", "effect", "fail"):
            txt = e.get(k)
            if not isinstance(txt, str) or not txt:
                continue

            # Roaming guard explicit
            for m in _ROAM_RE.finditer(txt):
                n = int(m.group(1))
                blob = m.group(2)
                for nn, nm in _split_spawn_names(f"{n} {blob}"):
                    _add(nm, nn)

            # Patrol blocks: "Spawn a patrol consisting of 3 X ..."
            for m in _PATROL_RE.finditer(txt):
                n = int(m.group(1))
                blob = m.group(2)
                for nn, nm in _split_spawn_names(f"{n} {blob}"):
                    _add(nm, nn)

            # Patrol additions: "add 1 X to that patrol"
            for m in _ADD_RE.finditer(txt):
                n = int(m.group(1))
                blob = m.group(2)
                for nn, nm in _split_spawn_names(f"{n} {blob}"):
                    _add(nm, nn)

            # Generic spawn: "Spawn 1 Giant Rat ..."
            for m in _SPAWN_RE.finditer(txt):
                n = int(m.group(1))
                blob = m.group(2)
                for nn, nm in _split_spawn_names(f"{n} {blob}"):
                    _add(nm, nn)

    return counts


def _canonicalize_unit_name(raw: str, name_map_lower: Dict[str, str]) -> Optional[str]:
    """Best-effort normalization for unit names extracted from prose (events/rooms)."""
    s = (raw or "").strip()
    if not s:
        return None
    # Trim punctuation and trailing clauses
    s = re.split(r"[\.,;\(\)\[\]\{\}]", s, maxsplit=1)[0].strip()
    s = re.split(r"\b(?:that|which|who|they|it|will|as|at|near|within|behind|guarding|overlooking|under|over|along|by|from)\b", s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    if not s:
        return None

    sl = s.lower()
    # Reject overly-generic one-word extracts
    GENERIC_BAD = {"giant", "greater", "lesser", "young", "ancient", "dire", "dark", "elder"}
    if sl in GENERIC_BAD:
        return None
    if sl in name_map_lower:
        return name_map_lower[sl]

    # crude plural fixes: Flys -> Fly, Flies -> Fly, Wolves -> Wolf
    cand = sl
    if cand.endswith("ies") and len(cand) > 3:
        cand2 = cand[:-3] + "y"
        if cand2 in name_map_lower:
            return name_map_lower[cand2]
    if cand.endswith("ves") and len(cand) > 3:
        cand2 = cand[:-3] + "f"
        if cand2 in name_map_lower:
            return name_map_lower[cand2]
        cand3 = cand[:-3] + "fe"
        if cand3 in name_map_lower:
            return name_map_lower[cand3]
    if cand.endswith("s") and len(cand) > 2:
        cand2 = cand[:-1]
        if cand2 in name_map_lower:
            return name_map_lower[cand2]

    # Prefer longest prefix match against known names
    # (more reliable than substring matching and avoids "giant" false hits)
    candidates = sorted(name_map_lower.items(), key=lambda kv: len(kv[0]), reverse=True)
    for nlow, orig in candidates:
        if sl.startswith(nlow):
            return orig

    # Secondary: whole-word containment (only if the known name is >=2 words)
    # Helps with "spawn 1 rat swarm (giant)" type phrasing
    for nlow, orig in candidates:
        if " " not in nlow:
            continue  # require multi-word names here
        # whole-word match
        if re.search(rf"\b{re.escape(nlow)}\b", sl):
            return orig

    return None


def _extract_unit_counts_from_texts(texts: List[str], enemy_units: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract Spawn patterns from arbitrary text blocks (events, pressure cards, etc.)."""
    counts: Dict[str, int] = {}
    if not texts:
        return counts

    name_map_lower: Dict[str, str] = {}
    for u in enemy_units or []:
        nm = (u.get("name") or "").strip()
        if nm:
            name_map_lower[nm.lower()] = nm

    def _add(raw_name: str, n: int):
        nm = _canonicalize_unit_name(raw_name, name_map_lower)
        if not nm:
            return
        counts[nm] = counts.get(nm, 0) + int(n)

    for txt in texts:
        if not isinstance(txt, str) or not txt:
            continue

        # Roaming guard
        for m in _ROAM_RE.finditer(txt):
            n = int(m.group(1))
            blob = m.group(2)
            for nn, nm in _split_spawn_names(f"{n} {blob}"):
                _add(nm, nn)

        # Patrol wording
        for m in _PATROL_RE.finditer(txt):
            n = int(m.group(1))
            blob = m.group(2)
            for nn, nm in _split_spawn_names(f"{n} {blob}"):
                _add(nm, nn)

        # "add X to that patrol"
        for m in _ADD_RE.finditer(txt):
            n = int(m.group(1))
            blob = m.group(2)
            for nn, nm in _split_spawn_names(f"{n} {blob}"):
                _add(nm, nn)

        # Generic spawn
        for m in _SPAWN_RE.finditer(txt):
            n = int(m.group(1))
            blob = m.group(2)
            for nn, nm in _split_spawn_names(f"{n} {blob}"):
                _add(nm, nn)

    return counts


def build_briefing(
    *,
    timestamp: str,
    room_plan: Optional[List[Dict[str, Any]]] = None,
    inputs: Dict[str, Any],
    biome: Optional[Dict[str, Any]] = None,
    scenario_type: Optional[Dict[str, Any]],
    scenario_setup: str,
    objective: Optional[Dict[str, Any]],
    threats: List[str],
    leadership_tier: str,
    leader_block: Optional[Dict[str, Any]],
    leader_traits: List[Dict[str, Any]],
    enemy_types: List[Dict[str, Any]],
    enemy_counts: Dict[str, int],
    event_enemy_types: Optional[List[Dict[str, Any]]] = None,
    event_enemy_counts: Optional[Dict[str, int]] = None,
    total_enemies: int,
    clue_picks: List[Dict[str, Any]],
    event_picks: List[Dict[str, Any]],
    items: Optional[List[Dict[str, Any]]] = None,
    pressure_cards: Optional[Dict[str, Dict[str, Any]]] = None,
    total_gold: int,
    completion_xp: int,
    size_label: str,
    footnotes: List[str],
) -> str:
    lines: List[str] = []
    rooms_mode = bool(room_plan)

    rolled_items: Set[str] = set()
    item_index: Dict[str, Dict[str, Any]] = { (i.get('name') or ''): i for i in (items or []) if i.get('name') }

    lines.append("SCENARIO BRIEFING")
    lines.append("-" * 60)

    mode_name = str(inputs.get("mode", "Custom")).strip().upper()
    lines.append(f"Mode: {mode_name}")
    lines.append(f"Generated: {timestamp}")

    lines.append("Inputs:")
    for k in ("players", "allied_combatants", "difficulty"):
        if k in inputs:
            lines.append(f"  - {k}: {inputs[k]}")
    lines.append("")

    quest_entry = inputs.get("quest_board_entry")
    campaign_key = inputs.get("campaign_key")
    if quest_entry:
        lines.append("Quest Board Selection")
        lines.append("-" * 60)
        if campaign_key:
            lines.append(f"Campaign Key: {campaign_key}")
        lines.append(f"Scenario: {quest_entry.get('scenario_type_name', 'Scenario')}")
        lines.append(f"Objective: {quest_entry.get('objective_name', 'Objective')}")
        flavor = quest_entry.get("flavor")
        if flavor:
            lines.append(f"Flavor: {flavor}")
        lines.append("")

    lines.append("Scenario Type")
    lines.append("-" * 60)
    if scenario_type:
        lines.append(f"Type: {scenario_type.get('name','Unknown')}")
        if scenario_setup:
            lines.append(f"Setup: {scenario_setup}")
    else:
        lines.append("Type: None")
    lines.append("")

    # Encounter Summary (moved up for rooms-based types)
    lines.append("Encounter Summary")
    lines.append("-" * 60)
    lines.append(f"Objective: {objective['name'] if objective else 'Unknown Objective'}")
    if objective and objective.get("description"):
        lines.append(f"Brief: {objective['description']}")
    if biome and biome.get("name"):
        lines.append(f"Biome: {biome.get('name')}")
    lines.append("")

    if biome:
        lines.append("Biome Effects")
        lines.append("-" * 60)
        boon = str(biome.get("boon") or "").strip()
        bane = str(biome.get("bane") or "").strip()
        if boon:
            lines.append(f"Boon: {boon}")
        if bane:
            lines.append(f"Bane: {bane}")
        biome_tags = {str(t).strip().lower() for t in (biome.get("tags") or set()) if str(t).strip()}
        matched_threats = [t for t in threats if str(t).strip().lower() in biome_tags]
        for mt in matched_threats:
            lines.append(f"[{mt}] may ignore bane effects.")
        lines.append("")

    if room_plan:
        lines.append("Delve Layout")
        lines.append("-" * 60)
        lines.append("Rules: Rooms are unique; transitions may repeat.")
        lines.append("Pressure is baseline from Pressure= fields (and simple text scans); fail effects are not simulated.")
        lines.append("")
        # Connections (graph)
        labels = [(chr(ord("A") + i) if i < 26 else f"Z+{i-25}") for i in range(len(room_plan))]
        edges = None
        if room_plan and isinstance(room_plan[0], dict):
            edges = room_plan[0].get("_graph_edges")

        # Boss room label (if marked)
        boss_idx = None
        for i, e in enumerate(room_plan):
            if e.get("_boss_room") or e.get("_is_boss_room"):
                boss_idx = i
                break

        # Objective room label (if marked)
        objective_idx = None
        for i, e in enumerate(room_plan):
            # Single source of truth: explicit marker set during plan build
            if e.get("_objective_room") or e.get("_is_objective_room"):
                objective_idx = i
                break

        if edges:
            # Build adjacency
            adj = {i: set() for i in range(len(labels))}
            for a, b in edges:
                if 0 <= a < len(labels) and 0 <= b < len(labels):
                    adj[a].add(b)
                    adj[b].add(a)

            # Spine = non-branch rooms in listed order
            spine = [i for i, e in enumerate(room_plan) if not e.get("_branch")]
            spine_labels = [labels[i] for i in spine]
            if len(spine_labels) >= 2:
                main = "↔".join(spine_labels)
            elif spine_labels:
                main = spine_labels[0]
            else:
                main = ""

            # Compress branch attachments like D↔J/K
            extras = []
            spine_set = set(spine)
            for idx_in_spine, node in enumerate(spine):
                prev = spine[idx_in_spine - 1] if idx_in_spine - 1 >= 0 else None
                nxt = spine[idx_in_spine + 1] if idx_in_spine + 1 < len(spine) else None
                base_neighbors = {n for n in (prev, nxt) if n is not None}
                branch_neighbors = [labels[n] for n in sorted(adj[node] - base_neighbors) if n in adj]
                if branch_neighbors:
                    extras.append(f"{labels[node]}↔" + "/".join(branch_neighbors))

            # Also list edges between non-spine nodes (rare) in compact form
            non_spine_edges = set()
            for a, b in edges:
                if a not in spine_set or b not in spine_set:
                    # avoid duplicating simple spine attachments (already shown in extras)
                    if a in spine_set and b not in spine_set:
                        continue
                    if b in spine_set and a not in spine_set:
                        continue
                    key = tuple(sorted((a, b)))
                    non_spine_edges.add(key)
            for a, b in sorted(non_spine_edges):
                extras.append(f"{labels[a]}↔{labels[b]}")

            conn_line = "Connections: " + (main if main else ", ".join(extras))
            if extras:
                conn_line += "; " + "; ".join(extras)


            lines.append(conn_line)

            # Compact adjacency list (table-facing)
            lines.append("Adjacency:")
            for i in range(len(labels)):
                neigh = sorted(adj[i])
                if not neigh:
                    continue
                lines.append(f"  {labels[i]}: " + ", ".join(labels[n] for n in neigh))

        else:
            # Fallback linear
            if len(labels) >= 2:
                lines.append("Connections: " + "↔".join(labels))
            elif labels:
                lines.append("Connections: " + labels[0])

        # Leader and objective locations (table-facing)
        leader_idx = boss_idx
        if leader_idx is None and leader_block:
            # If no explicit boss marker, assume leader is at the last spine room (or last room)
            if edges:
                spine = [i for i, e in enumerate(room_plan) if not e.get("_branch")]
                leader_idx = (spine[-1] if spine else (len(room_plan) - 1))
            else:
                leader_idx = len(room_plan) - 1

        if leader_block and leader_idx is not None and 0 <= leader_idx < len(labels):
            lines.append(f"Leader Location: {labels[leader_idx]}")
        if objective_idx is not None and 0 <= objective_idx < len(labels):
            lines.append(f"Objective Location: {labels[objective_idx]}")
        lines.append("")

        for idx, e in enumerate(room_plan):
            label = chr(ord("A")+idx) if idx < 26 else f"Z+{idx-25}"
            name = e.get("name") or e.get("Name") or "Unknown"
            # Make branches obvious at the table: "(branch from D, end)"
            if e.get("_branch"):
                bf = e.get("_branch_from")
                bf_lab = labels[bf] if (isinstance(bf, int) and 0 <= bf < len(labels)) else "?"
                suffix = f"branch from {bf_lab}"
                if e.get("_branch_end"):
                    suffix += ", end"
                name = f"{name} ({suffix})"
            size = str(e.get("size","")).strip() or "unknown"
            kind = "transition" if e.get("kind") == "transition" else "room"
            delta = int(e.get("pressure_delta", 0))
            # Header (simple table-facing line)
            role_bits = []
            try:
                # Table-facing role tags should be authoritative from the generated plan,
                # not from room data flags (data flags are eligibility hints).
                if idx == 0 or _has_flag(e, "entrance"):
                    role_bits.append("ENTRANCE")
                if (objective_idx is not None and idx == objective_idx):
                    role_bits.append("OBJECTIVE")
                if (boss_idx is not None and idx == boss_idx):
                    role_bits.append("BOSS")
                if _has_flag(e, "exit"):
                    role_bits.append("EXIT")
                if _has_flag(e, "secret"):
                    role_bits.append("SECRET")
                if "nest" in (e.get("tags", set()) or set()):
                    role_bits.append("NEST")
            except Exception:
                pass
            role_tag = (" [" + ", ".join(role_bits) + "]") if role_bits else ""

            if kind == "transition":
                lines.append(f"{label}. {name}{role_tag} (transition) Pressure{delta:+d}")
            else:
                lines.append(f"{label}. {name}{role_tag} ({size} room) Pressure{delta:+d}")

            # Detail lines (GM-facing), avoid repeating the name/ID/tags
            if e.get("entering"):
                lines.append(f"    Entering: {e['entering']}")
            if e.get("description"):
                lines.append(f"    Description: {e['description']}")
            if e.get("effect"):
                lines.append(f"    Effect: {e['effect']}")
            if e.get("check"):
                lines.append(f"    Check: {e['check']}")
            if e.get("fail"):
                lines.append(f"    Fail: {e['fail']}")

            if e.get("loot"):
                raw_loot = str(e["loot"]).strip()

                # Table-facing: remove "Roll N" phrases anywhere in the loot string (not just at the start).
                # Examples:
                #   "+10 gold; Roll 1 Fairlight Leaf" -> "+10 gold; Fairlight Leaf"
                #   "+15 gold ... or Roll 1 Crossbow" -> "+15 gold ... or Crossbow"
                loot_txt = re.sub(r"\bRoll\s+\d+\s+", "", raw_loot, flags=re.IGNORECASE)
                loot_txt = re.sub(r"\bRoll\s+", "", loot_txt, flags=re.IGNORECASE)
                loot_txt = re.sub(r"\s+", " ", loot_txt).strip()
                lines.append(f"    Loot: {loot_txt}")

                # Capture item names for glossary (best-effort).
                # Split on common separators, keeping gold fragments but stripping inline roll directives.
                for part in re.split(r",|;|\bor\b|\band\b", loot_txt, flags=re.IGNORECASE):
                    nm = part.strip()
                    if not nm:
                        continue
                    nm = re.sub(r"\bRoll\s+\d+\s+", "", nm, flags=re.IGNORECASE).strip()
                    nm = re.sub(r"\bRoll\s+", "", nm, flags=re.IGNORECASE).strip()
                    nm = nm.strip(" .")
                    if nm:
                        rolled_items.add(nm)
    lines.append("")

    lines.append("Enemy Forces")
    lines.append("-" * 60)
    lines.append(f"Initial Enemies (minimum, including leader if present): {total_enemies}")
    lines.append(f"Leadership Tier: {leadership_tier}")
    lines.append("")

    if leader_block:
        lines.append("Leader")
        lines.append("." * 60)
        lines.append(f"Base Unit: {leader_block['name']}")
        lines.append(f"Final Stats: {leader_block['stat_final']}")
        if leader_traits:
            lines.append("Traits:")
            for t in leader_traits:
                mod = (t.get("stat_mod") or "").strip()
                extra = f" ({mod})" if mod else ""
                lines.append(f"  - {t['name']}: {t.get('description','')}{extra}")

        # Rolled enemy unit stats (rooms-based: show here; avoids a separate Enemy Types section)
        if enemy_types or (event_enemy_types and event_enemy_counts):
            lines.append("Rolled Units:")
            if enemy_types:
                lines.append("  Initial Spawns")
                for u in enemy_types:
                    nm = u.get("name","Unknown")
                    cnt = enemy_counts.get(nm, 0)
                    lines.append(f"    - {nm} x{cnt}: {u.get('stat_final','')}")
            if event_enemy_types and event_enemy_counts:
                lines.append("  Event Spawns (possible)")
                for u in event_enemy_types:
                    nm = u.get("name","Unknown")
                    cnt = event_enemy_counts.get(nm, 0)
                    lines.append(f"    - {nm} x{cnt}: {u.get('stat_final','')}")
        lines.append("")

    if not rooms_mode:
        lines.append("Enemy Types")
        lines.append("." * 60)
        for u in enemy_types:
            nm = u["name"]
            cnt = enemy_counts.get(nm, 0)
            lines.append(f"- {nm} x{cnt}")
            lines.append(f"  Final Stats: {u['stat_final']}")
        lines.append("")

    if not rooms_mode:
        lines.append("Recommended Battlefield")
        lines.append("-" * 60)
        if size_label == "Normal":
            obstacles, rough = 6, 4
        elif size_label == "Epic":
            obstacles, rough = 8, 6
        else:
            obstacles, rough = 10, 8

        if (scenario_type and (scenario_type.get("ID") or "").strip().lower() == "defensive"):
            lines.append("Setup Notes: You are attacked during travel/camping/in town.")
            lines.append("Suggested Defensible Pieces: barricades, wagons, walls, choke points, high ground.")

        lines.append(f"Terrain Obstacles: {obstacles}")
        lines.append(f"Rough Terrain Patches: {rough}")
        lines.append("")


    # Pressure-trigger events (not in the base deck; add when thresholds are reached)
    if pressure_cards:
        lines.append("Pressure Trigger Events (not in base deck)")
        lines.append("-" * 60)
        # Stable ordering for readability
        stid = (inputs.get("scenario_type_id") or (scenario_type.get("ID") if scenario_type else "") or "").strip().lower()
        if stid == "lair":
            order = ["lair_alert", "beast_stirs"]
        else:
            order = ["alarm", "lockdown", "villain_flees"]

        for key in order:
            e = pressure_cards.get(key) if isinstance(pressure_cards, dict) else None
            if not e:
                continue
            nm = (e.get("name") or "Unknown").strip()
            desc = (e.get("description") or "").strip()
            # Avoid printing the title twice when the description starts with "Name. ..."
            if nm and desc.lower().startswith((nm + ".").lower()):
                desc = desc[len(nm) + 1:].lstrip()
            lines.append(f"- {nm}: {desc}")
        lines.append("")
    lines.append("Events (shuffle as a deck; keep the last card as the Final event)")
    lines.append("-" * 60)
    cards = assign_event_cards(len(event_picks))
    for card, e in zip(cards, event_picks):
        lines.append(f"{card}: {e['name']} — {e.get('description','')}")
    lines.append("")

    die_used = 20 if len(clue_picks) <= 12 else 100
    ranges = assign_clue_ranges(clue_picks, die_sides=die_used)

    if not rooms_mode:
        lines.append(f"Clues (roll d{die_used} when searching)")
        lines.append("-" * 60)
        for i, (a, b, c) in enumerate(ranges, start=1):
            xp = parse_int_maybe(c.get("xp"), 0)
            rng = f"{a}-{b}" if a != b else f"{a}"
            lines.append(f"Dice Roll {rng} (Clue #{i}, XP {xp}): {c['name']} — {c.get('description','')}")
        lines.append("")

    lines.append("Rewards")
    lines.append("-" * 60)
    for u in enemy_types:
        xp_each = parse_int_maybe(u.get("xp"), 0)
        lines.append(f"{xp_each} XP for each {u['name']} defeated")

    if leader_block:
        leader_xp = parse_int_maybe(leader_block.get("xp"), 0)
        lines.append(f"{leader_xp} XP for defeating the {leader_block['name']} (leader)")

    for idx, c in enumerate(clue_picks, start=1):
        cxp = parse_int_maybe(c.get("xp"), 0)
        if cxp > 0:
            lines.append(f"{cxp} XP for finding clue #{idx}")


    if event_enemy_types and event_enemy_counts:
        lines.append("")
        lines.append("Event Spawn XP (only if these units appear and are defeated)")
        lines.append("." * 60)
        for u in event_enemy_types:
            xp_each = parse_int_maybe(u.get("xp"), 0)
            lines.append(f"{xp_each} XP for each {u['name']} defeated (event spawn)")
    lines.append(f"{completion_xp} XP for finishing the scenario")
    lines.append(f"Gold: {total_gold}")
    lines.append("")


    # Glossary (rooms-based convenience): rolled loot items
    if rooms_mode and rolled_items:
        lines.append("Glossary")
        lines.append("-" * 60)
        lines.append("Rolled Items")
        lines.append("." * 60)
        for nm in sorted(rolled_items):
            it = item_index.get(nm)
            if it:
                stat = (it.get("stat") or it.get("statline") or "").strip()
                desc = (it.get("description") or "").strip()
                extra = f" | {stat}" if stat else ""
                if desc:
                    lines.append(f"- {nm}{extra}: {desc}")
                else:
                    lines.append(f"- {nm}{extra}")
            else:
                lines.append(f"- {nm}")
        lines.append("")
    lines.append("Footnotes")
    lines.append("-" * 60)
    if footnotes:
        for f in footnotes:
            lines.append(f"- {f}")
    else:
        lines.append("None.")

    return "\n".join(lines)


# ============================================================
# #GENERATOR — Core scenario generation (IO-FREE)
# ============================================================

def generate_scenario(data: DataBundle, user_inputs: Dict[str, Any]) -> Tuple[str, str]:
    """
    Main core entrypoint.
    Returns (suggested_filename, briefing_text).
    """
    footnotes: List[str] = []
    settings = data.settings

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scenario_{ts}.txt"


    # Data references (no IO)
    enemy_units = data.enemy_units
    traits = data.traits
    events = data.events
    clues = data.clues
    scen_entries = data.scen_entries
    items = data.items
    spells = data.spells

    # Inputs
    players = int(user_inputs.get("players", s_get_int(settings, "default.players", 2)))
    allied = int(user_inputs.get("allied_combatants", s_get_int(settings, "default.allied_combatants", 2)))
    allies_total = players + allied

    difficulty = str(user_inputs.get("difficulty", s_get(settings, "default.difficulty", "Normal"))).strip().title()
    if difficulty not in DIFFICULTY_ORDER:
        difficulty = "Normal"

    # Scenario type selection
    scenario_type_id = (user_inputs.get("scenario_type_id") or "").strip()
    if not scenario_type_id or scenario_type_id.lower() == "random":
        # Robust fallback if CLI passes Random/blank
        scenario_type_id = pick_scenario_type_id(scen_entries, settings)

    selected_scenario_type = find_scenario_type_by_id(scen_entries, scenario_type_id)
    if not selected_scenario_type:
        selected_scenario_type = {"id": scenario_type_id or "unknown", "name": scenario_type_id or "Unknown", "tags": set()}
        footnotes.append(f"WARNING: scenario_type_id={scenario_type_id!r} not found in scenario_objectives.txt; using fallback.")

    st_tags = set((selected_scenario_type.get('tags') or [])) if selected_scenario_type else set()
    st_id_norm = (scenario_type_id or "").strip().lower()
    rooms_based = ('rooms_based' in st_tags) or (st_id_norm in {"site", "lair"})

    # Biome resolution by mode
    mode_name = str(user_inputs.get("mode", "Custom")).strip().lower()
    campaign_key = str(user_inputs.get("campaign_key") or "").strip()
    biome_id = str(user_inputs.get("biome_id") or "").strip()
    biome_entry: Optional[Dict[str, Any]] = None
    if mode_name == "quick":
        biome_id = ""
    elif mode_name in ("custom", "questboard"):
        if campaign_key and not biome_id:
            biome_id = _campaign_biome_id_from_key(campaign_key)
        if biome_id:
            biome_entry = _find_entry_by_id(getattr(data, "biomes", []) or [], biome_id)
            if not biome_entry:
                footnotes.append(f"WARNING: biome_id={biome_id!r} not found in biomes.txt; biome effects omitted.")

    # Threat tags
    threat1 = (user_inputs.get("threat_tag_1") or "").strip()
    threat2 = (user_inputs.get("threat_tag_2") or "").strip()
    threat_tags = {t.strip() for t in (threat1, threat2) if t and t.strip().lower() != "none"}

    threat_tags = sanitize_threat_tags(threat_tags, data.all_threats, settings, footnotes)

    # Lair mode is a monsters-only pool (threat-independent).
    if st_id_norm == "lair" and s_get_bool(settings, "lair.force_monsters", True):
        if threat_tags != {"monsters"}:
            footnotes.append(f"LAIR: overriding threats {sorted(threat_tags)} -> ['monsters'].")
        threat_tags = {"monsters"}

    # Mode-specific enemy pool (lair draws only from global monsters).
    enemy_units_mode = enemy_units
    if st_id_norm == "lair":
        enemy_units_mode = [u for u in enemy_units if "monsters" in (u.get("threats", set()) or set())]

    footnotes.append(f"POOL_CHECK: {pool_summary(enemy_units_mode, threat_tags)}")

    if not threat_tags:
        footnotes.append("ABORT: No threat tags selected.")
        brief = build_briefing(
            encounter_kind="Scenario",
            timestamp=ts,
            inputs=user_inputs,
            biome=biome_entry,
            scenario_type=selected_scenario_type,
            scenario_setup=(selected_scenario_type.get("setup") or selected_scenario_type.get("setup_hint") or "").strip()
                if selected_scenario_type else "",
            objective=None,
            threats=[],
            leadership_tier="None",
            leader_block=None,
            leader_traits=[],
            enemy_types=[],
            enemy_counts={},
            total_enemies=0,
            clue_picks=[],
            event_picks=[],
            total_gold=0,
            completion_xp=COMPLETION_XP,
            size_label="Normal",
            room_plan=[],
            footnotes=footnotes,
        )
        return filename, brief


    # Objective
    objective_choice = str(user_inputs.get("objective", "Random"))
    objective = pick_objective_for_scenario(scen_entries, objective_choice, selected_scenario_type, settings, footnotes)

    # Leadership min requirements
    scenario_min = minlead_from_entry(selected_scenario_type)
    objective_min = minlead_from_entry(objective)
    must_min = must_min_leadership(objective)
    min_required = max_leadership(scenario_min, objective_min, must_min)

    # If scenario is BBEG (or minlead=BBEG), leadership is forced
    if leadership_rank(min_required) >= leadership_rank("BBEG"):
        leadership_tier = "BBEG"
    else:
        leadership_req = str(user_inputs.get("leadership_tier", s_get(settings, "default.leadership_tier", "Random")))
        leadership_tier = choose_leadership_tier(leadership_req, allies_total, min_required, footnotes)

    # Monster lair always uses a Boss-tier leader.
    if st_id_norm == "lair":
        if leadership_tier != "Boss":
            footnotes.append(f"LAIR: leadership forced to Boss (was {leadership_tier}).")
        leadership_tier = "Boss"

    if leadership_rank(leadership_tier) < leadership_rank(min_required):
        footnotes.append(f"MUST: leadership upgraded from {leadership_tier} to {min_required} (minlead).")
        leadership_tier = min_required

    obj_id = (objective.get("ID") if objective else "") or ""
    obj_id = obj_id.strip().lower()

    # Duel: enforce a champion-style fight (traits apply cleanly)
    if obj_id == "duel" and leadership_rank(leadership_tier) < leadership_rank("Boss"):
        footnotes.append(f"DUEL: leadership upgraded from {leadership_tier} to Boss.")
        leadership_tier = "Boss"

    # Enemy types (minions)
    enemy_types = select_enemy_types_v2(enemy_units_mode, threat_tags, footnotes, players=players, max_types=6)

    # Leader selection
    has_leader = leadership_tier != "None"
    leader_block: Optional[Dict[str, Any]] = None
    leader_traits: List[Dict[str, Any]] = []

    if has_leader:
        leader = select_leader_unit(enemy_units_mode, threat_tags, footnotes)
        if not leader:
            footnotes.append("ABORT: Leadership required but no leader-eligible unit exists (needs tier=leader and xp=).")
            enemy_types = []
            has_leader = False

    # If duel: ignore minion types (champion only)
    if obj_id == "duel":
        enemy_types = []

    # Totals + distribution
    total_enemies = roll_enemy_total(allies_total, difficulty, scenario_type_id, leadership_tier)
    if obj_id == "duel":
        total_enemies = 1  # champion only

    enemy_counts = distribute_minions(total_enemies, has_leader, enemy_types)

    enemy_counts = enforce_role_model_caps(
        enemy_types,
        enemy_counts,
        total_enemies=total_enemies,
        has_leader=has_leader,
        players=players,
        footnotes=footnotes,
    )

    # Build leader block (final stats + traits)
    if has_leader:
        leader = select_leader_unit(enemy_units, threat_tags, footnotes)
        if leader:
            leader_stats = parse_statline(leader.get("stat", ""))
            apply_difficulty_mods(leader_stats, leader.get("tags", set()) or set(), difficulty, is_leader=True)
            apply_leadership_mods(leader_stats, leadership_tier)

            leader_traits = roll_leader_traits(traits, leadership_tier, footnotes)
            for tr in leader_traits:
                apply_stat_mods(leader_stats, tr.get("stat_mod"))

            leader_block = dict(leader)
            leader_block["stat_final"] = statline_to_string(leader_stats)
        else:
            footnotes.append("WARNING: Leadership requested but no leader resolved in final pass.")
            leader_block = None
            leader_traits = []

    # Placeholder sources: chosen units in this scenario
    chosen_units_for_placeholders = [u["name"] for u in enemy_types]
    if leader_block:
        chosen_units_for_placeholders.append(leader_block["name"])

    # Expand objective description
    if objective and objective.get("description"):
        objective = dict(objective)
        text = objective.get("description", "")
        objective["description"] = expand_templates(
            text,
            chosen_unit_names=chosen_units_for_placeholders,
            allowed_threats=threat_tags,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            footnotes=footnotes,
        )

    # Finalize printed stats for minion types
    for u in enemy_types:
        s = parse_statline(u.get("stat", ""))
        apply_difficulty_mods(s, u.get("tags", set()) or set(), difficulty, is_leader=False)
        u["stat_final"] = statline_to_string(s)

    # Counts
    clue_count = roll_clue_count(allies_total, difficulty, leadership_tier)
    event_count = roll_event_count(allies_total, difficulty, leadership_tier, scenario_type_id)
    if rooms_based:
        clue_count = 0
        event_count = max(event_count, 20)
    footnotes.append(f"ROLL: clue_count={clue_count}, event_count={event_count}")

    obj_tags = objective_required_tags(objective)

    # Must clue requirements + tag caps
    req_clue_tags = must_clue_requirements(objective, footnotes)
    if req_clue_tags:
        footnotes.append(f"MUST: required_clues={req_clue_tags}")

    clue_tag_caps = {"weather": 2}
    clue_tag_counts: Dict[str, int] = {}

    picked_clues: List[Dict[str, Any]] = []
    picked_clue_names: Set[str] = set()

    if rooms_based:
        # Rooms-based scenarios do not use clues; quest details are embedded in rooms.
        picked_clues = []
        picked_clue_names = set()
    else:

        required_clues = pick_required_clues_by_tag(
            clues=clues,
            required=req_clue_tags,
            chosen_unit_names=chosen_units_for_placeholders,
            allowed_threats=threat_tags,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            picked_names=picked_clue_names,
            footnotes=footnotes,
            tag_counts=clue_tag_counts,
            tag_caps=clue_tag_caps,
        )
        picked_clues.extend(required_clues)

        remaining = max(0, clue_count - len(picked_clues))
        picked_clues.extend(
            pick_weighted_clues(
                clues=clues,
                count=remaining,
                obj_tags=obj_tags,
                chosen_unit_names=chosen_units_for_placeholders,
                allowed_threats=threat_tags,
                enemy_units=enemy_units,
                items=items,
                spells=spells,
                picked_names=picked_clue_names,
                footnotes=footnotes,
                tag_counts=clue_tag_counts,
                tag_caps=clue_tag_caps,
            )
        )

    
# Events (rooms-based decks come from tag=room_event; non-rooms uses legacy objective/generic weighting)
    picked_event_names: Set[str] = set()
    pressure_cards: Dict[str, Dict[str, Any]] = {}

    if rooms_based:
        picked_events, pressure_cards = build_rooms_based_event_deck(
            events=events,
            count=event_count,
            scenario_type_id=scenario_type_id,
            difficulty=difficulty,
            chosen_unit_names=chosen_units_for_placeholders,
            allowed_threats=threat_tags,
            enemy_units=enemy_units_mode,
            items=items,
            spells=spells,
            settings=settings,
            footnotes=footnotes,
        )
        if pressure_cards:
            footnotes.append("PRESSURE_CARDS: " + ", ".join(sorted(pressure_cards.keys())) + ".")
    else:
        picked_events = pick_weighted_events(
            events=events,
            count=event_count,
            obj_tags=obj_tags,
            difficulty=difficulty,
            chosen_unit_names=chosen_units_for_placeholders,
            allowed_threats=threat_tags,
            enemy_units=enemy_units,
            items=items,
            spells=spells,
            picked_names=picked_event_names,
            footnotes=footnotes,
        )

    # Rewards totals -> gold
    total_xp = 0
    name_to_unit = {u["name"]: u for u in enemy_types}

    for nm, cnt in enemy_counts.items():
        unit = name_to_unit.get(nm)
        if unit:
            total_xp += parse_int_maybe(unit.get("xp"), 0) * cnt

    if leader_block:
        total_xp += parse_int_maybe(leader_block.get("xp"), 0)

    for c in picked_clues:
        total_xp += parse_int_maybe(c.get("xp"), 0)

    total_xp += COMPLETION_XP
    total_gold = total_xp * GOLD_PER_XP
    # Alias for template expansion / room plan
    chosen_unit_names = chosen_units_for_placeholders

    size_label = battle_size_label(allies_total, total_enemies, difficulty, leadership_tier)

    scenario_setup = ""
    if selected_scenario_type:
        scenario_setup = (selected_scenario_type.get("setup") or selected_scenario_type.get("setup_hint") or "").strip()

    # Rooms / Delve layout (v1.1) — rooms unique, transitions may repeat
    rooms_entries = getattr(data, "rooms", []) or []
    room_plan: List[Dict[str, Any]] = []
    if rooms_based:
        room_plan = build_room_plan(
            rooms_entries=rooms_entries,
            threat_tags=threat_tags,
            players=players,
            difficulty=difficulty,
            settings=settings,
            chosen_unit_names=chosen_units_for_placeholders,
            enemy_units=enemy_units_mode,
            items=items,
            spells=spells,
            footnotes=footnotes,
            leadership_tier=leadership_tier,
            scenario_type_id=scenario_type_id,
            objective_id=(objective.get("ID") or ""),
        )

        # Structural validation (never silent)
        try:
            _msgs = validate_room_graph(
                room_plan,
                objective_id=(objective.get("ID") or ""),
                scenario_type_id=scenario_type_id,
                leadership_tier=leadership_tier,
                leader_block=leader_block,
                settings=settings,
            )
            for m in _msgs:
                footnotes.append(m)
        except Exception as ex:
            footnotes.append(f"WARNING: validate_room_graph crashed: {ex}")


# Rooms-based: derive enemy roster from actual room spawns (and event spawns).
    event_enemy_types: List[Dict[str, Any]] = []
    event_enemy_counts: Dict[str, int] = {}
    if rooms_based and room_plan:
        # --- Initial spawns from rooms/transitions ---
        raw_initial = _extract_unit_counts_from_room_plan(room_plan)

        # --- Potential spawns from events (base deck + pressure-trigger events) ---
        event_texts: List[str] = []
        for e in (picked_events or []):
            d = e.get("description")
            if isinstance(d, str) and d:
                event_texts.append(d)
        if isinstance(pressure_cards, dict):
            for e in pressure_cards.values():
                d = e.get("description")
                if isinstance(d, str) and d:
                    event_texts.append(d)
        raw_events = _extract_unit_counts_from_texts(event_texts, enemy_units_mode)

        # Canonicalize names against the loaded enemy_units list
        name_to_unit = { (u.get('name') or '').strip(): u for u in (enemy_units_mode or []) if (u.get('name') or '').strip() }
        name_map_lower = { n.lower(): n for n in name_to_unit.keys() }

        def _canon_counts(raw: Dict[str,int]) -> Dict[str,int]:
            out: Dict[str,int] = {}
            for nm, n in (raw or {}).items():
                resolved = _canonicalize_unit_name(nm, name_map_lower) or nm.strip()
                if resolved in name_to_unit:
                    out[resolved] = out.get(resolved, 0) + int(n)
                else:
                    footnotes.append(f"WARNING: spawned unit name could not be resolved: {nm!r}.")
            return out

        derived_counts = _canon_counts(raw_initial)
        event_enemy_counts = _canon_counts(raw_events)

        def _finalize_unit(u: Dict[str, Any], *, is_leader: bool) -> Dict[str, Any]:
            uu = dict(u)
            s = parse_statline(uu.get("stat", ""))
            apply_difficulty_mods(s, uu.get("tags", set()) or set(), difficulty, is_leader=is_leader)
            uu["stat_final"] = statline_to_string(s)
            return uu

        derived_types: List[Dict[str, Any]] = []
        for nm in sorted(derived_counts.keys()):
            u = name_to_unit.get(nm)
            if not u:
                continue
            derived_types.append(_finalize_unit(u, is_leader=False))

        event_enemy_types = []
        for nm in sorted(event_enemy_counts.keys()):
            u = name_to_unit.get(nm)
            if not u:
                continue
            event_enemy_types.append(_finalize_unit(u, is_leader=False))

        enemy_types = derived_types
        enemy_counts = derived_counts

        # Total enemies now reflects only the initial spawns (+ leader if present).
        total_enemies = sum(derived_counts.values()) + (1 if leader_block else 0)

        # Recompute rewards totals/gold based on initial spawns only.
        total_xp = 0
        for u in enemy_types:
            total_xp += parse_int_maybe(u.get("xp"), 0) * enemy_counts.get(u.get("name",""), 0)
        if leader_block:
            total_xp += parse_int_maybe(leader_block.get("xp"), 0)
        total_xp += COMPLETION_XP
        total_gold = total_xp * GOLD_PER_XP

        # Recompute size label using the corrected totals.
        size_label = battle_size_label(allies_total, total_enemies, difficulty, leadership_tier)


    brief = build_briefing(
        timestamp=ts,
        room_plan=room_plan,
        inputs=user_inputs,
        biome=biome_entry,
        scenario_type=selected_scenario_type,
        scenario_setup=scenario_setup,
        objective=objective,
        threats=sorted(threat_tags),
        leadership_tier=leadership_tier,
        leader_block=leader_block,
        leader_traits=leader_traits,
        enemy_types=enemy_types,
        enemy_counts=enemy_counts,
        event_enemy_types=event_enemy_types,
        event_enemy_counts=event_enemy_counts,
        total_enemies=total_enemies,
        clue_picks=picked_clues,
        event_picks=picked_events,
        items=items,
        pressure_cards=pressure_cards,
        total_gold=total_gold,
        completion_xp=COMPLETION_XP,
        size_label=size_label,
        footnotes=footnotes,
    )

    return filename, brief
