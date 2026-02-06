"""Shared logic for Rangers at the Borderlands (IO-free helpers).

This module centralizes cross-script helpers so CLI and core stay in sync
without duplicating logic in multiple files.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

from rbtl_data import s_get_float, s_get_list


# ============================================================
# #CONFIG — Shared enums and rank mappings
# ============================================================

DIFFICULTY_ORDER = ["Easy", "Normal", "Hard", "Brutal"]
DIFF_IDX = {d: i for i, d in enumerate(DIFFICULTY_ORDER)}

LEAD_RANK = {"None": 0, "Lieutenant": 1, "Boss": 2, "BBEG": 3}


# ============================================================
# #LEADERSHIP — Normalization + minimum requirements
# ============================================================

def normalize_leadership_value(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "Random"
    ss = s.lower()
    if ss == "random":
        return "Random"
    if ss == "none":
        return "None"
    if ss in ("lieutenant", "lt"):
        return "Lieutenant"
    if ss == "boss":
        return "Boss"
    if ss == "bbeg":
        return "BBEG"
    return s.title()


def leadership_rank(tier: str) -> int:
    return LEAD_RANK.get(normalize_leadership_value(tier), 0)


def max_leadership(*tiers: str) -> str:
    best = "None"
    best_r = -1
    for t in tiers:
        tt = normalize_leadership_value(t)
        r = leadership_rank(tt)
        if r > best_r:
            best = tt
            best_r = r
    return best


def minlead_from_entry(e: Optional[Dict[str, Any]]) -> str:
    """Read minimum leadership requirement from a data row."""
    if not e:
        return "None"
    raw = (e.get("minlead") or e.get("leadership") or e.get("leadrship") or "").strip()
    if not raw or raw.lower() == "random":
        return "None"
    return normalize_leadership_value(raw)


def _parse_must_specs(obj: Optional[Dict[str, Any]]) -> List[str]:
    if not obj:
        return []
    specs: List[str] = []
    must_raw = str(obj.get("must", "") or "").strip()
    if must_raw:
        for part in must_raw.split(","):
            part = part.strip()
            if part:
                specs.append(part)
    for d in obj.get("directives", []) or []:
        if d.lower().startswith("must-"):
            specs.append(d.strip())
    return specs


def objective_requires_leader(obj: Optional[Dict[str, Any]]) -> bool:
    """Return True if an objective must include a leader target."""
    if not obj:
        return False

    leadership_field = (obj.get("leadership") or "").strip().lower()
    if leadership_field in ("lieutenant", "boss", "bbeg"):
        return True

    for spec in _parse_must_specs(obj):
        if "target:tier=leader" in spec.strip().lower():
            return True

    return False


def must_min_leadership(obj: Optional[Dict[str, Any]]) -> str:
    """must=target:tier=leader => at least Lieutenant."""
    return "Lieutenant" if objective_requires_leader(obj) else "None"


def leadership_options_from_min(min_required: str, allow_random: bool = True) -> List[str]:
    base = ["None", "Lieutenant", "Boss", "BBEG"]
    min_r = leadership_rank(min_required)
    opts = [o for o in base if leadership_rank(o) >= min_r]
    if allow_random:
        opts = ["Random"] + opts
    return opts


# ============================================================
# #THREATS — Eligibility + weighted selection (settings-aware)
# ============================================================

def eligible_enemy_unit(e: Dict[str, Any]) -> bool:
    return "xp" in e and str(e.get("xp", "")).strip() != ""


def gather_threats_from_units(enemy_units: List[Dict[str, Any]]) -> List[str]:
    threats: Set[str] = set()
    for u in enemy_units:
        if eligible_enemy_unit(u):
            threats |= (u.get("threats", set()) or set())
    return sorted(threats)


def threat_pool_from_settings(all_threats: List[str], settings: Dict[str, str]) -> List[Tuple[str, float]]:
    enabled = set(s_get_list(settings, "enabled.threats")) | set(s_get_list(settings, "enable_threat"))
    disabled = set(s_get_list(settings, "disabled.threats")) | set(s_get_list(settings, "disable_threat"))

    pool = [t for t in all_threats if (not enabled or t in enabled)]
    pool = [t for t in pool if t not in disabled]

    weighted: List[Tuple[str, float]] = []
    for t in pool:
        w = s_get_float(settings, f"threat_weight.{t}", 1.0)
        if w <= 0:
            continue
        weighted.append((t, w))
    return weighted


def pick_weighted_tag(weighted_pool: List[Tuple[str, float]], exclude: Optional[Set[str]] = None) -> Optional[str]:
    exclude = exclude or set()
    options = [(t, w) for (t, w) in weighted_pool if t not in exclude]
    if not options:
        return None
    tags = [t for (t, _) in options]
    weights = [w for (_, w) in options]
    return random.choices(tags, weights=weights, k=1)[0]


def roll_threat_pair(settings: Dict[str, str], all_threats: List[str], mode: str) -> Tuple[str, str]:
    weighted_pool = threat_pool_from_settings(all_threats, settings)
    if not weighted_pool:
        return "none", "none"

    t1 = pick_weighted_tag(weighted_pool, exclude=set()) or "none"
    p_second = 0.45 if mode.lower() == "now" else 0.25
    if t1 != "none" and random.random() < p_second:
        t2 = pick_weighted_tag(weighted_pool, exclude={t1}) or "none"
        return t1, t2
    return t1, "none"


# ============================================================
# #SCENARIO TYPES — Weighted selection (settings-aware)
# ============================================================

def _entry_weight(entry: Dict[str, Any], settings: Dict[str, str]) -> float:
    ident = (entry.get("ID") or entry.get("id") or "").strip()
    base = s_get_float(settings, f"scenariotype_weight.{ident}", 1.0)
    raw_w = str(entry.get("weight", "")).strip()
    try:
        row_weight = float(raw_w) if raw_w else 1.0
    except Exception:
        row_weight = 1.0
    return max(0.0001, row_weight * base)


def pick_scenario_type_id(scen_entries: List[Dict[str, Any]], settings: Dict[str, str]) -> str:
    scenario_types = [e for e in scen_entries if "scenariotype" in (e.get("tags", set()) or set())]
    enabled_ids = set(s_get_list(settings, "enabled.scenario_types"))
    if enabled_ids:
        scenario_types = [e for e in scenario_types if (e.get("ID") or "").strip() in enabled_ids]
    if not scenario_types:
        return ""

    weights = [_entry_weight(e, settings) for e in scenario_types]
    picked = random.choices(scenario_types, weights=weights, k=1)[0]
    return (picked.get("ID") or "").strip()
