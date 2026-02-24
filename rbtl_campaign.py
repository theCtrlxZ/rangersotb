# rbtl_campaign.py
from __future__ import annotations

import os
import random
import secrets
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rbtl_core import weighted_choice, entry_weight
from rbtl_data import DataBundle, s_get_float, s_get_list


# ============================================================
# #LABEL: BALANCE KNOBS + HARD CAPS
# What this section does: Defines difficulty tables and placement caps.
# ============================================================

MAX_PLAYERS = 8

DIFFICULTY_THREATS = {
    "easy": 1,
    "normal": 3,
    "hard": 3,
    "brutal": 4,
}

THREAT_LEVELS = {
    "easy": {"main": 5, "secondary": 4},
    "normal": {"main": 6, "secondary": 5},
    "hard": {"main": 7, "secondary": 6},
    "brutal": {"main": 8, "secondary": 7},
}

MAP_BASE_SIDE = {
    "easy": 2,
    "normal": 4,
    "hard": 6,
    "brutal": 8,
}

MAX_THREAT_ATTEMPTS_PER_SLOT = 250
MAX_SETTLEMENT_COUNT_ATTEMPTS = 100
MAX_MAP_PLACEMENT_ATTEMPTS = 500

WATER_BIOME_ID = "013"
COAST_BIOME_ID = "007"
URBAN_BIOME_ID = "011"
ROAD_BIOME_ID = "012"


# ============================================================
# #LABEL: NORMALIZATION HELPERS
# What this section does: Makes DataBundle entries behave like your older parser.
# ============================================================

def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def _csv_set(value: Any) -> Set[str]:
    raw = str(value or "").strip()
    if not raw:
        return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _wrap_paragraphs(text: str, width: int = 92) -> List[str]:
    """Wrap text into display-friendly lines, preserving blank lines."""
    text = (text or "").strip()
    if not text:
        return []
    out: List[str] = []
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            out.append("")
            continue
        out.extend(textwrap.fill(p, width=width).split("\n"))
    return out


def pick_intro_start(intro_starts: List[Dict[str, Any]], main_threat_tags: Set[str]) -> Optional[Dict[str, Any]]:
    """Pick a campaign intro; prefer tag=intro rows, then threat-matched rows within that pool."""
    if not intro_starts:
        return None

    intro_rows = [i for i in intro_starts if "intro" in (i.get("tags") or set())]
    if not intro_rows:
        intro_rows = intro_starts

    tagged = [i for i in intro_rows if (i.get("tags") or set()) and (set(i.get("tags") or set()) & (main_threat_tags or set()))]
    if tagged:
        return random.choice(tagged)

    untagged = [i for i in intro_rows if not (i.get("tags") or set())]
    if untagged:
        return random.choice(untagged)

    return random.choice(intro_rows)


def _intro_id(entry: Optional[Dict[str, Any]]) -> str:
    if not entry:
        return ""
    return str(entry.get("id") or entry.get("ID") or "").strip()


def _tags(entry: Dict[str, Any]) -> Set[str]:
    # DataBundle loader stores tags as a set, but doesn’t lowercase.
    return {_norm(t) for t in (entry.get("tags") or set()) if _norm(t)}


def _not_tokens(entry: Dict[str, Any]) -> Set[str]:
    # Campaign lists use not=... (csv). DataBundle keeps it as kv string.
    return _csv_set(entry.get("not", ""))


def _types(entry: Dict[str, Any]) -> Set[str]:
    # settlement entries use type=village,hamlet,town,city,any
    return _csv_set(entry.get("type", ""))


def _rarity(entry: Dict[str, Any]) -> str:
    return _norm(entry.get("rarity", "common")) or "common"


def _fmt_tags(tags: Set[str]) -> str:
    return ", ".join(sorted(tags)) if tags else ""


def _resolve_seed(inputs: Dict[str, Any]) -> int:
    raw = inputs.get("seed")
    if raw is None:
        raw = os.environ.get("RBTL_SEED")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return secrets.randbits(31)


def _build_campaign_key(truth_packet: Dict[str, Any]) -> str:
    seed = str(truth_packet.get("seed", "")).strip()
    biome_id = str(truth_packet.get("biome_id", "")).strip()
    main_pressure_id = str(truth_packet.get("main_pressure_id", "")).strip()
    sub_pressure_id = str(truth_packet.get("sub_pressure_id", "")).strip()
    intro_id = str(truth_packet.get("intro_id", "")).strip() or "none"
    biome_ids = [str(b).strip() for b in (truth_packet.get("biome_ids") or []) if str(b).strip()]
    biome_blob = ".".join(biome_ids) if biome_ids else "none"
    threats = ".".join(str(t) for t in (truth_packet.get("threat_ids") or []) if str(t).strip()) or "none"
    return f"RBTL-CAMP-{seed}-{biome_id}-{main_pressure_id}-{sub_pressure_id}-{intro_id}-{biome_blob}-{threats}"


def _dominant_region_biome_id(
    biome_grid: Dict[Tuple[int, int], str],
    *,
    excluded_ids: Optional[Set[str]] = None,
) -> str:
    excluded = {WATER_BIOME_ID}
    if excluded_ids:
        excluded |= {str(x).strip() for x in excluded_ids if str(x).strip()}

    counts: Dict[str, int] = {}
    for bid in biome_grid.values():
        norm = str(bid or "").strip()
        if not norm or norm in excluded:
            continue
        counts[norm] = counts.get(norm, 0) + 1

    if not counts:
        return ""

    return max(sorted(counts.keys()), key=lambda bid: counts[bid])


# ============================================================
# #LABEL: CAMPAIGN INTRO (intro_start.txt)
# What this section does: Picks a random opening blurb for the campaign.
#
# Design:
#   - no tag= is treated as generic
#   - generic vs tagged is a 50/50 bucket choice *only when both exist*
#   - no theme/overlap bias (very weak / non-existent leaning)
# ============================================================

def _intro_text(entry: Dict[str, Any]) -> str:
    """Intros can store text in name and/or description. Return a printable block."""
    name = str(entry.get("name") or "").strip()
    desc = str(entry.get("description") or "").strip()
    if name and desc:
        return f"{name}\n{desc}"
    return name or desc


def pick_campaign_intro(intros: List[Dict[str, Any]], *, context_tags: Optional[Set[str]] = None) -> str:
    """Pick an intro from intro_start.txt.

    context_tags is accepted for API stability but intentionally unused for weighting.
    """
    if not intros:
        return ""

    tagged: List[Dict[str, Any]] = []
    generic: List[Dict[str, Any]] = []

    for e in intros:
        text = _intro_text(e)
        if not text:
            continue

        tags = _tags(e)
        # no tag= is generic; tag=generic is also generic
        real_tags = {t for t in tags if t and t != "generic"}
        if real_tags:
            tagged.append(e)
        else:
            generic.append(e)

    # Fallbacks
    if not tagged and not generic:
        return ""
    if not tagged:
        pick = random.choice(generic)
        return _intro_text(pick).strip()
    if not generic:
        pick = random.choice(tagged)
        return _intro_text(pick).strip()

    # Both exist: 50/50 bucket selection (equal chance tagged vs generic)
    bucket = tagged if random.random() < 0.5 else generic
    pick = random.choice(bucket)
    return _intro_text(pick).strip()


# ============================================================
# #LABEL: FOOTNOTES + ABORT
# What this section does: Standardizes how rejects/warnings/aborts are recorded.
# ============================================================

def _log_reject(footnotes: List[str], label: str, entry: Dict[str, Any], reasons: List[str]) -> None:
    reason_text = "; ".join(reasons) if reasons else "rejected"
    footnotes.append(f"[REJECT] {label}: {entry.get('name', entry.get('id'))} → {reason_text}")


def _warn_once(footnotes: List[str], msg: str) -> None:
    if not any(msg in n for n in footnotes):
        footnotes.append(msg)


def _abort(footnotes: List[str], msg: str) -> None:
    footnotes.append(f"[ABORT] {msg}")
    raise SystemExit(msg)


# ============================================================
# #LABEL: CAMPAIGN PRESSURES (ROLE=main/sub/either)
# What this section does: Enforces role rules and picks main + sub pressures.
# ============================================================

VALID_PRESSURE_ROLES = {"main", "sub", "either"}


def validate_pressures_have_roles(pressures: List[Dict[str, Any]], footnotes: List[str]) -> None:
    bad = [p for p in pressures if _norm(p.get("role")) not in VALID_PRESSURE_ROLES]
    if not bad:
        return

    for p in bad[:20]:
        _log_reject(footnotes, "Campaign Pressure (role validation)", p, ["missing/invalid role (main/sub/either)"])
    _abort(footnotes, f"{len(bad)} campaign_pressures entries missing/invalid role=. Fix campaign_pressures.txt.")


def pick_pressure_for_slot(
    pressures: List[Dict[str, Any]],
    slot: str,
    used_ids: Set[str],
    footnotes: List[str],
) -> Dict[str, Any]:
    """Main slot allows main/either. Sub slot allows sub/either."""

    slot = _norm(slot)
    if slot == "main":
        allowed = {"main", "either"}
    elif slot == "sub":
        allowed = {"sub", "either"}
    else:
        _abort(footnotes, f"Internal error: unknown pressure slot '{slot}'.")

    candidates = [p for p in pressures if p.get("id") not in used_ids and _norm(p.get("role")) in allowed]
    if not candidates:
        _abort(footnotes, f"No valid campaign pressures for slot '{slot}'. Check role= assignments.")

    pick = weighted_choice(candidates)
    if not pick:
        _abort(footnotes, f"No valid campaign pressures for slot '{slot}'.")
    used_ids.add(pick["id"])
    return pick


# ============================================================
# #LABEL: THREAT SELECTION
# What this section does: Finite rerolls, role restrictions, not= exclusions, and tag overlap.
# ============================================================

def threat_eligible_by_role(threat: Dict[str, Any], is_main: bool) -> bool:
    role = _norm(threat.get("role"))
    if is_main and role == "secondary_only":
        return False
    if (not is_main) and role == "main_only":
        return False
    return True


def violates_not(threat: Dict[str, Any], forbidden_tokens: Set[str]) -> bool:
    ttags = _tags(threat)
    if ttags.intersection(forbidden_tokens):
        return True
    if _norm(threat.get("id")) in forbidden_tokens:
        return True
    if _norm(threat.get("name")) in forbidden_tokens:
        return True
    return False


def has_required_overlap(threat: Dict[str, Any], context_tags: Set[str]) -> bool:
    return len(_tags(threat).intersection(context_tags)) >= 1


def pick_one_threat_slot(
    threats: List[Dict[str, Any]],
    *,
    label: str,
    is_main_slot: bool,
    used_ids: Set[str],
    context_tags: Set[str],
    forbidden_tokens: Set[str],
    footnotes: List[str],
) -> Dict[str, Any]:
    """Picks a threat for a slot with finite attempts and rejection logging."""

    pool = [t for t in threats if t.get("id") not in used_ids]
    if not pool:
        _abort(footnotes, f"No threats available for {label} (all already used).")

    require_overlap = True
    if not context_tags:
        require_overlap = False
        _warn_once(
            footnotes,
            "[WARN] Threat theme matching skipped: context tags were empty (no tag= on rolled biome/pressures). Using role + not= only.",
        )

    for _ in range(MAX_THREAT_ATTEMPTS_PER_SLOT):
        pick = weighted_choice(pool)
        if not pick:
            break

        reasons: List[str] = []
        if not threat_eligible_by_role(pick, is_main=is_main_slot):
            reasons.append("role restriction")
        if violates_not(pick, forbidden_tokens):
            reasons.append("excluded by not=")
        if require_overlap and not has_required_overlap(pick, context_tags):
            reasons.append("no tag overlap with context")

        if not reasons:
            used_ids.add(pick["id"])
            return pick

        _log_reject(footnotes, label, pick, reasons)
        pool = [t for t in pool if t.get("id") != pick.get("id")]
        if not pool:
            break

    _abort(footnotes, f"No valid candidates for {label} after exclusions/role/tag rules.")


def adjusted_threat_count(players: int, difficulty: str) -> int:
    base = DIFFICULTY_THREATS[difficulty]
    if players < 3:
        return max(1, base - 1)
    return base

def _find_by_id(entries: List[Dict[str, Any]], entry_id: str, *, key: str = "id") -> Optional[Dict[str, Any]]:
    entry_id = str(entry_id or "").strip()
    if not entry_id:
        return None
    for e in entries:
        if str(e.get(key, "")).strip() == entry_id:
            return e
    return None


def roll_campaign_context(
    *,
    biomes: List[Dict[str, Any]],
    pressures: List[Dict[str, Any]],
    footnotes: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Set[str], Set[str]]:
    """Roll biome + pressures and compute context tags + forbidden tokens.
    Used by the main script to compute eligibility during Custom threat picking.
    """
    if not biomes:
        _abort(footnotes, "Missing or empty required file: biomes.txt")
    if not pressures:
        _abort(footnotes, "Missing or empty required file: campaign_pressures.txt")

    validate_pressures_have_roles(pressures, footnotes)

    biome = weighted_choice(biomes) or biomes[0]

    used_pressure_ids: Set[str] = set()
    main_pressure = pick_pressure_for_slot(pressures, "main", used_pressure_ids, footnotes)
    sub_pressure = pick_pressure_for_slot(pressures, "sub", used_pressure_ids, footnotes)

    context_tags: Set[str] = set()
    context_tags |= _tags(biome)
    context_tags |= _tags(main_pressure)
    context_tags |= _tags(sub_pressure)

    forbidden_tokens: Set[str] = set()
    forbidden_tokens |= _not_tokens(biome)
    forbidden_tokens |= _not_tokens(main_pressure)
    forbidden_tokens |= _not_tokens(sub_pressure)
    forbidden_tokens = {_norm(x) for x in forbidden_tokens if _norm(x)}

    return biome, main_pressure, sub_pressure, context_tags, forbidden_tokens


def available_threat_tags_from_pool(threat_pool: List[Dict[str, Any]]) -> List[str]:
    """Union of theme tags available in the provided threat pool (lowercased, sorted)."""
    tags: Set[str] = set()
    for t in threat_pool or []:
        tags |= _tags(t)
    # Remove tokens that are structural rather than thematic (rare, but keep safe).
    tags = {x for x in tags if x and x not in {"main", "secondary", "any"}}
    return sorted(tags)


def eligible_threat_candidates(
    threats: List[Dict[str, Any]],
    *,
    is_main_slot: bool,
    used_ids: Set[str],
    context_tags: Set[str],
    forbidden_tokens: Set[str],
    settings: Optional[Dict[str, str]] = None,
    required_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Filter threats to those eligible for the given slot + current context.

    - Enforces role=main/secondary/any via threat_eligible_by_role().
    - Enforces not= tokens via violates_not().
    - If context_tags is non-empty, requires tag overlap via has_required_overlap().
    - Optional required_tag hard-filter.
    - Applies settings-enabled/disabled threat tag collections (enabled.threats / disabled.threats).
    """
    settings = settings or {}

    enabled = {t.strip().lower() for t in (s_get_list(settings, "enabled.threats") + s_get_list(settings, "enable_threat")) if t.strip()}
    disabled = {t.strip().lower() for t in (s_get_list(settings, "disabled.threats") + s_get_list(settings, "disable_threat")) if t.strip()}

    req = _norm(required_tag)
    if req in {"", "random", "none"}:
        req = ""

    pool: List[Dict[str, Any]] = []
    for t in threats or []:
        tid = str(t.get("id", "")).strip()
        if tid and tid in used_ids:
            continue

        if not threat_eligible_by_role(t, is_main=is_main_slot):
            continue
        if violates_not(t, forbidden_tokens):
            continue
        if context_tags and not has_required_overlap(t, context_tags):
            continue

        t_tags = _tags(t)
        if req and req not in t_tags:
            continue
        if disabled and (t_tags & disabled):
            continue
        if enabled and not (t_tags & enabled):
            continue

        pool.append(t)

    return pool


def threat_weight_with_settings(
    threat: Dict[str, Any],
    settings: Optional[Dict[str, str]] = None,
    context_tags: Optional[Set[str]] = None,
) -> float:
    """Weight for random.choices() when selecting from an eligible threat pool.

    Base: rarity weight (same as rbtl_core.weighted_choice).
    Multipliers:
      - threat row weight=... if present (float)
      - max settings threat_weight.<tag> across threat tags
      - small boost for context overlap (prefers on-theme picks without forcing)
    """
    settings = settings or {}
    context_tags = context_tags or set()

    w = float(max(1, entry_weight(threat)))

    raw_w = str(threat.get("weight", "")).strip()
    if raw_w:
        try:
            w *= float(raw_w)
        except Exception:
            pass

    t_tags = _tags(threat)
    if t_tags:
        tag_mults = [s_get_float(settings, f"threat_weight.{t}", 1.0) for t in t_tags]
        if tag_mults:
            w *= max(tag_mults)

    overlap = t_tags & set(context_tags)
    if overlap:
        w *= (1.0 + 0.25 * len(overlap))

    # Ensure non-zero weight for random.choices
    return max(0.0001, float(w))



def pick_threats(
    threats: List[Dict[str, Any]],
    *,
    players: int,
    difficulty: str,
    context_tags: Set[str],
    forbidden_tokens: Set[str],
    footnotes: List[str],
    override_total: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns (main_threat, secondary_threats).
    NOTE: Output formatting decides how threats map to pressures.
    """

    used: Set[str] = set()
    main_threat = pick_one_threat_slot(
        threats,
        label="Main Threat",
        is_main_slot=True,
        used_ids=used,
        context_tags=context_tags,
        forbidden_tokens=forbidden_tokens,
        footnotes=footnotes,
    )

    count_total = override_total if (override_total and override_total > 0) else adjusted_threat_count(players, difficulty)
    count_total = max(1, count_total)

    secondary_needed = max(0, count_total - 1)
    secondary: List[Dict[str, Any]] = []
    for i in range(secondary_needed):
        secondary.append(
            pick_one_threat_slot(
                threats,
                label=f"Secondary Threat #{i + 1}",
                is_main_slot=False,
                used_ids=used,
                context_tags=context_tags,
                forbidden_tokens=forbidden_tokens,
                footnotes=footnotes,
            )
        )

    return main_threat, secondary


# ============================================================
# #LABEL: SETTLEMENT GENERATION
# What this section does: Picks a Town + 0–2 Villages + 0–2 Hamlets, and assigns aspects.
# ============================================================

def choose_settlement_counts(footnotes: List[str], override_total: Optional[int] = None) -> Dict[str, int]:
    """Returns counts dict with keys Town/Village/Hamlet."""

    # Custom override: total settlements including Town.
    if override_total and override_total > 0:
        total = int(override_total)
        if total < 2 or total > 5:
            _abort(footnotes, "Settlement override must be between 2 and 5 total.")
        remaining = total - 1
        # Greedy split within caps.
        ham = min(2, remaining)
        remaining -= ham
        vill = min(2, remaining)
        remaining -= vill
        if remaining != 0:
            _abort(footnotes, "Settlement override could not be satisfied with hamlet/village caps.")
        return {"Town": 1, "Village": vill, "Hamlet": ham}

    for _ in range(MAX_SETTLEMENT_COUNT_ATTEMPTS):
        ham = random.randint(0, 2)
        vill = random.randint(0, 2)
        town = 1
        total = ham + vill + town
        if 2 <= total <= 5:
            return {"Town": 1, "Village": vill, "Hamlet": ham}

    _abort(footnotes, f"Could not generate settlement counts within constraints after {MAX_SETTLEMENT_COUNT_ATTEMPTS} attempts.")


def _aspect_segments(entry: Dict[str, Any]) -> int:
    raw = entry.get("segments")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _aspect_requires(entry: Dict[str, Any]) -> Set[str]:
    return _csv_set(entry.get("requires", ""))


def _aspect_name(entry: Dict[str, Any]) -> str:
    return str(entry.get("name") or "").strip()


def _is_aspect_entry(entry: Dict[str, Any]) -> bool:
    entry_id = str(entry.get("id") or "").strip().upper()
    if entry_id.startswith("T"):
        return False
    return True


def _settlement_segment_capacity(settlement_type: str) -> int:
    return {
        "hamlet": 3,
        "village": 5,
        "town": 8,
        "city": 12,
    }.get(_norm(settlement_type), 3)


def _settlement_random_aspect_count(settlement_type: str) -> int:
    return {
        "hamlet": 1,
        "village": 2,
        "town": 3,
        "city": 4,
    }.get(_norm(settlement_type), 1)


def _biome_ignores_constraints(biome: Dict[str, Any]) -> bool:
    name = _norm(biome.get("name"))
    return name in {"urban", "road"}


def _biome_allows_aspect(biome: Dict[str, Any], entry: Dict[str, Any]) -> bool:
    if _biome_ignores_constraints(biome):
        return True
    biome_name = _norm(biome.get("name"))
    biome_req = _csv_set(entry.get("biome_req", ""))
    biome_not = _csv_set(entry.get("biome_not", ""))
    if biome_req and biome_name not in biome_req:
        return False
    if biome_not and biome_name in biome_not:
        return False
    return True


def pick_settlement_types(
    settlement_types: List[Dict[str, Any]],
    counts: Dict[str, int],
    biome: Dict[str, Any],
    footnotes: List[str],
) -> List[Dict[str, Any]]:
    """Picks settlement aspects. Enforces size, biome, and segment constraints."""

    results: List[Dict[str, Any]] = []
    if not settlement_types:
        for stype, n in counts.items():
            for _ in range(n):
                capacity = _settlement_segment_capacity(stype)
                results.append({"type": stype, "aspects": [], "open_segments": capacity})
        return results

    aspects_pool = [e for e in settlement_types if _is_aspect_entry(e)]
    if not aspects_pool:
        footnotes.append("[WARN] No settlement aspects found (check settlement_types.txt).")

    for settlement_type, n in counts.items():
        for _ in range(n):
            size_norm = _norm(settlement_type)
            capacity = _settlement_segment_capacity(settlement_type)
            random_count = _settlement_random_aspect_count(settlement_type)
            picked_aspects: List[Dict[str, Any]] = []
            used_ids: Set[str] = set()
            picked_names: Set[str] = set()
            used_segments = 0

            def eligible(entry: Dict[str, Any], remaining: int) -> bool:
                if entry.get("id") in used_ids:
                    return False
                types = _types(entry)
                if types and ("any" not in types) and (size_norm not in types):
                    return False
                if not _biome_allows_aspect(biome, entry):
                    return False
                segments = _aspect_segments(entry)
                if segments >= 3:
                    return False
                if segments > remaining:
                    return False
                if segments == 2 and size_norm not in {"town", "city"}:
                    return False
                requires = _aspect_requires(entry)
                if requires and not requires.issubset(picked_names):
                    return False
                if segments == 2 and requires:
                    return False
                return True

            for _ in range(random_count):
                remaining = capacity - used_segments
                candidates = [e for e in aspects_pool if eligible(e, remaining)]
                if not candidates:
                    footnotes.append(f"[WARN] Not enough eligible aspects for {size_norm} settlement in biome '{_norm(biome.get('name'))}'.")
                    break
                pick = weighted_choice(candidates)
                if not pick:
                    break
                used_ids.add(pick.get("id"))
                name = _aspect_name(pick)
                if name:
                    picked_names.add(_norm(name))
                segments = _aspect_segments(pick)
                used_segments += segments
                picked_aspects.append(
                    {
                        "name": name,
                        "description": str(pick.get("description") or "").strip(),
                        "effects": str(pick.get("effects") or pick.get("effect") or "").strip(),
                        "segments": segments,
                    }
                )

            open_segments = max(0, capacity - used_segments)
            results.append(
                {
                    "type": settlement_type,
                    "aspects": picked_aspects,
                    "open_segments": open_segments,
                }
            )

    return results


# ============================================================
# #LABEL: MAP + SITE PLACEMENT
# What this section does: Battleship-grid coordinates + spacing rules for settlements and sites.
# ============================================================

def excel_col_name(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA, etc."""
    s = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def coord_str(x: int, y: int) -> str:
    return f"{excel_col_name(x + 1)}{y + 1}"


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def all_cells(side: int) -> List[Tuple[int, int]]:
    return [(x, y) for x in range(side) for y in range(side)]


def weighted_center_cells(side: int) -> List[Tuple[int, int]]:
    cx = (side - 1) / 2
    cy = (side - 1) / 2
    out: List[Tuple[int, int]] = []
    for (x, y) in all_cells(side):
        d = abs(x - cx) + abs(y - cy)
        w = int(max(1, round((side * 1.5) - d)))
        out.extend([(x, y)] * w)
    return out


def ring_cells(side: int, center: Tuple[int, int], dmin: int, dmax: int, used: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cx, cy = center
    out: List[Tuple[int, int]] = []
    for x in range(side):
        for y in range(side):
            if (x, y) in used:
                continue
            d = abs(x - cx) + abs(y - cy)
            if dmin <= d <= dmax:
                out.append((x, y))
    return out


def compute_map_side(players: int, difficulty: str, footnotes: List[str]) -> int:
    if players < 1:
        _abort(footnotes, "Players must be >= 1.")
    if players > MAX_PLAYERS:
        _abort(footnotes, f"Max players is {MAX_PLAYERS}.")
    return MAP_BASE_SIDE[difficulty] + players


def pick_spaced_points(
    *,
    candidates: List[Tuple[int, int]],
    used: Set[Tuple[int, int]],
    count: int,
    min_between: int,
    min_from: List[Tuple[Tuple[int, int], int]],
    attempts: int = MAX_MAP_PLACEMENT_ATTEMPTS,
) -> List[Tuple[int, int]]:
    placed: List[Tuple[int, int]] = []
    cand = [c for c in candidates if c not in used]

    for _ in range(attempts):
        if len(placed) >= count or not cand:
            break
        p = random.choice(cand)

        ok = True
        for pt, md in min_from:
            if manhattan(p, pt) < md:
                ok = False
                break
        if ok:
            for q in placed:
                if manhattan(p, q) < min_between:
                    ok = False
                    break

        cand.remove(p)
        if not ok:
            continue

        placed.append(p)
        used.add(p)

    return placed


def generate_sites(
    *,
    side: int,
    town_xy: Tuple[int, int],
    used: Set[Tuple[int, int]],
    settlement_coords_xy: List[Tuple[int, int]],
    threats_ordered: List[Dict[str, Any]],
    footnotes: List[str],
) -> List[Dict[str, Any]]:
    """Creates Delve, Unexplored, Camp per threat (revealed), and Hideout per threat (hidden)."""

    sites: List[Dict[str, Any]] = []
    cells = all_cells(side)

    # Delve: 2–4 away from town (soften if needed)
    delve_pool = ring_cells(side, town_xy, 2, 4, used)
    if not delve_pool:
        delve_pool = [c for c in cells if c not in used and manhattan(c, town_xy) >= 2]
        footnotes.append("[WARN] Delve distance band softened due to tight map.")

    delve_xy = random.choice(delve_pool)
    used.add(delve_xy)
    sites.append({"kind": "delve", "name": "Delve", "xy": delve_xy, "coord": coord_str(*delve_xy), "reveal_coord": True})

    # Unexplored: 3+ away from town (soften if needed)
    unexplored_pool = [c for c in cells if c not in used and manhattan(c, town_xy) >= 3]
    if not unexplored_pool:
        unexplored_pool = [c for c in cells if c not in used and manhattan(c, town_xy) >= 2]
        footnotes.append("[WARN] Unexplored distance softened due to tight map.")

    unexplored_xy = random.choice(unexplored_pool)
    used.add(unexplored_xy)
    sites.append({"kind": "unexplored", "name": "Unexplored Location", "xy": unexplored_xy, "coord": coord_str(*unexplored_xy), "reveal_coord": True})

    # Camps: 1 per threat
    camp_count = len(threats_ordered)
    camp_min_from_town = 3 if side <= 6 else 4
    camp_min_from_settlements = 2
    camp_min_between = 2

    camp_candidates = [c for c in cells if c not in used]
    profiles = [
        ("strict", camp_min_from_town, camp_min_from_settlements, camp_min_between),
        ("soften_between", camp_min_from_town, camp_min_from_settlements, max(1, camp_min_between - 1)),
        ("soften_town", max(2, camp_min_from_town - 1), camp_min_from_settlements, max(1, camp_min_between - 1)),
    ]

    placed_camps: Optional[List[Tuple[int, int]]] = None
    for pname, d_town, d_set, d_between in profiles:
        used_snapshot = set(used)
        camps_snapshot: List[Tuple[int, int]] = []

        min_from = [(town_xy, d_town)] + [(sc, d_set) for sc in settlement_coords_xy]
        candidates = [c for c in camp_candidates if c not in used_snapshot]

        for _ in range(camp_count):
            placed = pick_spaced_points(
                candidates=candidates,
                used=used_snapshot,
                count=1,
                min_between=d_between,
                min_from=min_from + [(cxy, d_between) for cxy in camps_snapshot],
                attempts=MAX_MAP_PLACEMENT_ATTEMPTS,
            )
            if not placed:
                break
            camps_snapshot.append(placed[0])
            candidates = [c for c in candidates if c not in used_snapshot]

        if len(camps_snapshot) == camp_count:
            if pname != "strict":
                footnotes.append(f"[WARN] Camp spacing softened ({pname}).")
            used.clear()
            used.update(used_snapshot)
            placed_camps = camps_snapshot
            break

    if placed_camps is None:
        _abort(footnotes, "Map placement failed: could not place threat camps with spacing constraints.")

    for th, cxy in zip(threats_ordered, placed_camps):
        base = (th.get("name") or "Threat").strip()
        camp_title = str(th.get("camp") or "").strip()
        if camp_title:
            site_name = f"{base} Camp - {camp_title}"
        else:
            site_name = f"{base} Camp"
        sites.append({"kind": "camp", "name": site_name, "xy": cxy, "coord": coord_str(*cxy), "reveal_coord": True})

    # Hideouts: placed but coordinates withheld
    hideout_count = len(threats_ordered)
    hideout_min_from_settlements = 2
    hideout_min_between = 2

    hideout_profiles = [
        ("strict", hideout_min_from_settlements, hideout_min_between),
        ("soften_between", hideout_min_from_settlements, 1),
        ("soften_settlement", 1, 1),
    ]

    placed_hideouts: Optional[List[Tuple[int, int]]] = None
    for pname, d_set, d_between in hideout_profiles:
        used_snapshot = set(used)
        hideouts_snapshot: List[Tuple[int, int]] = []

        for _th in threats_ordered:
            pool = [
                c
                for c in cells
                if c not in used_snapshot
                and all(manhattan(c, sc) >= d_set for sc in settlement_coords_xy)
                and all(manhattan(c, hx) >= d_between for hx in hideouts_snapshot)
            ]
            if not pool:
                break
            pick = random.choice(pool)
            used_snapshot.add(pick)
            hideouts_snapshot.append(pick)

        if len(hideouts_snapshot) == hideout_count:
            if pname != "strict":
                footnotes.append(f"[WARN] Hideout spacing softened ({pname}).")
            used.clear()
            used.update(used_snapshot)
            placed_hideouts = hideouts_snapshot
            break

    if placed_hideouts is None:
        _abort(footnotes, "Map placement failed: could not place threat hideouts.")

    for th, hxy in zip(threats_ordered, placed_hideouts):
        base = (th.get("name") or "Threat").strip()
        sites.append({"kind": "hideout", "name": f"{base} Hideout (Hidden)", "xy": hxy, "coord": None, "reveal_coord": False})

    return sites


def _paint_cluster(
    grid: Dict[Tuple[int, int], str],
    side: int,
    biome_id: str,
    target: int,
    blocked: Set[Tuple[int, int]],
) -> None:
    if target <= 0:
        return
    starts = [c for c in all_cells(side) if c not in blocked and grid.get(c, "") == ""]
    if not starts:
        return
    start = random.choice(starts)
    frontier = [start]
    while frontier and target > 0:
        cell = frontier.pop(random.randrange(len(frontier)))
        if cell in blocked or grid.get(cell, ""):
            continue
        grid[cell] = biome_id
        target -= 1
        x, y = cell
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        random.shuffle(neighbors)
        for nx, ny in neighbors:
            if 0 <= nx < side and 0 <= ny < side and (nx, ny) not in blocked and not grid.get((nx, ny), ""):
                if random.random() < 0.85:
                    frontier.append((nx, ny))


def _generate_biome_grid(
    *,
    side: int,
    biome_entries: List[Dict[str, Any]],
    settlement_coords: List[Tuple[int, int]],
    footnotes: List[str],
) -> Dict[Tuple[int, int], str]:
    biome_ids = {str(b.get("id", "")).strip() for b in biome_entries}
    grid: Dict[Tuple[int, int], str] = {c: "" for c in all_cells(side)}

    # Water body + coast band
    water_target = max(2, int(round(side * side * 0.12)))
    _paint_cluster(grid, side, WATER_BIOME_ID, water_target, blocked=set())
    for x in range(side):
        for y in range(side):
            c = (x, y)
            if grid[c] == WATER_BIOME_ID:
                continue
            n4 = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            if any(0 <= nx < side and 0 <= ny < side and grid[(nx, ny)] == WATER_BIOME_ID for nx, ny in n4):
                if random.random() < 0.65:
                    grid[c] = COAST_BIOME_ID

    blocked = {c for c in all_cells(side) if grid[c] in {WATER_BIOME_ID, COAST_BIOME_ID}}

    # Primary clumps
    primary = ["001", "002", "004", "006", "008"]
    target_each = max(4, int(round((side * side * 0.45) / max(1, len(primary)))))
    for bid in primary:
        if bid in biome_ids:
            _paint_cluster(grid, side, bid, target_each, blocked=blocked)

    # Filler grassland/swamp
    for c in all_cells(side):
        if grid[c]:
            continue
        if "003" in biome_ids:
            grid[c] = "003"
        elif "005" in biome_ids and random.random() < 0.2:
            grid[c] = "005"
        else:
            grid[c] = "001" if "001" in biome_ids else next(iter(biome_ids), "")

    # Rare secondaries
    for bid in ("009", "010"):
        if bid not in biome_ids:
            continue
        for _ in range(max(1, side // 4)):
            cand = random.choice(all_cells(side))
            if cand in settlement_coords:
                continue
            if grid[cand] not in {WATER_BIOME_ID, COAST_BIOME_ID}:
                grid[cand] = bid

    # Urban on settlements only
    for c in settlement_coords:
        grid[c] = URBAN_BIOME_ID

    # Lightweight roads connecting settlements to town center
    if settlement_coords:
        town = settlement_coords[0]
        for sx, sy in settlement_coords[1:]:
            x, y = sx, sy
            while (x, y) != town:
                if random.random() < 0.5 and x != town[0]:
                    x += -1 if x > town[0] else 1
                elif y != town[1]:
                    y += -1 if y > town[1] else 1
                elif x != town[0]:
                    x += -1 if x > town[0] else 1
                if grid[(x, y)] not in {WATER_BIOME_ID, URBAN_BIOME_ID}:
                    grid[(x, y)] = ROAD_BIOME_ID

    if WATER_BIOME_ID not in biome_ids:
        footnotes.append("[WARN] Water biome id=013 missing from biomes.txt; map water generation may be inconsistent.")

    # Policy update: tundra (006) cannot be orthogonally adjacent to desert (002).
    if "006" in biome_ids and "002" in biome_ids:
        tundra_desert_repairs = 0
        for _ in range(8):
            changed = False
            for x in range(side):
                for y in range(side):
                    c = (x, y)
                    here = grid.get(c, "")
                    if here not in {"002", "006"}:
                        continue
                    other = "006" if here == "002" else "002"
                    n4 = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                    if not any(0 <= nx < side and 0 <= ny < side and grid.get((nx, ny), "") == other for nx, ny in n4):
                        continue

                    fallback = "003" if "003" in biome_ids else ("001" if "001" in biome_ids else "008")
                    if fallback == here:
                        fallback = "005" if "005" in biome_ids else fallback
                    if fallback in {"002", "006", "", WATER_BIOME_ID, COAST_BIOME_ID, URBAN_BIOME_ID, ROAD_BIOME_ID}:
                        continue

                    grid[c] = fallback
                    tundra_desert_repairs += 1
                    changed = True
            if not changed:
                break
        if tundra_desert_repairs > 0:
            footnotes.append(f"Biome adjacency rule applied: separated tundra/desert at {tundra_desert_repairs} cell(s).")

    return grid


def _biome_symbol(biome_name: str, used: Set[str]) -> str:
    letters = [ch for ch in (biome_name or "").upper() if "A" <= ch <= "Z"]
    for ch in letters:
        if ch not in used:
            used.add(ch)
            return ch
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if ch not in used:
            used.add(ch)
            return ch
    return "?"


def _render_ascii_biome_map(
    *,
    side: int,
    biome_grid: Dict[Tuple[int, int], str],
    biome_name_by_id: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    used_symbols: Set[str] = set()
    ids_in_use: List[str] = []
    for y in range(side):
        for x in range(side):
            bid = str(biome_grid.get((x, y), "") or "").strip()
            if bid and bid not in ids_in_use:
                ids_in_use.append(bid)

    symbol_by_id: Dict[str, str] = {}
    for bid in ids_in_use:
        name = str(biome_name_by_id.get(bid, bid) or bid)
        symbol_by_id[bid] = _biome_symbol(name, used_symbols)

    header = "   " + " ".join(chr(ord('A') + x) for x in range(side))
    lines = [header]
    for y in range(side):
        row_cells: List[str] = []
        for x in range(side):
            bid = str(biome_grid.get((x, y), "") or "").strip()
            row_cells.append(symbol_by_id.get(bid, "?"))
        lines.append(f"{y+1:>2} " + " ".join(row_cells))

    legend = [f"{symbol_by_id[bid]} = {biome_name_by_id.get(bid, bid)}" for bid in ids_in_use]
    return lines, legend


def generate_map_layout(
    *,
    players: int,
    difficulty: str,
    settlements: List[Dict[str, Any]],
    threats_ordered: List[Dict[str, Any]],
    biome_entries: List[Dict[str, Any]],
    footnotes: List[str],
) -> Dict[str, Any]:
    side = compute_map_side(players, difficulty, footnotes)
    used: Set[Tuple[int, int]] = set()
    cells = all_cells(side)

    # Town coordinate (center-biased)
    town_xy = random.choice(weighted_center_cells(side))
    used.add(town_xy)

    settlement_rows: List[Dict[str, Any]] = []
    settlement_coords_xy: List[Tuple[int, int]] = [town_xy]
    town_assigned = False

    def near_town_pool(max_dist: int) -> List[Tuple[int, int]]:
        return [c for c in cells if c not in used and manhattan(c, town_xy) <= max_dist]

    # Place settlements with dist>=3 between settlements.
    for s in settlements:
        stype = _norm(s.get("type"))
        if stype == "town" and not town_assigned:
            coord = town_xy
            town_assigned = True
        else:
            placed: Optional[Tuple[int, int]] = None
            for dist in (2, 3, 4, 5):
                pool = near_town_pool(dist)
                pool = [c for c in pool if all(manhattan(c, sc) >= 3 for sc in settlement_coords_xy)]
                if pool:
                    placed = random.choice(pool)
                    break

            if placed is None:
                pool = [c for c in cells if c not in used and all(manhattan(c, sc) >= 3 for sc in settlement_coords_xy)]
                if not pool:
                    pool = [c for c in cells if c not in used]
                    footnotes.append("[WARN] Settlement spacing softened due to tight map.")
                placed = random.choice(pool)

            used.add(placed)
            settlement_coords_xy.append(placed)
            coord = placed

        settlement_rows.append(
            {
                "type": s.get("type", ""),
                "aspects": s.get("aspects", []),
                "open_segments": s.get("open_segments", 0),
                "coord": coord_str(coord[0], coord[1]),
            }
        )

    if not town_assigned:
        footnotes.append("[WARN] No Town settlement found; using first settlement as Town.")

    sites = generate_sites(
        side=side,
        town_xy=town_xy,
        used=used,
        settlement_coords_xy=settlement_coords_xy,
        threats_ordered=threats_ordered,
        footnotes=footnotes,
    )

    biome_grid = _generate_biome_grid(
        side=side,
        biome_entries=biome_entries,
        settlement_coords=settlement_coords_xy,
        footnotes=footnotes,
    )
    biome_name_by_id = {str(b.get("id", "")).strip(): b.get("name", "") for b in biome_entries}
    for s in settlement_rows:
        cx = ord(str(s.get("coord", "A1"))[0:1].upper() or "A") - ord("A")
        cy = int(str(s.get("coord", "A1"))[1:] or "1") - 1
        s["biome_id"] = biome_grid.get((cx, cy), "")
        s["biome_name"] = biome_name_by_id.get(s["biome_id"], "")

    for site in sites:
        if not site.get("reveal_coord"):
            continue
        xy = site.get("xy")
        if not isinstance(xy, tuple):
            continue
        bid = biome_grid.get(xy, "")
        site["biome_id"] = bid
        site["biome_name"] = biome_name_by_id.get(bid, "")

    return {"side": side, "settlements_with_coords": settlement_rows, "sites": sites, "biome_grid": biome_grid, "biome_name_by_id": biome_name_by_id}


# ============================================================
# #LABEL: OUTPUT FORMATTING
# What this section does: Produces the final briefing text (no file IO).
# ============================================================

def format_campaign_briefing(
    *,
    players: int,
    difficulty: str,
    biome: Dict[str, Any],
    main_pressure: Dict[str, Any],
    sub_pressure: Dict[str, Any],
    main_threat: Dict[str, Any],
    secondary_threats: List[Dict[str, Any]],
    layout: Dict[str, Any],
    intro_start: Optional[Dict[str, Any]] = None,
    truth_packet: Optional[Dict[str, Any]] = None,
    footnotes: List[str],
) -> str:
    lines: List[str] = []
    intro_title = ""
    intro_desc = ""

    lines.append("CAMPAIGN BRIEFING")
    lines.append("=" * 60)
    lines.append("")
    # Intro (optional)
    if intro_start:
        lines.append("Intro")
        lines.append("-" * 60)
        intro_title = str(intro_start.get("name") or "").strip()
        if intro_title:
            lines.append(intro_title)
        intro_desc = str(intro_start.get("description") or "").strip()
        if intro_desc:
            intro_desc = intro_desc.replace("[threat1]", str(main_threat.get("name", "the threat")))
            lines.extend(_wrap_paragraphs(intro_desc))
        lines.append("")
    if truth_packet:
        key = _build_campaign_key(truth_packet)
        lines.append(f"Campaign Key: {key}")
        lines.append("")
    lines.append(f"Players: {players}")
    lines.append(f"Difficulty: {difficulty.title()}")
    lines.append("")

    # Region
    lines.append("Region")
    lines.append("-" * 60)
    lines.append("Region Name: _______________________________")
    lines.append(f"Biome: {biome.get('name', 'Unknown')}")
    if biome.get("description"):
        lines.append(str(biome["description"]).strip())
    if biome.get("terrain"):
        lines.append(f"Recommended Terrain: {str(biome['terrain']).strip()}")
    if biome.get("rough"):
        lines.append(f"Recommended Rough Terrain: {str(biome['rough']).strip()}")
    lines.append("")

    # Pressures + Threats (combined)
    levels = THREAT_LEVELS[difficulty]
    main_level = levels["main"]
    secondary_level = levels["secondary"]

    ordered_threats = [main_threat] + list(secondary_threats)

    # Ensure: if we have 2+ threats, at least one appears under each pressure.
    # Policy:
    #   - Threat 1 always under Main Pressure
    #   - If there are 2+ threats, Threat 2 is forced under Sub Pressure
    #   - Any remaining threats attach to Main Pressure (so main keeps momentum)
    main_block_idxs: List[int] = [1]
    sub_block_idxs: List[int] = []
    if len(ordered_threats) >= 2:
        sub_block_idxs.append(2)
        if len(ordered_threats) >= 3:
            main_block_idxs.extend(list(range(3, len(ordered_threats) + 1)))

    def lvl(i: int) -> int:
        return main_level if i == 1 else secondary_level

    lines.append("Campaign Pressures and Threats")
    lines.append("-" * 60)

    lines.append(f"Main Pressure — {main_pressure.get('name', 'Unknown')}")
    if main_pressure.get("description"):
        lines.append(str(main_pressure["description"]).strip())
    lines.append("." * 60)
    for i in main_block_idxs:
        t = ordered_threats[i - 1]
        lines.append(f"Threat {i} (Level {lvl(i)}): {t.get('name', 'Unknown')}")
        if t.get("description"):
            lines.append(str(t["description"]).strip())
        lines.append("." * 60)
    lines.append("")

    lines.append(f"Sub Pressure — {sub_pressure.get('name', 'Unknown')}")
    if sub_pressure.get("description"):
        lines.append(str(sub_pressure["description"]).strip())
    lines.append("." * 60)
    if not sub_block_idxs:
        lines.append("(none)")
        lines.append("." * 60)
    else:
        for i in sub_block_idxs:
            t = ordered_threats[i - 1]
            lines.append(f"Threat {i} (Level {lvl(i)}): {t.get('name', 'Unknown')}")
            if t.get("description"):
                lines.append(str(t["description"]).strip())
            lines.append("." * 60)
    lines.append("")

    # Map Overview
    lines.append("Map Overview")
    lines.append("-" * 60)
    side = int(layout.get("side", 0) or 0)
    lines.append(f"Grid Size: {side}x{side}")
    lines.append("")

    lines.append("Settlements")
    for s in layout.get("settlements_with_coords", []):
        lines.append("Settlement Name: __________________________")
        category = _norm(s.get("type", ""))
        if category:
            category = category.lower()
        lines.append(f"Category: {category}  @ {s.get('coord', '')}")
        aspects = s.get("aspects", []) or []
        for aspect in aspects:
            name = str(aspect.get("name") or "").strip()
            desc = str(aspect.get("description") or "").strip()
            effect = str(aspect.get("effects") or aspect.get("effect") or "").strip()
            if name:
                lines.append(name)
            if desc:
                lines.extend(_wrap_paragraphs(desc))
            if effect:
                lines.append(f"Effect: {effect}")
        open_segments = int(s.get("open_segments", 0) or 0)
        for _ in range(max(0, open_segments)):
            lines.append("______")
        lines.append("")

    lines.append("Sites")
    lines.append("-" * 60)
    order = {"delve": 0, "unexplored": 1, "camp": 2, "hideout": 3}
    sites_sorted = sorted(layout.get("sites", []), key=lambda s: order.get(s.get("kind", ""), 99))
    for s in sites_sorted:
        if s.get("reveal_coord"):
            lines.append(f"- {s.get('name', 'Site')} @ {s.get('coord', '')}")
        else:
            lines.append(f"- {s.get('name', 'Site')}: ____________________")
    lines.append("")

    biome_grid = layout.get("biome_grid") or {}
    biome_name_by_id = layout.get("biome_name_by_id") or {}
    if side > 0 and biome_grid and biome_name_by_id:
        map_lines, legend_lines = _render_ascii_biome_map(
            side=side,
            biome_grid=biome_grid,
            biome_name_by_id=biome_name_by_id,
        )
        lines.append("Biome Map")
        lines.append("-" * 60)
        lines.extend(map_lines)
        lines.append("")
        lines.append("Legend")
        lines.append("-" * 60)
        lines.extend([f"- {ln}" for ln in legend_lines])
        lines.append("")

    # Footnotes
    lines.append("Footnotes")
    lines.append("-" * 60)
    if footnotes:
        for n in footnotes:
            lines.append(f"- {n}")
    else:
        lines.append("- (none)")

    return "\n".join(lines) + "\n"


# ============================================================
# #LABEL: GENERATOR ENTRYPOINT
# What this section does: Orchestrates the flow and returns (filename, text).
# ============================================================

def generate_campaign(data: DataBundle, inputs: Dict[str, Any]) -> Tuple[str, str]:
    seed = _resolve_seed(inputs)
    random.seed(seed)

    players = int(inputs.get("players", 1))
    difficulty = _norm(inputs.get("difficulty", "normal"))
    if difficulty not in DIFFICULTY_THREATS:
        difficulty = "normal"

    # Optional Custom-mode overrides
    override_threats = int(inputs.get("override_threats", 0) or 0)
    override_settlements = int(inputs.get("override_settlements", 0) or 0)
    override_threats = override_threats if override_threats > 0 else None
    override_settlements = override_settlements if override_settlements > 0 else None

    footnotes: List[str] = []

    # Required datasets
    biomes = list(getattr(data, "biomes", []) or [])
    pressures = list(getattr(data, "campaign_pressures", []) or [])
    threats = list(getattr(data, "campaign_threats", []) or [])
    settlement_types = list(getattr(data, "settlement_types", []) or [])
    intro_starts = list(getattr(data, "intro_starts", []) or [])

    if not biomes:
        _abort(footnotes, "Missing or empty required file: biomes.txt")
    if not pressures:
        _abort(footnotes, "Missing or empty required file: campaign_pressures.txt")
    if not threats:
        _abort(footnotes, "Missing or empty required file: threats.txt")

    validate_pressures_have_roles(pressures, footnotes)

    # Step 1/2: Biome + Pressures (support optional locked context from Custom mode)
    locked_ctx = inputs.get("locked_context") or {}
    locked_biome_id = str(locked_ctx.get("biome_id", "")).strip()
    locked_main_pressure_id = str(locked_ctx.get("main_pressure_id", "")).strip()
    locked_sub_pressure_id = str(locked_ctx.get("sub_pressure_id", "")).strip()
    locked_intro_id = str(inputs.get("locked_intro_id", "")).strip()

    biome = _find_by_id(biomes, locked_biome_id) if locked_biome_id else None
    if biome:
        footnotes.append(f"LOCKED: biome_id={locked_biome_id}")
    else:
        biome = weighted_choice(biomes) or biomes[0]

    used_pressure_ids: Set[str] = set()
    main_pressure = _find_by_id(pressures, locked_main_pressure_id) if locked_main_pressure_id else None
    if main_pressure:
        used_pressure_ids.add(str(main_pressure.get("id", "")).strip())
        footnotes.append(f"LOCKED: main_pressure_id={locked_main_pressure_id}")
    else:
        main_pressure = pick_pressure_for_slot(pressures, "main", used_pressure_ids, footnotes)

    sub_pressure = _find_by_id(pressures, locked_sub_pressure_id) if locked_sub_pressure_id else None
    if sub_pressure:
        # Avoid duplicates; if locked to same ID as main, reroll properly.
        sid = str(sub_pressure.get("id", "")).strip()
        if sid and sid in used_pressure_ids:
            footnotes.append("LOCKED: sub_pressure_id duplicated main_pressure_id; rerolling sub pressure.")
            sub_pressure = pick_pressure_for_slot(pressures, "sub", used_pressure_ids, footnotes)
        else:
            used_pressure_ids.add(sid)
            footnotes.append(f"LOCKED: sub_pressure_id={locked_sub_pressure_id}")
    else:
        sub_pressure = pick_pressure_for_slot(pressures, "sub", used_pressure_ids, footnotes)

    # Context tags
    context_tags: Set[str] = set()
    context_tags |= _tags(biome)
    context_tags |= _tags(main_pressure)
    context_tags |= _tags(sub_pressure)

    # Forbidden tokens from not=
    forbidden_tokens: Set[str] = set()
    forbidden_tokens |= _not_tokens(biome)
    forbidden_tokens |= _not_tokens(main_pressure)
    forbidden_tokens |= _not_tokens(sub_pressure)
    forbidden_tokens = {_norm(x) for x in forbidden_tokens if _norm(x)}

    # Step 3/4: Threats (support optional locked threat IDs from Custom mode)
    locked_threat_ids = inputs.get("locked_threat_ids") or []
    locked_threat_ids = [str(x).strip() for x in locked_threat_ids if str(x).strip()]

    total_needed = int(override_threats or adjusted_threat_count(players, difficulty))

    if locked_threat_ids:
        picked_threats: List[Dict[str, Any]] = []
        used_ids: Set[str] = set()

        for tid in locked_threat_ids:
            t = _find_by_id(threats, tid)
            if not t:
                footnotes.append(f"LOCKED: threat_id not found: {tid}")
                continue
            t_id = str(t.get("id", "")).strip()
            if t_id in used_ids:
                continue
            used_ids.add(t_id)
            picked_threats.append(t)

        if not picked_threats:
            footnotes.append("LOCKED: No valid locked threats found; rolling threats normally.")
            main_threat, secondary_threats = pick_threats(
                threats,
                players=players,
                difficulty=difficulty,
                context_tags=context_tags,
                forbidden_tokens=forbidden_tokens,
                footnotes=footnotes,
                override_total=override_threats,
            )
        else:
            # If we need more threats than were locked, fill remaining slots with the normal rules.
            while len(picked_threats) < total_needed:
                is_main_slot = (len(picked_threats) == 0)
                label = "Main Threat" if is_main_slot else f"Secondary Threat {len(picked_threats)}"
                extra = pick_one_threat_slot(
                    threats,
                    label=label,
                    is_main_slot=is_main_slot,
                    used_ids=used_ids,
                    context_tags=context_tags,
                    forbidden_tokens=forbidden_tokens,
                    footnotes=footnotes,
                )
                picked_threats.append(extra)

            picked_threats = picked_threats[:total_needed]
            main_threat = picked_threats[0]
            secondary_threats = picked_threats[1:]
            footnotes.append(f"LOCKED: Using {len(picked_threats)} threat(s) from Custom selection.")
    else:
        main_threat, secondary_threats = pick_threats(
            threats,
            players=players,
            difficulty=difficulty,
            context_tags=context_tags,
            forbidden_tokens=forbidden_tokens,
            footnotes=footnotes,
            override_total=override_threats,
        )

    # Step 5: Settlements
    settlement_counts = choose_settlement_counts(footnotes, override_total=override_settlements)
    settlements = pick_settlement_types(settlement_types, settlement_counts, biome, footnotes)

    # Step 6: Map + Sites
    ordered_threats = [main_threat] + list(secondary_threats)
    layout = generate_map_layout(
        players=players,
        difficulty=difficulty,
        settlements=settlements,
        threats_ordered=ordered_threats,
        biome_entries=biomes,
        footnotes=footnotes,
    )

    
    intro_start = None
    if locked_intro_id:
        intro_start = _find_by_id(intro_starts, locked_intro_id)
        if intro_start:
            footnotes.append(f"LOCKED: intro_id={locked_intro_id}")
        else:
            footnotes.append(f"LOCKED: intro_id not found: {locked_intro_id}")
    if not intro_start:
        intro_start = pick_intro_start(intro_starts, set(main_threat.get("tags") or set()))

    region_biome_id = _dominant_region_biome_id(layout.get("biome_grid", {}), excluded_ids={ROAD_BIOME_ID})
    if not region_biome_id:
        region_biome_id = str(biome.get("id", "")).strip()
        footnotes.append("[WARN] Could not determine dominant region biome from map; falling back to rolled biome.")

    truth_packet = {
        "seed": seed,
        "biome_id": region_biome_id,
        "biome_ids": sorted({
            str(bid).strip()
            for bid in (layout.get("biome_grid", {}) or {}).values()
            if str(bid).strip() and str(bid).strip() != WATER_BIOME_ID
        }),
        "main_pressure_id": str(main_pressure.get("id", "")).strip(),
        "sub_pressure_id": str(sub_pressure.get("id", "")).strip(),
        "intro_id": _intro_id(intro_start),
        "threat_ids": [str(main_threat.get("id", "")).strip()] + [str(t.get("id", "")).strip() for t in secondary_threats],
    }
    text = format_campaign_briefing(
        players=players,
        difficulty=difficulty,
        biome=biome,
        main_pressure=main_pressure,
        sub_pressure=sub_pressure,
        main_threat=main_threat,
        secondary_threats=secondary_threats,
        layout=layout,
        intro_start=intro_start,
        truth_packet=truth_packet,
        footnotes=footnotes,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"campaign_{stamp}.txt"
    return filename, text
