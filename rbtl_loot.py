# rbtl_loot.py
# ============================================================
# Rangers at the Borderlands — Loot + Shop Generator (IO-FREE)
#
# Reads from DataBundle:
#   - items (items.txt)
#   - item_components (item_components.txt)
#   - spells (spells.txt)
#
# Key rules:
# - Random loot excludes shop_core/shop_extra items by default.
# - Shop generation:
#     hamlet: core + 1 random
#     village: core + 5 random
#     town: core + extra + 10 random
#   Random shop stock excludes items tagged shop_no.
# - build=... controls which component groups can roll.
#   Condition + Material are always required for gear (weapon/armor).
# - roll:* directives (canonical) are supported:
#     |roll:spells:tag=weapon_spell:count=1|
#   and component fields can also contain roll directives:
#     skill_mod=roll:spells:tag=weapon_spell:count=1
# - Numeric add fields support ranges / dice (e.g. charges_add=1-3, price_add=2d4+1)
# ============================================================

from __future__ import annotations

import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rbtl_data import DataBundle, parse_int_maybe, parse_tags

# ============================================================
# CONFIG
# ============================================================

RARITY_WEIGHTS = {"common": 60, "uncommon": 25, "rare": 10, "legendary": 5}
RARITY_ORDER = ["common", "uncommon", "rare", "legendary"]
RARITY_IDX = {r: i for i, r in enumerate(RARITY_ORDER)}

MECH_TIER_BY_RARITY = {"common": 1, "uncommon": 2, "rare": 3, "legendary": 4}

DEFAULT_PRICE_BY_RARITY = {"common": 40, "uncommon": 100, "rare": 250, "legendary": 600}
DEFAULT_DUR_BY_RARITY = {"common": 10, "uncommon": 12, "rare": 14, "legendary": 16}

WEAPON_TAGS = {"hand", "twohand", "ranged"}
ARMOR_TAG = "armor"
POTION_TAG = "potion"
HERB_TAG = "herb"

SHOP_CORE_TAG = "shop_core"
SHOP_EXTRA_TAG = "shop_extra"
SHOP_NO_TAG = "shop_no"

# Component slots
DEFAULT_SLOT_POS = {
    "condition": "prefix",
    "material": "prefix",
    "potency": "prefix",
    "infusion": "prefix",
    "bane": "prefix",
    "element": "suffix",
    "enchant": "suffix",
    "ward": "suffix",
    "weakness": "suffix",
    "quirk": "postfix",
}

DEFAULT_SLOT_CHANCES = {
    # by rarity
    "condition": {"common": 0.60, "uncommon": 0.80, "rare": 1.00, "legendary": 1.00},
    "material":  {"common": 0.50, "uncommon": 0.70, "rare": 1.00, "legendary": 1.00},
    "infusion":  {"common": 0.00, "uncommon": 0.25, "rare": 0.45, "legendary": 0.70},
    "element":   {"common": 0.00, "uncommon": 0.35, "rare": 0.60, "legendary": 0.80},
    "enchant":   {"common": 0.00, "uncommon": 0.25, "rare": 0.50, "legendary": 0.75},
    "quirk":     {"common": 0.00, "uncommon": 0.10, "rare": 0.25, "legendary": 0.50},
    "potency":   {"common": 1.00, "uncommon": 1.00, "rare": 1.00, "legendary": 1.00},
    "bane":      {"common": 0.00, "uncommon": 0.20, "rare": 0.40, "legendary": 0.60},
    "ward":      {"common": 0.00, "uncommon": 0.35, "rare": 0.55, "legendary": 0.75},
    "weakness":  {"common": 0.00, "uncommon": 0.35, "rare": 0.55, "legendary": 0.75},
}

# ============================================================
# UTIL
# ============================================================

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _tags(e: Dict[str, Any]) -> Set[str]:
    return set(e.get("tags", set()) or set())

def _rarity(e: Dict[str, Any]) -> str:
    r = _norm(str(e.get("rarity", "common")))
    return r if r in RARITY_IDX else "common"

def rarity_idx(r: str) -> int:
    return RARITY_IDX.get(_norm(r), 0)

def eligible_by_rarity(e: Dict[str, Any], max_rarity: str) -> bool:
    return rarity_idx(_rarity(e)) <= rarity_idx(max_rarity)

def is_weapon(it: Dict[str, Any]) -> bool:
    c = _norm(str(it.get("cat", "")))
    if c == "weapon":
        return True
    return bool(_tags(it).intersection(WEAPON_TAGS))

def is_armor(it: Dict[str, Any]) -> bool:
    c = _norm(str(it.get("cat", "")))
    if c == "armor":
        return True
    return ARMOR_TAG in _tags(it)

def is_potion(it: Dict[str, Any]) -> bool:
    c = _norm(str(it.get("cat", "")))
    if c == "potion":
        return True
    return POTION_TAG in _tags(it)

def is_herb(it: Dict[str, Any]) -> bool:
    c = _norm(str(it.get("cat", "")))
    if c == "herb":
        return True
    return HERB_TAG in _tags(it)

def item_category(it: Dict[str, Any]) -> str:
    c = _norm(str(it.get("cat", "")))
    if c in ("weapon", "armor", "potion", "herb", "trinket", "tool", "consumable"):
        return c
    if is_weapon(it):
        return "weapon"
    if is_armor(it):
        return "armor"
    if is_potion(it):
        return "potion"
    if is_herb(it):
        return "herb"
    return "item"

def roll_rarity(requested: str = "Random") -> str:
    req = _norm(requested)
    if req and req != "random":
        return req if req in RARITY_IDX else "common"
    return random.choices(RARITY_ORDER, weights=[RARITY_WEIGHTS[r] for r in RARITY_ORDER], k=1)[0]

def roll_rarity_capped(max_rarity: str) -> str:
    """Random rarity, but never above max_rarity."""
    mr = _norm(max_rarity)
    if mr not in RARITY_IDX:
        mr = "legendary"
    idx_max = RARITY_IDX[mr]
    sub = RARITY_ORDER[: idx_max + 1]
    return random.choices(sub, weights=[RARITY_WEIGHTS[r] for r in sub], k=1)[0]

def entry_weight(e: Dict[str, Any]) -> float:
    r_w = float(RARITY_WEIGHTS.get(_rarity(e), 1))
    w = 1.0
    raw = str(e.get("weight", "")).strip()
    if raw:
        try:
            w = float(raw)
        except Exception:
            w = 1.0
    if w <= 0:
        w = 1.0
    return max(0.0001, r_w * w)

def weighted_choice(pool: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pool:
        return None
    weights = [entry_weight(e) for e in pool]
    return random.choices(pool, weights=weights, k=1)[0]

def jitter_factor(pct: float = 0.25, mode: float = 1.0) -> float:
    try:
        pct = float(pct)
    except Exception:
        pct = 0.25
    pct = max(0.0, min(0.95, pct))
    low = 1.0 - pct
    high = 1.0 + pct
    m = max(low, min(high, float(mode)))
    return float(random.triangular(low, high, m))

def split_csv_words(s: Any) -> Set[str]:
    raw = str(s or "").strip()
    if not raw:
        return set()
    return {w.strip().lower() for w in raw.split(",") if w.strip()}

def parse_csv_field(e: Dict[str, Any], key: str) -> List[str]:
    raw = str(e.get(key, "") or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

def apply_num_mods(base: Optional[float], *, muls: List[float], adds: List[float]) -> Optional[float]:
    if base is None:
        return None
    v = float(base)
    for m in muls:
        v *= float(m)
    for a in adds:
        v += float(a)
    return v

# ============================================================
# NUMPARSE — ranges + dice
# ============================================================

_DICE_RE = re.compile(r"^(?P<n>\d+)d(?P<s>\d+)(?P<mod>[+-]\d+)?$", re.IGNORECASE)
_RANGE_RE = re.compile(r"^(?P<a>-?\d+)\s*-\s*(?P<b>-?\d+)$")

def parse_number_expr(x: Any) -> Optional[float]:
    """Parses numbers plus lightweight RNG expressions.

    Supported:
      - 10, -3, 2.5
      - 1-3 (inclusive integer range)
      - 2d4, 1d6+2, 3d8-1
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None

    m = _RANGE_RE.match(s)
    if m:
        a = int(m.group("a"))
        b = int(m.group("b"))
        lo, hi = (a, b) if a <= b else (b, a)
        return float(random.randint(lo, hi))

    m = _DICE_RE.match(s.replace(" ", ""))
    if m:
        n = max(1, int(m.group("n")))
        sides = max(1, int(m.group("s")))
        mod = int(m.group("mod") or 0)
        total = sum(random.randint(1, sides) for _ in range(n)) + mod
        return float(total)

    try:
        return float(s)
    except Exception:
        return None

# ============================================================
# MODS — stat_mod / skill_mod
# ============================================================

def _split_mod_terms(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]

def _parse_numeric_term(term: str) -> Optional[Tuple[str, int]]:
    """
    Accepts: Armor+1, Armor-2, Attack +1, "HP+3"
    Returns: ("Armor", 1)
    """
    t = (term or "").strip()
    if not t:
        return None
    t = t.replace(" ", "")
    m = re.match(r"^([A-Za-z_]+)([+-]\d+)$", t)
    if not m:
        return None
    key = m.group(1)
    try:
        val = int(m.group(2))
    except Exception:
        return None
    return key, val

def parse_mods_numeric(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for term in _split_mod_terms(s):
        pv = _parse_numeric_term(term)
        if not pv:
            continue
        k, v = pv
        out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}

def parse_traits_non_numeric(s: str) -> List[str]:
    """Pulls any non-numeric terms (e.g. 'Necrobane', 'Ice Vulnerability') out of a mod field."""
    out: List[str] = []
    for term in _split_mod_terms(s):
        if _parse_numeric_term(term):
            continue
        t = term.strip()
        if not t:
            continue
        if t.lower().startswith("roll:"):
            continue
        out.append(t)
    return out

def mods_to_str(mods: Dict[str, int]) -> str:
    if not mods:
        return ""
    bits = []
    for k in sorted(mods.keys()):
        v = mods[k]
        sign = "+" if v >= 0 else ""
        bits.append(f"{k}{sign}{v}")
    return ", ".join(bits)

def merge_mods(*mods: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in mods:
        for k, v in (m or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return {k: v for k, v in out.items() if v != 0}

# ============================================================
# TOKEN CONFLICTS — token/not system (optional)
# ============================================================

def entry_tokens(e: Dict[str, Any]) -> Set[str]:
    raw = str(e.get("token", "") or "").strip()
    if raw:
        return {t.strip().lower() for t in raw.split(",") if t.strip()}
    nm = _norm(str(e.get("name", "")))
    return {nm} if nm else set()

def entry_not_tokens(e: Dict[str, Any]) -> Set[str]:
    raw = str(e.get("not", "") or "").strip()
    if not raw:
        return set()
    return {t.strip().lower() for t in raw.split(",") if t.strip()}

def violates_tokens(candidate: Dict[str, Any], chosen_tokens: Set[str], forbidden_tokens: Set[str]) -> bool:
    c_tokens = entry_tokens(candidate)
    c_not = entry_not_tokens(candidate)
    if c_tokens.intersection(forbidden_tokens):
        return True
    if c_not.intersection(chosen_tokens):
        return True
    return False

# ============================================================
# COMPONENTS — selecting + rendering
# ============================================================

def slot_chance(base: Dict[str, Any], slot: str) -> Optional[float]:
    """Reads chance.<slot>=0.5 or chance_<slot>=0.5 from the base item row."""
    for k in (f"chance.{slot}", f"chance_{slot}"):
        raw = str(base.get(k, "") or "").strip()
        if raw:
            try:
                v = float(raw)
                return max(0.0, min(1.0, v))
            except Exception:
                return None
    return None

def component_pool(
    components: List[Dict[str, Any]],
    *,
    group: str,
    base_tags: Set[str],
    max_rarity: str,
) -> List[Dict[str, Any]]:
    g = _norm(group)
    out: List[Dict[str, Any]] = []
    for c in components:
        if _norm(str(c.get("group", ""))) != g:
            continue
        if not eligible_by_rarity(c, max_rarity):
            continue

        ctags = _tags(c)
        if ctags and not ctags.intersection(base_tags):
            continue
        out.append(c)
    return out

def should_roll_slot(base: Dict[str, Any], slot: str, rarity: str, required_slots: Set[str]) -> bool:
    s = _norm(slot)
    if s in required_slots:
        return True

    ch = slot_chance(base, s)
    if ch is not None:
        return random.random() < ch

    rr = _norm(rarity)
    by_r = DEFAULT_SLOT_CHANCES.get(s, {})
    if rr in by_r:
        return random.random() < float(by_r[rr])
    return False

def component_mech_tier(c: Dict[str, Any]) -> int:
    """Component mech_tier, default 0 if missing/invalid."""
    raw = str(c.get("mech_tier", "") or "").strip()
    if not raw:
        return 0
    try:
        return int(float(raw))
    except Exception:
        return 0

def merchant_bias_multiplier(*, slot: str, comp_tier: int, merchant_level: int, base_item: Dict[str, Any]) -> float:
    """Bias component weights upward toward the merchant's level (Pattern A).

    - Higher tier components (closer to merchant_level) are more likely.
    - Lower tier components remain possible.
    - Potency for potions/herbs gets a 'floor' boost so minor/basic options still appear.
    """
    ml = max(0, int(merchant_level))
    if ml <= 0:
        return 1.0

    s = _norm(slot)

    # Per-slot bias strength (smaller base => stronger upward pull)
    bias_base_by_slot = {
        "material": 0.70,
        "condition": 0.85,
        "potency": 0.75,
        "element": 0.80,
        "infusion": 0.80,
        "enchant": 0.80,
        "quirk": 0.85,
        "bane": 0.75,
        "ward": 0.80,
        "weakness": 0.80,
    }
    bias_base = bias_base_by_slot.get(s, 0.80)

    # Consumable exception: keep low-tier potencies available and not overly suppressed.
    if s == "potency" and (is_potion(base_item) or is_herb(base_item)):
        bias_base = 0.90  # milder upward pull for potency
        delta = max(0, ml - int(comp_tier))
        mult = bias_base ** delta
        if int(comp_tier) <= 1:
            mult *= 2.0  # "floor" boost for minor/basic potencies
        return float(mult)

    delta = max(0, ml - int(comp_tier))
    return float(bias_base ** delta)

def pick_component(
    pool: List[Dict[str, Any]],
    *,
    slot: str,
    base_item: Dict[str, Any],
    chosen_tokens: Set[str],
    forbidden_tokens: Set[str],
    used_ids: Set[str],
    merchant_level: Optional[int] = None,
    force_best: bool = False,
) -> Optional[Dict[str, Any]]:
    """Pick one component from a prepared pool.

    Pattern A (shop random additions):
      - if merchant_level is set: filter to mech_tier <= merchant_level (missing tier => 0)
      - then bias weights to favor higher tiers near merchant level
      - token/not conflicts + duplicate ID avoidance still apply
    """
    if not pool:
        return None

    ml = None
    if merchant_level is not None:
        try:
            ml = int(merchant_level)
        except Exception:
            ml = None
        if ml is not None:
            ml = max(0, ml)

    def allowed_by_merchant(c: Dict[str, Any]) -> bool:
        if ml is None:
            return True
        return component_mech_tier(c) <= ml

    candidates: List[Dict[str, Any]] = []
    for c in pool:
        if not allowed_by_merchant(c):
            continue
        cid = str(c.get("id", "")).strip()
        if cid and cid in used_ids:
            continue
        if violates_tokens(c, chosen_tokens, forbidden_tokens):
            continue
        candidates.append(c)

    if not candidates:
        for c in pool:
            if not allowed_by_merchant(c):
                continue
            if violates_tokens(c, chosen_tokens, forbidden_tokens):
                continue
            candidates.append(c)

    if not candidates:
        return None

    # If requested, pick the *best* tier available (used for consistent fixed stock).
    if force_best:
        try:
            best_tier = max(component_mech_tier(c) for c in candidates)
        except Exception:
            best_tier = None
        if best_tier is not None:
            best = [c for c in candidates if component_mech_tier(c) == best_tier]
            if best:
                wts = [max(0.0001, float(entry_weight(c))) for c in best]
                return random.choices(best, weights=wts, k=1)[0]

    # Weighted pick with optional merchant bias
    weights: List[float] = []
    for c in candidates:
        w = entry_weight(c)
        if ml is not None:
            w *= merchant_bias_multiplier(slot=_norm(slot), comp_tier=component_mech_tier(c), merchant_level=ml, base_item=base_item)
        weights.append(max(0.0001, float(w)))

    return random.choices(candidates, weights=weights, k=1)[0]

def component_text(c: Dict[str, Any], slot: str) -> str:
    silent = str(c.get("silent", "") or c.get("hide", "") or "").strip().lower()
    if silent in ("1", "true", "yes", "y", "on"):
        return ""

    s = _norm(slot)
    pos = _norm(str(c.get("pos") or c.get("position") or DEFAULT_SLOT_POS.get(s, "prefix")))
    if pos == "prefix":
        return str(c.get("prefix") or c.get("text") or c.get("name") or "").strip()
    if pos == "suffix":
        return str(c.get("suffix") or c.get("text") or c.get("name") or "").strip()
    if pos == "postfix":
        return str(c.get("postfix") or c.get("text") or c.get("name") or "").strip()
    return str(c.get("text") or c.get("name") or "").strip()

def build_item_name(base: Dict[str, Any], picks_by_slot: Dict[str, Dict[str, Any]]) -> str:
    base_name = str(base.get("name", "Unknown Item")).strip()

    prefix_slots = ["condition", "material", "infusion", "potency", "bane"]
    suffix_slots = ["element", "enchant", "ward", "weakness"]
    postfix_slots = ["quirk"]

    prefixes: List[str] = []
    suffixes: List[str] = []
    postfixes: List[str] = []

    for s in prefix_slots:
        if s in picks_by_slot:
            t = component_text(picks_by_slot[s], s)
            if t:
                prefixes.append(t)

    for s in suffix_slots:
        if s in picks_by_slot:
            t = component_text(picks_by_slot[s], s)
            if t:
                suffixes.append(t)

    for s in postfix_slots:
        if s in picks_by_slot:
            t = component_text(picks_by_slot[s], s)
            if t:
                if not (t.startswith("(") and t.endswith(")")):
                    t = f"({t})"
                postfixes.append(t)

    # Build name with punctuation-aware suffix joining.
    name_parts: List[str] = []
    if prefixes:
        name_parts.append(" ".join(prefixes))
    name_parts.append(base_name)

    name = " ".join(name_parts).strip()
    for suf in suffixes:
        suf = suf.strip()
        if not suf:
            continue
        # If suffix begins with punctuation, attach directly (e.g., ", Weak to Fire")
        if suf[0] in ",.;:)]}":
            name = f"{name}{suf}"
        else:
            name = f"{name} {suf}"

    if postfixes:
        name = f"{name} {' '.join(postfixes)}"
    return name

# ============================================================
# ROLL DIRECTIVES (spells/skills/abilities are in spells.txt)
# ============================================================

def _roll_parse_one(directive: str) -> Optional[Dict[str, Any]]:
    d = (directive or "").strip()
    if not d:
        return None
    # "spells:tag=weapon_spell:count=1"
    parts = [p.strip() for p in d.split(":") if p.strip()]
    if not parts:
        return None
    kind = _norm(parts[0])
    opts: Dict[str, str] = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[_norm(k)] = v.strip()
    count = 1
    try:
        if "count" in opts:
            v = parse_number_expr(str(opts["count"]).strip())
            if v is None:
                count = max(1, int(str(opts["count"]).strip()))
            else:
                count = max(1, int(round(float(v))))
    except Exception:
        count = 1
    tags = set()
    if "tag" in opts:
        tags = {t.strip() for t in str(opts["tag"]).split(",") if t.strip()}
    return {"kind": kind, "count": count, "tags": tags}

def parse_roll_directives(roll_str: str) -> List[Dict[str, Any]]:
    raw = (roll_str or "").strip()
    if not raw:
        return []
    directives = [d.strip() for d in raw.split(";") if d.strip()]
    out: List[Dict[str, Any]] = []
    for d in directives:
        one = _roll_parse_one(d)
        if one:
            out.append(one)
    return out

def pick_from_dataset(dataset: List[Dict[str, Any]], *, tags: Set[str], count: int, max_rarity: str = "legendary") -> List[Dict[str, Any]]:
    if not dataset or count <= 0:
        return []
    pool = []
    for e in dataset:
        if not eligible_by_rarity(e, max_rarity):
            continue
        if tags:
            et = _tags(e)
            if not tags.issubset(et):
                continue
        pool.append(e)
    if not pool:
        return []

    picks: List[Dict[str, Any]] = []
    used_ids: Set[str] = set()
    guard = 0
    while len(picks) < count and guard < 100:
        guard += 1
        e = weighted_choice(pool)
        if not e:
            break
        eid = str(e.get("id", "")).strip()
        if eid and eid in used_ids:
            continue
        used_ids.add(eid)
        picks.append(e)
    return picks

def resolve_rolls(
    data: DataBundle,
    *,
    roll_str: str,
    max_rarity: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for d in parse_roll_directives(roll_str):
        kind = d["kind"]
        tags = d["tags"]
        count = int(d["count"])
        if kind == "spells":
            out.extend(pick_from_dataset(getattr(data, "spells", []) or [], tags=tags, count=count, max_rarity=max_rarity))
        # future: items, traits, etc.
    return out

# ============================================================
# BUILD-UP ROLL PER BASE ITEM
# ============================================================

def slots_for_base(base: Dict[str, Any], *, auto_build: bool = True) -> List[str]:
    build = parse_csv_field(base, "build") or parse_csv_field(base, "slots")
    if build:
        return [_norm(s) for s in build if _norm(s)]

    if not auto_build:
        return []

    if is_potion(base):
        return ["potency"]
    if is_weapon(base) or is_armor(base):
        return ["condition", "material", "infusion", "element", "enchant", "quirk"]
    return []

def required_slots_for_base(base: Dict[str, Any]) -> Set[str]:
    req = {_norm(s) for s in (parse_csv_field(base, "required_slots") or parse_csv_field(base, "required") or []) if _norm(s)}
    if is_potion(base):
        req.add("potency")
    if is_weapon(base) or is_armor(base):
        req.add("condition")
        req.add("material")
    return req

def build_up_item(
    data: DataBundle,
    base: Dict[str, Any],
    *,
    components: List[Dict[str, Any]],
    target_rarity: str,
    auto_build: bool = True,
    merchant_level: Optional[int] = None,
    consistent_material: bool = False,
) -> Dict[str, Any]:
    """
    Returns a dict with:
      - final_name, picks_by_slot, rarity, mech_tier
      - stat_mods, skill_mods (numeric)
      - traits (non-numeric strings)
      - rolls (picked entries, typically spells/skills/abilities)
      - derived: price, dur, uses, charges_max, damagetype
      - weak/strong sets (optional passthrough)
    """
    slots = slots_for_base(base, auto_build=auto_build)
    required_slots = required_slots_for_base(base)

    base_tags = _tags(base)

    chosen_tokens: Set[str] = set()
    forbidden_tokens: Set[str] = set()
    used_comp_ids: Set[str] = set()

    picks_by_slot: Dict[str, Dict[str, Any]] = {}

    # component selection
    for slot in slots:
        if not should_roll_slot(base, slot, target_rarity, required_slots):
            continue
        pool = component_pool(components, group=slot, base_tags=base_tags, max_rarity=target_rarity)
        pick = pick_component(pool, slot=slot, base_item=base, chosen_tokens=chosen_tokens, forbidden_tokens=forbidden_tokens, used_ids=used_comp_ids, merchant_level=merchant_level, force_best=(consistent_material and _norm(slot)=="material"))
        if not pick:
            continue
        picks_by_slot[slot] = pick

        used_comp_ids.add(str(pick.get("id", "")).strip())
        chosen_tokens |= entry_tokens(pick)
        forbidden_tokens |= entry_not_tokens(pick)

    final_name = build_item_name(base, picks_by_slot) if picks_by_slot else str(base.get("name", "Unknown Item")).strip()

    # mech tier
    mech_tier = int(base.get("mech_tier", "") or 0) or MECH_TIER_BY_RARITY.get(_rarity(base), 1)
    for c in picks_by_slot.values():
        try:
            mt = int(str(c.get("mech_tier", "") or "").strip() or 0)
        except Exception:
            mt = 0
        mech_tier = max(mech_tier, mt or MECH_TIER_BY_RARITY.get(_rarity(c), 1))

    # mods + traits
    stat_mods = parse_mods_numeric(str(base.get("stat_mod", "") or ""))
    skill_mods = parse_mods_numeric(str(base.get("skill_mod", "") or ""))
    traits: List[str] = []
    traits.extend(parse_traits_non_numeric(str(base.get("skill_mod", "") or "")))
    traits.extend(parse_traits_non_numeric(str(base.get("stat_mod", "") or "")))  # allow if you ever do

    rolled_entries: List[Dict[str, Any]] = []
    # base roll directives (canonical pipe chunks)
    base_roll = str(base.get("roll", "") or "").strip()
    if base_roll:
        rolled_entries.extend(resolve_rolls(data, roll_str=base_roll, max_rarity=target_rarity))

    # also support bare pipe directives like |roll:spells:tag=...:count=1|
    for k in [k for k in base.keys() if str(k).strip().lower().startswith("roll:")]:
        rolled_entries.extend(resolve_rolls(data, roll_str=str(k).strip()[5:], max_rarity=target_rarity))

    damagetype_base = str(base.get("damagetype", "") or "").strip()

    # damagetype precedence:
    #   element (non-physical) overrides infusion magic; magic applies if no element; otherwise fall back to base
    element_dt: Optional[str] = None
    infusion_dt: Optional[str] = None

    # numeric effects from components + roll directives inside component fields
    price_muls: List[float] = []
    price_adds: List[float] = []
    dur_muls: List[float] = []
    dur_adds: List[float] = []
    uses_muls: List[float] = []
    uses_adds: List[float] = []
    charges_adds: List[float] = []
    charges_muls: List[float] = []

    weak_set: Set[str] = set(split_csv_words(base.get("weak")))
    strong_set: Set[str] = set(split_csv_words(base.get("strong")))

    for c in picks_by_slot.values():
        stat_mods = merge_mods(stat_mods, parse_mods_numeric(str(c.get("stat_mod", "") or "")))
        skill_mods = merge_mods(skill_mods, parse_mods_numeric(str(c.get("skill_mod", "") or "")))

        traits.extend(parse_traits_non_numeric(str(c.get("stat_mod", "") or "")))
        traits.extend(parse_traits_non_numeric(str(c.get("skill_mod", "") or "")))

        # component skill= ... (status/trait entries from spells.txt, or roll directives)
        raw_skill = str(c.get("skill", "") or "").strip()
        if raw_skill:
            if raw_skill.lower().startswith("roll:"):
                rolled_entries.extend(resolve_rolls(data, roll_str=raw_skill.split(":", 1)[1].strip(), max_rarity=target_rarity))
            else:
                # allow comma-separated names, lookup in spells dataset by name
                for nm in [x.strip() for x in raw_skill.split(",") if x.strip()]:
                    hit = None
                    for s in (getattr(data, "spells", []) or []):
                        if _norm(str(s.get("name", ""))) == _norm(nm):
                            hit = s
                            break
                    if hit:
                        rolled_entries.append(hit)
                    else:
                        traits.append(nm)

        # roll directives embedded in component fields (common use: infusion)
        for field in ("skill_mod", "stat_mod", "roll"):
            raw = str(c.get(field, "") or "").strip()
            if raw.lower().startswith("roll:"):
                rolled_entries.extend(resolve_rolls(data, roll_str=raw.split(":", 1)[1].strip(), max_rarity=target_rarity))

        # damage type candidates (precedence handled after loop)
        cand_dt = str(c.get("damagetype", "") or "").strip()
        if cand_dt:
            slot = _norm(str(c.get("group", "") or ""))  # group == slot
            cd = _norm(cand_dt)
            is_silent = str(c.get("silent", "") or c.get("hide", "") or "").strip().lower() in ("1","true","yes","y","on")

            # element sets element_dt only if non-physical and not silent
            if slot == "element" and (not is_silent) and cd and cd != "physical":
                element_dt = cand_dt

            # infusion sets infusion_dt (usually magic) if not silent
            if slot == "infusion" and (not is_silent) and not infusion_dt:
                infusion_dt = cand_dt

        # numeric fields (muls/adds can be dice)
        for key, target_muls, target_adds in (
            ("price_mul", price_muls, None),
            ("dur_mul", dur_muls, None),
            ("uses_mul", uses_muls, None),
            ("charges_mul", charges_muls, None),
        ):
            v = parse_number_expr(c.get(key))
            if v is not None:
                target_muls.append(float(v))

        for key, target_adds in (
            ("price_add", price_adds),
            ("dur_add", dur_adds),
            ("uses_add", uses_adds),
            ("charges_add", charges_adds),
        ):
            v = parse_number_expr(c.get(key))
            if v is not None:
                target_adds.append(float(v))

        weak_set |= split_csv_words(c.get("weak"))
        strong_set |= split_csv_words(c.get("strong"))

    # finalize damagetype
    damagetype = (element_dt or infusion_dt or damagetype_base or "")

    # derived fields
    base_r = _rarity(base)
    base_price = parse_int_maybe(base.get("price"))
    if base_price is None:
        base_price = int(DEFAULT_PRICE_BY_RARITY.get(base_r, DEFAULT_PRICE_BY_RARITY["common"]))

    # Only output durability/uses/charges when explicitly present (>0) on the base item.
    # If present, durability is clamped to at least 1 after modifiers.
    base_dur: Optional[int] = None
    base_dur_raw = parse_int_maybe(base.get("dur"))
    if base_dur_raw is not None and int(base_dur_raw) > 0:
        base_dur = int(base_dur_raw)

    base_uses: Optional[int] = parse_int_maybe(base.get("uses"))
    if base_uses is not None and int(base_uses) <= 0:
        base_uses = None

    charges_max: Optional[int] = parse_int_maybe(base.get("charges_max"))
    if charges_max is None:
        charges_max = parse_int_maybe(base.get("charges"))
    if charges_max is not None and int(charges_max) <= 0:
        charges_max = None

    # apply mul/add
    price_f = apply_num_mods(float(base_price), muls=price_muls, adds=price_adds) or float(base_price)
    dur_f: Optional[float] = None
    if base_dur is not None:
        dur_f = apply_num_mods(float(base_dur), muls=dur_muls, adds=dur_adds) or float(base_dur)

    uses_f: Optional[float] = None
    if base_uses is not None:
        uses_f = apply_num_mods(float(base_uses), muls=uses_muls, adds=uses_adds) or float(base_uses)

    charges_f: Optional[float] = None
    if charges_max is not None:
        charges_f = apply_num_mods(float(charges_max), muls=charges_muls, adds=charges_adds) or float(charges_max)

    # aggregate description (base + components + rolled entries) so we can omit a separate glossary
    desc_parts: List[str] = []
    base_desc = str(base.get("description", "") or "").strip()
    if base_desc:
        desc_parts.append(base_desc)
    for comp in picks_by_slot.values():
        cd = str(comp.get("description", "") or "").strip()
        if cd:
            desc_parts.append(cd)
    # NOTE: Rolled entries (skills/status/spells) are shown in a glossary section,
    # not embedded into item descriptions.
    full_desc = " ".join(desc_parts).strip()

    # attach
    return {
        "base": base,
        "description": full_desc,
        "final_name": final_name,
        "picks_by_slot": picks_by_slot,
        "rarity": target_rarity,
        "mech_tier": mech_tier,
        "stat_mod": mods_to_str(stat_mods),
        "skill_mod": mods_to_str(skill_mods),
        "traits": sorted({t for t in traits if t}),
        "rolled": rolled_entries,
        "price": int(round(price_f)),
        "dur": (max(1, int(round(dur_f))) if dur_f is not None else None),
        "uses": (max(1, int(round(uses_f))) if uses_f is not None else None),
        "charges_max": (max(1, int(round(charges_f))) if charges_f is not None else None),
        "damagetype": damagetype or None,
        "weak": sorted(weak_set) if weak_set else [],
        "strong": sorted(strong_set) if strong_set else [],
        "used_component_ids": sorted({str(c.get("id", "")).strip() for c in picks_by_slot.values() if str(c.get("id", "")).strip()}),
    }

# ============================================================
# PUBLIC: LOOT GENERATION
# ============================================================

def _filter_require(it: Dict[str, Any], require: str) -> bool:
    """require may match tag, cat, or slot (case-insensitive)."""
    req = _norm(require)
    if not req or req == "none":
        return True
    if req in {_norm(t) for t in _tags(it)}:
        return True
    if _norm(str(it.get("cat", ""))) == req:
        return True
    if _norm(str(it.get("slot", ""))) == req:
        return True
    return False

def _base_has_roll_directive(b: Dict[str, Any]) -> bool:
    # keys like "roll:spells:..." or values containing "roll:"
    for k in b.keys():
        if str(k).strip().lower().startswith("roll:"):
            return True
    for v in b.values():
        sv = str(v or "").strip().lower()
        if sv.startswith("roll:") or "roll:" in sv:
            return True
    return False

def _is_fixed_base(b: Dict[str, Any]) -> bool:
    # Fixed = no build= and no roll directives
    has_build = bool(str(b.get("build", "") or "").strip())
    if has_build:
        return False
    return not _base_has_roll_directive(b)

def roll_built_items(
    data: DataBundle,
    *,
    base_pool: List[Dict[str, Any]],
    count: int,
    rarity_req: str = "Random",
    auto_build: bool = True,
    unique: bool = True,
    jitter_pct: float = 0.0,
    merchant_level: Optional[int] = None,
    used_final_names: Optional[Set[str]] = None,
    used_fixed_base_names: Optional[Set[str]] = None,
    used_component_ids: Optional[Set[str]] = None,
    used_roll_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    if count <= 0 or not base_pool:
        return []

    components = list(getattr(data, "item_components", []) or [])
    used_final_names = used_final_names if used_final_names is not None else set()
    used_fixed_base_names = used_fixed_base_names if used_fixed_base_names is not None else set()
    used_component_ids = used_component_ids if used_component_ids is not None else set()
    used_roll_ids = used_roll_ids if used_roll_ids is not None else set()

    results: List[Dict[str, Any]] = []
    for _ in range(count):
        target_r = roll_rarity(rarity_req)

        rar_pool = [it for it in base_pool if eligible_by_rarity(it, target_r)]
        pick_pool = rar_pool if rar_pool else base_pool

        if unique:
            # Only prevent duplicates for FIXED bases (no build= and no roll directives).
            uniq_pool: List[Dict[str, Any]] = []
            for it in pick_pool:
                nm = str(it.get("name", "")).strip()
                if _is_fixed_base(it) and nm and nm in used_fixed_base_names:
                    continue
                uniq_pool.append(it)
            if uniq_pool:
                pick_pool = uniq_pool

        base = weighted_choice(pick_pool)
        if not base:
            continue

        built = build_up_item(
            data,
            base,
            components=components,
            target_rarity=target_r,
            auto_build=auto_build,
            merchant_level=merchant_level,
        )

        # best-effort uniqueness on final name
        if unique:
            guard = 0
            while built["final_name"] in used_final_names and guard < 10:
                built = build_up_item(
                    data,
                    base,
                    components=components,
                    target_rarity=target_r,
                    auto_build=auto_build,
                    merchant_level=merchant_level,
                )
                guard += 1

        used_final_names.add(built["final_name"])
        base_nm = str(base.get("name", "")).strip()
        if unique and _is_fixed_base(base) and base_nm:
            used_fixed_base_names.add(base_nm)

        used_component_ids |= set(built.get("used_component_ids", []) or [])
        for r in built.get("rolled", []) or []:
            rid = str(r.get("id", "")).strip()
            if rid:
                used_roll_ids.add(rid)

        if jitter_pct > 0:
            built["price"] = max(1, int(round(float(built["price"]) * jitter_factor(jitter_pct))))
            if built["dur"] is not None:
                built["dur"] = max(1, int(round(float(built["dur"]) * jitter_factor(jitter_pct))))

        results.append(built)

    return results

def generate_loot(data: DataBundle, inputs: Dict[str, Any]) -> Tuple[str, str]:
    """
    inputs:
      - count (int)
      - category: Random/weapon/armor/potion/herb/item/any
      - rarity: Random/common/uncommon/rare/legendary  (roll target per item)
      - auto_build: bool
      - unique: bool
      - include_shop: bool (default False)
      - require: optional tag/cat/slot filter (e.g. "hand", "ring", "trinket")
      - jitter_pct: float (price/dur jitter)
    """
    items = list(getattr(data, "items", []) or [])
    components = list(getattr(data, "item_components", []) or [])

    count = max(1, int(inputs.get("count", 10)))
    category_req = str(inputs.get("category", "Random")).strip().lower()
    rarity_req = str(inputs.get("rarity", "Random")).strip()
    auto_build = bool(inputs.get("auto_build", True))
    unique = bool(inputs.get("unique", True))
    include_shop = bool(inputs.get("include_shop", False))
    require = str(inputs.get("require", "") or "").strip()

    jitter_pct = float(inputs.get("jitter_pct", 0.25))

    def cat_ok(it: Dict[str, Any]) -> bool:
        c = item_category(it)
        if category_req in ("random", "any", ""):
            return True
        return c == category_req

    base_pool = []
    for it in items:
        if not include_shop and (SHOP_CORE_TAG in _tags(it) or SHOP_EXTRA_TAG in _tags(it)):
            continue
        if not cat_ok(it):
            continue
        if require and not _filter_require(it, require):
            continue
        base_pool.append(it)
    if not base_pool:
        base_pool = items[:]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"loot_{ts}.txt"

    lines: List[str] = []
    lines.append("LOOT ROLL")
    lines.append("-" * 60)
    lines.append(f"Generated: {ts}")
    lines.append(f"Count: {count}")
    lines.append(f"Category: {category_req or 'random'}")
    lines.append(f"Rarity: {rarity_req}")
    lines.append(f"Require: {require or 'None'}")
    lines.append(f"Auto-build: {auto_build}")
    lines.append("")

    used_final_names: Set[str] = set()
    used_fixed_base_names: Set[str] = set()
    used_component_ids: Set[str] = set()
    used_roll_ids: Set[str] = set()

    results = roll_built_items(
        data,
        base_pool=base_pool,
        count=count,
        rarity_req=rarity_req,
        auto_build=auto_build,
        unique=unique,
        jitter_pct=jitter_pct,
        used_final_names=used_final_names,
        used_fixed_base_names=used_fixed_base_names,
        used_component_ids=used_component_ids,
        used_roll_ids=used_roll_ids,
    )

    lines.append("Results")
    lines.append("-" * 60)

    # Sort alphabetically by final name
    results.sort(key=lambda r: _norm(str(r.get("final_name", ""))))

    for i, it in enumerate(results, start=1):
        lines.append(f"{i:02d}. {it['final_name']} (price={it['price']})")

        base = it.get('base', {}) or {}
        raw_cat = _norm(str(base.get('cat', '') or ''))
        cat = raw_cat if raw_cat else item_category(base)
        cat_label = 'miscellaneous item' if cat == 'misc' else cat

        rarity = str(it.get('rarity','common') or 'common').strip().lower()
        rarity_label = rarity.capitalize()

        desc = str(it.get('description', '') or '').strip()

        # line 2: rarity/category before description
        if desc:
            lines.append(f"    {rarity_label} {cat_label} — {desc}")
        else:
            lines.append(f"    {rarity_label} {cat_label}")

        # line 3: compact stats/mods (no labels for stat_mod/skill_mod)
        stats: List[str] = []

        sm = str(it.get('stat_mod', '') or '').strip()
        if sm:
            stats.append(sm)

        km = str(it.get('skill_mod', '') or '').strip()
        if km:
            stats.append(km)

        if it.get('dur') is not None:
            stats.append(f"durability={it['dur']}")
        # only show non-physical damage types
        dt = str(it.get('damagetype','') or '').strip()
        if dt and _norm(dt) != 'physical':
            stats.append(f"damagetype={dt}")
        if it.get('uses') is not None:
            stats.append(f"uses={it['uses']}")
        if it.get('charges_max') is not None:
            stats.append(f"charges_max={it['charges_max']}")
        rolled = it.get('rolled', []) or []
        if rolled:
            names = [str(r.get('name','')).strip() for r in rolled if str(r.get('name','')).strip()]
            if names:
                stats.append('skills: ' + ', '.join(names))

        if stats:
            lines.append('    ' + '; '.join(stats))

        # collect rolled entries for glossary
        for r in rolled:
            rid = str(r.get('id','') or '').strip()
            if rid:
                used_roll_ids.add(rid)

    # Skills/Status glossary (rolled entries only)
    spells = getattr(data, "spells", []) or []
    if spells and used_roll_ids:
        id_to_roll = {str(sp.get("id","")).strip(): sp for sp in spells}
        lines.append("")
        lines.append("Skills & Status Glossary")
        lines.append("-" * 60)

        rolls = []
        for rid in used_roll_ids:
            r = id_to_roll.get(rid)
            if r:
                rolls.append(r)
        rolls.sort(key=lambda r: _norm(str(r.get("name",""))))

        for r in rolls:
            nm = str(r.get("name","(unnamed)")).strip()
            desc = str(r.get("description","") or "").strip()
            lines.append(f"- {nm}")
            if desc:
                lines.append(f"  {desc}")
    lines.append("")
    return filename, "\n".join(lines)

# ============================================================
# PUBLIC: SHOP GENERATION
# ============================================================

def _shop_defaults_for_settlement(settlement: str) -> Tuple[int, str, bool]:
    """Returns (random_count, max_rarity, include_extra).

    Updated defaults:
      - hamlet:  core + 6 random
      - village: core + 12 random
      - town:    core + extra + 20 random
    """
    s = _norm(settlement)
    if s in ("hamlet", "outpost"):
        return 6, "uncommon", False
    if s in ("village",):
        return 12, "rare", False
    if s in ("town", "city"):
        return 20, "legendary", True
    return 12, "rare", False

def _default_merchant_level_for_settlement(settlement: str) -> int:
    s = _norm(settlement)
    if s in ("hamlet", "outpost"):
        return 1
    if s in ("village",):
        return 2
    if s in ("town", "city"):
        return 3
    return 2

def apply_merchant_overlay_fixed_stock(it: Dict[str, Any], merchant_level: int) -> Dict[str, Any]:
    """Pattern B: upgrade shop_core/shop_extra (weapons/armor) by merchant level.

    This is a deterministic overlay (no rolling) applied ONLY to fixed stock.
    It adjusts:
      - name prefix (Steel/Mythril/Adamant/Masterwork)
      - price multiplier
      - durability bonus (if durability exists)
    """
    ml = max(1, min(5, int(merchant_level)))

    cat = _norm(str(it.get("cat", "") or "")) or item_category(it)
    if cat not in ("weapon", "armor"):
        return it

    overlay = {
        1: {"prefix": "",          "price_mul": 1.0, "dur_add": 0},
        2: {"prefix": "Steel",     "price_mul": 1.5, "dur_add": 1},
        3: {"prefix": "Mythril",   "price_mul": 2.5, "dur_add": 2},
        4: {"prefix": "Adamant",   "price_mul": 4.0, "dur_add": 3},
        5: {"prefix": "Masterwork","price_mul": 6.0, "dur_add": 4},
    }[ml]

    eff = dict(it)

    base_name = str(eff.get("name", "") or "").strip()
    pref = overlay["prefix"]
    if pref:
        # Avoid double-prefixing if the name already starts with a known quality word
        known = ("Steel ", "Mythril ", "Adamant ", "Masterwork ")
        if not base_name.startswith(known):
            eff["name"] = f"{pref} {base_name}".strip()

    # Price overlay (only if price is present / numeric)
    p = parse_int_maybe(eff.get("price"))
    if p is not None:
        eff["price"] = max(1, int(round(float(p) * float(overlay["price_mul"]))))

    # Durability overlay (only if dur exists)
    d = parse_int_maybe(eff.get("dur"))
    if d is not None:
        eff["dur"] = max(1, int(d) + int(overlay["dur_add"]))

    return eff

def generate_shop(data: DataBundle, inputs: Dict[str, Any]) -> Tuple[str, str]:
    """
    inputs:
      - settlement: hamlet/village/town (default from settings if present)
      - random_count: override additional stock (optional)
      - max_rarity: cap for random stock (optional; default by settlement)
      - auto_build: bool
      - unique_base: bool (avoid base-name duplicates against fixed stock)
    """
    items = list(getattr(data, "items", []) or [])
    components = list(getattr(data, "item_components", []) or [])

    settlement = str(inputs.get("settlement") or data.settings.get("default.shop_settlement") or "Hamlet").strip()
    default_random_count, default_max_rarity, include_extra = _shop_defaults_for_settlement(settlement)

    # Merchant level (1-5) gates and biases components for random additions
    merchant_level = inputs.get("merchant_level")
    if merchant_level is None:
        merchant_level = data.settings.get("default.shop_merchant_level")
    if merchant_level is None:
        merchant_level = _default_merchant_level_for_settlement(settlement)
    try:
        merchant_level = int(merchant_level)
    except Exception:
        merchant_level = _default_merchant_level_for_settlement(settlement)
    merchant_level = max(1, min(5, merchant_level))


    random_count = inputs.get("random_count")
    if random_count is None:
        random_count = default_random_count
    random_count = max(0, int(random_count))

    max_rarity = _norm(str(inputs.get("max_rarity") or default_max_rarity))
    if max_rarity not in RARITY_IDX:
        max_rarity = default_max_rarity

    auto_build = bool(inputs.get("auto_build", True))
    unique_base = bool(inputs.get("unique_base", True))

    core = [it for it in items if SHOP_CORE_TAG in _tags(it)]
    extra = [it for it in items if SHOP_EXTRA_TAG in _tags(it)] if include_extra else []

    # fixed stock names for duplicate prevention
    fixed_names = { _norm(str(it.get("name",""))) for it in (core + extra) if str(it.get("name","")).strip() }

    # Random pool: exclude shop-only, exclude shop_no, and respect rarity cap
    pool = []
    for it in items:
        t = _tags(it)
        if SHOP_CORE_TAG in t or SHOP_EXTRA_TAG in t:
            continue
        if SHOP_NO_TAG in t:
            continue
        if not eligible_by_rarity(it, max_rarity):
            continue
        if unique_base and _norm(str(it.get("name",""))) in fixed_names:
            continue
        pool.append(it)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"shop_{_norm(settlement) or 'shop'}_{ts}.txt"

    lines: List[str] = []
    lines.append("SHOP STOCK")
    lines.append("-" * 60)
    lines.append(f"Generated: {ts}")
    lines.append(f"Settlement: {settlement}")
    lines.append(f"Merchant level: {merchant_level}")
    lines.append(f"Random additions: {random_count}")
    lines.append(f"Rarity cap: {max_rarity}")
    lines.append("")

    # Collect rolled skill/status IDs for glossary (fixed stock + random additions)
    used_roll_ids: Set[str] = set()

    # Fixed stock
    lines.append("Core Stock")
    lines.append("." * 60)

    core_sorted = sorted(core, key=lambda it: _norm(str(it.get("name", ""))))
    for base in core_sorted:
        built = build_up_item(
            data,
            base,
            components=components,
            target_rarity="legendary",
            auto_build=False,
            merchant_level=merchant_level,
            consistent_material=True,
        )

        nm = str(built.get("final_name", "") or str(base.get("name", "(unnamed)"))).strip()
        price = parse_int_maybe(built.get("price"))

        if price is not None:
            lines.append(f"- {nm} (price={price})")
        else:
            lines.append(f"- {nm}")

        desc = str(base.get("description", "") or "").strip()

        raw_cat = _norm(str(base.get("cat", "") or ""))
        cat = raw_cat if raw_cat else item_category(base)
        cat_label = "miscellaneous item" if cat == "misc" else cat

        r = _rarity(base)
        rarity_label = r.capitalize()

        if desc:
            lines.append(f"    {rarity_label} {cat_label} — {desc}")
        else:
            lines.append(f"    {rarity_label} {cat_label}")

        stats: List[str] = []

        sm = str(built.get("stat_mod", "") or "").strip()
        if sm:
            stats.append(sm)
        km = str(built.get("skill_mod", "") or "").strip()
        if km:
            stats.append(km)

        dur = built.get("dur")
        if dur is not None and int(dur) > 0:
            stats.append(f"durability={int(dur)}")

        dt = str(built.get("damagetype","") or "").strip()
        if dt and _norm(dt) != "physical":
            stats.append(f"damagetype={dt}")

        uses = built.get("uses")
        if uses is not None and int(uses) > 0:
            stats.append(f"uses={int(uses)}")

        cm = built.get("charges_max")
        if cm is not None and int(cm) > 0:
            stats.append(f"charges_max={int(cm)}")

        rolled = built.get("rolled", []) or []
        if rolled:
            names = [str(r.get("name","")).strip() for r in rolled if str(r.get("name","")).strip()]
            if names:
                stats.append("skills: " + ", ".join(names))
            for r0 in rolled:
                rid = str(r0.get("id","") or "").strip()
                if rid:
                    used_roll_ids.add(rid)

        if stats:
            lines.append("    " + "; ".join(stats))
    lines.append("")

    if extra:
        lines.append("Extra Stock")
        lines.append("." * 60)

        extra_sorted = sorted(extra, key=lambda it: _norm(str(it.get("name", ""))))
        for base in extra_sorted:
            built = build_up_item(
                data,
                base,
                components=components,
                target_rarity="legendary",
                auto_build=False,
                merchant_level=merchant_level,
                consistent_material=True,
            )

            nm = str(built.get("final_name", "") or str(base.get("name", "(unnamed)"))).strip()
            price = parse_int_maybe(built.get("price"))

            if price is not None:
                lines.append(f"- {nm} (price={price})")
            else:
                lines.append(f"- {nm}")

            desc = str(base.get("description", "") or "").strip()

            raw_cat = _norm(str(base.get("cat", "") or ""))
            cat = raw_cat if raw_cat else item_category(base)
            cat_label = "miscellaneous item" if cat == "misc" else cat

            r = _rarity(base)
            rarity_label = r.capitalize()

            if desc:
                lines.append(f"    {rarity_label} {cat_label} — {desc}")
            else:
                lines.append(f"    {rarity_label} {cat_label}")

            stats: List[str] = []

            sm = str(built.get("stat_mod", "") or "").strip()
            if sm:
                stats.append(sm)
            km = str(built.get("skill_mod", "") or "").strip()
            if km:
                stats.append(km)

            dur = built.get("dur")
            if dur is not None and int(dur) > 0:
                stats.append(f"durability={int(dur)}")

            dt = str(built.get("damagetype","") or "").strip()
            if dt and _norm(dt) != "physical":
                stats.append(f"damagetype={dt}")

            uses = built.get("uses")
            if uses is not None and int(uses) > 0:
                stats.append(f"uses={int(uses)}")

            cm = built.get("charges_max")
            if cm is not None and int(cm) > 0:
                stats.append(f"charges_max={int(cm)}")

            rolled = built.get("rolled", []) or []
            if rolled:
                names = [str(r.get("name","")).strip() for r in rolled if str(r.get("name","")).strip()]
                if names:
                    stats.append("skills: " + ", ".join(names))
                for r0 in rolled:
                    rid = str(r0.get("id","") or "").strip()
                    if rid:
                        used_roll_ids.add(rid)

            if stats:
                lines.append("    " + "; ".join(stats))
        lines.append("")

    # random additions (built items)
    lines.append("Random Additions")
    lines.append("." * 60)

    used_base_names: Set[str] = set(fixed_names)
    used_final_names: Set[str] = set()

    additions: List[Dict[str, Any]] = []
    for _ in range(random_count):
        if not pool:
            break
        target_r = roll_rarity_capped(max_rarity)
        rar_pool = [it for it in pool if eligible_by_rarity(it, target_r)]
        pick_pool = rar_pool if rar_pool else pool

        # base-name uniqueness
        if unique_base:
            pick_pool2 = [it for it in pick_pool if _norm(str(it.get("name",""))) not in used_base_names]
            if pick_pool2:
                pick_pool = pick_pool2

        base = weighted_choice(pick_pool)
        if not base:
            continue

        built = build_up_item(data, base, components=components, target_rarity=target_r, auto_build=auto_build, merchant_level=merchant_level)

        # avoid identical final names
        guard = 0
        while built["final_name"] in used_final_names and guard < 10:
            built = build_up_item(data, base, components=components, target_rarity=target_r, auto_build=auto_build, merchant_level=merchant_level)
            guard += 1

        used_base_names.add(_norm(str(base.get("name",""))))
        used_final_names.add(built["final_name"])
        additions.append(built)


    # collect rolled entries (random additions)

    if not additions:
        lines.append("(none)")
    else:
        # Sort alphabetically by final name
        additions.sort(key=lambda r: _norm(str(r.get("final_name", ""))))

        for it in additions:
            lines.append(f"- {it['final_name']} (price={it['price']})")

            base = it.get("base", {}) or {}
            raw_cat = _norm(str(base.get("cat", "") or ""))
            cat = raw_cat if raw_cat else item_category(base)
            cat_label = "miscellaneous item" if cat == "misc" else cat

            rarity = str(it.get("rarity","common") or "common").strip().lower()
            rarity_label = rarity.capitalize()

            desc = str(it.get("description", "") or "").strip()
            if desc:
                lines.append(f"    {rarity_label} {cat_label} — {desc}")
            else:
                lines.append(f"    {rarity_label} {cat_label}")

            stats: List[str] = []

            sm = str(it.get("stat_mod", "") or "").strip()
            if sm:
                stats.append(sm)

            km = str(it.get("skill_mod", "") or "").strip()
            if km:
                stats.append(km)

            if it.get("dur") is not None:
                stats.append(f"durability={it['dur']}")

            dt = str(it.get("damagetype","") or "").strip()
            if dt and _norm(dt) != "physical":
                stats.append(f"damagetype={dt}")

            if it.get("uses") is not None:
                stats.append(f"uses={it['uses']}")
            if it.get("charges_max") is not None:
                stats.append(f"charges_max={it['charges_max']}")

            rolled = it.get("rolled", []) or []
            if rolled:
                names = [str(r.get("name","")).strip() for r in rolled if str(r.get("name","")).strip()]
                if names:
                    stats.append("skills: " + ", ".join(names))
                for r in rolled:
                    rid = str(r.get("id","") or "").strip()
                    if rid:
                        used_roll_ids.add(rid)

            if stats:
                lines.append("    " + "; ".join(stats))

    # Skills & Status Glossary (rolled entries only)
    spells = getattr(data, "spells", []) or []
    if spells and used_roll_ids:
        id_to_roll = {str(sp.get("id", "")).strip(): sp for sp in spells}

        rolls: List[Dict[str, Any]] = []
        for rid in used_roll_ids:
            r = id_to_roll.get(rid)
            if r:
                rolls.append(r)

        if rolls:
            rolls.sort(key=lambda r: _norm(str(r.get("name",""))))
            lines.append("")
            lines.append("Skills & Status Glossary")
            lines.append("-" * 60)
            for r in rolls:
                nm = str(r.get("name","(unnamed)")).strip()
                desc = str(r.get("description","") or "").strip()
                lines.append(f"- {nm}")
                if desc:
                    lines.append(f"  {desc}")

    return filename, "\n".join(lines)
