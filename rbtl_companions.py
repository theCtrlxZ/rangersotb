# rbtl_companions.py
#
# Single-file Companion Generator (refactor-friendly, minimal script sprawl).
#
# REQUIREMENTS / ASSUMPTIONS
# - You already have rbtl_data.py providing DataBundle with:
#     data.items, data.spells, data.traits,
#     data.companion_names, data.companion_classes, data.companion_backgrounds
# - You already have rbtl_core.py providing:
#     parse_statline(stat_str) -> dict
#     apply_stat_mods(stats_dict, mod_str)
#     STAT_KEYS (ordered list of stats for printing)
#
# LOCKED (as per your project):
# - Companions only, batch generation, single output file, no state tracking
# - Trait-based naming + animal naming rule
# - Tags drive eligibility
# - Step 5 item randomization: Mode A placeholder replacement; Mode B optional (currently no-op)
# - NO duplicates for spells OR traits per companion across all sources (class/background/inline/step4)
#
# LOCKED ROLL GRAMMAR (canonical kinds):
#   weapons, spells, item, herb, armor, traits
# - weapons is a FILTERED roll from items.txt via tags: hand/twohand/ranged
# - spells from spells.txt
# - traits from traits.txt (excludes enemy_only by default)

# rbtl_companions.py
#
# Companion Generator (single module).
#
# Locked design rules supported:
# - Companions only, batch, single output
# - Trait-based naming + animal naming rule
# - Tags drive eligibility
# - Step 5 item randomization: placeholder replacement (Mode A); Mode B stubbed
# - No duplicates for spells or traits per companion (across class/background/trait-step/roll-tokens)

# rbtl_companions.py
from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rbtl_data import DataBundle, parse_int_maybe
from rbtl_core import weighted_choice, apply_stat_mods, parse_statline, STAT_KEYS
from rbtl_loot import roll_built_items

OUTPUT_DIR = "output"

RARITY_WEIGHTS = {"common": 60, "uncommon": 25, "rare": 10, "legendary": 5}

TRAIT_DISTRIBUTION = [
    (0.69, 0),
    (0.94, 1),
    (1.00, 2),
]

# Tags that imply an item is a weapon inside items.txt
WEAPON_TAGS = {"hand", "twohand", "ranged"}


# ============================================================
# #LABEL: TAG + SKILL HELPERS
# ============================================================

def entry_tags(e: Dict[str, Any]) -> Set[str]:
    tags = e.get("tags") or set()
    return {str(t).strip().lower() for t in tags if str(t).strip()}


def eligible_by_tags(entry: Dict[str, Any], required: Optional[Set[str]]) -> bool:
    if not required:
        return True
    et = entry_tags(entry)
    if not et:
        return True  # tagless entries are universal
    return bool(et.intersection(required))


def parse_skills(skill_string: str) -> Dict[str, Optional[int]]:
    skills: Dict[str, Optional[int]] = {}
    if not skill_string:
        return skills
    for part in skill_string.split(","):
        part = part.strip()
        if not part:
            continue
        if "+" in part:
            name, value = part.rsplit("+", 1)
            skills[name.strip()] = int(value.strip())
        else:
            skills[part] = None
    return skills


def apply_skill_mods(skills: Dict[str, Optional[int]], mod_string: str) -> None:
    if not mod_string:
        return
    for part in mod_string.split(","):
        part = part.strip()
        if not part or "+" not in part:
            continue
        name, value = part.rsplit("+", 1)
        key = name.strip()
        add = int(value.strip())
        cur = skills.get(key)
        if cur is None:
            cur = 0
        skills[key] = cur + add


def determine_trait_count() -> int:
    r = random.random()
    for threshold, count in TRAIT_DISTRIBUTION:
        if r <= threshold:
            return count
    return 0


def _parse_roll_params(rest: str) -> Dict[str, str]:
    """
    Accepts both:
      spells:tag=spells:count=2
      spells:tag=spells,count=2
    """
    if not rest:
        return {}
    chunks: List[str] = []
    for seg in rest.split(":"):
        seg = seg.strip()
        if not seg:
            continue
        chunks.extend([c.strip() for c in seg.split(",") if c.strip()])
    out: Dict[str, str] = {}
    for c in chunks:
        if "=" in c:
            k, v = c.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ============================================================
# #LABEL: UNIQUE PICKING
# ============================================================

def _pick_unique(pool: List[Dict[str, Any]], *, count: int, exclude: Set[str]) -> List[Dict[str, Any]]:
    picks: List[Dict[str, Any]] = []
    for _ in range(max(0, count)):
        candidates = [e for e in pool if e.get("name") not in exclude]
        if not candidates:
            break
        chosen = weighted_choice(candidates)
        if not chosen:
            break
        picks.append(chosen)
        exclude.add(chosen.get("name"))
    return picks


# ============================================================
# #LABEL: ROLL RESOLUTION
# ============================================================

def _pool_weapons(data: DataBundle, required: Optional[Set[str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in data.items:
        tags = entry_tags(it)
        if not tags.intersection(WEAPON_TAGS):
            continue
        if not eligible_by_tags(it, required):
            continue
        out.append(it)
    return out


def _pool_items(data: DataBundle, required: Optional[Set[str]]) -> List[Dict[str, Any]]:
    return [it for it in data.items if eligible_by_tags(it, required)]


def _roll_companion_items(
    data: DataBundle,
    *,
    base_pool: List[Dict[str, Any]],
    count: int,
    ctx: Dict[str, Any],
    rarity_req: str = "Random",
    merchant_level: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not base_pool or count <= 0:
        return []
    return roll_built_items(
        data,
        base_pool=base_pool,
        count=count,
        rarity_req=rarity_req,
        auto_build=True,
        unique=True,
        jitter_pct=0.0,
        merchant_level=merchant_level,
        used_final_names=ctx["picked_items"],
    )


def _pool_spells(data: DataBundle, required: Optional[Set[str]]) -> List[Dict[str, Any]]:
    return [sp for sp in data.spells if eligible_by_tags(sp, required)]


def _pool_traits(data: DataBundle, required: Optional[Set[str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tr in data.traits:
        ttags = entry_tags(tr)
        if "enemy_only" in ttags:
            continue
        if not eligible_by_tags(tr, required):
            continue
        out.append(tr)
    return out


def _collect_roll_fields(entry: Dict[str, Any]) -> List[str]:
    """
    Primary: entry["roll"] (now populated by rbtl_data for roll:... pipe chunks)
    Fallback: directives that start with roll:
    """
    out: List[str] = []
    rf = (entry.get("roll") or "").strip()
    if rf:
        out.append(rf)

    for d in (entry.get("directives") or []):
        ds = str(d).strip()
        if ds.lower().startswith("roll:"):
            out.append(ds[5:].strip())

    return out


def apply_roll_field(
    roll_field: str,
    *,
    companion: Dict[str, Any],
    data: DataBundle,
    ctx: Dict[str, Any],
    source_label: str,
    merchant_level: Optional[int] = None,
) -> None:
    """
    roll_field supports semicolon-separated directives:
      spells:tag=spells:count=2; weapons:tag=hand
      traits:tag=magic:count=1
    """
    if not roll_field:
        return

    directives = [d.strip() for d in str(roll_field).split(";") if d.strip()]
    for raw in directives:
        if ":" in raw:
            kind, rest = raw.split(":", 1)
        else:
            kind, rest = raw, ""

        kind = kind.strip().lower()
        params = _parse_roll_params(rest)

        count = max(1, parse_int_maybe(params.get("count"), 1))
        req = {t.strip().lower() for t in (params.get("tag", "").split(",")) if t.strip()} if "tag" in params else None

        # Track roll diagnostics
        ctx["debug_rolls"].append(f"{source_label}: {kind} (tag={','.join(sorted(req)) if req else 'ANY'}, count={count})")

        # Normalize synonyms to be forgiving
        if kind in ("spell", "spells"):
            pool = _pool_spells(data, req)
            picks = _pick_unique(pool, count=count, exclude=ctx["picked_spells"])
            for p in picks:
                companion["spells"].append(p["name"])
            if len(picks) < count:
                ctx["warnings"].append(
                    f"[{source_label}] roll:{kind} requested {count} but got {len(picks)} "
                    f"(pool={len(pool)}, tag={req or 'ANY'}, uniqueness exhausted or no matches)"
                )
            continue

        if kind in ("trait", "traits"):
            pool = _pool_traits(data, req)
            picks = _pick_unique(pool, count=count, exclude=ctx["picked_traits"])
            companion["traits"].extend(picks)
            for tr in picks:
                apply_stat_mods(companion["stats"], tr.get("stat_mod"))
                apply_skill_mods(companion["skills"], tr.get("skill_mod"))
            if len(picks) < count:
                ctx["warnings"].append(
                    f"[{source_label}] roll:{kind} requested {count} but got {len(picks)} "
                    f"(pool={len(pool)}, tag={req or 'ANY'}, uniqueness exhausted or no matches)"
                )
            continue

        if kind in ("weapon", "weapons"):
            pool = _pool_weapons(data, req)
            picks = _roll_companion_items(
                data,
                base_pool=pool,
                count=count,
                ctx=ctx,
                merchant_level=merchant_level,
            )
            companion["items"].extend(picks)
            if len(picks) < count:
                ctx["warnings"].append(
                    f"[{source_label}] roll:{kind} requested {count} but got {len(picks)} "
                    f"(pool={len(pool)}, tag={req or 'ANY'}, uniqueness exhausted or no matches)"
                )
            continue

        if kind in ("item", "items", "herb", "armor"):
            pool = _pool_items(data, req)
            picks = _roll_companion_items(
                data,
                base_pool=pool,
                count=count,
                ctx=ctx,
                merchant_level=merchant_level,
            )
            companion["items"].extend(picks)
            if len(picks) < count:
                ctx["warnings"].append(
                    f"[{source_label}] roll:{kind} requested {count} but got {len(picks)} "
                    f"(pool={len(pool)}, tag={req or 'ANY'}, uniqueness exhausted or no matches)"
                )
            continue

        if kind in ("ability", "abilities"):
            # optional pattern: pull from spells list tagged abilities if you ever do that
            pool = _pool_spells(data, req)
            picks = _pick_unique(pool, count=count, exclude=ctx["picked_abilities"])
            for p in picks:
                companion["abilities"].append(p["name"])
            if len(picks) < count:
                ctx["warnings"].append(
                    f"[{source_label}] roll:{kind} requested {count} but got {len(picks)} "
                    f"(pool={len(pool)}, tag={req or 'ANY'}, uniqueness exhausted or no matches)"
                )
            continue

        # Unknown kind: warn
        ctx["warnings"].append(f"[{source_label}] Unknown roll kind '{kind}' in directive '{raw}' (ignored)")


# ============================================================
# #LABEL: STEP 5 — PLACEHOLDER WEAPONS
# ============================================================

def replace_placeholder_weapons(
    companion: Dict[str, Any],
    *,
    data: DataBundle,
    ctx: Dict[str, Any],
    merchant_level: Optional[int] = None,
) -> None:
    out: List[Dict[str, Any]] = []
    for item in companion["items"]:
        name = item.get("final_name") if isinstance(item, dict) else str(item)
        if name == "Handweapon":
            pool = _pool_weapons(data, {"hand"})
            picks = _roll_companion_items(
                data,
                base_pool=pool,
                count=1,
                ctx=ctx,
                merchant_level=merchant_level,
            )
            if picks:
                out.append(picks[0])
            else:
                out.append({"final_name": name})
                ctx["warnings"].append("[Step5] Handweapon placeholder had no eligible unique hand weapons.")
        elif name == "Twohandweapon":
            pool = _pool_weapons(data, {"twohand"})
            picks = _roll_companion_items(
                data,
                base_pool=pool,
                count=1,
                ctx=ctx,
                merchant_level=merchant_level,
            )
            if picks:
                out.append(picks[0])
            else:
                out.append({"final_name": name})
                ctx["warnings"].append("[Step5] Twohandweapon placeholder had no eligible unique twohand weapons.")
        else:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append({"final_name": name})
    companion["items"] = out


# ============================================================
# #LABEL: NAME RULES
# ============================================================

def pick_unique_name(names: List[Dict[str, Any]], used: Set[str]) -> str:
    pool = [n for n in names if n.get("name") not in used] or names
    if not pool:
        return "Nameless"
    chosen = random.choice(pool).get("name", "Nameless")
    used.add(chosen)
    return chosen


def finalize_name(base: str, companion: Dict[str, Any]) -> str:
    tags = {t.lower() for t in (companion.get("tags") or set())}
    traits = companion.get("traits") or []
    if "animal" in tags and traits:
        return traits[0]["name"]
    if traits:
        return f"{base}, the {traits[0]['name']}"
    return base


def _resolve_item_entry(name: str, data: DataBundle) -> Dict[str, Any]:
    norm = name.strip().lower()
    base = None
    for it in data.items:
        if str(it.get("name", "")).strip().lower() == norm:
            base = it
            break
    entry: Dict[str, Any] = {"final_name": name}
    if base:
        entry["base"] = base
    return entry


# ============================================================
# #LABEL: GENERATE ONE COMPANION
# ============================================================

def generate_companion(
    data: DataBundle,
    *,
    used_names: Set[str],
    required_class_tags: Optional[Set[str]] = None,
    required_class_name: Optional[str] = None,
    allow_background_trait_rolls: bool = True,
    merchant_level: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    ctx = {
        "picked_spells": set(),
        "picked_traits": set(),
        "picked_abilities": set(),
        "picked_items": set(),     # NEW: prevents duplicate items per companion
        "warnings": [],            # NEW: collects warnings for debug footer
        "debug_rolls": [],         # NEW: roll diagnostics lines
    }

    # ---- Class selection ----
    cls = None
    if required_class_name:
        want = str(required_class_name).strip().lower()
        matches = [c for c in data.companion_classes if str(c.get("name", "")).strip().lower() == want]
        cls = matches[0] if matches else None

    if cls is None and required_class_tags:
        restricted = [c for c in data.companion_classes if required_class_tags.intersection(entry_tags(c))]
        cls = weighted_choice(restricted) if restricted else None

    if cls is None:
        cls = weighted_choice(data.companion_classes) or {"name": "Unknown"}

    bg = weighted_choice(data.companion_backgrounds) or {"name": "Unknown"}

    companion: Dict[str, Any] = {
        "class": cls,
        "background": bg,
        "stats": parse_statline(cls.get("stat", "")),
        "skills": parse_skills(cls.get("skills", "")),
        "items": [],
        "traits": [],
        "spells": [],
        "abilities": [],
        "tags": set(entry_tags(cls)) | set(entry_tags(bg)),
        "rp": 0,
        "name": "",
        "_ctx": ctx,
    }

    # Apply background mods
    apply_stat_mods(companion["stats"], bg.get("stat_mod"))
    apply_skill_mods(companion["skills"], bg.get("skill_mod"))

    # Base name
    base_name = pick_unique_name(data.companion_names, used_names)

    # Step 4: random traits
    trait_count = determine_trait_count()
    if trait_count > 0:
        pool = _pool_traits(data, {t.lower() for t in companion["tags"]})
        picks = _pick_unique(pool, count=trait_count, exclude=ctx["picked_traits"])
        companion["traits"].extend(picks)
        for tr in picks:
            apply_stat_mods(companion["stats"], tr.get("stat_mod"))
            apply_skill_mods(companion["skills"], tr.get("skill_mod"))
        if len(picks) < trait_count:
            ctx["warnings"].append(
                f"[Step4] Random trait step requested {trait_count} but got {len(picks)} (uniqueness/pool)."
            )

    # Gear tokens (roll:... in gear)
    for src, label in ((cls, "Class gear"), (bg, "Background gear")):
        gear = (src.get("gear") or "").strip()
        if not gear:
            continue
        for tok in [t.strip() for t in gear.split(",") if t.strip()]:
            low = tok.lower()
            if low.startswith("roll:"):
                apply_roll_field(
                    tok[5:].strip(),
                    companion=companion,
                    data=data,
                    ctx=ctx,
                    source_label=label,
                    merchant_level=merchant_level,
                )
            elif low.startswith("roll="):
                apply_roll_field(
                    tok.split("=", 1)[1].strip(),
                    companion=companion,
                    data=data,
                    ctx=ctx,
                    source_label=label,
                    merchant_level=merchant_level,
                )
            else:
                # NEW: prevent duplicate literal gear items too
                if tok not in ctx["picked_items"]:
                    companion["items"].append(_resolve_item_entry(tok, data))
                    ctx["picked_items"].add(tok)

    # Step 5: placeholder replacement
    replace_placeholder_weapons(companion, data=data, ctx=ctx, merchant_level=merchant_level)

    # Step 6: class/background roll directives from pipe segments
    for src, label in ((cls, "Class roll"), (bg, "Background roll")):
        rolls = _collect_roll_fields(src)
        if debug and label == "Class roll":
            print(f"DEBUG class={cls.get('name')} roll_fields={rolls}")
        for rf in rolls:
            apply_roll_field(
                rf,
                companion=companion,
                data=data,
                ctx=ctx,
                source_label=label,
                merchant_level=merchant_level,
            )

    # Optional: background extra trait roll field (future use)
    if allow_background_trait_rolls:
        tr = (bg.get("trait_roll") or "").strip()
        if tr:
            apply_roll_field(
                tr,
                companion=companion,
                data=data,
                ctx=ctx,
                source_label="Background trait_roll",
                merchant_level=merchant_level,
            )

    # Final name + RP
    companion["name"] = finalize_name(base_name, companion)

    rp_total = parse_int_maybe(cls.get("rp"), 0) + parse_int_maybe(bg.get("rp"), 0)
    for tr in companion["traits"]:
        rp_total += parse_int_maybe(tr.get("rp"), 0)
    companion["rp"] = rp_total

    return companion


# ============================================================
# #LABEL: OUTPUT FORMATTING
# ============================================================

def _item_name(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("final_name") or item.get("name") or "").strip()
    return str(item or "").strip()


def _item_field(item: Any, field: str) -> Optional[Any]:
    if not isinstance(item, dict):
        return None
    if field in item and item[field] not in (None, ""):
        return item[field]
    base = item.get("base")
    if isinstance(base, dict):
        return base.get(field)
    return None


def _format_numeric_mods(mod_str: Optional[str]) -> List[str]:
    mods: List[str] = []
    for chunk in (mod_str or "").split(","):
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
        if not key:
            continue
        num = parse_int_maybe(val, None)
        if num is None:
            continue
        sign = "+" if num >= 0 else ""
        mods.append(f"{key} {sign}{num}")
    return mods


def _item_stat_line(item: Any) -> str:
    stat_mod = _item_field(item, "stat_mod")
    skill_mod = _item_field(item, "skill_mod")
    parts: List[str] = []
    parts.extend(_format_numeric_mods(str(stat_mod) if stat_mod is not None else ""))
    parts.extend(_format_numeric_mods(str(skill_mod) if skill_mod is not None else ""))
    return ", ".join(parts)


def _item_skill_names(item: Any) -> List[str]:
    names: List[str] = []
    rolled = _item_field(item, "rolled") or []
    for entry in rolled:
        nm = str(entry.get("name", "") or "").strip()
        if nm:
            names.append(nm)
    traits = _item_field(item, "traits") or []
    for tr in traits:
        nm = str(tr or "").strip()
        if nm:
            names.append(nm)
    return names


def _item_stats_detail_line(item: Any) -> str:
    stats: List[str] = []
    numeric = _item_stat_line(item)
    if numeric:
        stats.append(numeric)
    dur = _item_field(item, "dur")
    if dur is not None:
        stats.append(f"durability={dur}")
    uses = _item_field(item, "uses")
    if uses is not None:
        stats.append(f"uses={uses}")
    charges = _item_field(item, "charges_max")
    if charges is not None:
        stats.append(f"charges_max={charges}")
    dt = str(_item_field(item, "damagetype") or "").strip()
    if dt and dt.lower() != "physical":
        stats.append(f"damagetype={dt}")
    skills = _item_skill_names(item)
    if skills:
        stats.append("skills: " + ", ".join(sorted(set(skills), key=str.lower)))
    return "; ".join(stats)


def format_companion(c: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Name: {c['name']}")
    lines.append(f"Class: {c['class']['name']}")
    if c["class"].get("description"):
        lines.append(str(c["class"]["description"]).strip())

    lines.append("")
    lines.append(f"Background: {c['background']['name']}")
    if c["background"].get("description"):
        lines.append(str(c["background"]["description"]).strip())

    if c.get("traits"):
        lines.append("")
        lines.append("Traits:")
        for t in c["traits"]:
            desc = (t.get("description") or "").strip()
            lines.append(f"- {t['name']}: {desc}" if desc else f"- {t['name']}")

    lines.append("")
    lines.append("Stats:")
    for stat in STAT_KEYS:
        val = c["stats"].get(stat, 0)
        if stat in ("Fight", "Shoot", "Will") and val >= 0:
            lines.append(f"{stat} +{val}")
        else:
            lines.append(f"{stat} {val}")

    if c.get("skills"):
        lines.append("")
        lines.append("Skills:")
        for name, value in sorted(c["skills"].items(), key=lambda kv: kv[0].lower()):
            if value is None:
                lines.append(name)
            else:
                sign = "+" if value >= 0 else ""
                lines.append(f"{name} {sign}{value}")

    combined = list(c.get("spells", [])) + list(c.get("abilities", []))
    if combined:
        lines.append("")
        lines.append("Spells / Abilities:")
        for x in combined:
            lines.append(f"- {x}")

    if c.get("items"):
        lines.append("")
        lines.append("Equipment:")
        for it in c["items"]:
            name = _item_name(it)
            stat_line = _item_stat_line(it)
            if stat_line:
                lines.append(f"- {name} - {stat_line}")
            else:
                lines.append(f"- {name}")

    lines.append("")
    lines.append(f"Recruitment Cost: {c.get('rp', 0)} RP")
    return "\n".join(lines)


# ============================================================
# #LABEL: GENERATE MANY + GLOSSARY + DEBUG FOOTER
# ============================================================

def generate_companions(data: DataBundle, inputs: Dict[str, Any]) -> Tuple[str, str]:
    count = max(1, int(inputs.get("count", 1)))

    raw_tags = inputs.get("required_class_tags") or []
    required_class_tags = {str(t).strip().lower() for t in raw_tags if str(t).strip()} if raw_tags else None
    required_class_name = (inputs.get("required_class_name") or "").strip() or None

    allow_bg_trait_rolls = bool(inputs.get("allow_background_trait_rolls", True))
    merchant_level_raw = parse_int_maybe(inputs.get("merchant_level"), None)
    merchant_level = merchant_level_raw if merchant_level_raw and merchant_level_raw > 0 else None
    debug = bool(inputs.get("debug", False))

    used_names: Set[str] = set()
    companions: List[Dict[str, Any]] = []

    # For glossary grouping
    used_items: Dict[str, Dict[str, Any]] = {}
    used_skill_status: Set[str] = set()

    # For debug footer (global)
    debug_warnings: List[str] = []
    debug_rolls: List[str] = []

    for _ in range(count):
        c = generate_companion(
            data,
            used_names=used_names,
            required_class_tags=required_class_tags,
            required_class_name=required_class_name,
            allow_background_trait_rolls=allow_bg_trait_rolls,
            merchant_level=merchant_level,
            debug=debug,
        )
        companions.append(c)

        for it in c.get("items", []):
            name = _item_name(it)
            if name and name not in used_items:
                used_items[name] = it
            used_skill_status.update(_item_skill_names(it))

        used_skill_status.update(c.get("spells", []))
        used_skill_status.update(c.get("abilities", []))  # abilities share glossary grouping

        ctx = c.get("_ctx", {})
        debug_warnings.extend(ctx.get("warnings", []))
        debug_rolls.extend(ctx.get("debug_rolls", []))

    body = ("\n" + "-" * 40 + "\n").join(format_companion(c) for c in companions)

    spell_lookup = {e["name"]: e for e in data.spells}

    # --- Equipment Glossary ---
    gloss: List[str] = []
    gloss.append("")
    gloss.append("=" * 40)
    gloss.append("Equipment Glossary")
    gloss.append("=" * 40)

    for name in sorted(used_items, key=str.lower):
        item = used_items[name]
        base = _item_field(item, "base") or {}
        rarity = str(_item_field(item, "rarity") or base.get("rarity") or "common").strip().lower()
        cat = str(base.get("cat") or base.get("slot") or "item").strip().lower()
        desc = str(_item_field(item, "description") or base.get("description") or "").strip()
        gloss.append("")
        gloss.append(name)
        line_two = f"{rarity.title()} {cat}"
        if desc:
            line_two = f"{line_two} — {desc}"
        gloss.append(f"    {line_two}")
        stats_line = _item_stats_detail_line(item)
        if stats_line:
            gloss.append(f"    {stats_line}")

    # --- Skills & Status Glossary ---
    gloss.append("")
    gloss.append("Skills & Status Glossary")
    gloss.append("-" * 60)
    if used_skill_status:
        for name in sorted(used_skill_status, key=str.lower):
            entry = spell_lookup.get(name)
            desc = (entry.get("description") or "").strip() if entry else ""
            gloss.append(f"- {name}")
            if desc:
                gloss.append(f"  {desc}")
    else:
        gloss.append("- (none)")

    # --- Debug Footer ---
    dbg: List[str] = []
    dbg.append("")
    dbg.append("=" * 40)
    dbg.append("DEBUG")
    dbg.append("=" * 40)
    dbg.append("")
    dbg.append("Roll Diagnostics")
    if debug_rolls:
        for line in debug_rolls:
            dbg.append(f"- {line}")
    else:
        dbg.append("- (none)")

    dbg.append("")
    dbg.append("Warnings")
    if debug_warnings:
        for w in debug_warnings:
            dbg.append(f"- {w}")
    else:
        dbg.append("- (none)")

    text = body + "\n" + "\n".join(gloss) + "\n" + "\n".join(dbg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"companions_{stamp}.txt"
    return filename, text

