# rbtl_cli.py
from typing import Any, Dict, List, Optional

from rbtl_data import s_get, s_get_bool, s_get_int
from rbtl_shared import (
    DIFFICULTY_ORDER,
    gather_threats_from_units,
    pick_scenario_type_id,
    pick_weighted_tag,
    roll_threat_pair,
    threat_pool_from_settings,
)

# Sentinel commands used by nav prompts
BACK = "__BACK__"
RESTART = "__RESTART__"
QUIT = "__QUIT__"


# ----------------------------
# Settings helpers (read-only)
# ----------------------------
def s_get_str(settings: Dict[str, str], key: str, default: str = "") -> str:
    """Return a stripped string for CLI prompts (wrapper around rbtl_data.s_get)."""
    return (s_get(settings, key, default) or "").strip()

def pick_class_filter_tag(data):
    # Collect tags from companion classes
    tag_to_classes = {}
    for c in data.companion_classes:
        tags = {t.lower() for t in (c.get("tags") or set()) if str(t).strip()}
        for t in tags:
            tag_to_classes.setdefault(t, []).append(c.get("name", "Unknown"))

    # Sort tags by how many classes they affect (descending), then alphabetically
    tag_rows = sorted(tag_to_classes.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    print("\nCustom: Class Filter")
    print("0) Any class")
    for i, (tag, cls_names) in enumerate(tag_rows, start=1):
        examples = ", ".join(cls_names[:3])
        extra = "" if len(cls_names) <= 3 else f" (+{len(cls_names)-3} more)"
        print(f"{i}) {tag}  ({len(cls_names)} classes: {examples}{extra})")

    choice = input("\nSelect a number: ").strip()
    try:
        idx = int(choice)
    except ValueError:
        return None  # treat as "Any"

    if idx <= 0:
        return None
    if idx > len(tag_rows):
        return None

    return tag_rows[idx - 1][0]  # the selected tag string



# ----------------------------
# DataBundle access (attr or dict)
# ----------------------------
def dget(data: Any, key: str, default: Any = None) -> Any:
    if hasattr(data, key):
        return getattr(data, key)
    if isinstance(data, dict):
        return data.get(key, default)
    return default


# ----------------------------
# CLI prompts
# ----------------------------
def prompt_choice_nav(title: str, options: List[str], default_idx: int = 0) -> str:
    """
    Choice prompt with:
      - Enter to accept default
      - b=back r=restart q=quit
      - numeric selection
      - exact/prefix/contains name fallback
    """
    print(f"\n{title}")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")

    if not options:
        return QUIT

    default_idx = 0 if default_idx < 0 or default_idx >= len(options) else default_idx
    default_label = options[default_idx]

    raw = input(f"Select [Enter={default_label}] (b=back r=restart q=quit): ").strip()
    low = raw.lower()

    if low in ("b", "back"):
        return BACK
    if low in ("r", "restart", "home"):
        return RESTART
    if low in ("q", "quit", "exit"):
        return QUIT
    if not raw:
        return default_label

    # numeric
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx]
    except Exception:
        pass

    # exact (case-insensitive)
    exact_map = {opt.lower(): opt for opt in options}
    if low in exact_map:
        return exact_map[low]

    # unique prefix
    prefix = [opt for opt in options if opt.lower().startswith(low)]
    if len(prefix) == 1:
        return prefix[0]

    # unique contains
    contains = [opt for opt in options if low in opt.lower()]
    if len(contains) == 1:
        return contains[0]

    return default_label


def prompt_int_nav(title: str, default: int) -> Any:
    raw = input(f"(b=back r=restart q=quit) [Enter={default}]    {title}: ").strip().lower()

    if raw in ("b", "back"):
        return BACK
    if raw in ("r", "restart", "home"):
        return RESTART
    if raw in ("q", "quit", "exit"):
        return QUIT
    if not raw:
        return default

    try:
        return int(raw)
    except Exception:
        return default


def _idx_or_zero(options: List[str], value: str) -> int:
    try:
        return options.index(value)
    except Exception:
        return 0


# ----------------------------
# Scenario/objective filtering
# ----------------------------
def objectives_allowed_for_scenario(picked_scen: Dict[str, Any], objectives_all: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scenario row: allow_obj_IDs=eliminate,leader,ritual
    Objective rows: ID=leader, etc.
    If allow_obj_IDs blank/missing => allow all.
    """
    allow_raw = (picked_scen.get("allow_obj_IDs") or "").strip()
    if not allow_raw:
        return objectives_all[:]
    allowed_ids = {x.strip() for x in allow_raw.split(",") if x.strip()}
    return [o for o in objectives_all if (o.get("ID") or "").strip() in allowed_ids]


# ----------------------------
# Mode gatherers
# ----------------------------
def gather_inputs_now(data: Any) -> Dict[str, Any]:
    settings: Dict[str, str] = dget(data, "settings", {}) or {}
    units: List[Dict[str, Any]] = dget(data, "enemy_units", []) or []
    scen_entries: List[Dict[str, Any]] = dget(data, "scenario_objectives", []) or dget(data, "scen_entries", []) or []

    inputs: Dict[str, Any] = {"mode": "Now"}

    # NOW: do NOT randomize these — always defaults from settings
    inputs["players"] = s_get_int(settings, "default.players", 2)
    inputs["allied_combatants"] = s_get_int(settings, "default.allied_combatants", 2)

    d = s_get_str(settings, "default.difficulty", "Normal")
    inputs["difficulty"] = d if d in DIFFICULTY_ORDER else "Normal"

    # Scenario type
    if s_get_bool(settings, "randomize.scenario_types", True):
        st_id = pick_scenario_type_id(scen_entries, settings)
    else:
        st_id = s_get_str(settings, "default.scenario_type_id", "")
        if not st_id or st_id.lower() == "random":
            st_id = pick_scenario_type_id(scen_entries, settings)

    inputs["scenario_type_id"] = st_id or ""
    inputs["encounter_kind"] = "Defensive Battle" if (st_id or "").strip().lower() == "defensive" else "Scenario"

    # Threats
    all_threats = gather_threats_from_units(units)
    if s_get_bool(settings, "randomize.threats", True):
        t1, t2 = roll_threat_pair(settings, all_threats, mode="now")
    else:
        weighted_pool = threat_pool_from_settings(all_threats, settings)
        t1 = s_get_str(settings, "default.threat_tag_1", "none")
        if t1.lower() == "random":
            t1 = pick_weighted_tag(weighted_pool, exclude=set()) or "none"
        t2 = s_get_str(settings, "default.threat_tag_2", "none")
        if t2.lower() == "random":
            t2 = pick_weighted_tag(weighted_pool, exclude={t1}) or "none"

    inputs["threat_tag_1"] = t1
    inputs["threat_tag_2"] = t2

    # Objective + leadership
    inputs["objective"] = "Random" if s_get_bool(settings, "randomize.objectives", True) else s_get_str(settings, "default.objective", "Random")
    inputs["leadership_tier"] = s_get_str(settings, "default.leadership_tier", "Random")

    return inputs


def gather_inputs_quick(data: Any) -> Dict[str, Any]:
    settings: Dict[str, str] = dget(data, "settings", {}) or {}
    units: List[Dict[str, Any]] = dget(data, "enemy_units", []) or []
    scen_entries: List[Dict[str, Any]] = dget(data, "scenario_objectives", []) or dget(data, "scen_entries", []) or []

    inputs: Dict[str, Any] = {"mode": "Quick"}

    # QUICK: prompt players + difficulty only (allies from settings)
    inputs["players"] = int(prompt_int_nav("Players (Rangers)", s_get_int(settings, "default.players", 2)))
    inputs["allied_combatants"] = s_get_int(settings, "default.allied_combatants", 2)

    default_diff = s_get_str(settings, "default.difficulty", "Normal")
    diff = prompt_choice_nav("Difficulty:", DIFFICULTY_ORDER, default_idx=_idx_or_zero(DIFFICULTY_ORDER, default_diff if default_diff in DIFFICULTY_ORDER else "Normal"))
    if diff in (BACK, RESTART, QUIT):
        raise SystemExit(0)
    inputs["difficulty"] = diff

    # Scenario type
    if s_get_bool(settings, "randomize.scenario_types", True):
        st_id = pick_scenario_type_id(scen_entries, settings)
    else:
        st_id = s_get_str(settings, "default.scenario_type_id", "")
        if not st_id or st_id.lower() == "random":
            st_id = pick_scenario_type_id(scen_entries, settings)

    inputs["scenario_type_id"] = st_id or ""
    inputs["encounter_kind"] = "Defensive Battle" if (st_id or "").strip().lower() == "defensive" else "Scenario"

    # Threats
    all_threats = gather_threats_from_units(units)
    if s_get_bool(settings, "randomize.threats", True):
        t1, t2 = roll_threat_pair(settings, all_threats, mode="quick")
    else:
        weighted_pool = threat_pool_from_settings(all_threats, settings)
        t1 = s_get_str(settings, "default.threat_tag_1", "none")
        if t1.lower() == "random":
            t1 = pick_weighted_tag(weighted_pool, exclude=set()) or "none"
        t2 = s_get_str(settings, "default.threat_tag_2", "none")
        if t2.lower() == "random":
            t2 = pick_weighted_tag(weighted_pool, exclude={t1}) or "none"

    inputs["threat_tag_1"] = t1
    inputs["threat_tag_2"] = t2

    # Objective + leadership
    inputs["objective"] = "Random" if s_get_bool(settings, "randomize.objectives", True) else s_get_str(settings, "default.objective", "Random")
    inputs["leadership_tier"] = s_get_str(settings, "default.leadership_tier", "Random")

    return inputs


def gather_inputs_custom(data: Any) -> Dict[str, Any]:
    settings: Dict[str, str] = dget(data, "settings", {}) or {}
    units: List[Dict[str, Any]] = dget(data, "enemy_units", []) or []
    scen_entries: List[Dict[str, Any]] = dget(data, "scenario_objectives", []) or dget(data, "scen_entries", []) or []

    inputs: Dict[str, Any] = {"mode": "Custom"}

    scenario_types = [e for e in scen_entries if "scenariotype" in (e.get("tags", set()) or set())]
    objectives_all = [e for e in scen_entries if "objective" in (e.get("tags", set()) or set())]

    # Threat display pool
    all_threats = gather_threats_from_units(units)
    weighted_pool = threat_pool_from_settings(all_threats, settings)
    display_threats = [t for (t, w) in weighted_pool if w > 0] or all_threats[:]

    steps = ["scenario", "players", "allies", "difficulty", "threats", "objective", "leadership"]
    i = 0

    picked_scen: Optional[Dict[str, Any]] = None
    picked_obj: Optional[Dict[str, Any]] = None

    while 0 <= i < len(steps):
        step = steps[i]

        # ---- Scenario type ----
        if step == "scenario":
            st_names = [e["name"] for e in scenario_types]
            cur = (inputs.get("scenario_type_name") or "").strip()
            pick = prompt_choice_nav("Scenario Type:", st_names, default_idx=_idx_or_zero(st_names, cur))

            if pick == BACK:
                i = max(0, i - 1); continue
            if pick == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if pick == QUIT:
                raise SystemExit(0)

            picked_scen = next(e for e in scenario_types if e["name"] == pick)
            inputs["scenario_type_id"] = (picked_scen.get("ID") or "").strip()
            inputs["scenario_type_name"] = picked_scen["name"]
            inputs["encounter_kind"] = "Defensive Battle" if inputs["scenario_type_id"].lower() == "defensive" else "Scenario"

            picked_obj = None
            i += 1
            continue

        # ---- Players ----
        if step == "players":
            cur = int(inputs.get("players", s_get_int(settings, "default.players", 2)))
            val = prompt_int_nav("Players (Rangers)", cur)

            if val == BACK:
                i = max(0, i - 1); continue
            if val == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if val == QUIT:
                raise SystemExit(0)

            inputs["players"] = int(val)
            i += 1
            continue

        # ---- Allies ----
        if step == "allies":
            cur = int(inputs.get("allied_combatants", s_get_int(settings, "default.allied_combatants", 2)))
            val = prompt_int_nav("Allied combatants", cur)

            if val == BACK:
                i = max(0, i - 1); continue
            if val == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if val == QUIT:
                raise SystemExit(0)

            inputs["allied_combatants"] = int(val)
            i += 1
            continue

        # ---- Difficulty ----
        if step == "difficulty":
            cur = (inputs.get("difficulty") or s_get_str(settings, "default.difficulty", "Normal")).strip()
            if cur not in DIFFICULTY_ORDER:
                cur = "Normal"
            pick = prompt_choice_nav("Difficulty:", DIFFICULTY_ORDER, default_idx=_idx_or_zero(DIFFICULTY_ORDER, cur))

            if pick == BACK:
                i = max(0, i - 1); continue
            if pick == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if pick == QUIT:
                raise SystemExit(0)

            inputs["difficulty"] = pick
            i += 1
            continue

        # ---- Threats ----
        if step == "threats":
            scen_id = (inputs.get("scenario_type_id") or "").strip().lower()
            if scen_id == "lair":
                # Monster lairs always use the monsters pool; do not ask for threats.
                inputs["threat_tag_1"] = "monsters"
                inputs["threat_tag_2"] = "none"
                i += 1
                continue
            t1_options = ["Random"] + display_threats
            cur1 = (inputs.get("threat_tag_1") or s_get_str(settings, "default.threat_tag_1", "Random")).strip()
            pick1 = prompt_choice_nav("Threat 1 (required):", t1_options, default_idx=_idx_or_zero(t1_options, cur1))

            if pick1 == BACK:
                i = max(0, i - 1); continue
            if pick1 == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if pick1 == QUIT:
                raise SystemExit(0)

            if pick1 == "Random":
                t1 = pick_weighted_tag(weighted_pool, exclude=set()) or "none"
            else:
                t1 = pick1

            if not t1 or t1.lower() == "none":
                print("Threat 1 cannot be 'none' in Custom. Pick a threat.")
                continue

            t2_options = ["none", "Random"] + [t for t in display_threats if t != t1]
            cur2 = (inputs.get("threat_tag_2") or s_get_str(settings, "default.threat_tag_2", "none")).strip()
            pick2 = prompt_choice_nav("Threat 2 (optional):", t2_options, default_idx=_idx_or_zero(t2_options, cur2))

            if pick2 == BACK:
                i = max(0, i - 1); continue
            if pick2 == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if pick2 == QUIT:
                raise SystemExit(0)

            if pick2 == "Random":
                t2 = pick_weighted_tag(weighted_pool, exclude={t1}) or "none"
            else:
                t2 = pick2

            inputs["threat_tag_1"] = t1
            inputs["threat_tag_2"] = t2

            picked_obj = None
            i += 1
            continue

        # ---- Objective (filtered by scenario hard rules) ----
        if step == "objective":
            if not picked_scen:
                i = 0
                continue

            allowed = objectives_allowed_for_scenario(picked_scen, objectives_all)
            if not allowed:
                print("No objectives allowed for this scenario type. Fix allow_obj_IDs in scenario_objectives.txt.")
                i = 0
                continue

            obj_names = [o["name"] for o in allowed]
            cur = (inputs.get("objective") or "").strip()
            pick = prompt_choice_nav("Objective:", obj_names, default_idx=_idx_or_zero(obj_names, cur))

            if pick == BACK:
                i = max(0, i - 1); continue
            if pick == RESTART:
                inputs = {"mode": "Custom"}; picked_scen = None; picked_obj = None; i = 0; continue
            if pick == QUIT:
                raise SystemExit(0)

            picked_obj = next(o for o in allowed if o["name"] == pick)
            inputs["objective"] = picked_obj["name"]
            i += 1
            continue

        # ---- Leadership (with minlead logic + skip for BBEG scenario) ----
        if step == "leadership":
            # Leadership tier is forced to Boss for rooms-based scenarios (and allowed elsewhere).
            inputs["leadership_tier"] = "Boss"
            i += 1
            continue

    # Finished walking the wizard steps
    return inputs



# ----------------------------
# Public entry point
# ----------------------------
def run_cli(data: Any) -> Dict[str, Any]:
    """Return an inputs dict for generate_scenario()."""
    settings = dget(data, "settings", {}) or {}

    # Choose top-level mode
    mode_opts = ["Now", "Quick", "Custom"]
    pick = prompt_choice_nav("Mode:", mode_opts, default_idx=0)
    if pick == QUIT:
        raise SystemExit(0)
    if pick == BACK or pick == RESTART:
        # treat as Now
        pick = "Now"

    if pick == "Now":
        inputs = gather_inputs_now(data)
    elif pick == "Quick":
        inputs = gather_inputs_quick(data)
    else:
        inputs = gather_inputs_custom(data)

    # If the user backed out/cancelled inside a gather_inputs_* flow, exit cleanly.
    # Some gather functions return None/QUIT sentinels rather than a dict.
    if not isinstance(inputs, dict):
        raise SystemExit(0)

    # Hard enforcement: rooms-based scenarios always use Boss leadership
    scen_id = (inputs.get("scenario_type_id") or "").strip().lower()
    if scen_id in ("lair", "site", "delve"):
        inputs["leadership_tier"] = "Boss"

    # Hard enforcement: lair uses monsters pool and no threats
    if scen_id == "lair":
        inputs["threat_tag_1"] = "monsters"
        inputs["threat_tag_2"] = "none"

    return inputs
