# rbtl_data.py

# ============================================================
# RBTL DEV RULES — IO + Architecture Guardrails (REFERENCE)
# ============================================================
# (Keep these as comment-only rules to prevent “mystery bugs.”)
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

# rbtl_data.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


# ============================================================
# Parse helpers (NON-IO)
# ============================================================

def parse_tags(tag_str: str) -> Set[str]:
    if not tag_str:
        return set()
    return {t.strip() for t in tag_str.split(",") if t.strip()}


def parse_int_maybe(s: Optional[str], default: int = 0) -> int:
    if s is None:
        return default
    s = str(s).strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        try:
            return int(s.replace("+", ""))
        except ValueError:
            return default


def parse_kv_chunks(chunks: List[str]) -> Dict[str, str]:
    """
    Pipe row chunks support:
      - key=value
      - roll:<directive>   (canonical, project-wide)

    Stores:
      entry["roll"] = "spells:tag=spells:count=2"
    If multiple roll:* chunks exist, they are concatenated with '; '.
    """
    out: Dict[str, str] = {}
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue

        # Canonical roll directive pipe chunks: roll:spells:tag=spells:count=2
        if c.lower().startswith("roll:"):
            v = c.split(":", 1)[1].strip()
            if v:
                out["roll"] = f"{out['roll']}; {v}" if out.get("roll") else v
            continue

        if "=" in c:
            k, v = c.split("=", 1)
            out[k.strip()] = v.strip()
            continue

    return out


def parse_directives(chunks: List[str]) -> List[str]:
    """
    Keeps non-kv chunks for systems that want them.
    Note: roll:* is handled by parse_kv_chunks.
    """
    directives: List[str] = []
    for c in chunks:
        c = (c or "").strip()
        if not c or c.startswith("#"):
            continue
        if "=" in c:
            continue
        if ":" in c and not c.lower().startswith("roll:"):
            directives.append(c)
    return directives


# ============================================================
# Settings (IO) + Settings accessors (for rbtl_core.py)
# ============================================================

def load_settings(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def s_get(settings: Dict[str, str], key: str, default: str = "") -> str:
    return settings.get(key, default)


def s_get_bool(settings: Dict[str, str], key: str, default: bool = False) -> bool:
    v = settings.get(key, "")
    if not v:
        return default
    v = v.strip().lower()
    if v in ("true", "on", "yes", "1"):
        return True
    if v in ("false", "off", "no", "0"):
        return False
    return default


def s_get_int(settings: Dict[str, str], key: str, default: int = 0) -> int:
    v = settings.get(key, "")
    try:
        return int(v)
    except Exception:
        return default


def s_get_float(settings: Dict[str, str], key: str, default: float = 1.0) -> float:
    v = settings.get(key, "")
    try:
        return float(v)
    except Exception:
        return default


def s_get_list(settings: Dict[str, str], key: str) -> List[str]:
    v = settings.get(key, "").strip()
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


# ============================================================
# Pipe-file loader (IO)
# ============================================================

def load_pipe_file(path: str) -> List[Dict[str, Any]]:
    """
    Format:
      id|Name|k=v|k=v|roll:<directive>|...

    Special:
      - tag=...    -> entry["tags"] set
      - threat=... -> entry["threats"] set
      - tier=...   -> entry["tier"]
    """
    entries: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return entries

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue

            entry_id = parts[0]
            name = parts[1]
            chunks = parts[2:] if len(parts) > 2 else []

            kv = parse_kv_chunks(chunks)
            directives = parse_directives(chunks)

            tags = parse_tags(kv.get("tag", ""))
            threats = parse_tags(kv.get("threat", ""))

            tier = (kv.get("tier") or "").strip().lower()
            if not tier:
                tier = "leader" if "leader" in tags else "minion"

            e: Dict[str, Any] = {
                "id": entry_id,
                "name": name,
                **kv,
                "tags": tags,
                "threats": threats,
                "directives": directives,
                "tier": tier,
            }
            entries.append(e)

    return entries


# ============================================================
# DataBundle
# ============================================================

@dataclass(frozen=True)
class DataPaths:
    project_root: str
    data_dir: str
    settings_path: str

    enemy_units_path: str
    traits_path: str
    events_path: str
    clues_path: str
    scenario_objectives_path: str
    items_path: str
    item_components_path: str
    spells_path: str

    # NEW: Rooms (delves/lairs)
    rooms_path: str

    companion_names_path: str
    companion_classes_path: str
    companion_backgrounds_path: str

    # Campaign generator lists
    biomes_path: str
    campaign_pressures_path: str
    campaign_threats_path: str
    settlement_types_path: str
    intro_start_path: str


@dataclass
class DataBundle:
    paths: DataPaths
    settings: Dict[str, str]

    enemy_units: List[Dict[str, Any]]
    traits: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    clues: List[Dict[str, Any]]
    scen_entries: List[Dict[str, Any]]
    items: List[Dict[str, Any]]
    item_components: List[Dict[str, Any]]
    spells: List[Dict[str, Any]]
    rooms: List[Dict[str, Any]]

    companion_names: List[Dict[str, Any]]
    companion_classes: List[Dict[str, Any]]
    companion_backgrounds: List[Dict[str, Any]]

    # Campaign generator lists
    biomes: List[Dict[str, Any]]
    campaign_pressures: List[Dict[str, Any]]
    campaign_threats: List[Dict[str, Any]]
    settlement_types: List[Dict[str, Any]]
    intro_starts: List[Dict[str, Any]]

    all_threats: List[str]


def gather_threats_from_units(enemy_units: List[Dict[str, Any]]) -> List[str]:
    threats: Set[str] = set()
    for u in enemy_units:
        threats |= (u.get("threats", set()) or set())
    return sorted(threats)


def build_paths(project_root: str, data_folder: str = "data") -> DataPaths:
    data_dir = os.path.join(project_root, data_folder)
    return DataPaths(
        project_root=project_root,
        data_dir=data_dir,
        settings_path=os.path.join(data_dir, "settings.txt"),

        enemy_units_path=os.path.join(data_dir, "enemy_units.txt"),
        traits_path=os.path.join(data_dir, "traits.txt"),
        events_path=os.path.join(data_dir, "events.txt"),
        clues_path=os.path.join(data_dir, "clues.txt"),
        scenario_objectives_path=os.path.join(data_dir, "scenario_objectives.txt"),
        items_path=os.path.join(data_dir, "items.txt"),
        item_components_path=os.path.join(data_dir, "item_components.txt"),
        spells_path=os.path.join(data_dir, "spells.txt"),

        rooms_path=os.path.join(data_dir, "rooms.txt"),

        companion_names_path=os.path.join(data_dir, "names.txt"),
        companion_classes_path=os.path.join(data_dir, "classes.txt"),
        companion_backgrounds_path=os.path.join(data_dir, "backgrounds.txt"),

        # Campaign generator lists
        biomes_path=os.path.join(data_dir, "biomes.txt"),
        campaign_pressures_path=os.path.join(data_dir, "campaign_pressures.txt"),
        campaign_threats_path=os.path.join(data_dir, "threats.txt"),
        settlement_types_path=os.path.join(data_dir, "settlement_types.txt"),
        intro_start_path=os.path.join(data_dir, "intro_start.txt"),
    )


def load_data_bundle(project_root: str, data_folder: str = "data") -> DataBundle:
    paths = build_paths(project_root, data_folder=data_folder)
    settings = load_settings(paths.settings_path)

    enemy_units = load_pipe_file(paths.enemy_units_path)
    traits = load_pipe_file(paths.traits_path)
    events = load_pipe_file(paths.events_path)
    clues = load_pipe_file(paths.clues_path)
    scen_entries = load_pipe_file(paths.scenario_objectives_path)
    items = load_pipe_file(paths.items_path)
    item_components = load_pipe_file(paths.item_components_path)
    spells = load_pipe_file(paths.spells_path)

    rooms = load_pipe_file(paths.rooms_path) if os.path.exists(paths.rooms_path) else []

    companion_names = load_pipe_file(paths.companion_names_path)
    companion_classes = load_pipe_file(paths.companion_classes_path)
    companion_backgrounds = load_pipe_file(paths.companion_backgrounds_path)

    # Campaign generator lists
    biomes = load_pipe_file(paths.biomes_path)
    campaign_pressures = load_pipe_file(paths.campaign_pressures_path)
    campaign_threats = load_pipe_file(paths.campaign_threats_path)
    settlement_types = load_pipe_file(paths.settlement_types_path)
    intro_starts = load_pipe_file(paths.intro_start_path)

    all_threats = gather_threats_from_units(enemy_units)

    return DataBundle(
        paths=paths,
        settings=settings,

        enemy_units=enemy_units,
        traits=traits,
        events=events,
        clues=clues,
        scen_entries=scen_entries,
        items=items,
        item_components=item_components,
        spells=spells,
        rooms=rooms,

        companion_names=companion_names,
        companion_classes=companion_classes,
        companion_backgrounds=companion_backgrounds,

        biomes=biomes,
        campaign_pressures=campaign_pressures,
        campaign_threats=campaign_threats,
        settlement_types=settlement_types,
        intro_starts=intro_starts,

        all_threats=all_threats,
    )
