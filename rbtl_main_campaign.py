# rbtl_main_campaign.py
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import rbtl_campaign as rbtl_campaign
print("Using rbtl_campaign from:", rbtl_campaign.__file__)

from rbtl_cli import BACK, QUIT, RESTART, _parse_campaign_key, _quest_board_entries, prompt_choice_nav, prompt_int_nav
from rbtl_campaign import (
    adjusted_threat_count,
    available_threat_tags_from_pool,
    eligible_threat_candidates,
    generate_campaign,
    roll_campaign_context,
    threat_weight_with_settings,
    violates_not,
)
from rbtl_companions import generate_companions
from rbtl_core import generate_scenario
from rbtl_data import load_data_bundle
from rbtl_loot import generate_shop


def write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _write_in_dir(out_dir: str, filename: str, text: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _find_threat_by_id(threats: List[Dict[str, Any]], tid: str) -> Optional[Dict[str, Any]]:
    for t in threats or []:
        if str(t.get("id", "")).strip() == str(tid).strip():
            return t
    return None


def _auto_roll_context_and_threats(data: Any, players: int, difficulty: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    footnotes: List[str] = []
    biomes = list(getattr(data, "biomes", []) or [])
    pressures = list(getattr(data, "campaign_pressures", []) or [])
    threats = list(getattr(data, "campaign_threats", []) or [])
    settings = getattr(data, "settings", {}) or {}

    biome, main_pressure, sub_pressure, context_tags, forbidden_tokens = roll_campaign_context(
        biomes=biomes,
        pressures=pressures,
        footnotes=footnotes,
    )

    max_threats = adjusted_threat_count(int(players), str(difficulty).lower())
    used_ids: set[str] = set()
    picked: List[Dict[str, Any]] = []

    for idx in range(1, max_threats + 1):
        is_main_slot = (idx == 1)
        pool = eligible_threat_candidates(
            threats,
            is_main_slot=is_main_slot,
            used_ids=used_ids,
            context_tags=context_tags,
            forbidden_tokens=forbidden_tokens,
            settings=settings,
            required_tag=None,
        )
        if not pool:
            break
        weights = [threat_weight_with_settings(t, settings, context_tags) for t in pool]
        chosen = random.choices(pool, weights=weights, k=1)[0]
        used_ids.add(str(chosen.get("id")))
        picked.append(chosen)

    return {
        "biome": biome,
        "main_pressure": main_pressure,
        "sub_pressure": sub_pressure,
        "context_tags": context_tags,
        "forbidden_tokens": forbidden_tokens,
    }, picked, {"footnotes": footnotes}


def _campaign_folder_name(key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.=-]+", "_", key.strip())
    return cleaned or "campaign"


def _extract_campaign_key(campaign_text: str) -> str:
    m = re.search(r"^Campaign Key:\s*(.+)$", campaign_text, flags=re.MULTILINE)
    return (m.group(1).strip() if m else "")


def _extract_settlements(campaign_text: str) -> List[Dict[str, str]]:
    lines = campaign_text.splitlines()
    out: List[Dict[str, str]] = []
    for i, line in enumerate(lines):
        if not line.startswith("Category:"):
            continue
        m = re.search(r"Category:\s*([a-zA-Z]+)\s*@\s*([A-Z]\d+)", line)
        if not m:
            continue
        out.append({"type": m.group(1).strip().lower(), "coord": m.group(2).strip().upper(), "idx": str(len(out) + 1)})
    return out


def _render_questboard_text(entries: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("Quest Board")
    lines.append("-" * 60)
    for idx, entry in enumerate(entries, start=1):
        title = str(entry.get("entry_name") or entry.get("objective_name") or f"Quest {idx}")
        scen = str(entry.get("scenario_type_name") or "Scenario")
        obj = str(entry.get("objective_name") or "Objective")
        enemy = str(entry.get("enemy_types") or "unknown enemies")
        flavor = str(entry.get("flavor") or "").strip()
        lines.append(f"{idx}. {title}")
        lines.append(f"   {scen} - {obj} ({enemy})")
        if flavor:
            lines.append(f"   {flavor}")
    lines.append("")
    return "\n".join(lines)


def _inject_questboard(campaign_text: str, entries: List[Dict[str, Any]]) -> str:
    block = _render_questboard_text(entries)
    marker = "\nFootnotes\n"
    pos = campaign_text.find(marker)
    if pos == -1:
        if not campaign_text.endswith("\n"):
            campaign_text += "\n"
        return campaign_text + "\n" + block
    return campaign_text[:pos] + "\n" + block + campaign_text[pos:]


def _companion_count_for_settlement(settlement_type: str) -> int:
    return {
        "hamlet": 4,
        "village": 6,
        "town": 8,
        "city": 10,
    }.get(str(settlement_type).strip().lower(), 6)


def _rare_count_for_settlement(settlement_type: str) -> int:
    return {
        "hamlet": 3,
        "village": 8,
        "town": 14,
        "city": 20,
    }.get(str(settlement_type).strip().lower(), 5)


def _merchant_level_for_settlement(settlement_type: str) -> int:
    return {
        "hamlet": 1,
        "village": 2,
        "town": 3,
        "city": 3,
    }.get(str(settlement_type).strip().lower(), 2)


def _run_complete_campaign(project_root: str, data: Any) -> None:
    players = prompt_int_nav("Number of players", default=2)
    if players in (BACK, RESTART, QUIT):
        return

    diff = prompt_choice_nav(
        "Difficulty",
        ["Easy", "Normal", "Hard", "Brutal"],
        default_idx=1,
    )
    if diff in (BACK, RESTART, QUIT):
        return

    players_i = int(players)
    diff_s = str(diff).lower()

    while True:
        rolled, threats, _ = _auto_roll_context_and_threats(data, players_i, diff_s)
        main_pressure = rolled["main_pressure"]
        sub_pressure = rolled["sub_pressure"]

        print("\n--- Complete Campaign Roll ---")
        print(f"Main Pressure: {main_pressure.get('name', '(unknown)')}")
        print(f"Sub Pressure: {sub_pressure.get('name', '(unknown)')}")
        if threats:
            print("Enemy Types:")
            for idx, t in enumerate(threats, start=1):
                print(f"  Threat {idx}: {t.get('name', 'Unknown')} ({', '.join(sorted(t.get('tags', set()) or set()))})")
        else:
            print("Enemy Types: (none)")
        print("------------------------------\n")

        keep = prompt_choice_nav("Keep this complete-campaign roll?", ["Accept", "Reroll"], default_idx=0)
        if keep in (BACK, RESTART, QUIT):
            return
        if keep == "Reroll":
            continue

        locked_context = {
            "biome_id": str(rolled["biome"].get("id") or ""),
            "main_pressure_id": str(main_pressure.get("id") or ""),
            "sub_pressure_id": str(sub_pressure.get("id") or ""),
        }
        locked_threat_ids = [str(t.get("id") or "") for t in threats if str(t.get("id") or "").strip()]

        campaign_filename, campaign_text = generate_campaign(
            data,
            {
                "players": players_i,
                "difficulty": diff_s,
                "locked_context": locked_context,
                "locked_threat_ids": locked_threat_ids,
            },
        )

        campaign_key = _extract_campaign_key(campaign_text)
        if not campaign_key:
            path = write_output(project_root, campaign_filename, campaign_text)
            print(f"\nWrote: {path}")
            print("Could not find Campaign Key in output; skipping complete-campaign expansion.\n")
            return

        entries = _quest_board_entries(data, campaign_key) or []
        campaign_text = _inject_questboard(campaign_text, entries)

        out_dir = os.path.join(project_root, "output", _campaign_folder_name(campaign_key))
        os.makedirs(out_dir, exist_ok=True)

        campaign_path = _write_in_dir(out_dir, "campaign.txt", campaign_text)
        print(f"\nWrote: {campaign_path}")

        parsed = _parse_campaign_key(campaign_key) or {}
        biome_id = parsed.get("biome_id", "")
        allied_combatants = players_i * 2

        for idx, entry in enumerate(entries, start=1):
            scenario_inputs: Dict[str, Any] = {
                "mode": "Questboard",
                "campaign_key": campaign_key,
                "players": players_i,
                "allied_combatants": allied_combatants,
                "difficulty": str(diff).title(),
                "quest_board_entry": entry,
                "scenario_type_id": entry.get("scenario_type_id", ""),
                "scenario_type_name": entry.get("scenario_type_name", ""),
                "objective": entry.get("objective_name", ""),
                "threat_tag_1": entry.get("threat_tag_1", "none"),
                "threat_tag_2": entry.get("threat_tag_2", "none"),
                "encounter_kind": "Defensive Battle" if str(entry.get("scenario_type_id", "")).lower() == "defensive" else "Scenario",
            }
            if biome_id:
                scenario_inputs["biome_id"] = biome_id

            _, scen_text = generate_scenario(data, scenario_inputs)
            scen_name = f"scenario_{idx}.txt"
            _write_in_dir(out_dir, scen_name, scen_text)

        settlements = _extract_settlements(campaign_text)
        for s in settlements:
            coord = s["coord"]
            stype = s["type"]

            _, shop_text = generate_shop(
                data,
                {
                    "settlement": stype,
                    "auto_build": False,
                    "merchant_level": _merchant_level_for_settlement(stype),
                    "random_count": _rare_count_for_settlement(stype),
                },
            )
            _write_in_dir(out_dir, f"{coord}-shop.txt", shop_text)

            _, companions_text = generate_companions(
                data,
                {
                    "count": _companion_count_for_settlement(stype),
                    "merchant_level": _merchant_level_for_settlement(stype),
                    "allow_background_trait_rolls": True,
                },
            )
            _write_in_dir(out_dir, f"{coord}-companions.txt", companions_text)

        print(f"Wrote complete campaign package in: {out_dir}\n")
        return


def run():
    project_root = os.getcwd()
    data = load_data_bundle(project_root)

    while True:
        mode = prompt_choice_nav(
            "Campaign Generator",
            ["Quick", "Custom", "Complete Campaign", "Quit"],
            default_idx=0,
        )

        if mode in (QUIT, "Quit"):
            return
        if mode == RESTART:
            continue

        if mode == "Complete Campaign":
            _run_complete_campaign(project_root, data)
            continue

        players = prompt_int_nav("Number of players", default=2)
        if players in (BACK, RESTART, QUIT):
            if players == BACK:
                continue
            if players == RESTART:
                continue
            return

        diff = prompt_choice_nav(
            "Difficulty",
            ["Easy", "Normal", "Hard", "Brutal"],
            default_idx=1,
        )
        if diff in (BACK, RESTART, QUIT):
            if diff == BACK:
                continue
            if diff == RESTART:
                continue
            return

        inputs = {
            "players": int(players),
            "difficulty": str(diff).lower(),
        }

        if mode.startswith("Custom"):
            # Custom threat picking.
            # NOTE: We let the user choose threats first (full tag list), then roll biome+pressures afterwards.
            # This avoids the "context tags" shrinking the menu before the player makes any choices.
            footnotes = []
            biomes = list(getattr(data, "biomes", []) or [])
            pressures = list(getattr(data, "campaign_pressures", []) or [])
            threats = list(getattr(data, "campaign_threats", []) or [])
            settings = getattr(data, "settings", {}) or {}

            max_threats = adjusted_threat_count(int(players), str(diff).lower())
            used_ids: set[str] = set()
            locked_threat_ids: list[str] = []

            restart_flow = False

            # Threat 1 is the Main Threat; Threat 2+ are Secondary.
            # We do NOT pass context_tags/forbidden_tokens here (they will be rolled later).
            empty_tags: set[str] = set()
            empty_forbidden: set[str] = set()

            for idx in range(1, max_threats + 1):
                is_main_slot = (idx == 1)

                # Pool without any required tag (for menu pruning)
                base_pool = eligible_threat_candidates(
                    threats,
                    is_main_slot=is_main_slot,
                    used_ids=used_ids,
                    context_tags=empty_tags,
                    forbidden_tokens=empty_forbidden,
                    settings=settings,
                    required_tag=None,
                )

                if not base_pool:
                    print("\nNo eligible threats remain for this slot (check roles/settings).\n")
                    break

                tags = available_threat_tags_from_pool(base_pool)

                # Menu: Random, then tags, then (optional) None
                menu = ["Random"] + tags
                if idx > 2:
                    menu.append("None")

                while True:
                    choice = prompt_choice_nav(
                        f"Threat {idx} filter (Main)" if is_main_slot else f"Threat {idx} filter",
                        menu,
                        default_idx=0,
                    )

                    if choice in (BACK, RESTART, QUIT):
                        if choice == QUIT:
                            return
                        restart_flow = True
                        break

                    if choice == "None":
                        break

                    required_tag = None if choice == "Random" else str(choice).lower()

                    pool = eligible_threat_candidates(
                        threats,
                        is_main_slot=is_main_slot,
                        used_ids=used_ids,
                        context_tags=empty_tags,
                        forbidden_tokens=empty_forbidden,
                        settings=settings,
                        required_tag=required_tag,
                    )

                    if not pool:
                        print("\nNo eligible threats match that filter. Try a different tag.\n")
                        continue

                    # With no context yet, weights are just rarity/settings (no overlap boost).
                    weights = [threat_weight_with_settings(t, settings, empty_tags) for t in pool]
                    picked = random.choices(pool, weights=weights, k=1)[0]

                    used_ids.add(str(picked.get("id")))
                    locked_threat_ids.append(str(picked.get("id")))
                    break

                if restart_flow:
                    break

                if choice == "None":
                    break

            if restart_flow:
                continue

            # Roll biome + pressures AFTER threats are chosen.
            # We keep the "hard roll / reroll" experience here, without shrinking the earlier menus.
            #
            # - If the rolled context "forbids" a locked threat (via not=...), we auto-reroll.
            # - If the rolled context has tags that don't overlap the locked threats, we WARN and let you reroll.
            max_attempts = 50
            attempt = 0

            while True:
                biome, main_pressure, sub_pressure, context_tags, forbidden_tokens = roll_campaign_context(
                    biomes=biomes,
                    pressures=pressures,
                    footnotes=footnotes,
                )

                # Hard conflict: any locked threat violates forbidden tokens.
                conflicts: list[str] = []

                for tid in locked_threat_ids:
                    t = _find_threat_by_id(threats, tid)
                    if not t:
                        continue
                    if forbidden_tokens and violates_not(t, forbidden_tokens):
                        conflicts.append(tid)

                if conflicts:
                    attempt += 1
                    if attempt >= max_attempts:
                        print("\nWARNING: Could not roll a context that avoids all 'not=' conflicts. Keeping the latest roll.\n")
                        break
                    continue  # auto reroll (hard rule)

                # Show the rolled context and let the player keep or reroll.
                print("\n--- Rolled Campaign Context ---")
                print(f"Biome: {biome.get('name','(unknown)')}")
                print(f"Main Pressure: {main_pressure.get('name','(unknown)')}")
                print(f"Sub Pressure: {sub_pressure.get('name','(unknown)')}")
                print("------------------------------\n")

                choice2 = prompt_choice_nav(
                    "Keep this campaign context?",
                    ["Accept", "Reroll"],
                    default_idx=0,
                )

                if choice2 in (BACK, RESTART, QUIT):
                    if choice2 == QUIT:
                        return
                    restart_flow = True
                    break

                if choice2 == "Reroll":
                    attempt += 1
                    if attempt >= max_attempts:
                        print("\nWARNING: Hit reroll limit; keeping the latest roll.\n")
                        break
                    continue

                break

            if restart_flow:
                continue

            # Lock the rolled context + chosen threats into the generator so output matches what we picked.
            inputs["locked_context"] = {
                "biome_id": str(biome.get("id")),
                "main_pressure_id": str(main_pressure.get("id")),
                "sub_pressure_id": str(sub_pressure.get("id")),
            }
            inputs["locked_threat_ids"] = locked_threat_ids

        filename, text = generate_campaign(data, inputs)
        path = write_output(project_root, filename, text)
        print(f"\nWrote: {path}\n")


if __name__ == "__main__":
    run()
