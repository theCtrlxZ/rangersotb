# rbtl_main_campaign.py
import os
import random

import rbtl_campaign as rbtl_campaign
print("Using rbtl_campaign from:", rbtl_campaign.__file__)

from rbtl_cli import BACK, QUIT, RESTART, prompt_choice_nav, prompt_int_nav
from rbtl_campaign import (
    adjusted_threat_count,
    available_threat_tags_from_pool,
    eligible_threat_candidates,
    generate_campaign,
    roll_campaign_context,
    threat_weight_with_settings,
    violates_not,
)
from rbtl_data import load_data_bundle




def write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run():
    project_root = os.getcwd()
    data = load_data_bundle(project_root)

    while True:
        mode = prompt_choice_nav(
            "Campaign Generator",
            ["Quick", "Custom", "Quit"],
            default_idx=0,
        )

        if mode in (QUIT, "Quit"):
            return
        if mode == RESTART:
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

            def _find_threat_by_id(tid: str):
                tid = str(tid).strip()
                for t in threats or []:
                    if str(t.get("id", "")).strip() == tid:
                        return t
                return None

            while True:
                biome, main_pressure, sub_pressure, context_tags, forbidden_tokens = roll_campaign_context(
                    biomes=biomes,
                    pressures=pressures,
                    footnotes=footnotes,
                )

                # Hard conflict: any locked threat violates forbidden tokens.
                conflicts: list[str] = []

                for tid in locked_threat_ids:
                    t = _find_threat_by_id(tid)
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
