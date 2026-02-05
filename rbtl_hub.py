# rbtl_hub.py
"""
Rangers at the Borderlands — Hub Launcher

Purpose:
- Single entry-point that lets you launch any generator from one menu.

Notes:
- Uses rbtl_cli prompt style (b=back, r=restart, q=quit).
- Runs each generator in the project root directory (folder containing this file),
  so relative paths to /data and /output behave consistently.
"""

from __future__ import annotations

import contextlib
import os
import traceback
from typing import Optional

from rbtl_cli import BACK, QUIT, RESTART, prompt_choice_nav
from rbtl_data import DataBundle, load_data_bundle


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def _pushd(path: str):
    """Temporarily chdir (so modules using os.getcwd() behave consistently)."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _safe_print_exception(prefix: str = "Error"):
    print(f"\n{prefix}:\n{traceback.format_exc()}\n")


def _load_data(project_root: str) -> DataBundle:
    return load_data_bundle(project_root)


def _write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, filename)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)
    return outpath


def run_scenario(data: DataBundle, project_root: str) -> None:
    """Scenario / Encounter generator (uses shared DataBundle)."""
    from rbtl_cli import run_cli
    from rbtl_core import generate_scenario

    try:
        inputs = run_cli(data)  # may raise SystemExit on quit
    except SystemExit:
        return

    result = generate_scenario(data, inputs)

    outpath = None
    if isinstance(result, tuple) and len(result) == 2:
        filename, briefing_text = result
        outpath = _write_output(project_root, filename, briefing_text)
    elif isinstance(result, str):
        outpath = result
    else:
        raise RuntimeError(f"Unexpected return from generate_scenario: {type(result)}")

    print(f"\nWrote briefing: {outpath}\n")


def run_campaign(project_root: str) -> None:
    """Delegates to existing campaign main (keeps its own UX)."""
    import rbtl_main_campaign

    rbtl_main_campaign.run()


def run_companions(project_root: str) -> None:
    """Delegates to existing companions main (keeps its own UX)."""
    import rbtl_main_companions

    rbtl_main_companions.run()


def run_loot_shop(project_root: str) -> None:
    """Delegates to existing loot/shop main (keeps its own UX)."""
    import rbtl_main_loot

    rbtl_main_loot.run()


def _data_summary(data: Optional[DataBundle]) -> str:
    if not data:
        return "Data: (not loaded)"
    try:
        return (
            "Data loaded: "
            f"units={len(getattr(data, 'enemy_units', []) or [])}, "
            f"events={len(getattr(data, 'events', []) or [])}, "
            f"items={len(getattr(data, 'items', []) or [])}, "
            f"spells={len(getattr(data, 'spells', []) or [])}"
        )
    except Exception:
        return "Data: (loaded, summary unavailable)"


def main() -> None:
    # Always treat the folder containing this file as the root.
    project_root = PROJECT_ROOT

    data: Optional[DataBundle] = None
    with _pushd(project_root):
        try:
            data = _load_data(project_root)
            print(_data_summary(data))
        except Exception:
            _safe_print_exception("Failed to load data bundle")
            data = None

        while True:
            title = "RBTL Hub"
            subtitle = _data_summary(data)
            print(f"\n[{title}] {subtitle}")

            choice = prompt_choice_nav(
                "Choose a generator",
                [
                    "Scenario / Encounter",
                    "Campaign",
                    "Companions",
                    "Loot / Shop",
                    "Reload data",
                    "Quit",
                ],
                default_idx=0,
            )

            if choice in (QUIT, "Quit"):
                return
            if choice == RESTART:
                continue
            if choice == BACK:
                continue

            if choice == "Reload data":
                try:
                    data = _load_data(project_root)
                    print(_data_summary(data))
                except Exception:
                    _safe_print_exception("Failed to reload data bundle")
                    data = None
                continue

            # Run selection (guard so hub doesn't die on exceptions).
            try:
                if choice == "Scenario / Encounter":
                    if data is None:
                        print("\nData bundle is not loaded. Use 'Reload data' first.\n")
                        continue
                    run_scenario(data, project_root)
                elif choice == "Campaign":
                    run_campaign(project_root)
                elif choice == "Companions":
                    run_companions(project_root)
                elif choice == "Loot / Shop":
                    run_loot_shop(project_root)
                else:
                    print(f"\nUnknown option: {choice}\n")
            except SystemExit:
                # Some flows (like scenario CLI) use SystemExit for quit.
                continue
            except Exception:
                _safe_print_exception(f"Generator crashed ({choice})")
                continue


if __name__ == "__main__":
    main()
