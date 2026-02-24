# rbtl_hub.py
"""Rangers at the Borderlands — Hub Launcher

Single entry-point to launch any generator from one menu.

What this hub does
- Provides a unified menu for Scenario/Encounter, Campaign, allies, and Loot/Shop.
- Forces a consistent working directory (project root) so relative paths to ./data and ./output behave.
- Keeps the hub alive if a generator crashes.
- Writes full tracebacks to ./output/hub_errors.log for debugging.

Optional CLI shortcuts
  python rbtl_hub.py --mode scenario
  python rbtl_hub.py --mode campaign
  python rbtl_hub.py --mode allies
  python rbtl_hub.py --mode loot
  python rbtl_hub.py --validate
  python rbtl_hub.py --seed 12345

Notes
- Scenario uses a shared DataBundle loaded once by the hub.
- Campaign / allies / Loot delegate to existing *main* modules (keeping their UX).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import os
import random
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

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


def _ensure_output_dir(project_root: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _log_exception(project_root: str, label: str) -> str:
    """Append the current exception traceback to output/hub_errors.log."""
    out_dir = _ensure_output_dir(project_root)
    log_path = os.path.join(out_dir, "hub_errors.log")

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb = traceback.format_exc()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f"{stamp} | {label}\n")
        f.write(tb)
        if not tb.endswith("\n"):
            f.write("\n")

    return log_path


def _safe_print_exception(project_root: str, prefix: str) -> None:
    """Print a short error, and write full details to the hub error log."""
    log_path = _log_exception(project_root, prefix)
    exc_line = (traceback.format_exc().strip().splitlines() or ["(unknown error)"])[-1]
    print(f"\n{prefix}: {exc_line}\n(Full traceback logged to: {log_path})\n")


def _load_data(project_root: str) -> DataBundle:
    return load_data_bundle(project_root)


def _write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = _ensure_output_dir(project_root)
    outpath = os.path.join(out_dir, filename)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)
    return outpath


def _apply_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    # Optional: makes the seed visible to downstream scripts if they want to read it.
    os.environ["RBTL_SEED"] = str(seed)


def run_scenario(data: DataBundle, project_root: str) -> None:
    """Scenario / Encounter generator (uses shared DataBundle)."""
    from rbtl_cli import run_cli
    from rbtl_core import generate_scenario

    try:
        inputs = run_cli(data)  # may raise SystemExit on quit
    except SystemExit:
        return

    result = generate_scenario(data, inputs)

    outpath: Optional[str] = None
    if isinstance(result, tuple) and len(result) == 2:
        filename, briefing_text = result
        outpath = _write_output(project_root, filename, briefing_text)
    elif isinstance(result, str):
        outpath = result
    else:
        raise RuntimeError(f"Unexpected return from generate_scenario: {type(result)}")

    print(f"\nWrote briefing: {outpath}\n")


def _run_main(module_name: str) -> None:
    """Import <module_name> and run module.run()."""
    mod = __import__(module_name, fromlist=["run"])
    run_fn = getattr(mod, "run", None)
    if not callable(run_fn):
        raise RuntimeError(f"{module_name}.run() not found")
    run_fn()


def run_campaign(_: Optional[DataBundle], __: str) -> None:
    _run_main("rbtl_main_campaign")


def run_allies(_: Optional[DataBundle], __: str) -> None:
    _run_main("rbtl_main_allies")


def run_loot_shop(_: Optional[DataBundle], __: str) -> None:
    _run_main("rbtl_main_loot")


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


def _validate_data_files(project_root: str) -> tuple[bool, str]:
    """Lightweight integrity checks on ./data/*.txt.

    This is intentionally conservative: it doesn't try to fully parse every format,
    just catches the common footguns that waste the most time.
    """
    data_dir = os.path.join(project_root, "data")
    if not os.path.isdir(data_dir):
        return False, f"No data directory found at: {data_dir}"

    files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(data_dir, f))
    )
    if not files:
        return False, f"No .txt files found in: {data_dir}"

    problems: list[str] = []
    warnings: list[str] = []

    for fname in files:
        path = os.path.join(data_dir, fname)
        seen_ids: set[str] = set()
        dupes: set[str] = set()
        pipe_lines = 0
        bad_id_lines = 0

        with open(path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # Strip inline comments.
                if "#" in line:
                    line = line.split("#", 1)[0].strip()
                    if not line:
                        continue

                if "|" not in line:
                    # settings.txt and similar key=value files will land here.
                    continue

                pipe_lines += 1
                first = line.split("|", 1)[0].strip()
                if not first:
                    bad_id_lines += 1
                    warnings.append(f"{fname}:{lineno} missing id before first '|'")
                    continue

                # Keep ids as strings (some packs might use alphanumerics later).
                if first in seen_ids:
                    dupes.add(first)
                else:
                    seen_ids.add(first)

        if dupes:
            problems.append(f"{fname}: duplicate ids ({len(dupes)}): {', '.join(sorted(dupes)[:12])}{'...' if len(dupes) > 12 else ''}")
        if pipe_lines > 0 and bad_id_lines > 0:
            warnings.append(f"{fname}: {bad_id_lines} pipe-format lines have missing ids")

    # Also attempt a full bundle load (catches deeper parsing errors).
    try:
        _ = load_data_bundle(project_root)
    except Exception:
        _log_exception(project_root, "Data validation: load_data_bundle() failed")
        problems.append("load_data_bundle() failed (see hub_errors.log for full traceback)")

    if not problems and not warnings:
        return True, "Data validation: OK (no obvious issues found)."

    out: list[str] = ["Data validation report:"]
    if problems:
        out.append("\nProblems:")
        out.extend([f"- {p}" for p in problems])
    if warnings:
        out.append("\nWarnings:")
        out.extend([f"- {w}" for w in warnings[:30]])
        if len(warnings) > 30:
            out.append(f"- ... ({len(warnings) - 30} more)")

    return (len(problems) == 0), "\n".join(out)

def _validate_and_reload(data: Optional[DataBundle], project_root: str) -> Optional[DataBundle]:
    """Validate ./data then (if OK) reload the in-memory DataBundle.

    Returns the (possibly updated) bundle. If validation finds problems, the original
    bundle is returned unchanged.
    """
    ok, report = _validate_data_files(project_root)
    print(report)
    if not ok:
        print("\nValidation found problems; data was NOT reloaded.\n")
        return data
    try:
        new_data = _load_data(project_root)
        print("\nReloaded data bundle.\n" + _data_summary(new_data) + "\n")
        return new_data
    except Exception:
        _safe_print_exception(project_root, "Failed to reload data bundle after validation")
        return None




@dataclass(frozen=True)
class _Entry:
    label: str
    fn: Callable[[Optional[DataBundle], str], None]
    needs_data: bool = False


ENTRIES: list[_Entry] = [
    _Entry("Campaign", run_campaign),
    _Entry("Scenario / Encounter", lambda d, root: run_scenario(d, root), needs_data=True),
    _Entry("allies", run_allies),
    _Entry("Shop / Loot", run_loot_shop),
]


def _run_mode(mode: str, data: Optional[DataBundle], project_root: str) -> None:
    mode = mode.strip().lower()
    if mode in ("scenario", "encounter", "scenario/encounter"):
        if data is None:
            raise RuntimeError("Data bundle is not loaded")
        run_scenario(data, project_root)
    elif mode == "campaign":
        run_campaign(data, project_root)
    elif mode in ("allies", "ally"):
        run_allies(data, project_root)
    elif mode in ("loot", "shop", "loot/shop"):
        run_loot_shop(data, project_root)
    else:
        raise RuntimeError(f"Unknown mode: {mode}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--mode", type=str, default="", help="Run one generator then exit (scenario, campaign, allies, loot)")
    p.add_argument("--seed", type=int, default=None, help="Seed Python RNG for reproducible rolls")
    p.add_argument("--validate", action="store_true", help="Validate data/*.txt then exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    project_root = PROJECT_ROOT

    with _pushd(project_root):
        _apply_seed(args.seed)

        # Validation-only path.
        if args.validate:
            try:
                ok, report = _validate_data_files(project_root)
                print(report)
            except Exception:
                _safe_print_exception(project_root, "Validate data failed")
            return

        # Load bundle once (scenario depends on it; others may still load internally).
        data: Optional[DataBundle] = None
        try:
            data = _load_data(project_root)
            print(_data_summary(data))
        except Exception:
            _safe_print_exception(project_root, "Failed to load data bundle")
            data = None

        # One-shot CLI mode.
        if args.mode:
            try:
                _run_mode(args.mode, data, project_root)
            except SystemExit:
                return
            except Exception:
                _safe_print_exception(project_root, f"Mode crashed ({args.mode})")
            return

        # Interactive menu.
        while True:
            subtitle = _data_summary(data)
            seed_note = f" | Seed={args.seed}" if args.seed is not None else ""
            print(f"\n[RBTL Hub] {subtitle}{seed_note}")

            options = [e.label for e in ENTRIES] + [
                "Validate & Reload Data",
                "Quit",
            ]

            choice = prompt_choice_nav(
                "Choose a generator",
                options,
                default_idx=0,
            )

            if choice in (QUIT, "Quit"):
                return
            if choice in (RESTART, BACK):
                continue
            if choice == "Validate & Reload Data":
                try:
                    data = _validate_and_reload(data, project_root)
                except Exception:
                    _safe_print_exception(project_root, "Validate + Reload failed")
                continue

            entry = next((e for e in ENTRIES if e.label == choice), None)
            if entry is None:
                print(f"\nUnknown option: {choice}\n")
                continue

            if entry.needs_data and data is None:
                print("\nData bundle is not loaded. Use 'Reload data' first.\n")
                continue

            try:
                entry.fn(data, project_root)
            except SystemExit:
                # Some flows use SystemExit for quit.
                continue
            except Exception:
                _safe_print_exception(project_root, f"Generator crashed ({choice})")
                continue


if __name__ == "__main__":
    main()
