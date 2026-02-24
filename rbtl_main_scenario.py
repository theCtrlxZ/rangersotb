# generate_scenario.py
import os
from typing import Any, Tuple

from rbtl_data import load_data_bundle
from rbtl_cli import run_cli
from rbtl_core import generate_scenario

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, filename)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(text)
    return outpath


def main():
    data = load_data_bundle(PROJECT_ROOT)

    # This is the missing piece: NOW/QUICK/CUSTOM prompts using settings defaults.
    inputs = run_cli(data)

    result = generate_scenario(data, inputs)

    # Support either style:
    #  (filename, briefing_text)   OR   an already-written outpath string
    outpath = None
    if isinstance(result, tuple) and len(result) == 2:
        filename, briefing_text = result
        outpath = _write_output(PROJECT_ROOT, filename, briefing_text)
    elif isinstance(result, str):
        # If core already wrote the file, it likely returns a path
        outpath = result
    else:
        raise RuntimeError(f"Unexpected return from generate_scenario: {type(result)}")

    print(f"\nWrote briefing: {outpath}")


if __name__ == "__main__":
    main()
