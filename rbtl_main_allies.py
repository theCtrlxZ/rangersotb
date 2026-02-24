# rbtl_main_allies.py
import os

from rbtl_data import load_data_bundle
from rbtl_cli import prompt_choice_nav, prompt_int_nav, BACK, RESTART, QUIT
from rbtl_allies import generate_allies


def write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def collect_class_tags(classes):
    tags = set()
    for c in classes:
        for t in (c.get("tags") or set()):
            s = str(t).strip().lower()
            if s:
                tags.add(s)
    return sorted(tags)


def classes_by_tag(classes, tag: str):
    want = str(tag).strip().lower()
    out = []
    for c in classes:
        ctags = {str(t).strip().lower() for t in (c.get("tags") or set()) if str(t).strip()}
        if want in ctags:
            out.append(c)
    return out


def run():
    project_root = os.getcwd()
    data = load_data_bundle(project_root)

    while True:
        mode = prompt_choice_nav(
            "Ally Generator",
            ["Quick", "Custom", "Quit"],
            default_idx=0
        )
        if mode in (QUIT, "Quit"):
            return
        if mode == RESTART:
            continue

        count = prompt_int_nav("How many allies?", default=6)
        if count in (BACK, RESTART, QUIT):
            if count == BACK:
                continue
            if count == RESTART:
                continue
            return

        count = int(count)

        merchant_level = prompt_int_nav("Merchant level for gear rolls? (0 = none)", default=0)
        if merchant_level in (BACK, RESTART, QUIT):
            if merchant_level == BACK:
                continue
            if merchant_level == RESTART:
                continue
            return

        merchant_level = int(merchant_level)

        if mode.startswith("Quick"):
            filename, text = generate_allies(
                data,
                {"count": count, "merchant_level": merchant_level},
            )
            path = write_output(project_root, filename, text)
            print(f"\nWrote: {path}\n")
            continue

        # Custom: tag -> class
        tag_options = ["Random"] + collect_class_tags(data.ally_classes)
        picked_tag = prompt_choice_nav(
            "Pick a hero class tag (filters ONLY the hero class roll)",
            tag_options,
            default_idx=0
        )
        if picked_tag in (BACK, RESTART, QUIT):
            if picked_tag == BACK:
                continue
            if picked_tag == RESTART:
                continue
            return

        if picked_tag == "Random":
            pool = data.ally_classes
        else:
            pool = classes_by_tag(data.ally_classes, picked_tag)

        class_options = ["Random"] + [c.get("name", "Unknown") for c in pool]
        picked_class = prompt_choice_nav(
            "Pick a specific hero class (optional)",
            class_options,
            default_idx=0
        )
        if picked_class in (BACK, RESTART, QUIT):
            if picked_class == BACK:
                continue
            if picked_class == RESTART:
                continue
            return

        inputs = {
            "count": count,
            "allow_background_trait_rolls": True,
            "merchant_level": merchant_level,
        }

        if picked_tag != "Random":
            inputs["required_class_tags"] = [picked_tag]

        if picked_class != "Random":
            inputs["required_class_name"] = picked_class

        filename, text = generate_allies(data, inputs)
        path = write_output(project_root, filename, text)
        print(f"\nWrote: {path}\n")


if __name__ == "__main__":
    run()

