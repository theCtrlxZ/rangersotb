# rbtl_main_loot.py
import os

from rbtl_data import load_data_bundle
from rbtl_cli import prompt_choice_nav, prompt_int_nav, BACK, RESTART, QUIT
from rbtl_loot import generate_loot, generate_shop


def _default_merchant_level_for_settlement(settlement: str) -> int:
    s = str(settlement or '').strip().lower()
    if s in ('hamlet', 'outpost'):
        return 1
    if s in ('village',):
        return 2
    if s in ('town', 'city'):
        return 3
    return 2


def write_output(project_root: str, filename: str, text: str) -> str:
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def _merchant_level_menu_default(data, settlement: str) -> int:
    ml = data.settings.get("default.shop_merchant_level")
    if ml is None:
        return _default_merchant_level_for_settlement(settlement)
    ml_str = str(ml).strip().lower()
    if ml_str in ("random", "rand", "r"):
        return 0  # special: random
    try:
        return max(1, min(5, int(ml)))
    except Exception:
        return _default_merchant_level_for_settlement(settlement)


def run():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data = load_data_bundle(project_root)

    while True:
        mode = prompt_choice_nav(
            "Loot / Shop Generator",
            ["Shop (Quick)", "Shop (Custom)", "Loot Items (Quick)", "Loot Items (Custom)", "Quit"],
            default_idx=0,
        )
        if mode in (QUIT, "Quit"):
            return
        if mode == RESTART:
            continue

        # ----------------------------
        # SHOP
        # ----------------------------
        if mode.startswith("Shop"):
            default_counts = {"hamlet": 1, "village": 5, "town": 10, "city": 10}

            if mode == "Shop (Quick)":
                settlement = data.settings.get("default.shop_settlement", "Hamlet")
                def_cnt = default_counts.get(str(settlement).strip().lower(), 5)

                count = prompt_int_nav("How many RANDOM additions (in addition to core stock)?", default=def_cnt)
                if count in (BACK, RESTART, QUIT):
                    if count in (BACK, RESTART):
                        continue
                    return

                # merchant level (settings may be random)
                ml_default = _merchant_level_menu_default(data, settlement)
                if ml_default == 0:
                    import random as _random
                    s = str(settlement).strip().lower()
                    lo = 1
                    ml = _random.randint(lo, 5)
                else:
                    ml = ml_default

                filename, text = generate_shop(
                    data,
                    {
                        "settlement": settlement,
                        "random_count": int(count),
                        "auto_build": False,
                        "merchant_level": int(ml),
                    },
                )
                path = write_output(project_root, filename, text)
                print(f"\nWrote: {path}\n")
                continue

            # Custom shop
            settlement = prompt_choice_nav(
                "Settlement type",
                ["Random", "Hamlet", "Village", "Town", "Back"],
                default_idx=0,
            )
            if str(settlement).strip().lower() == "random":
                import random as _random
                settlement = _random.choice(["Hamlet", "Village", "Town"])

            if settlement in (BACK, "Back", RESTART, QUIT):
                if settlement in (BACK, "Back", RESTART):
                    continue
                return

            override = prompt_choice_nav(
                "Override the default random-additions count for this settlement?",
                ["No", "Yes"],
                default_idx=0,
            )
            if override in (BACK, RESTART, QUIT):
                if override in (BACK, RESTART):
                    continue
                return

            # Ask random additions immediately after override question
            random_count_override = None
            if str(override).strip().lower() == "yes":
                def_cnt = default_counts.get(str(settlement).strip().lower(), 5)
                count = prompt_int_nav("How many RANDOM additions?", default=def_cnt)
                if count in (BACK, RESTART, QUIT):
                    if count in (BACK, RESTART):
                        continue
                    return
                random_count_override = int(count)
            # Merchant level (1-5): 1 = basic, 5 = high-powered.
            ml_default = _merchant_level_menu_default(data, settlement)
            options = [
                "1 - Basic (Common)",
                "2 - Modest (Uncommon)",
                "3 - Strong (Rare)",
                "4 - Elite (Legendary)",
                "5 - High-Powered (Legendary)",
                "Random (1-5)",
                "Back",
            ]
            # Default selection: use settings/default settlement level if valid, otherwise Random.
            default_idx = 5  # Random
            try:
                if int(ml_default) in (1, 2, 3, 4, 5):
                    default_idx = int(ml_default) - 1
            except Exception:
                default_idx = 5

            prompt = "Merchant level (1=basic, 5=high-powered)"
            ml_choice = prompt_choice_nav(prompt, options, default_idx=default_idx)
            if ml_choice in (BACK, "Back", RESTART, QUIT):
                if ml_choice in (BACK, "Back", RESTART):
                    continue
                return

            if str(ml_choice).strip().lower().startswith("random"):
                import random as _random
                ml = _random.randint(1, 5)
            else:
                try:
                    ml = int(str(ml_choice).strip()[0])
                except Exception:
                    ml = _default_merchant_level_for_settlement(settlement)

            ml = max(1, min(5, int(ml)))

            inputs = {"settlement": settlement, "auto_build": False, "merchant_level": ml}
            if random_count_override is not None:
                inputs["random_count"] = int(random_count_override)

            filename, text = generate_shop(data, inputs)
            path = write_output(project_root, filename, text)
            print(f"\nWrote: {path}\n")
            continue

        # ----------------------------
        # LOOT ITEMS
        # ----------------------------
        count = prompt_int_nav("How many loot results?", default=10)
        if count in (BACK, RESTART, QUIT):
            if count in (BACK, RESTART):
                continue
            return
        count = int(count)

        if mode == "Loot Items (Quick)":
            filename, text = generate_loot(
                data,
                {"count": count, "category": "random", "rarity": "random", "auto_build": False, "unique": True},
            )
            path = write_output(project_root, filename, text)
            print(f"\nWrote: {path}\n")
            continue

        # Custom loot
        category = prompt_choice_nav(
            "Loot category",
            ["Random", "Weapon", "Armor", "Potion", "Herb", "Item", "Back"],
            default_idx=0,
        )
        if category in (BACK, "Back", RESTART, QUIT):
            if category in (BACK, "Back", RESTART):
                continue
            return

        ml_choice = prompt_choice_nav(
            "Loot quality (merchant level; 1=basic, 5=high-powered)",
            [
                "1 - Basic (Common)",
                "2 - Modest (Uncommon)",
                "3 - Strong (Rare)",
                "4 - Elite (Legendary)",
                "5 - High-Powered (Legendary)",
                "Random (1-5)",
                "Back",
            ],
            default_idx=5,
        )
        if ml_choice in (BACK, "Back", RESTART, QUIT):
            if ml_choice in (BACK, "Back", RESTART):
                continue
            return

        if str(ml_choice).strip().lower().startswith("random"):
            import random as _random
            merchant_level = _random.randint(1, 5)
        else:
            try:
                merchant_level = int(str(ml_choice).strip()[0])
            except Exception:
                merchant_level = 2
        merchant_level = max(1, min(5, int(merchant_level)))

        filename, text = generate_loot(
            data,
            {
                "count": count,
                "category": str(category).lower(),
                "merchant_level": merchant_level,
                "policy_settlement": "Town",
                "auto_build": False,
                "unique": True,
            },
        )
        path = write_output(project_root, filename, text)
        print(f"\nWrote: {path}\n")


if __name__ == "__main__":
    run()
