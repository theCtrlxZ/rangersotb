# Rangers of the Borderlands

A database-driven, GM-less tabletop wargame/RPG toolkit. The system uses plain .txt “content packs” (units, events, items, spells, rooms, campaign pressures, etc.) plus lightweight Python randomizers to generate ready-to-play briefings: scenarios/encounters, campaign setups, companions, and loot/shops.

The goal is simple: make it fast to get to the table, and easy to homebrew by editing text files rather than wrestling with spreadsheets or a heavy app.

What you can generate:

Scenario / Encounter briefings
Tactical setups and prompts for action (via generate_scenario.py + rbtl_core.py).

Campaign setups
A higher-level context roll (biome + pressures + threats), plus campaign-facing structure (via rbtl_main_campaign.py + rbtl_campaign.py).

Companions
Quick allies/NPCs with classes/backgrounds (via rbtl_main_companions.py + rbtl_companions.py).

Loot / Shops
Loot drops and settlement shopping lists (via rbtl_main_loot.py + rbtl_loot.py).

Ready to start? Use the hub to access all of the above systems (via rbtl_hub.py)

Quickstart
Requirements

Python 3.9+ recommended (standard library only).

Run the hub (recommended)

From the project folder:

python rbtl_hub.py


You’ll get a menu to launch any generator in one place:

Scenario / Encounter

Campaign

Companions

Loot / Shop

All generators write their results into ./output.

How the content files work

Most databases use a simple “pipe” format:

id|Name|key=value|key=value|tag=comma,separated,tags|threat=comma,separated,threats|roll:<directive>

Notes

Lines starting with # are comments.

Inline comments are supported: anything after # is ignored.

tag= becomes entry["tags"] (a set).

threat= becomes entry["threats"] (a set).

tier= is supported; if omitted it defaults to:

leader when the entry has the leader tag, otherwise minion.

Because these are plain text files, homebrew is as easy as adding new lines and tags.

Settings

data/settings.txt is simple key=value pairs. The CLI reads settings to provide reasonable defaults and keep your generators consistent between runs.

Example:

# settings.txt
difficulty_default=2
players_default=4


(Exact supported keys depend on what the generators read.)

Design philosophy

Text-first content authoring: tweak balance and flavor by editing .txt, not Python.

Stable architecture: rbtl_data.py owns data loading/paths; core logic stays IO-light to reduce “mystery bugs.”

Table-ready output: generators produce something you can print or paste straight into a session doc.

Modular generators: scenario, campaign, companions, and loot can be used together or independently.

Troubleshooting

“Data loaded: … items=0”
Your data/items.txt path is missing, misnamed, or empty.

Weird file-not-found behavior when launching scripts directly
Launch from the project root or use rbtl_hub.py (it forces a consistent working directory).

Crashes that mention tags/roll directives
Check the line that was just added to a .txt file — a missing | delimiter or a stray = is the most common culprit.
