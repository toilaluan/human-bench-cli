"""Maintainer-only authoring script. Evaluation always reads the committed catalog.

Run with --write to author the initial catalog, or --check to detect accidental drift.
Never rewrite a released catalog: add a new version and new IDs instead.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/human_bench/data/prompts.v1.jsonl"
ACTIONS = [
    "walking",
    "standing",
    "sitting",
    "lying down",
    "cycling",
    "eating an apple",
    "laughing",
    "crying",
    "playing catch with a ball",
    "jumping",
    "writing on paper",
]
MATERIALS = ["rubber", "plastic", "metal", "wood", "fluffy fabric", "glass", "leather"]
PEOPLE = ["adult man", "adult woman", "adult person"]
RELATIONS = [
    ("to the left of", "to the right of", "a chair"),
    ("to the right of", "to the left of", "a table"),
    ("above", "below", "a chair"),
    ("below", "above", "a table"),
    ("on top of", "beneath", "a table"),
    ("beside", "inside", "a large open bag"),
    ("in front of", "behind", "a chair"),
    ("behind", "in front of", "a table"),
    ("inside", "outside", "a large open bag"),
    ("between", "to the left of", "two chairs"),
    ("among", "outside the group of", "three chairs"),
    ("surrounded by", "to the left of", "a ring of chairs"),
]


def fact(key, capability, statement):
    return {"fact_id": key, "capability": capability, "statement": statement}


def build():
    records = []

    def add(family, index, text, slots, facts, counterfactual, contrast):
        levels = {
            1: "easy",
            2: "hard",
            3: "easy",
            4: "medium",
            5: "hard",
            6: "easy",
            7: "medium",
            8: "very_hard",
        }
        statement = next(f["statement"] for f in facts if f["fact_id"] == contrast)
        records.append(
            {
                "prompt_id": f"hb-v1-t{family}-{index + 1:03d}",
                "prompt": text,
                "family": f"T{family}",
                "difficulty": levels[family],
                "slots": slots,
                "facts": facts,
                "counterfactual": counterfactual,
                "negation": f"It is not the case that {statement[0].lower() + statement[1:]}",
                "contrast_fact_id": contrast,
            }
        )

    for i in range(20):
        who = PEOPLE[i % 2]
        action = ACTIONS[i % len(ACTIONS)]
        other = ACTIONS[(i + 5) % len(ACTIONS)]
        add(
            1,
            i,
            f"Exactly one {who} is {action}.",
            {"person": who, "action": action},
            [
                fact("count", "count", "Exactly one human figure is present."),
                fact("person", "entity", f"The figure is an {who}."),
                fact("action", "action", f"The person is {action}."),
            ],
            f"Exactly one {who} is {other}.",
            "action",
        )

    for i in range(20):
        a = ACTIONS[i % len(ACTIONS)]
        b = ACTIONS[(i + 1 + i // len(ACTIONS)) % len(ACTIONS)]
        other = ACTIONS[(i + 5) % len(ACTIONS)]
        text = f"Exactly two people: one adult man {a} and one adult woman {b}."
        add(
            2,
            i,
            text,
            {"action_man": a, "action_woman": b},
            [
                fact("count", "count", "Exactly two human figures are present."),
                fact("man_action", "action", f"An adult man is {a}."),
                fact("woman_action", "action", f"An adult woman is {b}."),
                fact(
                    "binding",
                    "binding",
                    "The specified actions belong to two distinct people, "
                    "with the man's action and the woman's action assigned as requested.",
                ),
            ],
            f"Exactly two people: one adult man {other} and one adult woman {b}.",
            "man_action",
        )

    for i in range(20):
        who, material = PEOPLE[i % 3], MATERIALS[i % 7]
        other = MATERIALS[(i + 1) % 7]
        add(
            3,
            i,
            f"Exactly one human-shaped figure of an {who}, made entirely of {material}.",
            {"person": who, "material": material},
            [
                fact("count", "count", "Exactly one human-shaped figure is present."),
                fact("person", "entity", f"The figure has the form of an {who}."),
                fact(
                    "material",
                    "material",
                    f"The figure's body is made of {material}, "
                    "rather than merely wearing that material.",
                ),
            ],
            f"Exactly one human-shaped figure of an {who}, made entirely of {other}.",
            "material",
        )

    for family in (4, 5):
        for i in range(20):
            who = PEOPLE[(i + i // 12) % 3]
            relation, opposite, obj = RELATIONS[i % len(RELATIONS)]
            material = MATERIALS[i % 7]
            object_phrase = obj if family == 4 else f"{obj} made of {material}"
            text = f"Exactly one {who} is {relation} {object_phrase}."
            if "left" in relation or "right" in relation:
                text += " Left and right are from the viewer's perspective."
            facts = [
                fact("count", "count", "Exactly one human figure is present."),
                fact("person", "entity", f"The person is an {who}."),
                fact(
                    "spatial",
                    "spatial",
                    f"The person is {relation} {obj}, "
                    "using the viewer's left and right and visible depth cues where applicable.",
                ),
            ]
            if family == 5:
                facts.extend(
                    [
                        fact(
                            "material",
                            "material",
                            f"The referenced object(s) are made of {material}.",
                        ),
                        fact(
                            "binding",
                            "binding",
                            "The requested material belongs to the referenced "
                            "object(s), not merely the person's clothes or another item.",
                        ),
                    ]
                )
            cf = text.replace(f"is {relation}", f"is {opposite}", 1)
            add(
                family,
                i,
                text,
                {
                    "person": who,
                    "relation": relation,
                    "object": obj,
                    **({"material": material} if family == 5 else {}),
                },
                facts,
                cf,
                "spatial",
            )

    for family in (6, 7):
        for i in range(20):
            count = 2 + i % 6
            group = ["adult men", "adult women", "adults", "older adults"][i // 6]
            action = ACTIONS[i % len(ACTIONS)]
            text = f"Exactly {count} {group} are present, with no other people."
            facts = [
                fact("count", "count", f"Exactly {count} human figures are present."),
                fact("group", "entity", f"All counted people are {group}."),
            ]
            if family == 7:
                text += f" All {count} are {action}."
                facts.extend(
                    [
                        fact("action", "action", f"The action '{action}' is depicted."),
                        fact("binding", "binding", f"Every one of the {count} people is {action}."),
                    ]
                )
            cf = text.replace(f"Exactly {count}", f"Exactly {count + 1}", 1)
            if family == 7:
                cf = cf.replace(f"All {count}", f"All {count + 1}", 1)
            add(
                family,
                i,
                text,
                {"count": count, "group": group, **({"action": action} if family == 7 else {})},
                facts,
                cf,
                "count",
            )

    for i in range(20):
        men, women = 2 + i % 4, 2 + i // 4 % 4
        a, b = ACTIONS[i % 11], ACTIONS[(i + 4) % 11]
        text = (
            f"Exactly {men} adult men are {a} and exactly {women} adult women are {b}. "
            "These are distinct groups, with no other people present."
        )
        add(
            8,
            i,
            text,
            {"men": men, "women": women, "action_man": a, "action_woman": b},
            [
                fact("men_count", "count", f"Exactly {men} adult men are present."),
                fact("women_count", "count", f"Exactly {women} adult women are present."),
                fact("total_count", "count", f"Exactly {men + women} people are present in total."),
                fact("man_action", "action", f"The men are {a}."),
                fact("woman_action", "action", f"The women are {b}."),
                fact(
                    "binding",
                    "binding",
                    f"All {men} men perform '{a}' and all {women} women "
                    f"perform '{b}', with no swapped or overlapping group assignments.",
                ),
            ],
            text.replace(f"Exactly {men} adult men", f"Exactly {men + 1} adult men", 1),
            "men_count",
        )
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = build()
    if args.write:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(content)
        print(f"Wrote 160 prompts to {TARGET}")
    elif args.check:
        if TARGET.read_text() != content:
            raise SystemExit("Catalog drift detected; do not rewrite released prompt IDs")
        print("Catalog matches its authoring source")
    else:
        parser.print_help()
