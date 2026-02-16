#!/usr/bin/env python3

import json
import sys

"""
"""


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python3 parse_workbook.py <input>.workbook <output>.md")
        print(
            "Usage: python3 parse_workbook.py <input>.workbook // output workbook will default to <input>.md"
        )
        sys.exit(1)

    input_file = (
        sys.argv[1] + ".workbook"
        if not sys.argv[1].endswith(".workbook")
        else sys.argv[1]
    )
    output_file = input_file.replace(".workbook", ".md")

    if len(sys.argv) == 3:
        output_file = (
            sys.argv[2] + ".md" if not sys.argv[2].endswith(".md") else sys.argv[2]
        )

    # Load the workbook JSON structure
    with open(input_file, "r", encoding="utf-8") as f:
        workbook = json.load(f)

    cells = workbook["cells"]
    cells_order = map(str, workbook["cellsOrder"])

    with open(output_file, "w", encoding="utf-8") as out:
        for cell_id in cells_order:
            cell = cells[cell_id]
            if cell is None:
                # no such cell
                continue

            cell_type = cell["type"]
            out.write(f"<id:{cell_id}><type:{cell_type}>\n")

            print(f"{cell_id}: {cell_type}")
            if cell_type == "formalizationChecker":
                exercise = cell["data"]["exercise"]
                desc = exercise["description"].replace("\n", "; ")
                consts = exercise["constants"].replace("\n", "; ")
                preds = exercise["predicates"].replace("\n", "; ")
                constraints = exercise["constraints"].replace("\n", "; ")

                out.write(f"description: {desc}\n")
                out.write(f"constants: {consts}\n")
                out.write(f"predicates: {preds}\n")
                out.write(f"constraints: {constraints}\n\n")

                for prop in exercise["propositions"]:
                    prop_id = prop["proposition_id"]
                    prop_text = prop["proposition"]
                    solution = prop["solution"]

                    out.write(f"<prop_id:{prop_id}>\n")
                    out.write(f"proposition: {prop_text}\n")
                    out.write(f"solution: {solution}\n\n")

                out.write("---\n")
                # ---
                # <id:<id>><type:formalizationChecker>
                # description: <desc>
                # constants: <consts>
                # predicates: <preds>
                # constraints: <constraints>
                #
                # <prop_id:<prop_id>>
                # proposition: <prop>
                # solution: <solution>
                #
                # <prop_id:<prop_id>>
                # proposition: <prop>
                # solution: <solution>
                #
                # ...
                # ---
            else:
                out.write(f"{cell.get('data')}\n\n")
                out.write("---\n")
                # ---
                # <id:<id>><type:<type>>
                # <data>
                #
                # ---


if __name__ == "__main__":
    main()
