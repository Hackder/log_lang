#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys

"""
"""


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage:")
        print(
            "\tpython3 create_workbook.py <input.md> <based_on.workbook> <output.workbook>"
        )
        print("\tpython3 create_workbook.py <input.md> <based_on&output.workbook>")
        sys.exit(1)

    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        based_on_file = sys.argv[2]
        output_file = based_on_file
    elif len(sys.argv) == 4:
        input_file = sys.argv[1]
        based_on_file = sys.argv[2]
        output_file = sys.argv[3]
    else:
        print("Invalid number of arguments")
        sys.exit(1)

    # Read the markdown file and extract cells
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Load additional fields from the based_on workbook file
    with open(based_on_file, "r", encoding="utf-8") as f:
        old_workbook = json.load(f)

    cells = {}
    cells_order = []
    cell_header_regex = re.compile(r"^<id:(\d+)><type:([^>]+)>$")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        header_match = cell_header_regex.match(line)
        if header_match:
            cell_id = header_match.group(1)
            cell_type = header_match.group(2)
            i += 1
            # accumulate data until a line that is exactly '---'
            data_lines = []
            while i < len(lines) and lines[i].strip() != "---":
                data_lines.append(lines[i].rstrip("\n"))
                i += 1
            # skip the separator line '---'
            i += 1

            try:
                cell = old_workbook["cells"][cell_id]
            except KeyError:
                cell = {
                    "id": int(cell_id),
                    "type": cell_type,
                    "data": "",
                    "idCounter": 0,
                    "comments": {"ids": [], "entities": {}},
                }

            assert cell_type == cell["type"], (
                f"Cell type mismatch: {cell_type} != {cell['type']}"
            )

            print(f"{cell_id}: {cell_type}")
            if cell_type == "formalizationChecker":
                tableau_data = cell["data"]["exercise"]

                # find <prop_id:...> lines
                prop_id_regex = re.compile(r"^<prop_id:(\d+)>$")
                for j in range(len(data_lines)):
                    prop_id_match = prop_id_regex.match(data_lines[j])
                    if prop_id_match:
                        prop_id = prop_id_match.group(1)
                        solution = data_lines[j + 2].split("solution:")[1].strip()

                        for ex in tableau_data["propositions"]:
                            if ex["proposition_id"] == int(prop_id):
                                ex["solution"] = (
                                    solution if solution not in ("", "None") else None
                                )

                cell["data"]["exercise"] = tableau_data

                cells[cell_id] = {
                    "id": int(cell_id),
                    "type": cell_type,
                    "data": cell["data"],
                    "idCounter": cell["idCounter"],
                    "comments": cell["comments"],
                }

                # ---
                # <id:<id>>
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
            elif cell_type == "tableauEditor":
                tableau_data = cell["data"]

                # Find the first non-empty line in data_lines
                while data_lines and data_lines[0].strip() == "":
                    data_lines.pop(0)

                assert data_lines, "No data found in cell"

                if "solve" in data_lines[0].lower():
                    # Need to solve it - write data to source.txt
                    assert len(data_lines) > 1
                    with open("source.logic", "w") as f:
                        for line in data_lines[1:]:
                            f.write(line + "\n")

                    # Run the loglang command and capture output
                    python_path = os.path.expanduser("~/log_lang/.venv/bin/python3")
                    script_path = os.path.expanduser("~/log_lang/main.py")

                    print("    solving with loglang")
                    result = subprocess.run(
                        [python_path, script_path, "tableau", "source.logic", "json"],
                        capture_output=True,
                        text=True,
                    )
                    tableau_data = result.stdout.strip()

                    # Delete temporary file
                    os.remove("source.txt")

                    if result.returncode != 0:
                        print("Error running loglang")
                        print(result.stderr)
                        sys.exit(1)

                    print("    solved")

                else:
                    # Already solved, it's a JSON object
                    tableau_data = data_lines[0]

                # Convert tableau_data string to json object
                if tableau_data and tableau_data != "None":
                    tableau_data = tableau_data.replace("'", '"')  # make it valid json
                    tableau_data = json.loads(tableau_data)
                else:
                    tableau_data = None

                cell["data"] = tableau_data

                cells[cell_id] = {
                    "id": int(cell_id),
                    "type": cell_type,
                    "data": cell["data"],
                    "idCounter": cell["idCounter"],
                    "comments": cell["comments"],
                }
            else:
                data = "\n".join(data_lines)

                # remove the last newline character
                if data[-1] == "\n":
                    data = data[:-1]

                cells[cell_id] = {
                    "id": int(cell_id),
                    "type": cell_type,
                    "data": data,
                    "idCounter": cell["idCounter"],
                    "comments": cell["comments"],
                }
                # ---
                # id:<id>
                # <data>
                #
                # ---

            cells_order.append(int(cell_id))
        else:
            i += 1

    # Build the new workbook merging the cells from md input and fields from the old workbook
    workbook = {
        "versionNumber": 1,
        "cells": cells,
        "cellsOrder": cells_order,
        "settings": old_workbook["settings"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(workbook, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
