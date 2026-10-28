#!/usr/bin/env python3
"""Repair the ``axis`` field of stored personality assignments.

``_personality_items`` used to take a task's ``axis`` from a different ``arm_pairs``
entry than its ``simulator``, so every PatientSim assignment was stored under the
wrong axis. The field is metadata only - it never reached the rater, who always saw
the transcript named by ``conv_id`` and ``simulator`` and rated all six axes - so
submitted labels stay valid and only the label needs correcting.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling-gui"))
import storage


def axis_of_simulator(study_config: Path) -> dict[str, str]:
    arm_pairs = yaml.safe_load(study_config.read_text())["personality"]["arm_pairs"]
    return {
        pair[arm]: pair["axis"] for pair in arm_pairs for arm in ("pwp", "patientsim")
    }


def repair(labels_dir: Path, study_config: Path, write: bool) -> int:
    storage.init(str(labels_dir))
    axis_of = axis_of_simulator(study_config)

    repaired = 0
    for user_id in storage.list_users():
        data = storage.load_labels(user_id, "personality")
        if not data:
            continue

        unknown = sorted(
            {a["simulator"] for a in data["task_assignments"].values()} - set(axis_of)
        )
        if unknown:
            raise ValueError(f"{user_id}: no arm_pairs entry for {unknown}")

        wrong = {
            task_id: (assignment["axis"], axis_of[assignment["simulator"]])
            for task_id, assignment in data["task_assignments"].items()
            if assignment["axis"] != axis_of[assignment["simulator"]]
        }
        for task_id, (stored, correct) in sorted(wrong.items()):
            labelled = "labelled" if task_id in data["labels"] else "unlabelled"
            print(f"  {user_id} {task_id}: axis {stored} -> {correct} ({labelled})")

        if wrong and write:
            for task_id in wrong:
                assignment = data["task_assignments"][task_id]
                assignment["axis"] = axis_of[assignment["simulator"]]
            storage.save_labels(user_id, "personality", data)
        repaired += len(wrong)

    return repaired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    repaired = repair(args.labels_dir, args.study_config, args.write)
    verb = "Repaired" if args.write else "Would repair"
    print(f"{verb} {repaired} assignment(s)")
    if repaired and not args.write:
        print("Re-run with --write to apply")


if __name__ == "__main__":
    main()
