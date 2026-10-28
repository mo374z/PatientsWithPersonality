#!/usr/bin/env python3
import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

TEXT_COLUMNS = ["doctor_question", "real_response", "simulated_response"]
ELLIPSIS = re.compile(r"[ \t]*(?:\.{2,}|…)+[ \t]*")
CLAUSE_END = ".!?,;:-–—"


def _replacement(before: str, after: str) -> str:
    if not after or after[0] == "\n":
        return "." if before and before[-1] not in CLAUSE_END else ""
    if not before or before[-1] in CLAUSE_END:
        return " "
    return ", "


def strip_ellipses(text: str) -> str:
    stripped = ELLIPSIS.sub(
        lambda m: _replacement(
            text[: m.start()].rstrip(" \t"), text[m.end() :].lstrip(" \t")
        ),
        text,
    )
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def strip_turns_file(path: Path) -> int:
    frame = pd.read_csv(path)
    replaced = 0
    for column in TEXT_COLUMNS:
        if column not in frame:
            raise KeyError(f"{path} has no column '{column}'")
        original = frame[column]
        text = original.fillna("").astype(str)
        replaced += int(text.str.count(ELLIPSIS).sum())
        frame[column] = text.map(strip_ellipses).where(original.notna(), original)
    if replaced:
        frame.to_csv(path, index=False)
    return replaced


def strip_results(results_dir: str, backup_dir: str):
    results, backup = Path(results_dir), Path(backup_dir)
    if not results.is_dir():
        raise FileNotFoundError(f"results directory not found: {results}")
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")

    shutil.copytree(results, backup)
    print(f"copied {results} -> {backup}", file=sys.stderr)

    files = sorted(results.rglob("turns.csv"))
    if not files:
        raise FileNotFoundError(f"no turns.csv below {results}")

    replaced = sum(strip_turns_file(path) for path in files)
    print(f"stripped {replaced} ellipses in {len(files)} turns.csv", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--backup-dir", required=True)
    args = parser.parse_args()
    strip_results(args.results_dir, args.backup_dir)


if __name__ == "__main__":
    main()
