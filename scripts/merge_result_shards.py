#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import sys

import pandas as pd

CONV_CSV = "all_conversations.csv"


def merge(shard_dirs, target_dir):
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    target_csv = target / CONV_CSV

    frames = []
    if target_csv.exists():
        frames.append(pd.read_csv(target_csv))

    for shard_dir in shard_dirs:
        shard = Path(shard_dir)
        shard_csv = shard / CONV_CSV
        if not shard_csv.exists():
            raise FileNotFoundError(f"no {CONV_CSV} in {shard}")

        for child in shard.iterdir():
            if child.name == CONV_CSV:
                continue
            destination = target / child.name
            if destination.exists():
                for sub in child.rglob("*"):
                    if sub.is_file():
                        moved = destination / sub.relative_to(child)
                        moved.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(sub), moved)
            else:
                shutil.move(str(child), destination)

        frame = pd.read_csv(shard_csv)
        frame["path"] = frame["path"].str.replace(
            str(shard), str(target), n=1, regex=False
        )
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True).drop_duplicates("path", keep="last")
    missing = [p for p in merged["path"] if not (Path(p) / "turns.csv").exists()]
    if missing:
        raise ValueError(
            f"{len(missing)} merged rows point at a missing turns.csv, e.g. {missing[0]}"
        )

    merged.to_csv(target_csv, index=False)
    print(
        f"merged {len(frames)} frames -> {len(merged)} rows in {target_csv}",
        file=sys.stderr,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--shard-dirs", required=True, nargs="+")
    args = parser.parse_args()
    merge(args.shard_dirs, args.target_dir)


if __name__ == "__main__":
    main()
