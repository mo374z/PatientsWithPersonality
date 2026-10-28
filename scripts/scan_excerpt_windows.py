#!/usr/bin/env python3
"""
Pick the most informative excerpt window per case for the doctor study.

Windows are scored on the real transcript, so every arm of a case is shown the
same turn range and the choice cannot favour any simulator. The one exception
is the goodbye penalty: some simulators end their conversation early and then
loop on farewells for the rest of it, independently of what the real
transcript says at that turn. This is invisible in the reference transcript,
so it is detected by scanning every arm's simulated_response for farewell
language and penalising windows any arm loops on, regardless of which arm it
is - the penalty never favours one simulator over another.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

WORD_WEIGHT = 0.01
GOODBYE_PATTERN = re.compile(
    r"\b(goodbye|good-bye|bye|take care|have a (?:nice|good) day)\b", re.IGNORECASE
)
GOODBYE_PENALTY = 2.0


def _disclosed_fields(fact_map) -> set[str]:
    if not isinstance(fact_map, str):
        return set()
    return {fact["field"] for fact in json.loads(fact_map)}


def _model_dir(results_dir: Path, arm: str) -> str:
    models = [d.name for d in (results_dir / arm).iterdir() if d.is_dir()]
    if len(models) != 1:
        raise ValueError(
            f"Expected exactly one model dir in {results_dir / arm}, found {models}"
        )
    return models[0]


def _load_reference_turns(results_dir: Path, arm: str, conv_id: str) -> pd.DataFrame:
    model = _model_dir(results_dir, arm)
    return pd.read_csv(results_dir / arm / model / conv_id / "turns.csv")


def _goodbye_turns(results_dir: Path, arms: list[str], conv_id: str, n_turns: int) -> list[bool]:
    flagged = [False] * n_turns
    for arm in arms:
        path = results_dir / arm / _model_dir(results_dir, arm) / conv_id / "turns.csv"
        if not path.exists():
            continue
        responses = pd.read_csv(path)["simulated_response"].astype(str)
        for turn, response in enumerate(responses[:n_turns]):
            if GOODBYE_PATTERN.search(response):
                flagged[turn] = True
    return flagged


def _score_windows(
    df: pd.DataFrame, window: int, goodbye: list[bool]
) -> list[tuple[float, int, int, int]]:
    fields = [_disclosed_fields(m) for m in df["turn_fact_map"]]
    words = df["real_response"].astype(str).str.split().str.len().tolist()
    return [
        (
            len(set().union(*fields[start : start + window]))
            + WORD_WEIGHT * sum(words[start : start + window])
            - GOODBYE_PENALTY * sum(goodbye[start : start + window]),
            len(set().union(*fields[start : start + window])),
            sum(words[start : start + window]),
            sum(goodbye[start : start + window]),
        )
        for start in range(len(df) - window + 1)
    ]


def scan(results_dir: Path, arm: str, window: int) -> dict:
    model = _model_dir(results_dir, arm)
    conv_ids = sorted(d.name for d in (results_dir / arm / model).iterdir() if d.is_dir())
    arms = sorted(d.name for d in results_dir.iterdir() if d.is_dir())

    windows = {}
    for conv_id in conv_ids:
        df = _load_reference_turns(results_dir, arm, conv_id)
        if len(df) < window:
            raise ValueError(
                f"{conv_id} has {len(df)} turns, fewer than the {window}-turn window"
            )
        goodbye = _goodbye_turns(results_dir, arms, conv_id, len(df))
        scored = _score_windows(df, window, goodbye)
        score, fields, words, goodbye_turns = max(scored)
        start = scored.index((score, fields, words, goodbye_turns))
        windows[conv_id] = {
            "start": start,
            "fields": fields,
            "patient_words": words,
            "goodbye_turns": goodbye_turns,
            "n_turns": len(df),
        }
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--reference-arm", required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    windows = scan(Path(args.results_dir), args.reference_arm, args.window)
    payload = {
        "turn_window": args.window,
        "reference_arm": args.reference_arm,
        "windows": windows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")

    starts = [w["start"] for w in windows.values()]
    print(f"Wrote {len(windows)} windows to {out}")
    print(
        f"  start turn: min {min(starts)}, median {sorted(starts)[len(starts) // 2]}, max {max(starts)}"
    )
    print(f"  windows starting at turn 0: {sum(s == 0 for s in starts)}")

    contaminated = sorted(
        ((cid, w["goodbye_turns"]) for cid, w in windows.items() if w["goodbye_turns"] > 0),
        key=lambda item: -item[1],
    )
    if contaminated:
        print(
            f"  windows still touching a goodbye/farewell turn despite the penalty: "
            f"{len(contaminated)}/{len(windows)}"
        )
        for cid, count in contaminated[:10]:
            print(f"    {cid}: {count}/{args.window} turns")


if __name__ == "__main__":
    main()
