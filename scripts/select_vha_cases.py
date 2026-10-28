#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.utils import parse_transcript_file
from patient_simulator.patients.pwp import LEISURE_FIELDS, MEDICAL_FIELDS

SCOREABLE_FIELDS = LEISURE_FIELDS + MEDICAL_FIELDS
UNKNOWN_VALUES = {"unknown", "", "n/a"}


def is_known(value):
    return str(value).strip().lower() not in UNKNOWN_VALUES


def case_stats(case_id, conversations_dir, case_descriptions_dir):
    turns = parse_transcript_file(f"{conversations_dir}/{case_id}.txt")
    with open(f"{case_descriptions_dir}/{case_id}.json") as f:
        profile = json.load(f)

    missing = [f for f in SCOREABLE_FIELDS if f not in profile]
    if missing:
        raise ValueError(f"{case_id}: profile is missing scoreable fields {missing}")

    patient_turns = [t for speaker, t in turns if speaker != "DOCTOR"]
    return {
        "case": case_id,
        "n_patient_turns": len(patient_turns),
        "patient_tokens": sum(len(t.split()) for t in patient_turns),
        "known_fields": sum(is_known(profile[f]) for f in SCOREABLE_FIELDS),
        "known_medical": sum(is_known(profile[f]) for f in MEDICAL_FIELDS),
    }


def select(
    conversations_dir,
    case_descriptions_dir,
    max_patient_turns,
    min_known_fields,
    exclude=(),
    sample_n=None,
    seed=None,
):
    case_ids = sorted(
        p.stem
        for p in Path(conversations_dir).glob("*.txt")
        if (Path(case_descriptions_dir) / f"{p.stem}.json").exists()
    )
    if not case_ids:
        raise ValueError(
            f"no cases with both a transcript and a profile in {conversations_dir}"
        )

    df = pd.DataFrame(
        case_stats(c, conversations_dir, case_descriptions_dir) for c in case_ids
    )
    keep = df[
        (df["n_patient_turns"] <= max_patient_turns)
        & (df["known_fields"] >= min_known_fields)
        & (~df["case"].isin(set(exclude)))
    ].sort_values("case")
    if keep.empty:
        raise ValueError(
            f"no case satisfies max_patient_turns={max_patient_turns} and "
            f"min_known_fields={min_known_fields}"
        )

    if sample_n is not None:
        if sample_n > len(keep):
            raise ValueError(
                f"sample_n={sample_n} exceeds the {len(keep)} eligible cases"
            )
        keep = keep.sample(n=sample_n, random_state=seed).sort_values("case")

    return df, keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations-dir", required=True)
    parser.add_argument("--case-descriptions-dir", required=True)
    parser.add_argument("--max-patient-turns", type=int, required=True)
    parser.add_argument("--min-known-fields", type=int, required=True)
    parser.add_argument("--stats-out")
    parser.add_argument("--exclude-file", help="case list to hold out, one id per line")
    parser.add_argument("--sample-n", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    exclude = Path(args.exclude_file).read_text().split() if args.exclude_file else ()

    df, keep = select(
        args.conversations_dir,
        args.case_descriptions_dir,
        args.max_patient_turns,
        args.min_known_fields,
        exclude=exclude,
        sample_n=args.sample_n,
        seed=args.seed,
    )

    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        df.assign(selected=df["case"].isin(keep["case"])).to_csv(
            args.stats_out, index=False
        )

    print(
        f"# {len(keep)} of {len(df)} cases: "
        f"n_patient_turns <= {args.max_patient_turns}, "
        f"known_fields >= {args.min_known_fields} of {len(SCOREABLE_FIELDS)}",
        file=sys.stderr,
    )
    print(
        f"# patient turns {keep.n_patient_turns.median():.0f} median "
        f"({keep.n_patient_turns.min()}-{keep.n_patient_turns.max()}), "
        f"known fields {keep.known_fields.mean():.2f} mean "
        f"vs {df.known_fields.mean():.2f} over the full corpus",
        file=sys.stderr,
    )
    print("conversations:")
    for case in keep["case"]:
        print(f"  - {case}")


if __name__ == "__main__":
    main()
