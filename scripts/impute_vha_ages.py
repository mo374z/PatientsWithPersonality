#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
from pathlib import Path
import sys

import pandas as pd
import pydantic
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.utils import (
    create_llm_instance,
    parse_transcript_file,
    retry_on_transient_error,
)

log = logging.getLogger(__name__)

AGE_PROMPT = """You are reading a transcript of a real primary-care consultation.

The patient's age was never stated explicitly, but a clinician reading this
transcript could estimate it from context: life stage, retirement or work
references, the children or grandchildren mentioned, age-related conditions,
screening programmes discussed, and how long ago past events are said to be.

Estimate the patient's age in years as a single integer between 18 and 95, and
state the evidence in the transcript you based it on.

TRANSCRIPT:
{transcript}"""


class AgeEstimate(pydantic.BaseModel):
    age: int = pydantic.Field(description="Estimated patient age in years, 18-95.")
    basis: str = pydantic.Field(
        description="Evidence in the transcript for the estimate."
    )


def has_numeric_age(profile):
    try:
        int(str(profile.get("age")).strip())
        return True
    except (TypeError, ValueError):
        return False


async def impute(cfg):
    profiles_dir = Path(cfg["case_descriptions_dir"])
    conversations_dir = Path(cfg["conversations_dir"])
    audit_path = Path(cfg["audit_path"])

    cases = [
        c.strip() for c in Path(cfg["cases_file"]).read_text().split() if c.strip()
    ]
    pending = []
    for c in cases:
        profile_path = profiles_dir / f"{c}.json"
        if not profile_path.exists():
            raise FileNotFoundError(f"no profile for case {c} at {profile_path}")
        if not has_numeric_age(json.loads(profile_path.read_text())):
            pending.append(c)

    log.info("%d of %d cases need an imputed age", len(pending), len(cases))
    if not pending:
        return

    llm = create_llm_instance(cfg["model_cfg"], keys_file=cfg["keys_file"])
    semaphore = asyncio.Semaphore(cfg["concurrency"])
    progress = tqdm(total=len(pending), desc="Imputing ages")
    records = []

    async def worker(case):
        turns = parse_transcript_file(str(conversations_dir / f"{case}.txt"))
        transcript = "\n".join(f"{speaker}: {text}" for speaker, text in turns)

        async with semaphore:
            try:
                response = await retry_on_transient_error(
                    lambda: llm.generate_response(
                        prompt=AGE_PROMPT.format(transcript=transcript),
                        outlines_class=AgeEstimate,
                    )
                )
            except Exception as e:
                log.error("Failed to impute age for %s: %s", case, e, exc_info=True)
                progress.update(1)
                return

        parsed = response["response"]
        estimate = (
            parsed
            if isinstance(parsed, AgeEstimate)
            else AgeEstimate.model_validate(parsed)
        )
        if not 18 <= estimate.age <= 95:
            log.error("%s: age %d outside 18-95, skipping", case, estimate.age)
            progress.update(1)
            return

        profile_path = profiles_dir / f"{case}.json"
        profile = json.loads(profile_path.read_text())
        profile["age"] = str(estimate.age)
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))

        records.append(
            {"case": case, "imputed_age": estimate.age, "basis": estimate.basis}
        )
        progress.update(1)

    await asyncio.gather(*(worker(c) for c in pending))
    progress.close()

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values("case")
    if audit_path.exists():
        frame = (
            pd.concat([pd.read_csv(audit_path), frame], ignore_index=True)
            .drop_duplicates("case", keep="last")
            .sort_values("case")
        )
    frame.to_csv(audit_path, index=False)
    log.info("Imputed %d ages -> %s", len(records), audit_path)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations-dir", required=True)
    parser.add_argument("--case-descriptions-dir", required=True)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--keys-file", required=True)
    parser.add_argument("--audit-path", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    cfg = {
        "conversations_dir": args.conversations_dir,
        "case_descriptions_dir": args.case_descriptions_dir,
        "cases_file": args.cases_file,
        "model_cfg": OmegaConf.to_container(
            OmegaConf.load(args.model_config), resolve=True
        ),
        "keys_file": args.keys_file,
        "audit_path": args.audit_path,
        "concurrency": args.concurrency,
    }
    asyncio.run(impute(cfg))


if __name__ == "__main__":
    main()
