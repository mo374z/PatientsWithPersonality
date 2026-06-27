#!/usr/bin/env python3
"""Build data/helpmed/dataset.csv with LLM-extracted patient profiles.

Reads the raw HELPMed CSVs, extracts a structured PatientProfile for each scenario
via LLM, and writes a flat CSV combining profiles with ground-truth targets.
Run this once before running run_helpmed_benchmark.py.

Usage:
  python scripts/create_helpmed_dataset.py
  python scripts/create_helpmed_dataset.py concurrency=10
  python scripts/create_helpmed_dataset.py extractor_model=...
"""

import ast
import asyncio
import json
import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.data_extraction import (
    PatientProfile,
    extract_patient_profile_from_case_description,
)
from patient_simulator.misc.utils import create_llm_instance, retry_on_transient_error

log = logging.getLogger(__name__)

PROFILE_FIELDS = list(PatientProfile.model_fields.keys())
DATASET_COLUMNS = [
    "scenario_id",
    "condition",
    "urgency",
    "urgency_text",
    "full_differential",
    "red_flags",
    "context_text",
] + PROFILE_FIELDS


def load_raw_helpmed(data_dir: Path) -> pd.DataFrame:
    clean = pd.read_csv(data_dir / "main" / "clean_scenarios.csv").drop_duplicates("id")
    gold = pd.read_csv(data_dir / "main" / "scenarios.csv")
    return clean.merge(gold, left_on="id", right_on="scenario_id")


def format_case_description(context_json: dict) -> str:
    parts = [context_json.get("description", "")]
    for i in range(1, 5):
        heading = context_json.get(f"heading_{i}")
        body = context_json.get(f"body_{i}")
        if heading and body:
            parts.append(f"{heading}: {body}")
    return "\n\n".join(parts)


async def extract_one(row: pd.Series, llm, sem: asyncio.Semaphore) -> dict:
    context = json.loads(row["context_json"])
    context_text = format_case_description(context)

    async with sem:
        profile = await retry_on_transient_error(
            lambda: extract_patient_profile_from_case_description(context_text, llm)
        )

    full_diff = (
        ast.literal_eval(row["full_differential"])
        if isinstance(row["full_differential"], str)
        else list(row["full_differential"])
    )
    red_flags = (
        ast.literal_eval(row["red_flags"])
        if isinstance(row["red_flags"], str)
        else list(row["red_flags"])
    )

    return {
        "scenario_id": int(row["id"]),
        "condition": row["condition"],
        "urgency": int(row["urgency"]),
        "urgency_text": row["urgency_text"],
        "full_differential": json.dumps([c.lower() for c in full_diff]),
        "red_flags": json.dumps([c.lower() for c in red_flags]),
        "context_text": context_text,
        **profile,
    }


async def _run(cfg: DictConfig):
    data_dir = Path(cfg.data_dir)
    dataset_path = Path(cfg.dataset_path)
    concurrency = int(cfg.concurrency)

    merged = load_raw_helpmed(data_dir)
    log.info("Loaded %d scenarios from %s", len(merged), data_dir)

    llm = create_llm_instance(cfg.extractor_model, use_cache=False)
    sem = asyncio.Semaphore(concurrency)

    tasks = [extract_one(row, llm, sem) for _, row in merged.iterrows()]
    rows = await atqdm.gather(*tasks, desc="Extracting profiles")

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    df.to_csv(dataset_path, index=False)
    log.info("Saved %d rows to %s", len(df), dataset_path)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="helpmed_dataset",
)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
