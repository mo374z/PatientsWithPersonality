#!/usr/bin/env python3
import asyncio
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.utils import create_llm_instance
from patient_simulator.patients.pwp import PatientsWithPersonality


CEFR_CHOICES = ["A", "B", "C"]


async def _build_dataset(
    profiles: list[dict[str, Any]],
    meta_llm,
    rows_per_profile: int,
    seed: int,
    register: str,
    answer_form: float,
    test_fraction: float,
) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    shuffled = list(range(len(profiles)))
    rng.shuffle(shuffled)
    test_profiles = set(shuffled[: max(1, round(len(profiles) * test_fraction))])

    for profile_idx, profile in enumerate(profiles):
        for sample_idx in range(rows_per_profile):
            cefr_level = rng.choice(CEFR_CHOICES)
            h_trait = rng.randint(1, 3)
            e_trait = rng.randint(1, 3)
            x_trait = rng.randint(1, 3)
            a_trait = rng.randint(1, 3)
            c_trait = rng.randint(1, 3)
            o_trait = rng.randint(1, 3)

            patient = PatientsWithPersonality(
                case_description=profile,
                h=h_trait,
                e=e_trait,
                x=x_trait,
                a=a_trait,
                c=c_trait,
                o=o_trait,
                llm=meta_llm,
                meta_llm=meta_llm,
                level=cefr_level,
                register=register,
                answer_form=answer_form,
                dynamic_case_description=False,
                verbosity=0,
            )
            await patient._initialize_dynamic_prompts()

            rows.append(
                {
                    "row_id": f"profile_{profile_idx}_sample_{sample_idx}",
                    "x": patient.build_role_inputs(),
                    "profile_json": json.dumps(profile),
                    "cefr_level": cefr_level,
                    "h_trait": h_trait,
                    "e_trait": e_trait,
                    "x_trait": x_trait,
                    "a_trait": a_trait,
                    "c_trait": c_trait,
                    "o_trait": o_trait,
                    "split": "test" if profile_idx in test_profiles else "train",
                }
            )

    return pd.DataFrame(rows)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="metaprompt_tuning_dataset",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    with Path(cfg.profiles_path).open("r", encoding="utf-8") as f:
        profiles_data = json.load(f)

    profiles = profiles_data["profiles"]

    total_rows = int(cfg.total_rows)

    if total_rows % len(profiles) != 0:
        raise ValueError("total_rows must be divisible by number of selected profiles.")

    rows_per_profile = total_rows // len(profiles)

    meta_llm = create_llm_instance(cfg.meta_model)
    dataset = asyncio.run(
        _build_dataset(
            profiles=profiles,
            meta_llm=meta_llm,
            rows_per_profile=rows_per_profile,
            seed=int(cfg.seed),
            register=str(cfg.register),
            answer_form=float(cfg.answer_form),
            test_fraction=float(cfg.test_fraction),
        )
    )

    if len(dataset) != total_rows:
        raise ValueError(
            f"Created dataset must contain {total_rows} rows, got {len(dataset)}"
        )

    output_path = Path(str(cfg.output_dataset_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    print(f"Saved tuning dataset to {output_path} with {len(dataset)} rows")


if __name__ == "__main__":
    main()
