#!/usr/bin/env python3
import asyncio
import json
import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.eval import evaluate_conversation
from patient_simulator.misc.utils import create_llm_instance, retry_on_transient_error

log = logging.getLogger(__name__)
logging.getLogger("mistral_common").setLevel(logging.WARNING)


FILTER_COLUMN_MAP = {
    "conversations": "conversation_name",
    "patient_types": "patient_type",
    "patient_names": "patient_name",
    "models": "model_name",
}


def apply_filters(df: pd.DataFrame, filters: DictConfig | None) -> pd.DataFrame:
    if filters is None:
        return df
    for filter_key, column in FILTER_COLUMN_MAP.items():
        values = filters.get(filter_key)
        if values is None:
            continue
        values = list(values)
        if not values:
            continue
        if column not in df.columns:
            raise KeyError(
                f"Filter '{filter_key}' requires column '{column}' in all_conversations.csv"
            )
        df = df[df[column].isin(values)]
    return df


async def rerun_evals(cfg: DictConfig, eval_llm):
    conv_res_path = Path(cfg.output_dir) / "all_conversations.csv"
    if not conv_res_path.exists():
        raise FileNotFoundError(f"all_conversations.csv not found at {conv_res_path}")

    all_conv_df = pd.read_csv(conv_res_path)

    filtered = apply_filters(all_conv_df, cfg.get("filters"))
    if filtered.empty:
        raise ValueError(
            "No rows in all_conversations.csv match the configured filters"
        )

    metrics = list(cfg.metrics)
    log.info("Rerunning metrics %s on %d runs", metrics, len(filtered))

    for row_idx, row in tqdm(filtered.iterrows(), total=len(filtered), desc="Runs"):
        results_dir = Path(row["path"])
        turns_path = results_dir / "turns.csv"
        if not turns_path.exists():
            log.warning("turns.csv missing at %s, skipping", turns_path)
            continue

        turns_df = pd.read_csv(turns_path)

        case_desc = None
        case_desc_path = (
            Path(cfg.case_descriptions_dir) / f"{row['conversation_name']}.json"
        )
        if case_desc_path.exists():
            with open(case_desc_path, "r") as f:
                case_desc = json.load(f)
        elif "profile_fidelity" in metrics:
            log.warning(
                "Case description not found at %s; profile_fidelity will be skipped for this run",
                case_desc_path,
            )

        try:
            turn_res, conv_res, sim_profile = await retry_on_transient_error(
                lambda: evaluate_conversation(
                    turns_df,
                    eval_llm,
                    case_description=case_desc,
                    metrics=metrics,
                )
            )
        except Exception as e:
            log.error("Error re-evaluating %s: %s", results_dir, e, exc_info=True)
            continue

        turn_res.to_csv(turns_path, index=False)

        if "profile_fidelity" in metrics and case_desc is not None:
            with open(results_dir / "sim_profile.json", "w") as f:
                json.dump(sim_profile, f, indent=2)

        for key, value in conv_res.items():
            if key not in all_conv_df.columns:
                all_conv_df[key] = pd.Series([pd.NA] * len(all_conv_df), dtype=object)
            all_conv_df.at[row_idx, key] = value

    all_conv_df.to_csv(conv_res_path, index=False)
    log.info("Updated %s", conv_res_path)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="posthoc_eval",
)
def main(cfg: DictConfig):
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    if "metrics" not in cfg or not cfg.metrics:
        raise ValueError(
            "cfg.metrics must be a non-empty list of metric names to rerun"
        )

    eval_llm = create_llm_instance(cfg.llm_judge_model)

    asyncio.run(rerun_evals(cfg, eval_llm))

    if cfg.llm_judge_model.backend == "VLLM":
        eval_llm.cleanup()

    log.info("Post-hoc evaluation complete!")


if __name__ == "__main__":
    main()
