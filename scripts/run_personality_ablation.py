#!/usr/bin/env python3
import asyncio
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.personality_ablation import (
    collect_runs,
    parse_windows,
    score_runs,
)
from patient_simulator.misc.utils import create_llm_instance

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="personality_ablation",
)
def main(cfg: DictConfig):
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    windows = parse_windows(cfg.windows)
    prompt_variants = list(cfg.prompt_variants)
    runs = collect_runs(cfg.output_dir, list(cfg.patient_names), cfg.model_name)
    log.info(
        "Scoring %d runs x %d windows x %d judge prompts",
        len(runs),
        len(windows),
        len(prompt_variants),
    )

    judge_llm = create_llm_instance(cfg.llm_judge_model)
    scores = asyncio.run(
        score_runs(runs, judge_llm, windows, prompt_variants, int(cfg.concurrency))
    )

    scores_path = Path(cfg.output_dir) / cfg.scores_file
    scores.to_csv(scores_path, index=False)
    log.info("Wrote %d reconstructions to %s", len(scores), scores_path)


if __name__ == "__main__":
    main()
