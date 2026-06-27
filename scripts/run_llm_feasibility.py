#!/usr/bin/env python3
import asyncio
import json
import sys
from time import perf_counter
from pathlib import Path

import hydra
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.llm import LLM, VLLM
from patient_simulator.misc.utils import create_llm_instance
from patient_simulator.misc.llm_feasibility import (
    load_case_description,
    test_basic_simulation,
    test_downplay,
    test_emotional_state,
    test_personality_steering,
    test_relevant_field_classification,
)


TEST_REGISTRY = {
    "test_downplay": test_downplay,
    "test_emotional_state": test_emotional_state,
    "test_relevant_field_classification": test_relevant_field_classification,
    "test_basic_simulation": test_basic_simulation,
    "test_personality_steering": test_personality_steering,
}

META_TESTS = {
    "test_downplay",
    "test_emotional_state",
    "test_relevant_field_classification",
}

CONV_TESTS = {
    "test_basic_simulation",
    "test_personality_steering",
}


def append_results_row(
    output_csv: str | Path,
    model_name: str,
    score_results: dict[str, float],
    response_results: dict[str, object],
    meta_llm_score: float | None,
    conv_llm_score: float | None,
    runtime_seconds: float,
) -> None:
    """Append one row of feasibility results for a model to a CSV file."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, object] = {
        "model": model_name,
        "meta_llm_score": meta_llm_score,
        "conv_llm_score": conv_llm_score,
        "runtime_seconds": runtime_seconds,
    }

    for test_name in TEST_REGISTRY:
        row[test_name] = score_results.get(test_name)
        row[f"{test_name}_resp"] = json.dumps(response_results.get(test_name, []))

    new_row_df = pd.DataFrame([row])
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        all_columns = list(dict.fromkeys([*existing_df.columns, *new_row_df.columns]))
        existing_df = existing_df.reindex(columns=all_columns)
        new_row_df = new_row_df.reindex(columns=all_columns)
        out_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        out_df = new_row_df

    out_df.to_csv(output_path, index=False)


async def run_tests_for_model(cfg: DictConfig, model_name: str) -> dict[str, float]:
    """Run selected feasibility tests for one model and return score dictionary."""
    tests = list(cfg.tests)
    case_description = load_case_description(
        case_name=str(cfg.case_name),
        data_root=str(cfg.case_path),
    )

    for test_name in tests:
        if test_name not in TEST_REGISTRY:
            raise ValueError(f"Unknown test: {test_name}")

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["name"] = model_name

    LLM._response_cache.clear()
    llm = create_llm_instance(model_cfg)
    start = perf_counter()

    score_results: dict[str, float] = {}
    response_results: dict[str, object] = {}

    for test_name in tqdm(tests, desc=f"tests ({model_name})", leave=False):
        fn = TEST_REGISTRY[test_name]
        if test_name in META_TESTS:
            score, responses = await fn(llm)
        elif test_name == "test_personality_steering":
            score, responses = await fn(llm, llm, case_description)
        else:
            score, responses = await fn(llm, case_description)
        score_results[test_name] = float(score)
        response_results[test_name] = responses
        tqdm.write(f"[{model_name}] {test_name}: {score:.3f}")

    meta_scores = [score_results[name] for name in META_TESTS if name in score_results]
    conv_scores = [score_results[name] for name in CONV_TESTS if name in score_results]
    meta_llm_score = sum(meta_scores) / len(meta_scores) if meta_scores else None
    conv_llm_score = sum(conv_scores) / len(conv_scores) if conv_scores else None

    if meta_llm_score is not None:
        print(f"meta_llm_score: {meta_llm_score:.3f}")
    if conv_llm_score is not None:
        print(f"conv_llm_score: {conv_llm_score:.3f}")

    runtime_seconds = perf_counter() - start
    print(f"runtime_seconds: {runtime_seconds:.2f}")

    output_csv = str(cfg.get("output_csv", ""))
    if output_csv:
        append_results_row(
            output_csv=output_csv,
            model_name=model_name,
            score_results=score_results,
            response_results=response_results,
            meta_llm_score=meta_llm_score,
            conv_llm_score=conv_llm_score,
            runtime_seconds=runtime_seconds,
        )

    if isinstance(llm, VLLM):
        llm.cleanup()

    return score_results


async def run_tests(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Run selected feasibility tests for all configured models."""
    all_results: dict[str, dict[str, float]] = {}
    model_names = (
        [cfg.model.name] if isinstance(cfg.model.name, str) else cfg.model.name
    )

    for model_name in tqdm(model_names, desc="models"):
        all_results[model_name] = await run_tests_for_model(cfg, model_name)

    return all_results


@hydra.main(
    version_base=None,
    config_path="../configs/experiment/",
    config_name="llm_feasibility",
)
def main(cfg: DictConfig) -> None:
    """Run selected feasibility tests with Hydra configuration."""
    load_dotenv()
    print(OmegaConf.to_yaml(cfg))
    asyncio.run(run_tests(cfg))


if __name__ == "__main__":
    main()
