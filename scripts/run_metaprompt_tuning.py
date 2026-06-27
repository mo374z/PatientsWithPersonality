#!/usr/bin/env python3
import asyncio
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
import random
import numpy as np
import torch

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from promptolution.helpers import get_optimizer, get_predictor, get_task
from promptolution.llms.base_llm import BaseLLM
from promptolution.utils.callbacks import (
    FileOutputCallback,
)
from promptolution.utils import ExperimentConfig

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.eval import (
    evaluate_fact_fidelity,
    evaluate_personal_information_presence,
    reconstruct_personality_params,
)
from patient_simulator.misc.llm import VLLM as PatientVLLM
from patient_simulator.misc.utils import (
    create_llm_instance,
    is_transient_api_error,
    load_openrouter_api_key,
)
from patient_simulator.patients.pwp import PatientsWithPersonality

PROBE_SPECS = [
    {
        "question": "Can you tell me when your symptoms first started and how they've changed since then? What have you done to manage your symptoms so far?",
        "axes": ["X", "E"],
    },
    {
        "question": "Can you think of anything that might be causing your symptoms?",
        "axes": ["O", "E"],
    },
    {
        "question": "Considering your symptoms, is there anything you are particularly worried about?",
        "axes": ["A", "X"],
    },
    {
        "question": "From what you've described so far, I don't think it's anything serious. This will probably go away on its own.",
        "axes": ["A", "O"],
    },
]


class TunedPatientsWithPersonality(PatientsWithPersonality):
    def __init__(self, *args: Any, forced_latent_role: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._forced_latent_role = forced_latent_role

    async def _generate_latent_role_description(self) -> str:
        return self._forced_latent_role


class PromptolutionVLLMAdapter(BaseLLM):
    """Expose patient-simulator VLLM via promptolution's BaseLLM interface."""

    def __init__(
        self,
        shared_vllm: PatientVLLM,
        config: ExperimentConfig,
    ) -> None:
        self._shared_vllm = shared_vllm
        super().__init__(config=config)
        # Enables promptolution token counting via tokenizer-aware path.
        self.tokenizer = self._shared_vllm.engine.get_tokenizer()

    def _get_response(self, prompts: list[str], system_prompts: list[str]) -> list[str]:
        async def _run_batch() -> list[str]:
            results = await self._shared_vllm.generate_batch(
                prompts=prompts,
                system_instructions=system_prompts,
            )
            return [str(item["response"]) for item in results]

        return asyncio.run(_run_batch())

    def set_generation_seed(self, seed: int) -> None:
        # Keep seed behavior aligned with promptolution optimizers.
        self._shared_vllm.default_sampling_kwargs["seed"] = int(seed)


def _personality_distance(
    reconstructed: dict[str, int | None],
    target: dict[str, int],
) -> float:
    total = 0.0
    max_missing_penalty = 2.0
    for trait_key, target_value in target.items():
        reconstructed_value = reconstructed.get(trait_key)
        if reconstructed_value is None:
            total += max_missing_penalty
            continue
        total += abs(target_value - reconstructed_value)
    # return the average absolute distance across traits
    return total / len(target.keys()) if target.keys() else 0.0


class RewardEvaluator:
    _DEFAULT_REWARD_CACHE_COLUMNS = [
        "meta_prompt",
        "row_id",
        "cefr_level",
        "h_trait",
        "e_trait",
        "x_trait",
        "a_trait",
        "c_trait",
        "o_trait",
        "h_recon",
        "e_recon",
        "x_recon",
        "a_recon",
        "c_recon",
        "o_recon",
        "information_penalty",
        "personality_penalty",
        "personal_info_penalty",
        "reward",
        "fidelity_distribution",
        "fidelity_extracted_facts",
        "fidelity_fact_verdicts",
        "fidelity_input_token_count",
        "fidelity_output_token_count",
        "personal_info_distribution",
        "personal_info_fields",
        "personal_info_presence",
        "personal_info_input_token_count",
        "personal_info_output_token_count",
        "score_async_input_token_count",
        "score_async_output_token_count",
        "probe_logs",
        "latent_role_prediction",
    ]

    def __init__(
        self,
        conversational_llm,
        patient_meta_llm,
        judge_llm,
        reward_log_path: str | Path | None = None,
        reward_cache_path: str | Path | None = None,
        resource_exhausted_max_attempts: int = 6,
        resource_exhausted_initial_delay_seconds: float = 15.0,
        resource_exhausted_max_delay_seconds: float = 300.0,
    ):
        self.conversational_llm = conversational_llm
        self.patient_meta_llm = patient_meta_llm
        self.judge_llm = judge_llm
        self.reward_log_path = (
            Path(reward_log_path) if reward_log_path is not None else None
        )
        self.reward_cache_path = (
            Path(reward_cache_path) if reward_cache_path is not None else None
        )
        self.resource_exhausted_max_attempts = int(resource_exhausted_max_attempts)
        self.resource_exhausted_initial_delay_seconds = float(
            resource_exhausted_initial_delay_seconds
        )
        self.resource_exhausted_max_delay_seconds = float(
            resource_exhausted_max_delay_seconds
        )
        self._pending_model_inputs: deque[str] = deque()
        self._optimization_step: int | None = None
        self._reward_cache_columns = list(self._DEFAULT_REWARD_CACHE_COLUMNS)
        self._reward_cache: dict[tuple[str, str], dict[str, Any]] = {}

        if self.reward_log_path is not None:
            self.reward_log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.reward_cache_path is not None:
            self.reward_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_reward_cache()

        if self.resource_exhausted_max_attempts <= 0:
            raise ValueError("resource_exhausted_max_attempts must be > 0.")
        if self.resource_exhausted_initial_delay_seconds <= 0:
            raise ValueError("resource_exhausted_initial_delay_seconds must be > 0.")
        if self.resource_exhausted_max_delay_seconds <= 0:
            raise ValueError("resource_exhausted_max_delay_seconds must be > 0.")

    @staticmethod
    def _is_fact_fidelity_length_mismatch_error(exc: Exception) -> bool:
        if not isinstance(exc, ValueError):
            return False
        return str(exc).startswith("Fact fidelity output length mismatch:")

    @staticmethod
    def _is_personal_info_length_mismatch_error(exc: Exception) -> bool:
        if not isinstance(exc, ValueError):
            return False
        return str(exc).startswith(
            "Personal information presence output length mismatch:"
        )

    @staticmethod
    def _is_empty_json_validation_error(exc: Exception) -> bool:
        message = str(exc)
        markers = [
            "Invalid JSON: EOF while parsing a value",
            "type=json_invalid",
        ]
        return all(marker in message for marker in markers)

    def register_model_inputs(self, inputs: list[str]) -> None:
        self._pending_model_inputs.extend(inputs)

    def set_optimization_step(self, step: int) -> None:
        self._optimization_step = int(step)

    def clear_optimization_step(self) -> None:
        self._optimization_step = None

    def _pop_meta_prompt(self, x_input: str) -> str:
        if not self._pending_model_inputs:
            raise ValueError("Missing predictor input for reward evaluation row.")

        full_input = self._pending_model_inputs.popleft()
        suffix = f"\n{x_input}"
        print("⚠️ Full predictor input:", repr(full_input))
        if not full_input.endswith(suffix):
            raise ValueError("Predictor input format mismatch.")
        return full_input[: -len(suffix)]

    def _flush_log_row(self, row: dict[str, Any]) -> None:
        if self.reward_log_path is None:
            return

        reward_log_columns = ["time", "optimization_step", *self._reward_cache_columns]
        missing_columns = [column for column in reward_log_columns if column not in row]
        if missing_columns:
            missing_columns_text = ", ".join(missing_columns)
            raise ValueError(
                f"Reward log row is missing required columns: {missing_columns_text}"
            )

        pd.DataFrame([row], columns=reward_log_columns).to_csv(
            self.reward_log_path,
            mode="a",
            header=not self.reward_log_path.exists(),
            index=False,
        )

    def _load_reward_cache(self) -> None:
        if self.reward_cache_path is None or not self.reward_cache_path.exists():
            return

        cache_df = pd.read_csv(self.reward_cache_path)
        required_columns = {"meta_prompt", "row_id", "reward"}
        missing_columns = required_columns - set(cache_df.columns)
        if missing_columns:
            missing_columns_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Reward cache file is missing required columns: {missing_columns_text}"
            )

        self._reward_cache_columns = [str(column) for column in cache_df.columns]
        for row in cache_df.itertuples(index=False):
            meta_prompt = str(getattr(row, "meta_prompt"))
            row_id = str(getattr(row, "row_id"))
            row_payload = {
                column: getattr(row, column, "")
                for column in self._reward_cache_columns
            }
            self._reward_cache[(meta_prompt, row_id)] = row_payload

    def _get_cached_reward_row(
        self,
        meta_prompt: str,
        row_id: str,
    ) -> dict[str, Any] | None:
        return self._reward_cache.get((meta_prompt, row_id))

    def _store_cached_reward_row(self, row: dict[str, Any]) -> None:
        meta_prompt = str(row["meta_prompt"])
        row_id = str(row["row_id"])
        key = (meta_prompt, row_id)
        if key in self._reward_cache:
            return

        cache_row = {
            column: row.get(column, "") for column in self._reward_cache_columns
        }
        self._reward_cache[key] = cache_row

        if self.reward_cache_path is None:
            return

        pd.DataFrame([cache_row], columns=self._reward_cache_columns).to_csv(
            self.reward_cache_path,
            mode="a",
            header=not self.reward_cache_path.exists(),
            index=False,
        )

    async def _score_async(
        self,
        prediction: str,
        row_id: str,
        x_input: str,
        profile_json: str,
        cefr_level: str,
        h_trait: int,
        e_trait: int,
        x_trait: int,
        a_trait: int,
        c_trait: int,
        o_trait: int,
        explicit_meta_prompt: str | None = None,
    ) -> float:
        token_usage = {
            "input_token_count": 0,
            "output_token_count": 0,
        }
        originals: list[tuple[Any, Any]] = []
        seen: set[int] = set()
        llms = [self.conversational_llm, self.patient_meta_llm, self.judge_llm]

        for llm in llms:
            llm_id = id(llm)
            if llm_id in seen:
                continue
            seen.add(llm_id)

            original_generate_response = getattr(llm, "generate_response", None)
            if original_generate_response is None:
                continue

            async def wrapped_generate_response(
                *args, _orig=original_generate_response, **kwargs
            ):
                result = await _orig(*args, **kwargs)
                token_usage["input_token_count"] += int(
                    result.get("prompt_token_count", 0)
                )
                token_usage["output_token_count"] += int(
                    result.get("output_token_count", 0)
                )
                return result

            setattr(llm, "generate_response", wrapped_generate_response)
            originals.append((llm, original_generate_response))

        try:
            case_description_raw = json.loads(profile_json)
            forced_latent_role = prediction.strip()

            patient = TunedPatientsWithPersonality(
                case_description=case_description_raw,
                h=int(h_trait),
                e=int(e_trait),
                x=int(x_trait),
                a=int(a_trait),
                c=int(c_trait),
                o=int(o_trait),
                llm=self.conversational_llm,
                meta_llm=self.patient_meta_llm,
                level=str(cefr_level),
                verbosity=1,
                forced_latent_role=forced_latent_role,
            )

            trait_targets = {
                "H": int(h_trait),
                "E": int(e_trait),
                "X": int(x_trait),
                "A": int(a_trait),
                "C": int(c_trait),
                "O": int(o_trait),
            }

            probe_logs: list[dict[str, Any]] = []
            personality_penalties: list[float] = []
            all_probe_responses: list[str] = []

            print("✨ Starting EVAL")

            for probe_idx, probe in enumerate(PROBE_SPECS, start=1):
                question = str(probe["question"])
                selected_axes = [
                    axis for axis in probe["axes"] if axis in trait_targets
                ]

                patient_response = await patient.get_response(question)
                print("PATIENT RESPONSE:", patient_response)
                all_probe_responses.append(patient_response)

                probe_conversation = f"Doctor: {question}\nPatient: {patient_response}"
                reconstructed = await reconstruct_personality_params(
                    conversation=probe_conversation,
                    llm=self.judge_llm,
                    subset=set(selected_axes),
                )
                personality_target_subset = {
                    axis: trait_targets[axis] for axis in selected_axes
                }
                personality_penalty_probe = _personality_distance(
                    reconstructed=reconstructed,
                    target=personality_target_subset,
                )
                personality_penalties.append(personality_penalty_probe)

                probe_logs.append(
                    {
                        "probe_idx": probe_idx,
                        "question": question,
                        "selected_axes": selected_axes,
                        "response": patient_response,
                        "reconstructed_axes": reconstructed,
                        "personality_penalty": personality_penalty_probe,
                    }
                )

            print("Evaluating information fidelity...")

            (
                fidelity_distribution_total,
                fact_fidelity_details,
            ) = await evaluate_fact_fidelity(
                patient_turns=all_probe_responses,
                personal_fields=x_input,
                case_data=patient.case_description,
                llm=self.judge_llm,
            )

            (
                personal_info_distribution,
                personal_info_details,
            ) = await evaluate_personal_information_presence(
                personal_fields=x_input,
                latent_role=forced_latent_role,
                llm=self.judge_llm,
            )

            total = sum(fidelity_distribution_total.values())
            if total == 0:
                information_penalty = 99.0
            else:
                information_penalty = (
                    fidelity_distribution_total["Hallucination"] / total
                )

            personal_info_total = sum(personal_info_distribution.values())
            if personal_info_total == 0:
                personal_info_penalty = 99.0
            else:
                personal_info_penalty = (
                    personal_info_distribution["Missing"] / personal_info_total
                )

            if personality_penalties:
                personality_penalty = float(
                    sum(personality_penalties) / len(personality_penalties)
                )
            else:
                personality_penalty = 99.0

            reconstructed_axis_values = {
                axis: [
                    float(probe_log["reconstructed_axes"][axis])
                    for probe_log in probe_logs
                    if axis in probe_log["reconstructed_axes"]
                ]
                for axis in trait_targets
            }
            reconstructed_axis_means = {
                axis: (sum(values) / len(values) if values else None)
                for axis, values in reconstructed_axis_values.items()
            }

            print(f"Information penalty: {information_penalty:.4f}")
            print(f"Personality penalty: {personality_penalty:.4f}")
            print(f"Personal information penalty: {personal_info_penalty:.4f}")

            reward = -(
                information_penalty + personality_penalty + personal_info_penalty
            )
            meta_prompt = (
                explicit_meta_prompt
                if explicit_meta_prompt is not None
                else self._pop_meta_prompt(x_input=x_input)
            )

            log_row = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "optimization_step": self._optimization_step,
                "meta_prompt": meta_prompt,
                "row_id": row_id,
                "cefr_level": str(cefr_level),
                "h_trait": int(h_trait),
                "e_trait": int(e_trait),
                "x_trait": int(x_trait),
                "a_trait": int(a_trait),
                "c_trait": int(c_trait),
                "o_trait": int(o_trait),
                "h_recon": reconstructed_axis_means.get("H"),
                "e_recon": reconstructed_axis_means.get("E"),
                "x_recon": reconstructed_axis_means.get("X"),
                "a_recon": reconstructed_axis_means.get("A"),
                "c_recon": reconstructed_axis_means.get("C"),
                "o_recon": reconstructed_axis_means.get("O"),
                "information_penalty": information_penalty,
                "personality_penalty": personality_penalty,
                "personal_info_penalty": personal_info_penalty,
                "reward": reward,
                "fidelity_distribution": json.dumps(fidelity_distribution_total),
                "fidelity_extracted_facts": json.dumps(
                    fact_fidelity_details["extracted_facts"],
                    ensure_ascii=False,
                ),
                "fidelity_fact_verdicts": json.dumps(
                    fact_fidelity_details["fact_verdicts"],
                    ensure_ascii=False,
                ),
                "fidelity_input_token_count": int(
                    fact_fidelity_details["input_token_count"]
                ),
                "fidelity_output_token_count": int(
                    fact_fidelity_details["output_token_count"]
                ),
                "personal_info_distribution": json.dumps(personal_info_distribution),
                "personal_info_fields": json.dumps(
                    personal_info_details["personal_info_fields"],
                    ensure_ascii=False,
                ),
                "personal_info_presence": json.dumps(
                    personal_info_details["personal_info_presence"],
                    ensure_ascii=False,
                ),
                "personal_info_input_token_count": int(
                    personal_info_details["input_token_count"]
                ),
                "personal_info_output_token_count": int(
                    personal_info_details["output_token_count"]
                ),
                "score_async_input_token_count": int(token_usage["input_token_count"]),
                "score_async_output_token_count": int(
                    token_usage["output_token_count"]
                ),
                "probe_logs": json.dumps(probe_logs),
                "latent_role_prediction": forced_latent_role,
            }
            self._flush_log_row(log_row)
            self._store_cached_reward_row(log_row)
            return reward
        finally:
            for llm, original_generate_response in originals:
                setattr(llm, "generate_response", original_generate_response)

    def __call__(
        self,
        prediction: str,
        **kwargs: Any,
    ) -> float:
        required_fields = {
            "row_id",
            "x_input",
            "profile_json",
            "cefr_level",
            "h_trait",
            "e_trait",
            "x_trait",
            "a_trait",
            "c_trait",
            "o_trait",
        }
        missing_fields = required_fields - set(kwargs.keys())
        if missing_fields:
            missing_fields_text = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Missing reward kwargs for RewardEvaluator: {missing_fields_text}"
            )

        row_id = str(kwargs["row_id"])
        x_input = str(kwargs["x_input"])
        meta_prompt = (
            str(kwargs["meta_prompt"])
            if "meta_prompt" in kwargs and kwargs["meta_prompt"] is not None
            else self._pop_meta_prompt(x_input=x_input)
        )

        cached_reward_row = self._get_cached_reward_row(
            meta_prompt=meta_prompt,
            row_id=row_id,
        )
        if cached_reward_row is not None:
            log_row = dict(cached_reward_row)
            log_row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_row["optimization_step"] = self._optimization_step
            log_row["meta_prompt"] = meta_prompt
            log_row["row_id"] = row_id
            self._flush_log_row(log_row)
            print(f"Using cached reward for row_id={row_id}.")
            return float(log_row["reward"])

        for attempt in range(self.resource_exhausted_max_attempts):
            try:
                reward = asyncio.run(
                    self._score_async(
                        prediction=prediction,
                        row_id=row_id,
                        x_input=x_input,
                        profile_json=str(kwargs["profile_json"]),
                        cefr_level=str(kwargs["cefr_level"]),
                        h_trait=int(kwargs["h_trait"]),
                        e_trait=int(kwargs["e_trait"]),
                        x_trait=int(kwargs["x_trait"]),
                        a_trait=int(kwargs["a_trait"]),
                        c_trait=int(kwargs["c_trait"]),
                        o_trait=int(kwargs["o_trait"]),
                        explicit_meta_prompt=meta_prompt,
                    )
                )
                return float(reward)
            except Exception as exc:
                is_last_attempt = attempt >= self.resource_exhausted_max_attempts - 1
                is_retryable = (
                    is_transient_api_error(exc)
                    or self._is_fact_fidelity_length_mismatch_error(exc)
                    or self._is_personal_info_length_mismatch_error(exc)
                    or self._is_empty_json_validation_error(exc)
                )
                if is_last_attempt or not is_retryable:
                    raise

                delay = min(
                    self.resource_exhausted_initial_delay_seconds * (2**attempt),
                    self.resource_exhausted_max_delay_seconds,
                )
                delay *= random.uniform(0.8, 1.2)
                print(
                    "Transient API error, temporary fact-fidelity output mismatch, or empty JSON response during reward eval; "
                    f"retrying in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{self.resource_exhausted_max_attempts})."
                )
                time.sleep(delay)

        raise RuntimeError("Exhausted retry loop in RewardEvaluator.__call__.")


def _load_initial_prompts(path: str) -> list[str]:
    prompts_path = Path(path)
    with prompts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "prompts" in data:
        prompts = data["prompts"]
    else:
        prompts = data

    if not isinstance(prompts, list) or not all(
        isinstance(prompt, str) for prompt in prompts
    ):
        raise ValueError("Initial prompts file must be a JSON list of strings.")

    return prompts


def _to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    out = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(out, dict):
        raise ValueError("Expected mapping config.")
    return out


def _run_experiment(
    train_df: pd.DataFrame,
    loop_config: ExperimentConfig,
    meta_llm: BaseLLM,
    reward_evaluator: RewardEvaluator,
    output_dir: Path,
) -> list[str]:
    train_df = train_df.copy()
    train_df["x_input"] = train_df["x"]

    predictor = get_predictor(meta_llm, config=loop_config)
    original_predict = predictor.predict
    original_get_response = predictor.llm.get_response

    def get_response_with_logging(
        prompts: list[str],
        system_prompts: list[str] | None = None,
    ) -> list[str]:
        reward_evaluator.register_model_inputs(list(prompts))
        return original_get_response(prompts, system_prompts=system_prompts)

    def predict_with_logging(
        prompts: str | list[str],
        xs: list[str],
        system_prompts: str | list[str] | None = None,
    ):
        predictor.llm.get_response = get_response_with_logging
        try:
            return original_predict(prompts, xs, system_prompts=system_prompts)
        finally:
            predictor.llm.get_response = original_get_response

    predictor.predict = predict_with_logging

    task_train = get_task(train_df, loop_config)

    optimizer = get_optimizer(
        predictor=predictor,
        meta_llm=meta_llm,
        task=task_train,
        config=loop_config,
    )

    original_step = optimizer._step
    optimization_step_counter = 0

    def step_with_logging_context():
        nonlocal optimization_step_counter
        optimization_step_counter += 1
        reward_evaluator.set_optimization_step(optimization_step_counter)
        try:
            return original_step()
        finally:
            reward_evaluator.clear_optimization_step()

    optimizer._step = step_with_logging_context

    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer.callbacks.extend(
        [
            FileOutputCallback(dir=str(output_dir), file_type="csv"),
        ]
    )

    prompts = optimizer.optimize(n_steps=loop_config.n_steps)

    prompt_texts = []
    for prompt in prompts:
        if hasattr(prompt, "construct_prompt"):
            prompt_texts.append(prompt.construct_prompt())
        else:
            prompt_texts.append(str(prompt))

    return prompt_texts


def _load_last_step_prompts(step_results_path: Path) -> pd.DataFrame:
    if not step_results_path.exists():
        raise FileNotFoundError(
            "Missing optimizer output file: "
            f"{step_results_path}. "
            "Optimization likely exited early due to upstream failures (for example quota exhaustion)."
        )

    step_results = pd.read_csv(step_results_path)
    if step_results.empty:
        raise ValueError(f"No rows found in {step_results_path}")

    last_step = pd.to_numeric(step_results["step"], errors="raise").max()
    last_step_rows = step_results.loc[
        step_results["step"] == last_step,
        ["prompt", "score"],
    ].copy()
    last_step_rows["score"] = pd.to_numeric(last_step_rows["score"], errors="raise")
    last_step_rows = (
        last_step_rows.drop_duplicates(subset=["prompt"])
        .rename(columns={"score": "last_step_score"})
        .reset_index(drop=True)
    )
    return last_step_rows


def _evaluate_last_step_prompts(
    test_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    loop_config: ExperimentConfig,
    meta_llm: BaseLLM,
    reward_evaluator: RewardEvaluator,
    eval_results_path: Path,
) -> pd.DataFrame:
    eval_df = test_df.copy().reset_index(drop=True)
    eval_df["x_input"] = eval_df["x"]

    predictor = get_predictor(meta_llm, config=loop_config)
    eval_records: list[dict[str, Any]] = []

    for row in prompts_df.itertuples(index=False):
        prompt = str(row.prompt)
        last_step_score = float(row.last_step_score)

        rewards: list[float] = []
        for sample_row in eval_df.itertuples(index=False):
            predictions, _ = predictor.predict(prompt, [str(sample_row.x)])
            if len(predictions) != 1:
                raise ValueError(
                    f"Expected 1 prediction per eval row, got {len(predictions)}."
                )

            prediction = str(predictions[0])
            reward = reward_evaluator(
                prediction=prediction,
                meta_prompt=prompt,
                row_id=str(sample_row.row_id),
                x_input=str(sample_row.x_input),
                profile_json=str(sample_row.profile_json),
                cefr_level=str(sample_row.cefr_level),
                h_trait=int(sample_row.h_trait),
                e_trait=int(sample_row.e_trait),
                x_trait=int(sample_row.x_trait),
                a_trait=int(sample_row.a_trait),
                c_trait=int(sample_row.c_trait),
                o_trait=int(sample_row.o_trait),
            )
            rewards.append(float(reward))

        rewards_series = pd.Series(rewards, dtype=float)
        eval_records.append(
            {
                "prompt": prompt,
                "last_step_score": last_step_score,
                "eval_mean_reward": float(rewards_series.mean()),
                "eval_std_reward": float(rewards_series.std(ddof=0)),
                "n_eval": int(len(rewards)),
            }
        )

    eval_results = pd.DataFrame(eval_records).sort_values(
        "eval_mean_reward", ascending=False
    )
    eval_results.to_csv(eval_results_path, index=False)
    return eval_results


def seed_everything(seed: int):
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _extract_profile_index(row_id: str) -> int:
    match = re.match(r"^profile_(\d+)_", row_id)
    if match is None:
        raise ValueError(f"Invalid row_id format: {row_id}")
    return int(match.group(1))


def _blocked_profile_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile_idx = dataset["row_id"].astype(str).map(_extract_profile_index)

    train_mask = profile_idx.between(0, 6)
    test_mask = profile_idx.between(7, 9)

    train_df = dataset.loc[train_mask].copy().reset_index(drop=True)
    test_df = dataset.loc[test_mask].copy().reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Blocked split produced an empty train set.")
    if test_df.empty:
        raise ValueError("Blocked split produced an empty test set.")

    return train_df, test_df


def _sample_absolute(
    split_df: pd.DataFrame,
    sample_n: int | None,
    random_seed: int,
    split_name: str,
) -> pd.DataFrame:
    if sample_n is None:
        return split_df.reset_index(drop=True)

    if sample_n <= 0:
        raise ValueError(f"{split_name}_sample_n must be > 0, got {sample_n}.")

    available = len(split_df)
    if sample_n > available:
        raise ValueError(
            f"{split_name}_sample_n={sample_n} exceeds available {split_name} rows={available}."
        )

    return split_df.sample(n=sample_n, random_state=random_seed).reset_index(drop=True)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
    config_name="metaprompt_tuning",
)
def main(cfg: DictConfig) -> None:
    random_seed = int(cfg.random_seed)
    seed_everything(random_seed)

    dataset = pd.read_csv(cfg.dataset_path)
    train_df, test_eval_df = _blocked_profile_split(dataset)

    train_sample_n = (
        int(cfg.train_sample_n) if cfg.get("train_sample_n") is not None else None
    )
    test_sample_n = (
        int(cfg.test_sample_n) if cfg.get("test_sample_n") is not None else None
    )

    train_df = _sample_absolute(
        split_df=train_df,
        sample_n=train_sample_n,
        random_seed=random_seed,
        split_name="train",
    )
    test_eval_df = _sample_absolute(
        split_df=test_eval_df,
        sample_n=test_sample_n,
        random_seed=random_seed,
        split_name="test",
    )

    promptolution_cfg = _to_plain_dict(cfg.promptolution)
    output_dir = Path(str(cfg.output_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    step_results_path = output_dir / "step_results.csv"
    eval_results_path = output_dir / "eval_results.csv"
    step_reward_log_path = output_dir / "step_reward.csv"
    eval_reward_log_path = output_dir / "eval_reward.csv"
    reward_cache_path = output_dir / "reward_cache.csv"

    shared_meta_llm = create_llm_instance(cfg["meta_model"], "keys.json")
    if not isinstance(shared_meta_llm, PatientVLLM):
        raise TypeError(
            "meta_model.backend must be VLLM for shared promptolution/patient meta usage."
        )

    # Keep promptolution config compatible while sourcing actual engine from meta_model.
    promptolution_cfg["model_id"] = f"vllm-{cfg.meta_model.name}"

    conversational_llm = create_llm_instance(cfg["conversational_model"], "keys.json")
    patient_meta_llm = shared_meta_llm
    judge_llm = create_llm_instance(cfg["llm_judge_model"], "keys.json")

    quota_retry_cfg = cfg.get("quota_retry")
    resource_exhausted_max_attempts = (
        int(quota_retry_cfg.max_attempts) if quota_retry_cfg is not None else 6
    )
    resource_exhausted_initial_delay_seconds = (
        float(quota_retry_cfg.initial_delay_seconds)
        if quota_retry_cfg is not None
        else 15.0
    )
    resource_exhausted_max_delay_seconds = (
        float(quota_retry_cfg.max_delay_seconds)
        if quota_retry_cfg is not None
        else 300.0
    )

    step_reward_evaluator = RewardEvaluator(
        conversational_llm=conversational_llm,
        patient_meta_llm=patient_meta_llm,
        judge_llm=judge_llm,
        reward_log_path=step_reward_log_path,
        reward_cache_path=reward_cache_path,
        resource_exhausted_max_attempts=resource_exhausted_max_attempts,
        resource_exhausted_initial_delay_seconds=resource_exhausted_initial_delay_seconds,
        resource_exhausted_max_delay_seconds=resource_exhausted_max_delay_seconds,
    )
    eval_reward_evaluator = RewardEvaluator(
        conversational_llm=conversational_llm,
        patient_meta_llm=patient_meta_llm,
        judge_llm=judge_llm,
        reward_log_path=eval_reward_log_path,
        reward_cache_path=reward_cache_path,
        resource_exhausted_max_attempts=resource_exhausted_max_attempts,
        resource_exhausted_initial_delay_seconds=resource_exhausted_initial_delay_seconds,
        resource_exhausted_max_delay_seconds=resource_exhausted_max_delay_seconds,
    )

    init_prompts = _load_initial_prompts(str(cfg.initial_prompts_path))

    experiment_kwargs = {
        "task_description": str(cfg.task_description),
        "prompts": init_prompts,
        "x_column": "x",
        "task_type": "reward",
        "reward_function": step_reward_evaluator,
        "reward_columns": [
            "row_id",
            "x_input",
            "profile_json",
            "cefr_level",
            "h_trait",
            "e_trait",
            "x_trait",
            "a_trait",
            "c_trait",
            "o_trait",
        ],
        "begin_marker": "<role>",
        "end_marker": "</role>",
    }
    promptolution_model_id = str(promptolution_cfg["model_id"]).strip()
    if not promptolution_model_id.startswith(("vllm-", "local-")):
        api_key = load_openrouter_api_key(keys_file="keys.json")
        promptolution_cfg["api_key"] = api_key

    for key, value in promptolution_cfg.items():
        if value is not None:
            experiment_kwargs[key] = value

    if (
        "model_storage_path" not in experiment_kwargs
        or not experiment_kwargs["model_storage_path"]
    ):
        experiment_kwargs["model_storage_path"] = str(Path.home() / "models" / "hf")

    loop_experiment_config = ExperimentConfig(**experiment_kwargs)
    promptolution_meta_llm = PromptolutionVLLMAdapter(
        shared_vllm=shared_meta_llm,
        config=loop_experiment_config,
    )

    if not step_results_path.exists():
        _run_experiment(
            train_df,
            loop_experiment_config,
            promptolution_meta_llm,
            step_reward_evaluator,
            output_dir,
        )

    if not step_results_path.exists():
        raise RuntimeError(
            "Optimization did not produce step_results.csv. "
            "Check earlier errors (for example RESOURCE_EXHAUSTED) and retry with higher quota backoff settings."
        )

    print("Running evaluation of last step prompts...")

    if not eval_results_path.exists():
        last_step_prompts = _load_last_step_prompts(step_results_path)
        _evaluate_last_step_prompts(
            test_df=test_eval_df,
            prompts_df=last_step_prompts,
            loop_config=loop_experiment_config,
            meta_llm=promptolution_meta_llm,
            reward_evaluator=eval_reward_evaluator,
            eval_results_path=eval_results_path,
        )

    print(
        f"Experiment complete. Step results at {step_results_path}, eval results at {eval_results_path}."
    )


if __name__ == "__main__":
    main()
