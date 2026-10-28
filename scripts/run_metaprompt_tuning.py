#!/usr/bin/env python3
import asyncio
import contextvars
import threading
import json
import multiprocessing as mp
import os
import sys
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
from promptolution.helpers import get_optimizer, get_task
from promptolution.predictors.maker_based_predictor import MarkerBasedPredictor
from promptolution.utils.formatting import extract_from_tag
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


MAX_FRACTION_PENALTY = 1.0
MAX_PERSONALITY_PENALTY = 2.0
INVALID_ROLE_REWARD = -3.0
MISSING_ROLE_MARKERS = "__missing_role_markers__"


class RoleMarkerPredictor(MarkerBasedPredictor):
    """Requires the role markers and keeps the generated casing.

    The base predictor lowercases every prediction and falls back to the raw text
    when the markers are absent, so an untagged generation scored identically to a
    tagged one and the downstream patient received a role the framework never sees.
    """

    def _extract_preds(self, preds: list[str]) -> list[str]:
        return [
            extract_from_tag(prediction, self.begin_marker, self.end_marker)
            if self.begin_marker in prediction and self.end_marker in prediction
            else MISSING_ROLE_MARKERS + prediction
            for prediction in preds
        ]


class _EventLoopRunner:
    """One event loop for the whole process.

    Third-party async clients keep long-lived asyncio primitives — google-genai's
    per-client auth lock, vLLM's engine lock — that bind to the first loop which
    contends on them. A fresh ``asyncio.run`` per batch therefore dies the first
    time such a lock is reused from a later loop, which for the auth lock means
    the first token refresh (roughly hourly) after any contention.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(
            target=self._loop.run_forever, name="reward-eval-loop", daemon=True
        ).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()


_LOOP_RUNNER: _EventLoopRunner | None = None


def _loop_runner() -> _EventLoopRunner:
    global _LOOP_RUNNER
    if _LOOP_RUNNER is None:
        _LOOP_RUNNER = _EventLoopRunner()
    return _LOOP_RUNNER


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

        return _loop_runner().run(_run_batch())

    def set_generation_seed(self, seed: int) -> None:
        # Keep seed behavior aligned with promptolution optimizers.
        self._shared_vllm.default_sampling_kwargs["seed"] = int(seed)


def _personality_distance(
    reconstructed: dict[str, int],
    target: dict[str, int],
) -> float:
    """Mean absolute HEXACO axis distance over the scored axes."""
    return sum(
        abs(target_value - reconstructed[axis]) for axis, target_value in target.items()
    ) / len(target)


_TOKEN_USAGE: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "reward_token_usage", default=None
)


def _install_token_accounting(llms: list[Any]) -> None:
    """Wrap each LLM once so token usage lands in the calling task's accumulator.

    The wrapper is permanent and context-local, so concurrent reward evaluations
    each accumulate into their own dict instead of racing on a shared patch.
    """
    for llm in {id(llm): llm for llm in llms if llm is not None}.values():
        original = getattr(llm, "generate_response", None)
        if original is None or getattr(original, "_token_accounted", False):
            continue

        async def wrapped_generate_response(*args, _orig=original, **kwargs):
            result = await _orig(*args, **kwargs)
            usage = _TOKEN_USAGE.get()
            if usage is not None:
                usage["input_token_count"] += int(result.get("prompt_token_count", 0))
                usage["output_token_count"] += int(result.get("output_token_count", 0))
            return result

        wrapped_generate_response._token_accounted = True
        setattr(llm, "generate_response", wrapped_generate_response)


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
        "role_length_penalty",
        "role_token_count",
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
        register: str,
        answer_form: float,
        role_length_weight: float,
        role_token_budget: int,
        personal_info_weight: float,
        reward_log_path: str | Path | None = None,
        reward_cache_path: str | Path | None = None,
        resource_exhausted_max_attempts: int = 6,
        resource_exhausted_initial_delay_seconds: float = 15.0,
        resource_exhausted_max_delay_seconds: float = 300.0,
    ):
        self.conversational_llm = conversational_llm
        self.patient_meta_llm = patient_meta_llm
        self.role_length_weight = float(role_length_weight)
        self.role_token_budget = int(role_token_budget)
        self.personal_info_weight = float(personal_info_weight)
        self.judge_llm = judge_llm
        self.register = register
        self.answer_form = answer_form
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

        if self.role_length_weight < 0:
            raise ValueError("role_length_weight must be >= 0.")
        if self.role_token_budget <= 0:
            raise ValueError("role_token_budget must be > 0.")
        if self.personal_info_weight < 0:
            raise ValueError("personal_info_weight must be >= 0.")
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
        missing_columns = set(self._DEFAULT_REWARD_CACHE_COLUMNS) - set(
            cache_df.columns
        )
        if missing_columns:
            missing_columns_text = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Reward cache {self.reward_cache_path} is missing columns: "
                f"{missing_columns_text}. It predates the current reward definition; "
                "point output_path at a fresh directory."
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

    def _role_token_count(self, role: str) -> int:
        return len(self.patient_meta_llm.engine.get_tokenizer().encode(role))

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
        _install_token_accounting(
            [self.conversational_llm, self.patient_meta_llm, self.judge_llm]
        )
        token_usage = {"input_token_count": 0, "output_token_count": 0}
        _TOKEN_USAGE.set(token_usage)

        case_description_raw = json.loads(profile_json)
        forced_latent_role = prediction.strip()

        role_token_count = self._role_token_count(forced_latent_role)
        role_length_penalty = (
            self.role_length_weight
            * max(0, role_token_count - self.role_token_budget)
            / self.role_token_budget
        )

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
            register=self.register,
            answer_form=self.answer_form,
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
        all_probe_responses: list[str] = []
        transcript_turns: list[str] = []

        print("✨ Starting EVAL")

        for probe_idx, probe in enumerate(PROBE_SPECS, start=1):
            question = str(probe["question"])
            patient_response = await patient.get_response(question)
            print("PATIENT RESPONSE:", patient_response)
            all_probe_responses.append(patient_response)
            transcript_turns.append(f"Doctor: {question}\nPatient: {patient_response}")

            probe_logs.append(
                {
                    "probe_idx": probe_idx,
                    "question": question,
                    "elicited_axes": list(probe["axes"]),
                    "response": patient_response,
                }
            )

        scored_axes = {axis for probe in PROBE_SPECS for axis in probe["axes"]}
        reconstructed_axes = await reconstruct_personality_params(
            conversation="\n\n".join(transcript_turns),
            llm=self.judge_llm,
            prompt_variant="anchored",
            subset=scored_axes,
        )
        personality_penalty = _personality_distance(
            reconstructed=reconstructed_axes,
            target={axis: trait_targets[axis] for axis in scored_axes},
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
            information_penalty = MAX_FRACTION_PENALTY
        else:
            information_penalty = fidelity_distribution_total["Hallucination"] / total

        personal_info_total = sum(personal_info_distribution.values())
        if personal_info_total == 0:
            personal_info_penalty = MAX_FRACTION_PENALTY
        else:
            personal_info_penalty = (
                personal_info_distribution["Missing"] / personal_info_total
            )

        print(f"Information penalty: {information_penalty:.4f}")
        print(f"Personality penalty: {personality_penalty:.4f}")
        print(f"Personal information penalty: {personal_info_penalty:.4f}")

        reward = -(
            information_penalty
            + personality_penalty
            + self.personal_info_weight * personal_info_penalty
            + role_length_penalty
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
            "h_recon": reconstructed_axes.get("H"),
            "e_recon": reconstructed_axes.get("E"),
            "x_recon": reconstructed_axes.get("X"),
            "a_recon": reconstructed_axes.get("A"),
            "c_recon": reconstructed_axes.get("C"),
            "o_recon": reconstructed_axes.get("O"),
            "information_penalty": information_penalty,
            "personality_penalty": personality_penalty,
            "personal_info_penalty": personal_info_penalty,
            "role_length_penalty": role_length_penalty,
            "role_token_count": role_token_count,
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
            "score_async_output_token_count": int(token_usage["output_token_count"]),
            "probe_logs": json.dumps(probe_logs),
            "latent_role_prediction": forced_latent_role,
        }
        self._flush_log_row(log_row)
        self._store_cached_reward_row(log_row)
        return reward

    def __call__(
        self,
        prediction: str,
        **kwargs: Any,
    ) -> float:
        return _loop_runner().run(self.score(prediction, **kwargs))

    def _reject_missing_role_markers(
        self,
        meta_prompt: str,
        row_id: str,
        unmarked_generation: str,
        kwargs: dict[str, Any],
    ) -> float:
        """Score a generation that ignored the role markers as maximally invalid.

        The generation itself is kept so truncated or unmarked output stays
        inspectable; only the role markers were missing, not the text.
        """
        reward = INVALID_ROLE_REWARD
        log_row = {column: None for column in self._reward_cache_columns}
        log_row.update(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "optimization_step": self._optimization_step,
                "meta_prompt": meta_prompt,
                "row_id": row_id,
                "cefr_level": str(kwargs["cefr_level"]),
                "information_penalty": MAX_FRACTION_PENALTY,
                "personality_penalty": MAX_PERSONALITY_PENALTY,
                "personal_info_penalty": MAX_FRACTION_PENALTY,
                "role_length_penalty": None,
                "role_token_count": self._role_token_count(unmarked_generation),
                "reward": reward,
                "latent_role_prediction": unmarked_generation,
            }
        )
        for axis in ["h", "e", "x", "a", "c", "o"]:
            log_row[f"{axis}_trait"] = int(kwargs[f"{axis}_trait"])

        self._flush_log_row(log_row)
        print(f"Rejected role without role markers for row_id={row_id}.")
        return reward

    async def score(
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

        if prediction.startswith(MISSING_ROLE_MARKERS):
            return self._reject_missing_role_markers(
                meta_prompt=meta_prompt,
                row_id=row_id,
                unmarked_generation=prediction[len(MISSING_ROLE_MARKERS) :],
                kwargs=kwargs,
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
                reward = await self._score_async(
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
                await asyncio.sleep(delay)

        raise RuntimeError("Exhausted retry loop in RewardEvaluator.score.")


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
    reward_concurrency: int,
) -> list[str]:
    train_df = train_df.copy()
    train_df["x_input"] = train_df["x"]

    predictor = RoleMarkerPredictor(meta_llm, config=loop_config)
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

    def evaluate_concurrently(xs, ys, preds):
        """Score a batch in one event loop instead of one blocking call per row.

        Meta prompts are popped up front, in batch order, because the evaluator
        pairs them to rows through a FIFO queue that concurrency would scramble.
        """
        kwargs_list = []
        for x in xs:
            row_kwargs = dict(task_train.kwargs_map[x])
            row_kwargs["meta_prompt"] = reward_evaluator._pop_meta_prompt(
                x_input=str(row_kwargs["x_input"])
            )
            kwargs_list.append(row_kwargs)

        async def score_all():
            semaphore = asyncio.Semaphore(reward_concurrency)

            async def one(pred, row_kwargs):
                async with semaphore:
                    return await reward_evaluator.score(pred, **row_kwargs)

            return await asyncio.gather(
                *(one(pred, kw) for pred, kw in zip(preds, kwargs_list))
            )

        return np.asarray(_loop_runner().run(score_all()), dtype=float)

    task_train._evaluate = evaluate_concurrently

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


def _step_results_fragments(step_results_path: Path) -> list[Path]:
    """Archived runs plus the live file, in run order."""
    archived = sorted(step_results_path.parent.glob("step_results_run*.csv"))
    return archived + ([step_results_path] if step_results_path.exists() else [])


def _rotate_step_results(step_results_path: Path) -> None:
    """Archive the previous run's step results before a new run truncates them.

    promptolution's CSV callback rewrites the file on its own first step, which
    restarts at 1 in every process, so a resumed run would erase the history it
    is resuming from.
    """
    if not step_results_path.exists():
        return

    archived = sorted(step_results_path.parent.glob("step_results_run*.csv"))
    step_results_path.rename(
        step_results_path.parent / f"step_results_run{len(archived) + 1:02d}.csv"
    )


def _resume_state(step_results_path: Path) -> tuple[int, pd.DataFrame | None]:
    """Steps already completed and the population they ended on.

    Step numbering restarts whenever a run is resumed, so completed steps are
    counted as contiguous blocks of equal step values rather than by their max.
    """
    fragments = _step_results_fragments(step_results_path)
    if not fragments:
        return 0, None

    step_results = pd.concat(
        [pd.read_csv(path) for path in fragments], ignore_index=True
    )
    if step_results.empty:
        return 0, None

    steps = pd.to_numeric(step_results["step"], errors="raise")
    blocks = (steps != steps.shift()).cumsum()
    final = step_results.loc[blocks == blocks.max(), ["prompt", "score"]].copy()
    final["score"] = pd.to_numeric(final["score"], errors="raise")
    final = final.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
    return int(blocks.max()), final


def _load_last_step_prompts(step_results_path: Path) -> pd.DataFrame:
    completed, final = _resume_state(step_results_path)
    if final is None or final.empty:
        raise FileNotFoundError(
            f"No optimizer results in {step_results_path}. "
            "Optimization likely exited early due to upstream failures (for example quota exhaustion)."
        )
    return final.rename(columns={"score": "last_step_score"})


def _evaluate_last_step_prompts(
    test_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    loop_config: ExperimentConfig,
    meta_llm: BaseLLM,
    reward_evaluator: RewardEvaluator,
    eval_results_path: Path,
    reward_concurrency: int,
) -> pd.DataFrame:
    eval_df = test_df.copy().reset_index(drop=True)
    eval_df["x_input"] = eval_df["x"]

    predictor = RoleMarkerPredictor(meta_llm, config=loop_config)
    eval_records: list[dict[str, Any]] = []

    for row in prompts_df.itertuples(index=False):
        prompt = str(row.prompt)
        last_step_score = float(row.last_step_score)

        eval_xs = eval_df["x"].astype(str).tolist()
        predictions, _ = predictor.predict([prompt] * len(eval_xs), eval_xs)
        if len(predictions) != len(eval_df):
            raise ValueError(
                f"Expected {len(eval_df)} predictions per prompt, got {len(predictions)}."
            )

        async def score_all() -> list[float]:
            semaphore = asyncio.Semaphore(reward_concurrency)

            async def one(prediction: str, sample_row: Any) -> float:
                async with semaphore:
                    return await reward_evaluator.score(
                        str(prediction),
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

            return await asyncio.gather(
                *(
                    one(prediction, sample_row)
                    for prediction, sample_row in zip(
                        predictions, eval_df.itertuples(index=False)
                    )
                )
            )

        rewards_series = pd.Series(_loop_runner().run(score_all()), dtype=float)
        eval_records.append(
            {
                "prompt": prompt,
                "last_step_score": last_step_score,
                "eval_mean_reward": float(rewards_series.mean()),
                "eval_std_reward": float(rewards_series.std(ddof=0)),
                "n_eval": int(len(rewards_series)),
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


def _case_blocked_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split on the case-blocked ``split`` column written by the dataset builder."""
    if "split" not in dataset.columns:
        raise ValueError("Tuning dataset is missing the 'split' column.")

    splits = {}
    for name in ("train", "test"):
        split_df = dataset.loc[dataset["split"] == name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Dataset contains no '{name}' rows.")
        splits[name] = split_df

    return splits["train"], splits["test"]


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
    train_df, test_eval_df = _case_blocked_split(dataset)

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
        register=str(cfg.register),
        answer_form=float(cfg.answer_form),
        role_length_weight=float(cfg.role_length_weight),
        role_token_budget=int(cfg.role_token_budget),
        personal_info_weight=float(cfg.personal_info_weight),
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
        register=str(cfg.register),
        answer_form=float(cfg.answer_form),
        role_length_weight=float(cfg.role_length_weight),
        role_token_budget=int(cfg.role_token_budget),
        personal_info_weight=float(cfg.personal_info_weight),
        reward_log_path=eval_reward_log_path,
        reward_cache_path=reward_cache_path,
        resource_exhausted_max_attempts=resource_exhausted_max_attempts,
        resource_exhausted_initial_delay_seconds=resource_exhausted_initial_delay_seconds,
        resource_exhausted_max_delay_seconds=resource_exhausted_max_delay_seconds,
    )

    _rotate_step_results(step_results_path)
    completed_steps, last_population = _resume_state(step_results_path)
    remaining_steps = int(promptolution_cfg["n_steps"]) - completed_steps
    promptolution_cfg["n_steps"] = max(remaining_steps, 0)

    if last_population is None:
        init_prompts = _load_initial_prompts(str(cfg.initial_prompts_path))
    else:
        init_prompts = (
            last_population.sort_values("score", ascending=False)["prompt"]
            .astype(str)
            .tolist()
        )
        print(
            f"Resuming: {completed_steps} steps already in {step_results_path}, "
            f"{promptolution_cfg['n_steps']} to go, seeded with {len(init_prompts)} evolved prompts."
        )

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

    if remaining_steps > 0:
        _run_experiment(
            train_df,
            loop_experiment_config,
            promptolution_meta_llm,
            step_reward_evaluator,
            output_dir,
            int(cfg.reward_concurrency),
        )
    else:
        print(f"Optimization already complete ({completed_steps} steps); evaluating.")

    if not _step_results_fragments(step_results_path):
        raise RuntimeError(
            "Optimization did not produce step results. "
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
            reward_concurrency=int(cfg.reward_concurrency),
        )

    print(
        f"Experiment complete. Step results at {step_results_path}, eval results at {eval_results_path}."
    )


if __name__ == "__main__":
    main()
