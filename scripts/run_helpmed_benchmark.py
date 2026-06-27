#!/usr/bin/env python3
"""Reproducing the benchmark of Bean et. al. 2026 using their HelpMED dataset"""

import asyncio
import difflib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Literal

import hydra
import pandas as pd
import pydantic
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.data_extraction import PatientProfile
from patient_simulator.misc.utils import create_llm_instance, retry_on_transient_error
from patient_simulator.patients import (
    AgentClinicPatient,
    CraftMDPatient,
    PatientSimPatient,
    PatientsWithPersonality,
    StateAwarePatient,
    VirtualPatient,
)

PATIENT_TYPE_MAP = {
    "AgentClinicPatient": AgentClinicPatient,
    "CraftMDPatient": CraftMDPatient,
    "PatientSimPatient": PatientSimPatient,
    "PatientsWithPersonality": PatientsWithPersonality,
    "StateAwarePatient": StateAwarePatient,
    "VirtualPatient": VirtualPatient,
}

log = logging.getLogger(__name__)

ASSISTANT_SYSTEM = (
    "You are a conversational assistant. Limit your answers to around 50 words. "
    "Do not refer to your word limit."
)


class Disposition(pydantic.BaseModel):
    disposition: Literal[
        "Self-care",
        "Routine GP",
        "Urgent Primary Care",
        "A&E",
        "Ambulance",
    ]


class Conditions(pydantic.BaseModel):
    conditions: list[str]


DISPOSITION_TO_INT = {
    "Self-care": 0,
    "Routine GP": 1,
    "Urgent Primary Care": 2,
    "A&E": 3,
    "Ambulance": 4,
}


FINAL_QUESTION = (
    "Based on your conversation with the medical assistant, please answer:\n"
    "1. What healthcare service do you need? Choose ONE: "
    "Self-care, Routine GP, Urgent Primary Care, A&E, Ambulance.\n"
    "2. List specific medical conditions you think might be relevant."
)

DIRECT_DISPOSITION_PROMPT = (
    "{scenario_text}\n\n"
    "Based on the above, what healthcare service does this patient need? "
)

DIRECT_CONDITIONS_PROMPT = (
    "{scenario_text}\n\n"
    "Based on the above, list specific medical conditions that might be relevant "
    "for this patient."
)

EXTRACT_DISPOSITION_PROMPT = (
    "From the response below, identify which healthcare service was chosen.\n\n"
    "Response: {response}"
)

EXTRACT_CONDITIONS_PROMPT = (
    "From the response below, identify all medical conditions named. "
    "If none are present, return an empty list.\n\n"
    "Response: {response}"
)


PROFILE_FIELDS = list(PatientProfile.model_fields.keys())


def load_helpmed_dataset(dataset_path: str) -> list[dict]:
    df = pd.read_csv(dataset_path)
    scenarios = []
    for _, row in df.iterrows():
        scenarios.append(
            {
                "id": int(row["scenario_id"]),
                "condition": row["condition"],
                "urgency": int(row["urgency"]),
                "urgency_text": row["urgency_text"],
                "full_differential": json.loads(row["full_differential"]),
                "red_flags": json.loads(row["red_flags"]),
                "context_text": row["context_text"],
                "persona_context": {f: str(row[f]) for f in PROFILE_FIELDS},
            }
        )
    return scenarios


def fuzzy_match_any(pred: str, gold_list: list[str], threshold: float = 0.8) -> bool:
    pred_lower = pred.lower()
    return any(
        difflib.SequenceMatcher(None, pred_lower, g.lower()).ratio() >= threshold
        for g in gold_list
    )


async def extract_disposition(text: str, extract_llm) -> Disposition:
    result = await extract_llm.generate_response(
        prompt=EXTRACT_DISPOSITION_PROMPT.format(response=text),
        outlines_class=Disposition,
    )
    return result["response"]


async def extract_conditions(text: str, extract_llm) -> list[str]:
    result = await extract_llm.generate_response(
        prompt=EXTRACT_CONDITIONS_PROMPT.format(response=text),
        outlines_class=Conditions,
    )
    return [c.strip() for c in result["response"].conditions if c.strip()]


def compute_scores(
    disposition: Disposition | None,
    conditions: list[str],
    scenario: dict,
) -> dict:
    disp_int = (
        DISPOSITION_TO_INT[disposition.disposition] if disposition is not None else None
    )
    return {
        "disposition_pred": disp_int,
        "disposition_correct": disp_int == scenario["urgency"]
        if disp_int is not None
        else False,
        "conditions_pred": ", ".join(conditions),
        "conditions_correct": any(
            fuzzy_match_any(c, scenario["full_differential"]) for c in conditions
        )
        if conditions
        else False,
        "red_flags_correct": any(
            fuzzy_match_any(c, scenario["red_flags"]) for c in conditions
        )
        if conditions
        else False,
        "n_conditions_mentioned": len(conditions),
    }


async def get_assistant_response(history: list[dict], assistant_llm) -> str:
    result = await assistant_llm.generate_response(
        prompt=history,
        system_instruction=ASSISTANT_SYSTEM,
    )
    return result["response"].strip()


async def run_direct(scenario: dict, assistant_llm) -> dict:
    disp_result = await assistant_llm.generate_response(
        prompt=DIRECT_DISPOSITION_PROMPT.format(scenario_text=scenario["context_text"]),
        outlines_class=Disposition,
    )
    cond_result = await assistant_llm.generate_response(
        prompt=DIRECT_CONDITIONS_PROMPT.format(scenario_text=scenario["context_text"]),
        outlines_class=Conditions,
    )
    conditions = [c.strip() for c in cond_result["response"].conditions if c.strip()]
    return compute_scores(disp_result["response"], conditions, scenario)


async def run_interactive(
    scenario: dict,
    patient,
    assistant_llm,
    extract_llm,
    max_turns: int,
) -> tuple[dict, list[dict]]:
    assistant_history: list[dict] = []
    turns: list[dict] = []

    benchmarking_setup = "You are going to converse with a medical assistant. After this conversation, you will have to answer the following questions:\n"
    "1. What healthcare service do you need? Choose ONE: "
    "Self-care, Routine GP, Urgent Primary Care, A&E, Ambulance.\n"
    "2. List specific medical conditions you think might be relevant."

    patient.conversation_history.append({"role": "user", "content": benchmarking_setup})
    patient_msg = await patient.get_response(
        "Please describe what is bothering you today."
    )
    turns.append({"role": "patient", "content": patient_msg})
    assistant_history.append({"role": "user", "content": patient_msg})

    for _ in range(max_turns):
        assistant_msg = await get_assistant_response(assistant_history, assistant_llm)
        turns.append({"role": "assistant", "content": assistant_msg})
        assistant_history.append({"role": "assistant", "content": assistant_msg})

        patient_msg = await patient.get_response(assistant_msg)
        turns.append({"role": "patient", "content": patient_msg})
        assistant_history.append({"role": "user", "content": patient_msg})

    final_msg = await patient.get_response(FINAL_QUESTION)
    turns.append({"role": "patient_final", "content": final_msg})

    disposition = await extract_disposition(final_msg, extract_llm)
    conditions = await extract_conditions(final_msg, extract_llm)

    return compute_scores(disposition, conditions, scenario), turns


def sample_params(params) -> dict:
    return {
        k: random.choice(list(v)) if isinstance(v, (list, ListConfig)) else v
        for k, v in params.items()
    }


async def _run(cfg: DictConfig):
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_helpmed_dataset(cfg.dataset_path)
    log.info("Loaded %d scenarios", len(scenarios))

    personas_df = pd.DataFrame(
        [
            {
                "scenario_id": s["id"],
                "condition": s["condition"],
                **s["persona_context"],
            }
            for s in scenarios
        ]
    )
    personas_df.to_csv(output_dir / "personas.csv", index=False)
    log.info("Saved personas to %s", output_dir / "personas.csv")

    patient_configs = OmegaConf.to_container(cfg.patient_configs, resolve=True)
    has_interactive = any(pc["patient_type"] != "direct" for pc in patient_configs)

    patient_llm = (
        create_llm_instance(cfg.patient_model, use_cache=False)
        if has_interactive
        else None
    )
    meta_llm = (
        create_llm_instance(cfg.meta_model, use_cache=False)
        if has_interactive
        else None
    )

    results_path = output_dir / "results.csv"
    header_written = results_path.exists()
    results = []

    completed = set()
    if results_path.exists():
        existing = pd.read_csv(results_path)
        completed = {
            (
                str(r.assistant_model),
                int(r.scenario_id),
                str(r.patient_config_type),
                str(r.persona_params),
                int(r.run),
            )
            for r in existing.itertuples(index=False)
        }
        log.info("Found %d completed runs in %s", len(completed), results_path)

    try:
        for asst_cfg in tqdm(cfg.assistant_models, desc="Assistant models"):
            assistant_llm = create_llm_instance(asst_cfg, use_cache=False)
            if asst_cfg.get("full_name"):
                model_name = asst_cfg.full_name
            else:
                model_name = asst_cfg.name

            for scenario in tqdm(scenarios, desc="Scenarios", leave=False):
                for pc in tqdm(patient_configs, desc="Patient configs", leave=False):
                    ptype = pc["patient_type"]
                    params_template = pc.get("params") or {}

                    for run_idx in range(cfg.n_runs):
                        params = sample_params(params_template)
                        params_json = json.dumps(params) if params else "{}"
                        run_key = (
                            model_name,
                            scenario["id"],
                            ptype,
                            params_json,
                            run_idx,
                        )
                        if run_key in completed:
                            log.info(
                                "Skipping completed run: %s scenario=%s ptype=%s params=%s run=%d",
                                model_name,
                                scenario["id"],
                                ptype,
                                params_json,
                                run_idx,
                            )
                            continue
                        turns = None

                        if ptype == "direct":
                            run_result = await retry_on_transient_error(
                                lambda: run_direct(scenario, assistant_llm)
                            )
                            pname = "direct"
                        elif ptype in PATIENT_TYPE_MAP:
                            patient_class = PATIENT_TYPE_MAP[ptype]
                            extra_kwargs = {}
                            if patient_class is PatientsWithPersonality:
                                extra_kwargs["meta_llm"] = meta_llm
                            patient = patient_class(
                                scenario["persona_context"],
                                llm=patient_llm,
                                **extra_kwargs,
                                **params,
                            )
                            pname = patient.__name__(short=False)

                            run_result, turns = await retry_on_transient_error(
                                lambda: run_interactive(
                                    scenario,
                                    patient,
                                    assistant_llm,
                                    meta_llm,
                                    cfg.max_turns,
                                )
                            )
                        else:
                            raise ValueError(f"Unknown patient config type: {ptype}")

                        if turns is not None:
                            turns_path = (
                                output_dir
                                / pname
                                / model_name
                                / str(scenario["id"])
                                / f"turns_{run_idx}.csv"
                            )
                            turns_path.parent.mkdir(parents=True, exist_ok=True)
                            pd.DataFrame(turns).to_csv(turns_path, index=False)

                        row = {
                            "assistant_model": model_name,
                            "patient_name": pname,
                            "patient_config_type": ptype,
                            "scenario_id": scenario["id"],
                            "scenario_condition": scenario["condition"],
                            "run": run_idx,
                            "persona_params": params_json,
                            **run_result,
                        }
                        results.append(row)
                        pd.DataFrame([row]).to_csv(
                            results_path,
                            mode="a",
                            header=not header_written,
                            index=False,
                        )
                        header_written = True
    finally:
        if patient_llm is not None and hasattr(patient_llm, "cleanup"):
            patient_llm.cleanup()

    df = pd.DataFrame(results)
    log.info("Saved %d rows to %s", len(df), results_path)

    summary = (
        df.groupby(["assistant_model", "patient_name"])[
            ["disposition_correct", "conditions_correct", "red_flags_correct"]
        ]
        .mean()
        .round(5)
    )
    summary.to_csv(output_dir / "summary.csv")

    print("\n=== Accuracy Summary ===")
    print(summary.to_string())


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
)
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
