#!/usr/bin/env python3
import asyncio
import json
import logging
import os
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from patient_simulator.misc.utils import (
    parse_transcript_file,
    create_llm_instance,
    expand_param_grid,
    retry_on_transient_error,
)
from patient_simulator.eval import evaluate_conversation
from patient_simulator.patients import (
    CraftMDPatient,
    AgentClinicPatient,
    StateAwarePatient,
    PatientSimPatient,
    PatientsWithPersonality,
    BaselinePatient,
    VirtualPatient,
    RealPatient,
)

log = logging.getLogger(__name__)
logging.getLogger("mistral_common").setLevel(logging.WARNING)


PATIENT_TYPE_MAP = {
    "RealPatient": RealPatient,
    "CraftMDPatient": CraftMDPatient,
    "AgentClinicPatient": AgentClinicPatient,
    "StateAwarePatient": StateAwarePatient,
    "PatientSimPatient": PatientSimPatient,
    "PatientsWithPersonality": PatientsWithPersonality,
    "BaselinePatient": BaselinePatient,
    "VirtualPatient": VirtualPatient,
}


async def run_simulations(
    cfg: DictConfig,
    conversations,
    patient_configs,
    conversational_llm,
    meta_llm,
    eval_llm,
):
    max_turns = cfg.get("max_turns", None)

    for c in tqdm(conversations, desc="Conversations"):
        case_desc_path = f"{cfg.case_descriptions_dir}/{c}.json"
        if not Path(case_desc_path).exists():
            log.warning("Case description not found at %s", case_desc_path)
            continue

        with open(case_desc_path, "r") as f:
            case_desc = json.load(f)

        conv_path = f"{cfg.conversations_dir}/{c}.txt"
        if not Path(conv_path).exists():
            log.warning("Transcript not found at %s", conv_path)
            continue

        conv = parse_transcript_file(conv_path)

        if max_turns is not None:
            conv = conv.head(max_turns)

        for patient_class, params in tqdm(
            patient_configs, leave=False, position=1, desc="Patient Configs"
        ):
            run_params = dict(params)
            align_doctor = True
            if patient_class == BaselinePatient and "real_responses" not in run_params:
                run_params["real_responses"] = [
                    text for speaker, text in conv if speaker == "PATIENT"
                ]

            if patient_class == RealPatient and "real_responses" not in run_params:
                run_params["real_responses"] = [
                    text for speaker, text in conv if speaker == "PATIENT"
                ]
                align_doctor = False

            if patient_class == PatientsWithPersonality:
                run_params["meta_llm"] = meta_llm
            patient = patient_class(case_desc, llm=conversational_llm, **run_params)
            patient_name = patient.__name__(short=False)
            model_name_short = cfg.conversational_model.name.split("/")[-1]
            results_dir = f"{cfg.output_dir}/{patient_name}/{model_name_short}/{c}"
            os.makedirs(results_dir, exist_ok=True)

            turns_path = Path(f"{results_dir}/turns.csv")
            conv_res_path = Path(f"{cfg.output_dir}/all_conversations.csv")

            already_recorded = (
                conv_res_path.exists()
                and str(results_dir) in pd.read_csv(conv_res_path)["path"].values
            )
            if already_recorded:
                continue

            if not turns_path.exists() or turns_path.stat().st_size <= 1:
                try:
                    sim_conv = await retry_on_transient_error(
                        lambda: patient.simulate_conversation(
                            conv=conv,
                            path=results_dir,
                            align_doctor=align_doctor,
                        )
                    )
                except Exception as e:
                    log.error(
                        "Error simulating conversation %s with %s: %s",
                        c,
                        patient_name,
                        e,
                        exc_info=True,
                    )
                    continue
                if patient_class == PatientsWithPersonality:
                    with open(f"{results_dir}/fuzzy_history.json", "w") as f:
                        json.dump(patient.fuzzy_history, f)
                    with open(f"{results_dir}/downplayed_fields.json", "w") as f:
                        json.dump(patient.downplayed_fields, f)
            else:
                sim_conv = pd.read_csv(turns_path)

            expected_patient_turns = sum(
                1 for speaker, _ in conv if speaker == "PATIENT"
            )
            if len(sim_conv) != expected_patient_turns:
                log.warning(
                    "Skipping eval for %s / %s: expected %d turns, got %d",
                    c,
                    patient_name,
                    expected_patient_turns,
                    len(sim_conv),
                )
                continue

            sim_conv = sim_conv[
                ["doctor_question", "real_response", "simulated_response"]
            ]
            try:
                turn_res, conv_res, sim_profile = await retry_on_transient_error(
                    lambda: evaluate_conversation(
                        sim_conv, eval_llm, case_description=case_desc
                    )
                )
            except Exception as e:
                log.error(
                    "Error evaluating conversation %s with %s: %s",
                    c,
                    patient_name,
                    e,
                    exc_info=True,
                )
                continue

            turn_res.to_csv(turns_path, index=False)
            with open(f"{results_dir}/sim_profile.json", "w") as f:
                json.dump(sim_profile, f, indent=2)

            conv_res["patient_name"] = patient_name
            conv_res["patient_type"] = patient_class.__name__
            conv_res["model_name"] = model_name_short
            conv_res["conversation_name"] = c
            for key, value in params.items():
                conv_res[f"param_{key}"] = value
            conv_res["path"] = results_dir

            new_row = pd.DataFrame([conv_res])
            if conv_res_path.exists():
                pd.concat(
                    [pd.read_csv(conv_res_path), new_row], ignore_index=True
                ).to_csv(conv_res_path, index=False)
            else:
                new_row.to_csv(conv_res_path, index=False)


@hydra.main(
    version_base=None,
    config_path="../configs/experiment",
)
def main(cfg: DictConfig):
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    patient_configs = []
    for patient_cfg in cfg.patient_configs:
        patient_type_name = patient_cfg.patient_type
        if patient_type_name not in PATIENT_TYPE_MAP:
            raise ValueError(
                f"Invalid patient type: {patient_type_name}. "
                f"Choose from: {list(PATIENT_TYPE_MAP.keys())}"
            )

        patient_class = PATIENT_TYPE_MAP[patient_type_name]
        if "params" in patient_cfg:
            params = OmegaConf.to_container(patient_cfg.params, resolve=True)
        else:
            params = {}

        param_combinations = expand_param_grid(params)
        for param_combo in param_combinations:
            patient_configs.append((patient_class, param_combo))

    if "conversations" in cfg:
        conversations = list(cfg.conversations)
    else:
        conversations = sorted(
            p.stem for p in Path(cfg.conversations_dir).glob("*.txt")
        )

    log.info("Running simulations for %d conversations", len(conversations))
    log.info("Patient configurations: %d", len(patient_configs))
    for patient_class, params in patient_configs:
        param_str = f" with {params}" if params else ""
        log.info("  - %s%s", patient_class.__name__, param_str)
    if cfg.get("max_turns"):
        log.info("Max turns per conversation: %d", cfg.max_turns)

    conversational_llm = create_llm_instance(cfg.conversational_model)
    meta_llm = create_llm_instance(cfg.meta_model)
    eval_llm = create_llm_instance(cfg.llm_judge_model)

    asyncio.run(
        run_simulations(
            cfg, conversations, patient_configs, conversational_llm, meta_llm, eval_llm
        )
    )

    for llm_instance, model_cfg in [
        (conversational_llm, cfg.conversational_model),
        (meta_llm, cfg.meta_model),
        (eval_llm, cfg.llm_judge_model),
    ]:
        if model_cfg.backend == "VLLM":
            llm_instance.cleanup()

    log.info("Simulations complete!")


if __name__ == "__main__":
    main()
