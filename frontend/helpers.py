import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from patient_simulator.patients import (
    CraftMDPatient,
    AgentClinicPatient,
    StateAwarePatient,
    PatientSimPatient,
    PatientsWithPersonality,
    VirtualPatient,
)
from patient_simulator.misc.utils import create_llm_instance


def initialize_patient(
    impl_type: str,
    case_description: str | dict,
    model: str,
    bias_present: str | None,
    params: dict = {},
    llm_backend: str = "API",
    llm_config: dict | None = None,
    meta_llm_config: dict | None = None,
):
    """Initialize patient simulator based on implementation type (sync wrapper for non-PatientSim)."""
    keys_file = Path(__file__).parent.parent / "keys.json"
    llm_cfg = {
        "backend": llm_backend,
        "name": model,
    }
    if llm_config:
        llm_cfg.update(llm_config)
    llm = create_llm_instance(llm_cfg, keys_file=keys_file)

    meta_llm = None
    if meta_llm_config and impl_type == "everyday":
        meta_llm_cfg = {
            "backend": meta_llm_config["backend"],
            "name": meta_llm_config["model"],
        }
        meta_runtime_config = meta_llm_config.get("runtime_config")
        if meta_runtime_config is None:
            meta_runtime_config = meta_llm_config.get("vllm_config", {})
        if meta_runtime_config:
            meta_llm_cfg.update(meta_runtime_config)
        if "api_key" in meta_llm_config:
            meta_llm_cfg["api_key"] = meta_llm_config["api_key"]
        meta_llm = create_llm_instance(meta_llm_cfg, keys_file=keys_file)

    if impl_type == "craftmd":
        patient = CraftMDPatient(case_description=case_description, llm=llm)
    elif impl_type == "agentclinic":
        patient = AgentClinicPatient(
            case_description=case_description, llm=llm, bias_present=bias_present
        )
    elif impl_type == "stateaware":
        patient = StateAwarePatient(case_description=case_description, llm=llm)
    elif impl_type == "patientsim":
        patient = PatientSimPatient(patient_profile=case_description, llm=llm, **params)
    elif impl_type == "everyday":
        everyday_params = {**params}
        if meta_llm:
            everyday_params["meta_llm"] = meta_llm
        patient = PatientsWithPersonality(
            case_description=case_description, llm=llm, **everyday_params
        )
    elif impl_type == "virtual":
        patient = VirtualPatient(case_description=case_description, llm=llm)
    else:
        raise ValueError(f"Unknown implementation type: {impl_type}")

    return patient
