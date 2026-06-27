import asyncio
import logging
import os
import datetime
import json
import random
import re
from itertools import product
from pathlib import Path
from typing import Awaitable, Callable, List, TypeVar

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from patient_simulator.misc.llm import APILLM, OpenRouterLLM, VLLM

T = TypeVar("T")


def log_experiment(
    task_type: str, hadm_id: str, payload: dict, enabled: bool = True
) -> str:
    """Save a structured run log under logs/<task_type>/<hadm_id>/<timestamp>.json.

    Returns the path to the created log file or an empty string if logging disabled.
    """

    if not enabled:
        return ""

    base_dir = os.path.join("logs", str(task_type), str(hadm_id))
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"{ts}.json"
    path = os.path.join(base_dir, filename)

    logger = logging.getLogger(f"clinical_benchmarking.log.{task_type}.{hadm_id}.{ts}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)

    try:
        payload.setdefault(
            "logged_at_utc", datetime.datetime.utcnow().isoformat() + "Z"
        )
        logger.info(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        logger.removeHandler(fh)
        fh.close()

    return path


def query_to_dataframe(result_and_columns):
    """Convert the result of a query to a pandas DataFrame."""

    result, column_names = result_and_columns
    df = pd.DataFrame(result, columns=column_names)
    return df


def extract_response(response: str, marker: str) -> str:
    """Extracts the part of the response between the given markers."""

    pattern = rf"<{re.escape(marker)}>(.*?)</{re.escape(marker)}>"
    m = re.search(pattern, response)
    return m.group(1).strip() if m else ""


def parse_transcript_file(path: str) -> List[tuple[str, str]]:
    """Parse a single raw transcript txt file into (speaker, text) turns."""
    turns: List[tuple[str, str]] = []
    if not os.path.isfile(path):
        return turns
    lines = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeError:
        # Fallback to utf-16 if utf-8 fails
        with open(path, "r", encoding="utf-16") as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Expect 'D: ...' or 'P: ...' and also tolerate 'D; ...' or 'P; ...'
        m = re.match(r"^([DdPp])[:;]\s*(.+)$", line)
        if not m:
            continue
        speaker_raw, text = m.groups()
        speaker = speaker_raw.upper()
        if speaker == "D":
            speaker = "DOCTOR"
        elif speaker == "P":
            speaker = "PATIENT"
        turns.append((speaker, text))
    return turns


def load_key(
    key_field: str,
    env_var: str | None = None,
    keys_file: str | Path | None = None,
) -> str:
    """Load a secret from an environment variable or keys.json."""
    if env_var:
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

    keys_file = Path(keys_file) if keys_file is not None else Path("keys.json")
    if not keys_file.exists():
        raise FileNotFoundError(f"keys.json not found at {keys_file}")

    keys_data = json.loads(keys_file.read_text())
    if key_field not in keys_data:
        raise KeyError(f"{key_field} not found in keys.json")

    return keys_data[key_field]


def load_openrouter_api_key(keys_file: str | Path | None = None) -> str:
    """Load OpenRouter API key from environment or keys.json."""
    return load_key("OPENROUTER_KEY", env_var="OPENROUTER_API_KEY", keys_file=keys_file)


def load_gcp_project(keys_file: str | Path | None = None) -> str:
    """Load Google Cloud project id from environment or keys.json."""
    return load_key(
        "GOOGLE_CLOUD_PROJECT", env_var="GOOGLE_CLOUD_PROJECT", keys_file=keys_file
    )


def create_llm_instance(
    model_cfg: DictConfig | dict,
    keys_file: str | Path | None = None,
    use_cache: bool = True,
):
    """Create an LLM instance from a config mapping."""
    cfg = (
        OmegaConf.to_container(model_cfg, resolve=True)
        if isinstance(model_cfg, DictConfig)
        else dict(model_cfg)
    )

    backend = str(cfg.get("backend", "OPENROUTER")).upper()
    model_name = str(cfg["name"])

    if backend == "VLLM":
        endpoint_id = cfg.get("endpoint_id")
        return VLLM(
            model=model_name,
            endpoint_id=endpoint_id,
            project=cfg.get("project") or load_gcp_project(keys_file)
            if endpoint_id
            else cfg.get("project"),
            location=cfg.get("location"),
            engine_kwargs=cfg.get("engine_kwargs", {}),
            sampling_kwargs=cfg.get("sampling_kwargs", {}),
            batch_size=cfg.get("batch_size"),
            use_cache=use_cache,
        )

    if backend in {"API", "APILLM"}:
        return APILLM(
            model=model_name,
            project=cfg.get("project") or load_gcp_project(keys_file),
            location=cfg.get("location"),
            vertexai=cfg.get("vertexai", True),
            api_key=cfg.get("api_key"),
            api_version=cfg.get("api_version"),
            generation_config=cfg.get("generation_config"),
            max_retries=cfg.get("max_retries", 5),
            initial_delay=cfg.get("initial_delay", 2.0),
            verbosity=cfg.get("verbosity", 0),
            use_cache=use_cache,
        )

    if backend == "OPENROUTER":
        api_key = cfg.get("api_key") or load_openrouter_api_key(keys_file=keys_file)
        return OpenRouterLLM(model=model_name, api_key=api_key, use_cache=use_cache)

    raise ValueError(f"Unknown LLM backend: {backend}")


def extract_tagged_text(text: str, tag: str) -> str:
    """Extract text enclosed in an XML-like tag."""
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()


def expand_param_grid(params: dict) -> list[dict]:
    if not params:
        return [{}]
    list_params = {k: v for k, v in params.items() if isinstance(v, list)}
    single_params = {k: v for k, v in params.items() if not isinstance(v, list)}
    if not list_params:
        return [params]
    keys = list(list_params.keys())
    combos = []
    for vals in product(*list_params.values()):
        combo = single_params.copy()
        combo.update(dict(zip(keys, vals)))
        combos.append(combo)
    return combos


def patient_name_from_config(patient_type: str, params: dict) -> str:
    if patient_type in (
        "BaselinePatient",
        "CraftMDPatient",
        "StateAwarePatient",
        "VirtualPatient",
    ):
        return patient_type
    if patient_type == "AgentClinicPatient":
        bias = params.get("bias_present")
        return f"AgentClinicPatient_{bias}" if bias else "AgentClinicPatient_nobias"
    if patient_type == "PatientSimPatient":
        return (
            f"PatientSimPatient_pers{params['personality_type']}_cefr{params['cefr_type']}"
            f"_dazed{params['dazed_level_type']}_recall{params['recall_level_type']}"
        )
    if patient_type == "PatientsWithPersonality":
        dynamic = params.get("dynamic_case_description", True)
        prefix = (
            "PatientsWithPersonality" if dynamic else "PatientsWithPersonality_Static"
        )
        return f"{prefix}_H{params['h']}_E{params['e']}_X{params['x']}_A{params['a']}_C{params['c']}_O{params['o']}_L{params['level']}"
    raise ValueError(f"Unknown patient type: {patient_type}")


def check_experiment_completeness(cfg: dict, delete_incomplete: bool = False) -> None:
    output_dir = Path(cfg["output_dir"])
    conversations_dir = Path(cfg["conversations_dir"])
    model_name_short = cfg["conversational_model"]["name"].split("/")[-1]
    max_turns = cfg.get("max_turns")

    if cfg.get("conversations"):
        conversations = list(cfg["conversations"])
    else:
        conversations = sorted(p.stem for p in conversations_dir.glob("*.txt"))

    patient_combos = [
        (
            patient_cfg["patient_type"],
            patient_name_from_config(patient_cfg["patient_type"], combo),
        )
        for patient_cfg in cfg["patient_configs"]
        for combo in expand_param_grid(patient_cfg.get("params") or {})
    ]

    conv_res_path = output_dir / "all_conversations.csv"
    eval_paths = (
        set(pd.read_csv(conv_res_path)["path"].values)
        if conv_res_path.exists()
        else set()
    )

    missing_sim, wrong_turns, missing_eval = [], [], []

    for conversation in conversations:
        orig_conv = parse_transcript_file(
            str(conversations_dir / f"{conversation}.txt")
        )
        if max_turns is not None:
            orig_conv = orig_conv[:max_turns]
        expected_turns = sum(1 for speaker, _ in orig_conv if speaker == "PATIENT")

        for _, patient_name in patient_combos:
            results_dir = output_dir / patient_name / model_name_short / conversation
            turns_path = results_dir / "turns.csv"

            if not turns_path.exists() or turns_path.stat().st_size <= 1:
                missing_sim.append((conversation, patient_name))
                continue

            actual_turns = len(pd.read_csv(turns_path))
            if actual_turns != expected_turns:
                wrong_turns.append(
                    (conversation, patient_name, expected_turns, actual_turns)
                )
                if delete_incomplete:
                    turns_path.unlink()

            if str(results_dir) not in eval_paths:
                missing_eval.append((conversation, patient_name))

    total = len(conversations) * len(patient_combos)
    print(f"Experiment : {output_dir}")
    print(
        f"Conversations: {len(conversations)}  |  Patient configs: {len(patient_combos)}  |  Total: {total}"
    )

    print(
        f"\n── Simulation ({'OK' if not missing_sim else f'{len(missing_sim)} missing'})"
    )
    for conv, pat in missing_sim:
        print(f"   {conv} / {pat}")

    print(
        f"\n── Turn count ({'OK' if not wrong_turns else f'{len(wrong_turns)} mismatch'})"
    )
    for conv, pat, exp, got in wrong_turns:
        suffix = " → deleted" if delete_incomplete else ""
        print(f"   {conv} / {pat}: expected {exp}, got {got}{suffix}")

    print(
        f"\n── Evaluation ({'OK' if not missing_eval else f'{len(missing_eval)} missing'})"
    )
    for conv, pat in missing_eval:
        print(f"   {conv} / {pat}")


log = logging.getLogger(__name__)


def is_transient_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 499, 500, 502, 503, 504}:
        return True

    response_json = getattr(exc, "response_json", None)
    if isinstance(response_json, dict):
        error = response_json.get("error")
        if isinstance(error, dict):
            status = str(error.get("status", "")).upper()
            if status in {
                "RESOURCE_EXHAUSTED",
                "CANCELLED",
                "INTERNAL",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
            }:
                return True

    message = str(exc).upper()
    transient_markers = [
        "RESOURCE_EXHAUSTED",
        "CANCELLED",
        "INTERNAL",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "SERVERERROR",
        "HTTP 500",
        "HTTP 502",
        "HTTP 503",
        "HTTP 504",
    ]
    return any(marker in message for marker in transient_markers)


async def retry_on_transient_error(
    coro_factory: Callable[[], Awaitable[T]],
    max_attempts: int = 10,
    initial_delay: float = 15.0,
    max_delay: float = 300.0,
    is_retryable: Callable[[Exception], bool] = is_transient_api_error,
) -> T:
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            is_last = attempt >= max_attempts - 1
            if is_last or not is_retryable(exc):
                raise
            delay = min(initial_delay * (2**attempt), max_delay)
            delay *= random.uniform(0.8, 1.2)
            log.warning(
                "Transient error, retrying in %.1fs (attempt %d/%d): %s",
                delay,
                attempt + 1,
                max_attempts,
                exc,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Exhausted retry loop.")
