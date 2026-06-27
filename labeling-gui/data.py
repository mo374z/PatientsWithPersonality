import json
import random
import re
import functools
from pathlib import Path

import pandas as pd
import yaml


CONVERSATION_WINDOW_SIZE = 5

_results_dir: Path = None
_data_dir: Path = None
_study_config: dict = {}


def init(results_dir: str, data_dir: str = None):
    global _results_dir, _data_dir
    _results_dir = Path(results_dir)
    if not _results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {_results_dir}")
    if data_dir is not None:
        _data_dir = Path(data_dir)
        if not _data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {_data_dir}")


def init_study_config(path: str):
    global _study_config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Study config not found: {config_path}")
    with open(config_path) as f:
        _study_config = yaml.safe_load(f)


def get_task_config(task: str) -> dict:
    if task not in _study_config:
        raise KeyError(f"Task '{task}' not found in study config")
    return _study_config[task]


def _get_results_dir() -> Path:
    if _results_dir is None:
        raise RuntimeError("data.init() must be called before loading data")
    return _results_dir


def _clean_response(text) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s*</response>\s*", "", text).strip()


@functools.lru_cache(maxsize=1)
def get_simulators() -> list[str]:
    results = _get_results_dir()
    return sorted(
        [
            d.name
            for d in results.iterdir()
            if d.is_dir() and d.name != "all_conversations.csv"
        ]
    )


@functools.lru_cache(maxsize=1)
def get_everyday_simulators() -> list[str]:
    return [s for s in get_simulators() if s.startswith("PatientsWithPersonality_")]


@functools.lru_cache(maxsize=1)
def get_non_baseline_simulators() -> list[str]:
    return [s for s in get_simulators() if s != "BaselinePatient"]


@functools.lru_cache(maxsize=256)
def _discover_model(simulator: str) -> str:
    sim_dir = _get_results_dir() / simulator
    models = [d.name for d in sim_dir.iterdir() if d.is_dir()]
    if len(models) != 1:
        raise ValueError(f"Expected exactly one model dir in {sim_dir}, found {models}")
    return models[0]


@functools.lru_cache(maxsize=256)
def get_conversation_ids(simulator: str) -> list[str]:
    model = _discover_model(simulator)
    conv_dir = _get_results_dir() / simulator / model
    return sorted([d.name for d in conv_dir.iterdir() if d.is_dir()])


@functools.lru_cache(maxsize=1)
def get_all_conversation_ids() -> list[str]:
    ids = set()
    for sim in get_simulators():
        ids.update(get_conversation_ids(sim))
    return sorted(ids)


@functools.lru_cache(maxsize=512)
def _load_turns_df(simulator: str, conversation_id: str) -> pd.DataFrame:
    model = _discover_model(simulator)
    path = _get_results_dir() / simulator / model / conversation_id / "turns.csv"
    return pd.read_csv(path)


def load_conversation(simulator: str, conversation_id: str) -> list[dict]:
    df = _load_turns_df(simulator, conversation_id)
    return [
        {
            "doctor": _clean_response(row["doctor_question"]),
            "patient": _clean_response(row["simulated_response"]),
        }
        for _, row in df.iterrows()
    ]


def load_real_conversation(conversation_id: str) -> list[dict]:
    ref_sim = get_simulators()[0]
    df = _load_turns_df(ref_sim, conversation_id)
    return [
        {
            "doctor": _clean_response(row["doctor_question"]),
            "patient": _clean_response(row["real_response"]),
        }
        for _, row in df.iterrows()
    ]


def load_source_conversation(source: str, conversation_id: str) -> list[dict]:
    if source == "real":
        return load_real_conversation(conversation_id)
    return load_conversation(source, conversation_id)


@functools.lru_cache(maxsize=2)
def _load_short_desc(dataset: str) -> dict:
    if _data_dir is None:
        return {}
    path = _data_dir / dataset / "short_desc.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def select_turn_window(
    turns: list[dict], seed_key: str, size: int = CONVERSATION_WINDOW_SIZE
) -> tuple[list[dict], int]:
    if len(turns) <= size:
        return turns, 0
    start = random.Random(hash(seed_key)).randint(0, len(turns) - size)
    return turns[start : start + size], start


def get_patient_summary(conversation_id: str) -> str:
    if conversation_id.startswith("VS"):
        desc = _load_short_desc("aci_bench")
        return desc.get(conversation_id, "")
    desc = _load_short_desc("osce_bench")
    return desc.get(f"{conversation_id}.json", "")
