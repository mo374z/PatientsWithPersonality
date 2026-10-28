import json
import re
import functools
from pathlib import Path

import pandas as pd
import yaml


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


RATER_ID_PATTERN = re.compile(r"^[a-z0-9]{4,}$")


def normalize_rater_id(user_id: str) -> str | None:
    normalized = (user_id or "").strip().lower()
    return normalized if RATER_ID_PATTERN.match(normalized) else None


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


@functools.lru_cache(maxsize=8)
def _load_dataset_file(dataset: str, name: str, generator: str) -> dict:
    if _data_dir is None:
        raise RuntimeError("data.init() must be called with a data_dir")
    path = _data_dir / dataset / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run {generator} first.")
    return json.loads(path.read_text())


def _dataset() -> str:
    if "dataset" not in _study_config:
        raise KeyError("Study config must specify 'dataset'")
    return _study_config["dataset"]


def _excerpts() -> dict:
    return _load_dataset_file(
        _dataset(), "excerpt_windows.json", "scripts/scan_excerpt_windows.py"
    )


def get_turn_window_size() -> int:
    return _excerpts()["turn_window"]


def get_excerpt_start(conversation_id: str) -> int:
    windows = _excerpts()["windows"]
    if conversation_id not in windows:
        raise KeyError(
            f"No excerpt window scanned for conversation '{conversation_id}'"
        )
    return windows[conversation_id]["start"]


def take_excerpt(turns: list[dict], conversation_id: str) -> list[dict]:
    start = get_excerpt_start(conversation_id)
    return turns[start : start + get_turn_window_size()]


def get_patient_summary(conversation_id: str) -> str:
    summaries = _load_dataset_file(
        _dataset(), "case_summaries.json", "scripts/shorten_case_summaries.py"
    )
    if conversation_id not in summaries:
        raise KeyError(f"No case summary for conversation '{conversation_id}'")
    return summaries[conversation_id]
