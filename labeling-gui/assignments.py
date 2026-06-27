import random

import data
import storage


def _make_rng(user_id: str) -> random.Random:
    return random.Random(hash(user_id))


def _require(cfg: dict, key: str, task: str):
    if key not in cfg:
        raise KeyError(f"{task} config must specify '{key}'")
    return cfg[key]


def _is_current_schema(existing: dict) -> bool:
    assignments = existing.get("task_assignments") or {}
    if not assignments:
        return False
    return all("conv_id" in v for v in assignments.values())


def get_realism_assignments(user_id: str) -> dict:
    existing = storage.load_labels(user_id, "realism")
    if _is_current_schema(existing):
        return existing

    rng = _make_rng(user_id)
    cfg = data.get_task_config("realism")

    sources = _require(cfg, "sources", "realism")
    cases = _require(cfg, "cases", "realism")
    n_samples_per_case = _require(cfg, "n_samples_per_case", "realism")

    task_assignments = {}
    order = []
    for conv_id in cases:
        available = [
            s for s in sources if s == "real" or conv_id in data.get_conversation_ids(s)
        ]
        if not available:
            continue
        picks = rng.sample(available, min(n_samples_per_case, len(available)))
        for k, source in enumerate(picks):
            task_id = f"{conv_id}#{k}"
            task_assignments[task_id] = {
                "conv_id": conv_id,
                "source": source,
                "is_real": source == "real",
            }
            order.append(task_id)

    result = {"task_assignments": task_assignments, "labels": {}, "order": order}
    storage.save_labels(user_id, "realism", result)
    return result


def get_personality_assignments(user_id: str) -> dict:
    existing = storage.load_labels(user_id, "personality")
    if _is_current_schema(existing):
        return existing

    rng = _make_rng(user_id)
    cfg = data.get_task_config("personality")

    simulators = _require(cfg, "patients", "personality")
    cases = _require(cfg, "cases", "personality")
    n_samples_per_case = _require(cfg, "n_samples_per_case", "personality")

    task_assignments = {}
    order = []
    for conv_id in cases:
        available = [s for s in simulators if conv_id in data.get_conversation_ids(s)]
        if not available:
            continue
        picks = rng.sample(available, min(n_samples_per_case, len(available)))
        for k, sim in enumerate(picks):
            task_id = f"{conv_id}#{k}"
            task_assignments[task_id] = {
                "conv_id": conv_id,
                "simulator": sim,
            }
            order.append(task_id)

    result = {"task_assignments": task_assignments, "labels": {}, "order": order}
    storage.save_labels(user_id, "personality", result)
    return result
