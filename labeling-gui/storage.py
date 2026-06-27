import json
import tempfile
from pathlib import Path

_labels_dir: Path = None


def init(labels_dir: str):
    global _labels_dir
    _labels_dir = Path(labels_dir)


def _get_labels_dir() -> Path:
    if _labels_dir is None:
        raise RuntimeError("storage.init() must be called before accessing labels")
    return _labels_dir


def _user_dir(user_id: str) -> Path:
    return _get_labels_dir() / user_id


def load_labels(user_id: str, task_type: str) -> dict:
    path = _user_dir(user_id) / f"{task_type}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_labels(user_id: str, task_type: str, data: dict):
    user_dir = _user_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{task_type}.json"
    tmp = tempfile.NamedTemporaryFile(
        mode="w", dir=user_dir, suffix=".tmp", delete=False
    )
    try:
        json.dump(data, tmp, indent=2)
        tmp.close()
        Path(tmp.name).replace(path)
    except Exception:
        Path(tmp.name).unlink(missing_ok=True)
        raise


def get_progress(user_id: str, task_type: str) -> tuple[int, int]:
    data = load_labels(user_id, task_type)
    assignments = data.get("task_assignments", {})
    labels = data.get("labels", {})
    return len(labels), len(assignments)


def list_users() -> list[str]:
    labels_dir = _get_labels_dir()
    if not labels_dir.exists():
        return []
    return sorted([d.name for d in labels_dir.iterdir() if d.is_dir()])
