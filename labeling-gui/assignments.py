import hashlib
import random
from itertools import combinations

import data
import storage

SCHEMA = 3


def _rng(*parts) -> random.Random:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest, 16))


def _require(cfg: dict, key: str, task: str):
    if key not in cfg:
        raise KeyError(f"{task} config must specify '{key}'")
    return cfg[key]


def arm_combinations(
    arms: list[str], arms_per_case: int, exclusive: list[str], always: list[str]
) -> list[list[str]]:
    unknown = [arm for arm in exclusive + always if arm not in arms]
    if unknown:
        raise ValueError(f"{unknown} are not among the configured arms")
    overlap = set(exclusive) & set(always)
    if overlap:
        raise ValueError(f"arms {sorted(overlap)} cannot be both exclusive and always-shown")
    free = [arm for arm in arms if arm not in exclusive and arm not in always]
    combos = [
        [one, *always, *rest]
        for one in exclusive
        for rest in combinations(free, arms_per_case - 1 - len(always))
    ]
    if not combos:
        raise ValueError(
            f"no valid selection of {arms_per_case} arms shows exactly one of "
            f"{exclusive} plus all of {always} alongside {len(free)} other arms"
        )
    return combos


def shown_arms(
    n_cases: int,
    arms: list[str],
    arms_per_case: int,
    exclusive: list[str],
    always: list[str],
    rng: random.Random,
) -> list[list[str]]:
    combos = arm_combinations(arms, arms_per_case, exclusive, always)
    pool = []
    while len(pool) < n_cases:
        cycle = list(combos)
        rng.shuffle(cycle)
        pool += cycle
    return pool[:n_cases]


def _realism_items(cfg: dict, user_id: str) -> list[dict]:
    arms = _require(cfg, "arms", "realism")
    cases = _require(cfg, "cases", "realism")
    arms_per_case = _require(cfg, "arms_per_case", "realism")
    exclusive = _require(cfg, "exclusive_arms", "realism")
    always = _require(cfg, "always_arms", "realism")

    rng = _rng("realism", user_id)
    items = []
    for case, shown in zip(
        cases, shown_arms(len(cases), arms, arms_per_case, exclusive, always, rng)
    ):
        rng.shuffle(shown)
        items.append({"conv_id": case, "arms": shown})
    rng.shuffle(items)
    return items


def _personality_items(cfg: dict, user_id: str) -> list[dict]:
    arm_pairs = _require(cfg, "arm_pairs", "personality")
    common_cases = _require(cfg, "common_cases", "personality")
    rotation_pool = _require(cfg, "rotation_cases", "personality")

    conditions = [
        ((pair["axis"], pair["pwp"]), (pair["axis"], pair["patientsim"]))
        for pair in arm_pairs
    ]
    rng = _rng("personality", user_id)
    rotation_cases = rng.sample(rotation_pool, len(conditions) - len(common_cases))
    blocks = {
        "common": zip(common_cases, conditions[: len(common_cases)]),
        "rotation": zip(rotation_cases, conditions[len(common_cases) :]),
    }
    items = [
        {"conv_id": case, "simulator": simulator, "axis": axis, "block": block}
        for block, assigned in blocks.items()
        for case, condition in assigned
        for axis, simulator in condition
    ]
    return _space_repeated_cases(items, rng)


def _space_repeated_cases(items: list[dict], rng: random.Random) -> list[dict]:
    by_case = {}
    for item in items:
        by_case.setdefault(item["conv_id"], []).append(item)
    for case_items in by_case.values():
        rng.shuffle(case_items)

    cases = list(by_case)
    rng.shuffle(cases)
    rounds = max(len(case_items) for case_items in by_case.values())
    return [
        by_case[case][k]
        for k in range(rounds)
        for case in cases
        if k < len(by_case[case])
    ]


BUILDERS = {"realism": _realism_items, "personality": _personality_items}


def get_assignments(user_id: str, task: str) -> dict:
    existing = storage.load_labels(user_id, task)
    if existing.get("schema") == SCHEMA:
        return existing

    items = BUILDERS[task](data.get_task_config(task), user_id)

    task_assignments = {}
    order = []
    seen = {}
    for item in items:
        conv_id = item["conv_id"]
        k = seen.get(conv_id, 0)
        seen[conv_id] = k + 1
        task_id = f"{conv_id}#{k}"
        task_assignments[task_id] = item
        order.append(task_id)

    result = {
        "schema": SCHEMA,
        "task_assignments": task_assignments,
        "labels": {},
        "order": order,
    }
    storage.save_labels(user_id, task, result)
    return result


def get_realism_assignments(user_id: str) -> dict:
    return get_assignments(user_id, "realism")


def get_personality_assignments(user_id: str) -> dict:
    return get_assignments(user_id, "personality")
