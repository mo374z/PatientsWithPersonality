"""Judge-prompt and conversation-window ablation for HEXACO personality recovery.

Re-scores stored runs under two judge prompts — the ``anchored`` one used in the
paper, whose level criteria paraphrase the persona-generating prompts, and a
``neutral`` one that names only the HEXACO axis and its scale direction — on
several turn windows. This separates recoverability from judge-prompt wording
overlap and from surface cues concentrated in the opening turns.
"""

import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.asyncio import tqdm as atqdm

from patient_simulator.eval import (
    HEXACO_AXES,
    format_transcript,
    reconstruct_personality_params,
)
from patient_simulator.misc.display_names import (
    canonical_patient_name,
    patient_display_name,
    patient_family,
    pwp_target_axes,
)
from patient_simulator.misc.llm import LLM
from patient_simulator.misc.utils import retry_on_transient_error


def parse_windows(windows) -> dict[str, tuple[int, Optional[int]]]:
    """Turn a config mapping ``{name: [start, end]}`` into 1-indexed inclusive bounds.

    ``end: null`` means "to the end of the conversation".
    """
    parsed = {}
    for name, bounds in dict(windows).items():
        start, end = list(bounds)
        if start < 1 or (end is not None and end < start):
            raise ValueError(f"Invalid turn window {name!r}: {bounds}")
        parsed[str(name)] = (int(start), None if end is None else int(end))
    return parsed


def slice_window(turns: pd.DataFrame, bounds: tuple[int, Optional[int]]):
    """Turns inside the window, or None if the conversation does not cover it."""
    start, end = bounds
    if len(turns) < (end if end is not None else start):
        return None
    return turns.iloc[start - 1 : end]


def collect_runs(output_dir: str, patient_names, model_name: str) -> pd.DataFrame:
    conversations = pd.read_csv(Path(output_dir) / "all_conversations.csv")
    wanted = {canonical_patient_name(name) for name in patient_names}
    canonical = conversations["patient_name"].map(canonical_patient_name)
    runs = conversations[
        canonical.isin(wanted) & (conversations["model_name"] == model_name)
    ]
    missing = wanted - set(canonical[runs.index])
    if missing:
        raise ValueError(f"No runs for patient names {sorted(missing)} in {output_dir}")
    return runs.reset_index(drop=True)


async def score_runs(
    runs: pd.DataFrame,
    llm: LLM,
    windows: dict[str, tuple[int, Optional[int]]],
    prompt_variants,
    concurrency: int,
) -> pd.DataFrame:
    """Reconstruct HEXACO for every (run x window x judge prompt) cell."""
    semaphore = asyncio.Semaphore(concurrency)

    async def score_cell(run, window_name, transcript, n_window_turns, variant):
        async with semaphore:
            reconstruction = await retry_on_transient_error(
                lambda: reconstruct_personality_params(
                    conversation=transcript,
                    llm=llm,
                    prompt_variant=variant,
                )
            )
        patient_name = canonical_patient_name(run["patient_name"])
        target = pwp_target_axes(patient_name) or {}
        return {
            "patient_name": patient_name,
            "patient_display_name": patient_display_name(patient_name),
            "simulator": patient_family(patient_name),
            "conversation_name": run["conversation_name"],
            "n_turns": run["n_turns"],
            "window": window_name,
            "n_window_turns": n_window_turns,
            "prompt_variant": variant,
            **{f"target_{axis}": target.get(axis) for axis in HEXACO_AXES},
            **{f"recon_{axis}": reconstruction[axis] for axis in HEXACO_AXES},
        }

    tasks = []
    for _, run in runs.iterrows():
        turns = pd.read_csv(Path(run["path"]) / "turns.csv")
        run = {**run, "n_turns": len(turns)}
        for window_name, bounds in windows.items():
            window_turns = slice_window(turns, bounds)
            if window_turns is None:
                continue
            transcript = format_transcript(window_turns)
            for variant in prompt_variants:
                tasks.append(
                    score_cell(
                        run, window_name, transcript, len(window_turns), variant
                    )
                )

    rows = await atqdm.gather(*tasks, desc="Reconstructions")
    return pd.DataFrame(rows)


def to_long(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (run, window, judge prompt, axis) with target and reconstructed level."""
    keys = [
        "patient_name",
        "patient_display_name",
        "simulator",
        "conversation_name",
        "window",
        "prompt_variant",
    ]
    frames = [
        scores[keys].assign(
            axis=axis,
            target=pd.to_numeric(scores[f"target_{axis}"], errors="coerce"),
            reconstructed=pd.to_numeric(scores[f"recon_{axis}"], errors="coerce"),
        )
        for axis in HEXACO_AXES
    ]
    return pd.concat(frames, ignore_index=True)


def _contrast(low: pd.Series, high: pd.Series) -> dict:
    """Mean shift from target level 1 to target level 3, with a Welch test and Cohen's d."""
    pooled_sd = np.sqrt((low.var(ddof=1) + high.var(ddof=1)) / 2)
    test = stats.ttest_ind(high, low, equal_var=False)
    return {
        "mean_low": low.mean(),
        "mean_high": high.mean(),
        "delta": high.mean() - low.mean(),
        "se": np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low)),
        "cohens_d": (high.mean() - low.mean()) / pooled_sd if pooled_sd else np.nan,
        "t": test.statistic,
        "p": test.pvalue,
        "n_low": len(low),
        "n_high": len(high),
    }


def axis_recovery(long: pd.DataFrame) -> pd.DataFrame:
    """Per (judge prompt, window, axis) discrimination between target levels 1 and 3."""
    scored = long.dropna(subset=["target", "reconstructed"])
    rows = []
    for (variant, window, axis), group in scored.groupby(
        ["prompt_variant", "window", "axis"], sort=False
    ):
        low = group.loc[group["target"] == 1, "reconstructed"]
        high = group.loc[group["target"] == 3, "reconstructed"]
        if low.empty or high.empty:
            raise ValueError(
                f"Missing target level 1 or 3 for {variant}/{window}/{axis}"
            )
        rows.append(
            {
                "prompt_variant": variant,
                "window": window,
                "axis": axis,
                "spearman": stats.spearmanr(
                    group["target"], group["reconstructed"]
                ).statistic,
                **_contrast(low, high),
            }
        )
    return pd.DataFrame(rows)


def overall_recovery(long: pd.DataFrame) -> pd.DataFrame:
    """Per (judge prompt, window) recovery pooled over all axes."""
    scored = long.dropna(subset=["target", "reconstructed"])
    rows = []
    for (variant, window), group in scored.groupby(
        ["prompt_variant", "window"], sort=False
    ):
        error = group["reconstructed"] - group["target"]
        rows.append(
            {
                "prompt_variant": variant,
                "window": window,
                "n": len(group),
                "spearman": stats.spearmanr(
                    group["target"], group["reconstructed"]
                ).statistic,
                "exact_match": (error == 0).mean(),
                "within_one": (error.abs() <= 1).mean(),
                "mae": error.abs().mean(),
                "bias": error.mean(),
                **_contrast(
                    group.loc[group["target"] == 1, "reconstructed"],
                    group.loc[group["target"] == 3, "reconstructed"],
                ),
            }
        )
    return pd.DataFrame(rows)


def manipulated_axis_contrast(
    scores: pd.DataFrame, persona_axis_map: dict[str, str]
) -> pd.DataFrame:
    """Extreme-minus-low recovery per simulator, for personas without numeric targets.

    For each simulator and axis: the mean reconstructed level of the persona that
    manipulates that axis, minus the mean level the same simulator's remaining
    single-axis personas receive on it. This is the cross-simulator comparison of
    the paper, which PatientSim personas support because they name a target axis
    even though they carry no HEXACO vector.
    """
    manipulated = scores[scores["patient_name"].isin(persona_axis_map)].copy()
    manipulated["manipulated_axis"] = manipulated["patient_name"].map(persona_axis_map)

    rows = []
    for (variant, window, simulator), group in manipulated.groupby(
        ["prompt_variant", "window", "simulator"], sort=False
    ):
        for axis in HEXACO_AXES:
            extreme = group.loc[group["manipulated_axis"] == axis, f"recon_{axis}"]
            low = group.loc[group["manipulated_axis"] != axis, f"recon_{axis}"]
            if extreme.empty or low.empty:
                raise ValueError(
                    f"No persona manipulating {axis} for {simulator} in {variant}/{window}"
                )
            rows.append(
                {
                    "prompt_variant": variant,
                    "window": window,
                    "simulator": simulator,
                    "axis": axis,
                    **_contrast(low, extreme),
                }
            )
    return pd.DataFrame(rows)


def window_agreement(long: pd.DataFrame, window_a: str, window_b: str) -> pd.DataFrame:
    """Agreement between two windows on identical runs, per judge prompt."""
    keys = ["patient_name", "conversation_name", "prompt_variant", "axis"]
    paired = long[long["window"] == window_a].merge(
        long[long["window"] == window_b],
        on=keys,
        suffixes=(f"_{window_a}", f"_{window_b}"),
    )
    rows = []
    for variant, group in paired.groupby("prompt_variant", sort=False):
        a = group[f"reconstructed_{window_a}"]
        b = group[f"reconstructed_{window_b}"]
        rows.append(
            {
                "prompt_variant": variant,
                "n": len(group),
                "spearman": stats.spearmanr(a, b).statistic,
                "exact_match": (a == b).mean(),
                "within_one": (a - b).abs().le(1).mean(),
                "mean_shift": (b - a).mean(),
            }
        )
    return pd.DataFrame(rows)


def judge_agreement(long_a: pd.DataFrame, long_b: pd.DataFrame, window: str) -> pd.DataFrame:
    """Agreement between two judges on identical runs, per judge prompt.

    ``persona_*`` compares the mean reconstructed level per persona and axis, the
    level at which the paper reports recovery; ``within_one`` compares individual
    conversations.
    """
    keys = ["patient_name", "conversation_name", "prompt_variant", "axis"]
    paired = long_a[long_a["window"] == window].merge(
        long_b[long_b["window"] == window], on=keys, suffixes=("_a", "_b")
    )
    rows = []
    for variant, group in paired.groupby("prompt_variant", sort=False):
        cells = group.groupby(["patient_name", "axis"])[["reconstructed_a", "reconstructed_b"]].mean()
        a, b = cells["reconstructed_a"], cells["reconstructed_b"]
        rows.append(
            {
                "prompt_variant": variant,
                "n_conversations": len(group[["patient_name", "conversation_name"]].drop_duplicates()),
                "n_cells": len(cells),
                "persona_spearman": stats.spearmanr(a, b).statistic,
                "persona_pearson": stats.pearsonr(a, b).statistic,
                "persona_mae": (a - b).abs().mean(),
                "within_one": (group["reconstructed_a"] - group["reconstructed_b"]).abs().le(1).mean(),
            }
        )
    return pd.DataFrame(rows)
