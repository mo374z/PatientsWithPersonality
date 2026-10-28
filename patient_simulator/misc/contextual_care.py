"""Analysis of the contextual-care benchmark (``scripts/run_contextual_care_benchmark.py``).

Every rate is pooled over the coded contextual factors of a case, and every interval comes
from a case bootstrap that reuses one resample of the cases for all arms of a doctor model,
so differences between arms are paired.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from patient_simulator.misc.plotting import PATIENT_TYPE_PALETTE

METRICS = [
    "cause_correct",
    "contextual_error",
    "plan_matches_real",
    "factor_elicited",
    "unsupported_cause",
]

SIMULATOR_NAMES = {
    "PatientsWithPersonality_H2_E1_X1_A2_C2_O1_LB": "PWP",
    "PatientsWithPersonality_H2_E2_X2_A2_C2_O2_LB": "PWP all-2",
    "PatientsWithPersonality_H3_E1_X1_A2_C3_O1_LB": "PWP withholding (H3 C3)",
    "PatientsWithPersonality_H2_E3_X1_A2_C2_O3_LB": "PWP emotional (E3 O3)",
    "PatientSimPatient_persplain_cefrB_dazednormal_recallhigh": "PatientSim",
    "AgentClinicPatient_nobias": "AgentClinic",
}

MODEL_NAMES = {
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
}

REAL_REFERENCES = ["real_no_plan", "real_transcript", "real_patient_only"]
BASELINES = ["PatientSim", "AgentClinic"]


def load_results(paths: list[Path]) -> pd.DataFrame:
    """Concatenate benchmark ``results.csv`` files into one frame of 0/1 metric columns.

    ``arm`` is the doctor arm, suffixed with the simulator for interview arms
    (e.g. ``strong:PWP``); ``model`` is the doctor model.
    """
    df = pd.concat([pd.read_csv(p, dtype={"case_id": str}) for p in paths])
    for col in ["cause_correct", "contextual_error", "plan_matches_real", "factor_elicited", "cause_identified"]:
        df[col] = df[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    df["unsupported_cause"] = np.where(df.has_factor, np.nan, df.cause_identified)
    simulator = df.patient_name.map(SIMULATOR_NAMES)
    df["arm"] = np.where(simulator.notna(), df.doctor_arm + ":" + simulator.fillna(""), df.doctor_arm)
    df["model"] = df.assistant_model
    return df


def case_bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Pooled rate and its bootstrap draws for every (model, arm, metric).

    Returns ``{(model, arm, metric): (estimate, draws)}``.
    """
    cases = np.array(sorted(set(df.case_id)))
    idx = np.random.default_rng(seed).integers(0, len(cases), (n_boot, len(cases)))
    draws = {}
    for (model, arm), d in df.groupby(["model", "arm"]):
        for metric in METRICS:
            g = d.dropna(subset=[metric]).groupby("case_id")[metric]
            if not len(g):
                continue
            hits = g.sum().reindex(cases, fill_value=0).values
            counts = g.count().reindex(cases, fill_value=0).values
            draws[(model, arm, metric)] = (
                hits.sum() / counts.sum(),
                hits[idx].sum(1) / counts[idx].sum(1),
            )
    return draws


def interval(estimate: float, draws: np.ndarray, mark_nonzero: bool = False) -> str:
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    star = "*" if mark_nonzero and (lo > 0 or hi < 0) else ""
    return f"{estimate:.2f} [{lo:.2f}, {hi:.2f}]{star}"


def estimates_frame(draws: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model,
                "arm": arm,
                "metric": metric,
                "est": est,
                "lo": np.nanpercentile(d, 2.5),
                "hi": np.nanpercentile(d, 97.5),
            }
            for (model, arm, metric), (est, d) in draws.items()
        ]
    )


def rate_table(draws: dict, arms: list[str], metric: str) -> pd.DataFrame:
    """Rate per arm (rows) and doctor model (columns), with the difference to ``arms[0]``."""
    reference = arms[0]
    rows = {}
    for arm in arms:
        rows[arm] = {}
        for model in MODEL_NAMES:
            est, d = draws[(model, arm, metric)]
            cell = interval(est, d)
            if arm != reference:
                ref_est, ref_d = draws[(model, reference, metric)]
                cell += f" ({interval(est - ref_est, d - ref_d, mark_nonzero=True)})"
            rows[arm][MODEL_NAMES[model]] = cell
    return pd.DataFrame(rows).T


def distance_to_real_table(draws: dict, reference: str, metric: str, doctor: str) -> pd.DataFrame:
    """|simulator - real| per doctor model and averaged over models.

    ``PWP - <baseline>`` below zero means PWP is closer to the real visit; ``*`` marks a
    paired interval that excludes zero.
    """
    simulators = ["PWP"] + BASELINES
    gaps = {sim: [] for sim in simulators}
    for model in MODEL_NAMES:
        ref_est, ref_d = draws[(model, reference, metric)]
        for sim in simulators:
            est, d = draws[(model, f"{doctor}:{sim}", metric)]
            gaps[sim].append((abs(est - ref_est), np.abs(d - ref_d)))
    for sim in simulators:
        gaps[sim].append(
            (np.mean([e for e, _ in gaps[sim]]), np.mean([d for _, d in gaps[sim]], axis=0))
        )
    rows = {}
    for i, model in enumerate([*MODEL_NAMES.values(), "Mean over models"]):
        rows[model] = {f"|{sim} - real|": interval(*gaps[sim][i]) for sim in simulators}
        for baseline in BASELINES:
            (e_pwp, d_pwp), (e_base, d_base) = gaps["PWP"][i], gaps[baseline][i]
            rows[model][f"PWP - {baseline}"] = interval(
                e_pwp - e_base, d_pwp - d_base, mark_nonzero=True
            )
    return pd.DataFrame(rows).T


def model_spread_table(draws: dict, reference: str, metric: str, doctor: str) -> pd.DataFrame:
    """Gap between the strongest and the weakest doctor model under each patient.

    A simulator that pushes every model towards the ceiling shrinks this gap relative to
    the real visit.
    """
    strongest, weakest = list(MODEL_NAMES)[-1], list(MODEL_NAMES)[0]
    arms = [reference] + [f"{doctor}:{sim}" for sim in ["PWP"] + BASELINES]
    spreads = {}
    for arm in arms:
        (e_hi, d_hi), (e_lo, d_lo) = draws[(strongest, arm, metric)], draws[(weakest, arm, metric)]
        spreads[arm] = (e_hi - e_lo, d_hi - d_lo)
    e_ref, d_ref = spreads[reference]
    e_pwp, d_pwp = spreads[arms[1]]
    rows = {}
    for arm, (e, d) in spreads.items():
        rows[arm] = {"spread": interval(e, d)}
        if arm != reference:
            rows[arm]["share of real spread"] = interval(e / e_ref, d / d_ref)
        if arm in arms[2:]:
            rows[arm]["PWP share - share"] = interval(
                (e_pwp - e) / e_ref, (d_pwp - d) / d_ref, mark_nonzero=True
            )
    return pd.DataFrame(rows).T.fillna("")


def persona_effect_table(
    df: pd.DataFrame, personas: list[str], default: str, n_boot: int, seed: int
) -> pd.DataFrame:
    """Persona minus default, pooled over doctor models and seeds, with a paired flag bootstrap."""
    d = df[df.has_factor].assign(flag=lambda x: x.case_id + "_" + x.flag_idx.astype(str))
    flags = np.array(sorted(d.flag.unique()))
    idx = np.random.default_rng(seed).integers(0, len(flags), (n_boot, len(flags)))
    rows = []
    for doctor in ["weak", "strong"]:
        for persona in personas:
            row = {"doctor": doctor, "persona": persona}
            for metric in ["cause_correct", "contextual_error", "factor_elicited"]:
                per_flag = [
                    d[d.arm == f"{doctor}:{p}"].groupby("flag")[metric].mean().reindex(flags)
                    for p in [persona, default]
                ]
                diff = (per_flag[0] - per_flag[1]).values
                row[metric] = interval(
                    np.nanmean(diff), np.nanmean(diff[idx], axis=1), mark_nonzero=True
                )
            rows.append(row)
    return pd.DataFrame(rows).set_index(["doctor", "persona"])


PLOT_ARMS = {
    "real_no_plan": ("Real visit", "#222222", "D"),
    "strong:PWP": ("PWP (ours)", PATIENT_TYPE_PALETTE["PatientsWithPersonality"], "o"),
    "strong:PatientSim": ("PatientSim", PATIENT_TYPE_PALETTE["PatientSim"], "s"),
    "strong:AgentClinic": ("AgentClinic", PATIENT_TYPE_PALETTE["AgentClinic"], "^"),
}

PLOT_PANELS = {
    "contextual_error": "Contextual error rate\n(↓ better model or easier task)",
    "cause_correct": "Cause correctly identified\n(↑ better model or easier task)",
}


def plot_benchmark(estimates: pd.DataFrame, dodge: float = 0.07) -> plt.Figure:
    """Per-model rate of each patient on both headline metrics, with 95% intervals."""
    est = estimates.set_index(["metric", "arm", "model"]).sort_index()
    x = np.arange(len(MODEL_NAMES))

    def line_style(arm, color):
        is_real = arm == "real_no_plan"
        return {
            "lw": 2.2 if is_real else 1.4,
            "ls": "--" if is_real else "-",
            "mec": color if is_real else "#222222",
        }

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.1))
    for ax, (metric, title) in zip(axes, PLOT_PANELS.items()):
        for i, (arm, (_, color, marker)) in enumerate(PLOT_ARMS.items()):
            rows = est.loc[(metric, arm)].loc[list(MODEL_NAMES)]
            xs = x + (i - (len(PLOT_ARMS) - 1) / 2) * dodge
            style = line_style(arm, color)
            ax.plot(xs, rows.est, color=color, lw=style["lw"], ls=style["ls"], zorder=2)
            ax.errorbar(
                xs,
                rows.est,
                yerr=[rows.est - rows.lo, rows.hi - rows.est],
                fmt=marker,
                color=color,
                ms=6,
                mec=style["mec"],
                mew=0.6,
                elinewidth=1,
                capsize=0,
                zorder=4,
            )
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(x, [name.replace(" ", "\n", 1) for name in MODEL_NAMES.values()], fontsize=9)
        ax.set_xlim(-0.35, len(MODEL_NAMES) - 0.65)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color="0.9", lw=0.6)

    axes[0].set_ylabel("Rate (95% CI)")
    handles = [
        Line2D([], [], color=color, marker=marker, ms=6, mew=0.6, label=label, **line_style(arm, color))
        for arm, (label, color, marker) in PLOT_ARMS.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(PLOT_ARMS), frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
