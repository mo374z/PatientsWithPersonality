"""Profile-level comparison utilities: token recall for H (leisure) and C (medical) axes."""

import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from patient_simulator.patients.pwp import LEISURE_FIELDS, MEDICAL_FIELDS

_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "and",
    "or",
    "is",
    "was",
    "to",
    "for",
    "at",
    "on",
    "with",
    "by",
    "from",
    "as",
    "no",
    "not",
    "had",
    "has",
}


def _token_recall(sim_val, orig_val):
    """Fraction of original (non-stopword) tokens present in the simulated value."""
    if not orig_val or orig_val == "Unknown":
        return np.nan
    if not sim_val or sim_val == "Unknown":
        return 0.0
    orig_tokens = {
        t for t in re.findall(r"\w+", orig_val.lower()) if t not in _STOPWORDS
    }
    sim_tokens = {t for t in re.findall(r"\w+", sim_val.lower()) if t not in _STOPWORDS}
    return len(orig_tokens & sim_tokens) / len(orig_tokens) if orig_tokens else np.nan


def compute_profile_recalls(
    res,
    data_dir="data/aci_bench/extracted_profiles",
    sim_filename="sim_profile.json",
    patient_type="PatientsWithPersonality",
):
    """Compute per-field token recall between sim_profile and original profile.

    Returns a DataFrame with one row per (conversation × field) containing:
    param_h, param_c, field, field_type (leisure|medical), recall.
    Only PatientsWithPersonality rows with a loadable sim_profile are included.
    """
    sub = res[res["patient_type"] == patient_type].reset_index(drop=True)

    rows = []
    for _, row in sub.iterrows():
        sim_path = os.path.join(row["path"], sim_filename)
        orig_path = os.path.join(data_dir, f"{row['conversation_name']}.json")
        if not os.path.isfile(sim_path) or not os.path.isfile(orig_path):
            continue
        with open(sim_path) as f:
            sim = json.load(f)
        with open(orig_path) as f:
            orig = json.load(f)

        for field in LEISURE_FIELDS:
            r = _token_recall(sim.get(field, "Unknown"), orig.get(field, "Unknown"))
            if not np.isnan(r):
                rows.append(
                    {
                        "param_h": row["param_h"],
                        "param_c": row["param_c"],
                        "field": field,
                        "field_type": "leisure",
                        "recall": r,
                    }
                )
        for field in MEDICAL_FIELDS:
            r = _token_recall(sim.get(field, "Unknown"), orig.get(field, "Unknown"))
            if not np.isnan(r):
                rows.append(
                    {
                        "param_h": row["param_h"],
                        "param_c": row["param_c"],
                        "field": field,
                        "field_type": "medical",
                        "recall": r,
                    }
                )

    return pd.DataFrame(rows)


def compute_persona_recalls(
    res,
    data_dir="data/aci_bench/extracted_profiles",
    sim_filename="sim_profile.json",
):
    """Compute per-field token recall for all rows in res.

    Requires res to have 'persona' and 'simulator' columns (already mapped from PERSONA_MAP).
    Returns a DataFrame with columns: simulator, persona, field, field_type, recall.
    """
    rows = []
    for _, row in res.iterrows():
        sim_path = os.path.join(row["path"], sim_filename)
        orig_path = os.path.join(data_dir, f"{row['conversation_name']}.json")
        if not os.path.isfile(sim_path) or not os.path.isfile(orig_path):
            continue
        with open(sim_path) as f:
            sim = json.load(f)
        with open(orig_path) as f:
            orig = json.load(f)

        for field in LEISURE_FIELDS:
            r = _token_recall(sim.get(field, "Unknown"), orig.get(field, "Unknown"))
            if not np.isnan(r):
                rows.append(
                    {
                        "simulator": row["simulator"],
                        "persona": row["persona"],
                        "field": field,
                        "field_type": "leisure",
                        "recall": r,
                    }
                )
        for field in MEDICAL_FIELDS:
            r = _token_recall(sim.get(field, "Unknown"), orig.get(field, "Unknown"))
            if not np.isnan(r):
                rows.append(
                    {
                        "simulator": row["simulator"],
                        "persona": row["persona"],
                        "field": field,
                        "field_type": "medical",
                        "recall": r,
                    }
                )
    return pd.DataFrame(rows)


def plot_information_handling_comparison(
    recall_df,
    simulator_order=None,
    extreme_personas=("Dishonest", "Disorganized"),
    figure_size=(6, 3.5),
    save_as=None,
):
    """Two-panel Δ-recall chart: change vs Default for leisure and medical fields.

    One panel per extreme persona (Dishonest, Disorganized). Each shows Δ token
    recall = recall(extreme) − recall(Default), broken out by field type and grouped
    by simulator. The double dissociation for PatientsWithPersonality (opposing signs across
    field types) vs uniform suppression in PatientSimPatient is the key story.
    No title; legend to the right.
    """
    import matplotlib.patches as mpatches
    from patient_simulator.misc.plotting import save_to_figures

    if simulator_order is None:
        simulator_order = list(dict.fromkeys(recall_df["simulator"].dropna()))

    means = recall_df.groupby(["simulator", "persona", "field_type"])["recall"].mean()
    sems = recall_df.groupby(["simulator", "persona", "field_type"])["recall"].sem()
    stats = pd.DataFrame({"mean": means, "sem": sems}).reset_index()

    default_stats = stats[stats["persona"] == "Default"].set_index(
        ["simulator", "field_type"]
    )

    delta_rows = []
    for _, row in stats[stats["persona"].isin(extreme_personas)].iterrows():
        key = (row["simulator"], row["field_type"])
        if key not in default_stats.index:
            continue
        d = default_stats.loc[key]
        delta_rows.append(
            {
                "simulator": row["simulator"],
                "persona": row["persona"],
                "field_type": row["field_type"],
                "delta": row["mean"] - d["mean"],
                "delta_sem": np.sqrt(row["sem"] ** 2 + d["sem"] ** 2),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    print(deltas)

    field_colors = {"leisure": "#4C72B0", "medical": "#DD8452"}
    field_labels = {"leisure": "Leisure fields", "medical": "Medical fields"}
    field_types = ["leisure", "medical"]

    fig, axes = plt.subplots(1, len(extreme_personas), figsize=figure_size, sharey=True)
    if len(extreme_personas) == 1:
        axes = [axes]

    x = np.arange(len(simulator_order))
    width = 0.35

    for ax_i, persona in enumerate(extreme_personas):
        ax = axes[ax_i]
        sub = deltas[deltas["persona"] == persona]

        for ft_i, field_type in enumerate(field_types):
            sub_ft = sub[sub["field_type"] == field_type].set_index("simulator")
            vals = [
                sub_ft.loc[s, "delta"] if s in sub_ft.index else np.nan
                for s in simulator_order
            ]
            errs = [
                sub_ft.loc[s, "delta_sem"] if s in sub_ft.index else np.nan
                for s in simulator_order
            ]
            ax.bar(
                x + (ft_i - 0.5) * width,
                vals,
                width,
                yerr=errs,
                color=field_colors[field_type],
                # alpha=0.85,
                edgecolor="black",
                linewidth=0.7,
                capsize=4,
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_title(f"Persona: {persona}")
        ax.set_xticklabels(["PatientsWithPersonality", "PatientSim"])
        ax.grid(axis="y", alpha=0.3)
        if ax_i == 0:
            ax.set_ylabel("Δ Token recall vs. Default")

    legend_handles = [
        mpatches.Patch(
            facecolor=field_colors[ft],
            label=field_labels[ft],
            edgecolor="black",
            linewidth=0.4,
        )
        for ft in field_types
    ]
    axes[-1].legend(
        handles=legend_handles,
        # loc="center left",
        # bbox_to_anchor=(1.02, 0.5),
    )
    plt.tight_layout()

    if save_as:
        save_to_figures(fig, save_as)

    plt.show()
    return fig


def plot_information_handling(
    recall_df,
    figure_size=(10, 4.5),
    save_as=None,
):
    """Two-panel bar chart: leisure recall by H level and medical recall by C level.

    No title; layout matches grouped bar plots elsewhere in the notebook.
    """
    from patient_simulator.misc.plotting import PATIENT_TYPE_PALETTE, save_to_figures

    leisure = recall_df[recall_df["field_type"] == "leisure"]
    medical = recall_df[recall_df["field_type"] == "medical"]

    h_levels = sorted(leisure["param_h"].dropna().unique())
    c_levels = sorted(medical["param_c"].dropna().unique())

    h_means = leisure.groupby("param_h")["recall"].mean().reindex(h_levels).values
    h_sems = leisure.groupby("param_h")["recall"].sem().reindex(h_levels).values
    c_means = medical.groupby("param_c")["recall"].mean().reindex(c_levels).values
    c_sems = medical.groupby("param_c")["recall"].sem().reindex(c_levels).values

    color = PATIENT_TYPE_PALETTE.get("PatientsWithPersonality (Ours)", "#888888")
    bar_kwargs = dict(
        color=color, alpha=0.85, edgecolor="black", linewidth=0.7, capsize=4
    )

    fig, axes = plt.subplots(1, 2, figsize=figure_size)

    ax = axes[0]
    x = np.arange(len(h_levels))
    ax.bar(x, h_means, yerr=h_sems, **bar_kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H = {int(h)}" for h in h_levels])
    ax.set_ylabel("Token recall (leisure fields)")
    ax.set_ylim(0, max(h_means + h_sems) * 1.25)
    ax.set_xlabel("Honesty-Humility (H)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    x = np.arange(len(c_levels))
    ax.bar(x, c_means, yerr=c_sems, **bar_kwargs)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C = {int(c)}" for c in c_levels])
    ax.set_ylabel("Token recall (medical fields)")
    ax.set_ylim(0, max(c_means + c_sems) * 1.25)
    ax.set_xlabel("Conscientiousness (C)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_as:
        save_to_figures(fig, save_as)

    plt.show()
    return fig
