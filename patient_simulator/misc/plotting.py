"""Plotting utilities for patient simulator evaluation."""

import ast
import json
import math
from functools import partial
from itertools import combinations, permutations
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Circle, Wedge
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.transforms import Bbox, ScaledTranslation

from patient_simulator.misc.display_names import HEXACO_AXES

plt.style.use(Path(__file__).parent / "style.mplstyle")

_FIGURES_DIR = Path(__file__).parent.parent.parent / "notebooks" / "figures"

PATIENT_TYPE_ORDER = [
    "PatientsWithPersonality",
    "PatientSim",
    "AgentClinic",
    "VirtualPatient",
    "StateAwarePatient",
    "CraftMD",
    "Human Rephrase",
    "Real Transcripts",
    "Human Actor",
    "Human Reference",
]

ALL_AXES = HEXACO_AXES

def _sort_patient_types(types):
    order = {pt: i for i, pt in enumerate(PATIENT_TYPE_ORDER)}
    return sorted(types, key=lambda x: (order.get(x, len(PATIENT_TYPE_ORDER)), x))


sort_patient_types = _sort_patient_types


def save_to_figures(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    path = _FIGURES_DIR / (name if Path(name).suffix else f"{name}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

PATIENT_TYPE_PALETTE = {
    "Human Rephrase": "#555555",
    "PatientsWithPersonality": sns.color_palette("Dark2")[0],
    "CraftMD": sns.color_palette("Dark2")[2],
    "AgentClinic": sns.color_palette("Dark2")[5],
    "StateAwarePatient": sns.color_palette("Dark2")[3],
    "PatientSim": sns.color_palette("Dark2")[1],
    "VirtualPatient": sns.color_palette("Dark2")[6],
    "Real Transcripts": "#1f4e79",
    "Human Actor": "#808080",
    "Human Reference": "#808080",
}

def plot_model_size_metrics_with_uncertainty(
    df,
    metric_cols,
    summary_col=None,
    model_size_col="model_size_b",
    display_name_col="display_name",
    model_col="model",
    title="Scores vs estimated model size",
    y_label="Score",
    y_lim=None,
    figure_height=None,
    palette=None,
    metric_display_names=None,
    model_order=None,
    hatch_map=None,
):
    if not metric_cols:
        raise ValueError("metric_cols must contain at least one metric column")

    all_cols = [*metric_cols, *([summary_col] if summary_col else [])]
    required_cols = {model_size_col, display_name_col, model_col, *all_cols}
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    plot_df = df.dropna(subset=[model_size_col]).copy()

    if model_order is not None:
        existing = set(plot_df[display_name_col].unique())
        ordered_names = [n for n in model_order if n in existing]
    else:
        model_order_df = (
            plot_df[[display_name_col, model_size_col]]
            .drop_duplicates()
            .sort_values([model_size_col, display_name_col])
        )
        ordered_names = model_order_df[display_name_col].tolist()

    if figure_height is None:
        figure_height = max(4.0, len(ordered_names) * 0.42)

    if palette is None:
        colors = sns.color_palette("tab20", n_colors=max(len(ordered_names), 1))
        palette = dict(zip(ordered_names, colors))

    if y_lim is None:
        metric_values = (
            plot_df[all_cols]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy()
            .ravel()
        )
        metric_values = metric_values[pd.notna(metric_values)]
        if len(metric_values) > 0:
            v_min = float(metric_values.min())
            v_max = float(metric_values.max())
            value_range = v_max - v_min
            if value_range == 0:
                pad = max(abs(v_min) * 0.05, 0.05)
            else:
                pad = value_range * 0.05
            y_lim = (v_min - pad, v_max + pad)

    n = len(metric_cols) + (1 if summary_col else 0)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, figure_height), sharey=True, sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, metric_col) in enumerate(zip(axes, all_cols)):
        summary = (
            plot_df.groupby([model_col, display_name_col, model_size_col], as_index=False)[metric_col]
            .agg(mean="mean", std="std")
        )
        summary["std"] = summary["std"].fillna(0.0)
        summary[display_name_col] = pd.Categorical(
            summary[display_name_col], categories=ordered_names, ordered=True
        )
        summary = summary.sort_values(display_name_col)

        bar_colors = [palette.get(name, "gray") for name in summary[display_name_col]]
        bars = ax.barh(
            summary[display_name_col],
            summary["mean"],
            xerr=summary["std"],
            capsize=4,
            color=bar_colors,
            edgecolor="none",
        )
        if hatch_map:
            for bar, name in zip(bars, summary[display_name_col]):
                h = hatch_map.get(str(name), "")
                if h:
                    bar.set_hatch(h)
                    bar.set_edgecolor("black")
                    bar.set_linewidth(0.5)

        col_title = metric_display_names.get(metric_col, metric_col) if metric_display_names else metric_col
        ax.set_title(col_title)
        ax.set_xlabel(y_label)
        if i > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Model (ordered by estimated size)")
        if y_lim is not None:
            ax.set_xlim(*y_lim)

    fig.suptitle(title, y=1.03, fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_boxplot(
    df,
    metric,
    category="patient_type",
    label=None,
    real_metric=None,
    figure_size=(6, 4),
    show_title=True,
    save_as=None,
):
    plot_df = df.copy()
    patient_types = _sort_patient_types(plot_df[category].unique().tolist())

    if real_metric:
        real_df = plot_df[["conversation_name", real_metric]].copy()
        real_df[category] = " Real"
        real_df = real_df.rename(columns={real_metric: metric})
        plot_df = pd.concat(
            [plot_df[["conversation_name", category, metric]], real_df],
            ignore_index=True,
        )
        patient_types.append(" Real")

    palette = {pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray") for pt in patient_types}
    plot_df[category] = pd.Categorical(plot_df[category], categories=list(reversed(patient_types)), ordered=True)
    plot_df = plot_df.sort_values(by=category)

    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(data=plot_df, x=metric, y=category, hue=category, palette=palette, ax=ax, legend=False)

    if show_title:
        ax.set_title(f"{label} by Patient Type" if label else f"{metric} by Patient Type")
    ax.set_xlabel(label if label else metric)
    ax.set_ylabel("Patient Type")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def _paired_errors(per_user, category, fraction_col, user_col, paired_reference):
    """Standard error of the within-rater difference to ``paired_reference``."""
    rates = per_user.pivot(index=user_col, columns=category, values=fraction_col)
    if paired_reference not in rates.columns:
        raise ValueError(f"paired_reference {paired_reference!r} not in {list(rates.columns)}")
    errors = {}
    for level in rates.columns:
        diff = (rates[level] - rates[paired_reference]).dropna()
        errors[level] = 0.0 if level == paired_reference else diff.std() / len(diff) ** 0.5
    return pd.Series(errors).rename("error").rename_axis(category).reset_index()


def _category_fraction_frame(
    df, category, value_col, target_value, user_col, fraction_col, error_type, categories,
    paired_reference,
):
    grouped = df.groupby(category, as_index=False).agg(
        **{fraction_col: (value_col, lambda s: (s == target_value).mean())}
    )

    if user_col in df.columns and error_type is not None:
        per_user = df.groupby([category, user_col])[value_col].apply(
            lambda s: (s == target_value).mean()
        ).reset_index(name=fraction_col)

        def compute_error(x):
            if error_type == "std":
                return x.std()
            if error_type == "sem":
                return x.std() / len(x) ** 0.5
            if error_type == "mad":
                from scipy.stats import median_abs_deviation
                return median_abs_deviation(x, nan_policy="omit")

        if error_type == "paired":
            errors = _paired_errors(per_user, category, fraction_col, user_col, paired_reference)
        else:
            errors = per_user.groupby(category)[fraction_col].agg(error=compute_error).reset_index()
        grouped = grouped.merge(errors, on=category)
    else:
        grouped["error"] = 0.0

    return (
        grouped.set_index(category)
        .reindex(categories)
        .fillna(0.0)
        .rename_axis(category)
        .reset_index()
    )


def _category_fraction_barh(ax, grouped, category, fraction_col, error_type, palette):
    ax.barh(
        grouped[category].tolist(),
        grouped[fraction_col],
        xerr=grouped["error"] if error_type is not None else None,
        capsize=4 if error_type is not None else 0,
        color=[palette[c] for c in grouped[category]],
    )


def plot_category_fraction(
    df,
    category,
    value_col,
    target_value,
    user_col="user",
    fraction_col="fraction",
    error_type=None,
    paired_reference=None,
    facet_col=None,
    ncols=3,
    title=None,
    x_label="Proportion",
    y_label=None,
    figure_size=(6, 4),
    save_as=None,
):
    if error_type not in (None, "std", "sem", "mad", "paired"):
        raise ValueError(
            f"error_type must be None, 'std', 'sem', 'mad', or 'paired', got {error_type!r}"
        )
    if (error_type == "paired") != (paired_reference is not None):
        raise ValueError("error_type='paired' and paired_reference must be given together")

    missing_cols = [
        col for col in [category, value_col] + ([facet_col] if facet_col else []) if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if facet_col is not None and error_type is not None:
        raise ValueError("error_type is undefined within a facet; pass error_type=None with facet_col")

    categories = list(reversed(_sort_patient_types(df[category].dropna().unique().tolist())))
    palette = {
        c: PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    }
    frame = partial(
        _category_fraction_frame,
        category=category,
        value_col=value_col,
        target_value=target_value,
        user_col=user_col,
        fraction_col=fraction_col,
        error_type=error_type,
        categories=categories,
        paired_reference=paired_reference,
    )

    if facet_col is None:
        grouped = frame(df)
        fig, ax = plt.subplots(figsize=figure_size)
        _category_fraction_barh(ax, grouped, category, fraction_col, error_type, palette)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label or category.replace("_", " ").title())
        ax.set_xlim(0, min(1.0, (grouped[fraction_col] + grouped["error"]).max() * 1.05))
    else:
        facets = sorted(df[facet_col].dropna().unique())
        grouped = pd.concat(
            [frame(df[df[facet_col] == f]).assign(**{facet_col: f}) for f in facets],
            ignore_index=True,
        )
        ncols = min(ncols, len(facets))
        nrows = math.ceil(len(facets) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(figure_size[0] * ncols, figure_size[1] * nrows),
            sharex=True, squeeze=False,
        )
        axes = axes.flatten()

        for ax, facet in zip(axes, facets):
            sub = grouped[grouped[facet_col] == facet]
            _category_fraction_barh(ax, sub, category, fraction_col, error_type, palette)
            n = len(df[df[facet_col] == facet])
            ax.set_title(f"{facet} (n={n})")
            ax.grid(axis="x", alpha=0.3)
        for ax in axes[len(facets):]:
            ax.set_visible(False)

        axes[0].set_xlim(0, 1)
        for i in range(nrows):
            axes[i * ncols].set_ylabel(y_label or category.replace("_", " ").title())
        for ax in axes[len(facets) - ncols:]:
            ax.set_xlabel(x_label)
            ax.tick_params(labelbottom=True)
        if title:
            fig.suptitle(title, fontweight="bold")

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return grouped


def plot_dist_by_category(
    df,
    distribution_col="relevance_distribution",
    category="patient_type",
    sharey=False,
    ncols=5,
    valid_categories=None,
    save_as=None,
):
    rows = []
    dist_values = set()
    patient_types = list(reversed(_sort_patient_types(df[category].dropna().unique().tolist())))

    for _, row in df.iterrows():
        raw = row[distribution_col]
        try:
            dist = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(dist, dict):
                continue
        except (json.JSONDecodeError, ValueError):
            try:
                dist = ast.literal_eval(raw)
                if not isinstance(dist, dict):
                    continue
            except (ValueError, SyntaxError):
                continue
        dist_values.update(dist.keys())
        total = sum(dist.values()) or 1
        for value, count in dist.items():
            rows.append({category: row[category], "dist_value": value, "proportion": count / total})

    dist_values = sorted(dist_values)
    if valid_categories is not None:
        filtered_out = set(dist_values) - set(valid_categories)
        if filtered_out:
            print(f"Filtered out categories: {filtered_out}")
            for v in filtered_out:
                count = sum(1 for row in rows if row["dist_value"] == v)
                print(f"  - {v}: {count} instances")
        dist_values = [v for v in dist_values if v in valid_categories]
    if not dist_values or not patient_types:
        return

    long_df = pd.DataFrame(rows)

    summary = long_df.groupby([category, "dist_value"], as_index=False).agg(
        mean=("proportion", "mean"),
        sem=("proportion", lambda x: x.std() / len(x) ** 0.5),
    )

    full_index = pd.MultiIndex.from_product(
        [patient_types, dist_values], names=[category, "dist_value"]
    )
    summary = (
        summary.set_index([category, "dist_value"])
        .reindex(full_index)
        .fillna(0)
        .reset_index()
    )
    palette = {pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray") for pt in patient_types}

    n_plots = len(dist_values)
    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=sharey)
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, value in zip(axes, dist_values):
        sub = summary[summary["dist_value"] == value]
        means = [
            sub.loc[sub[category] == pt, "mean"].iloc[0] if (sub[category] == pt).any() else 0
            for pt in patient_types
        ]
        sems = [
            sub.loc[sub[category] == pt, "sem"].iloc[0] if (sub[category] == pt).any() else 0
            for pt in patient_types
        ]

        ax.barh(patient_types, means, xerr=sems, capsize=4, color=[palette[pt] for pt in patient_types])
        ax.set_title(value)
        ax.set_xlabel("Mean Proportion")
        if ax == axes[0]:
            ax.set_ylabel("Patient Type")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_stacked_bar(
    df,
    value_cols,
    value_dispnames,
    category="patient_type",
    title="Stacked Bar Plot",
    sort_by=None,
    figure_size=(8, 6),
    save_as=None,
):
    plot_df = df[[category] + value_cols].copy()
    plot_df = plot_df.groupby(category).mean().reset_index()

    totals = plot_df[value_cols].sum(axis=1)
    missing_info = []
    for idx, row in plot_df.iterrows():
        total = totals[idx]
        if total < 1.0:
            missing = 1.0 - total
            missing_info.append({category: row[category], "missing": missing})

    if missing_info:
        print("Warning: Some categories do not add up to 100%:")
        for info in missing_info:
            print(f"  {info[category]}: missing {info['missing']:.2%}")

    for col in value_cols:
        plot_df[col] = plot_df[col] / totals

    plot_df = plot_df.melt(
        id_vars=[category], value_vars=value_cols, var_name="category", value_name="value"
    )

    canonical_order = list(reversed(_sort_patient_types(plot_df[category].unique().tolist())))
    plot_df[category] = pd.Categorical(plot_df[category], categories=canonical_order, ordered=True)
    plot_df = plot_df.sort_values([category, "category"])

    palette = ["#d73027", "#fee08b", "#1a9850"]

    fig, ax = plt.subplots(figsize=figure_size)

    left = pd.Series([0.0] * len(canonical_order), index=canonical_order)
    for i, col in enumerate(value_cols):
        subset = plot_df[plot_df["category"] == col]
        ax.barh(
            subset[category],
            subset["value"],
            color=palette[i],
            left=left[subset[category]].values,
            label=value_dispnames[i],
        )
        left[subset[category]] += subset["value"].values

    ax.set_title(title)
    ax.set_xlabel("Proportion")
    ax.set_ylabel(category)
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_lineplot_steps(
    df,
    metric,
    category="step",
    hue_col="patient_type",
    label=None,
    figure_size=(8, 6),
):
    """Plot line plot showing metric evolution over steps for each patient type."""
    patient_types = sorted(df[hue_col].unique().tolist())
    colors = sns.color_palette("Set2", n_colors=len(patient_types))
    palette = dict(zip(patient_types, colors))

    fig, ax = plt.subplots(figsize=figure_size)
    sns.lineplot(data=df, x=category, y=metric, hue=hue_col, palette=palette, ax=ax, linewidth=3)
    
    ax.set_title(f"{label} over Steps" if label else f"{metric} over Steps")
    ax.set_xlabel(category.replace("_", " ").title())
    ax.set_ylabel(label if label else metric)

    fig.tight_layout()
    plt.show()


def plot_hexaco_reconstruction_deviation(
    df,
    figure_size=(8, 5),
    title="HEXACO Reconstruction Deviation",
):
    """Plot mean signed deviation between reconstructed and real HEXACO values."""

    axis_mapping = [
        ("O", "param_o", "personality_reconstructed_openness"),
        ("C", "param_c", "personality_reconstructed_conscientiousness"),
        ("A", "param_a", "personality_reconstructed_agreeableness"),
        ("X", "param_x", "personality_reconstructed_extraversion"),
        ("E", "param_e", "personality_reconstructed_emotional_state"),
        ("H", "param_h", "personality_reconstructed_honesty"),
    ]

    deviations = []
    std_devs = []
    labels = []

    for axis_label, real_col, reconstructed_col in axis_mapping:
        if real_col not in df.columns or reconstructed_col not in df.columns:
            mean_deviation = 0.0
            std_deviation = 0.0
        else:
            real_vals = pd.to_numeric(df[real_col], errors="coerce")
            reconstructed_vals = pd.to_numeric(df[reconstructed_col], errors="coerce")

            valid_mask = (
                real_vals.notna()
                & reconstructed_vals.notna()
                & (real_vals % 1 == 0)
                & (reconstructed_vals % 1 == 0)
            )

            if valid_mask.any():
                diffs = (
                    reconstructed_vals[valid_mask].astype(int)
                    - real_vals[valid_mask].astype(int)
                )
                mean_deviation = float(diffs.mean())
                std_deviation = float(diffs.std())
            else:
                mean_deviation = 0.0
                std_deviation = 0.0

        labels.append(axis_label)
        deviations.append(mean_deviation)
        std_devs.append(std_deviation)

    positive_color = sns.color_palette("Set2")[0]
    negative_color = sns.color_palette("Set2")[1]
    colors = [positive_color if value >= 0 else negative_color for value in deviations]

    fig, ax = plt.subplots(figsize=figure_size)
    ax.barh(labels, deviations, xerr=std_devs, capsize=5, color=colors, ecolor='black', alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)

    max_abs = max((abs(dev) + std for dev, std in zip(deviations, std_devs)), default=1.0)
    if max_abs == 0:
        max_abs = 1.0
    ax.set_xlim(-max_abs, max_abs)

    ax.set_title(title)
    ax.set_xlabel("Deviation (Reconstructed - Real)")
    ax.set_ylabel("HEXACO Axis")

    fig.tight_layout()
    plt.show()

    return pd.DataFrame({"axis": labels, "mean_deviation": deviations, "std_deviation": std_devs})


def plot_hexaco_deviation_grid(
    df,
    persona_col,
    simulator_col,
    palette,
    persona_order=None,
    simulator_order=None,
    reference_vector=(2, 1, 1, 2, 2, 1),
    ncols=3,
    figure_size=(15, 7),
):
    """Grid of per-persona HEXACO deviation subplots with grouped bars per simulator.

    Deviation = mean reconstructed HEXACO value minus ``reference_vector``
    (ordered H, E, X, A, C, O). The reference is a fixed vector, so the
    deviation is well-defined for every (persona, simulator) — a failure to
    compute indicates missing data.

    Bars are colored by persona (``palette``); simulators are distinguished by
    hatch and alpha.
    """
    axis_mapping = [
        ("H", "personality_reconstructed_H"),
        ("E", "personality_reconstructed_E"),
        ("X", "personality_reconstructed_X"),
        ("A", "personality_reconstructed_A"),
        ("C", "personality_reconstructed_C"),
        ("O", "personality_reconstructed_O"),
    ]
    axis_labels = [a[0] for a in axis_mapping]
    rec_cols = [a[1] for a in axis_mapping]
    ref = dict(zip(rec_cols, reference_vector))

    personas = persona_order or list(df[persona_col].dropna().unique())
    simulators = simulator_order or sorted(df[simulator_col].dropna().unique())

    num = df[rec_cols].apply(pd.to_numeric, errors="coerce")
    work = pd.concat([df[[persona_col, simulator_col]].reset_index(drop=True),
                      num.reset_index(drop=True)], axis=1)

    records = []
    for persona in personas:
        for sim in simulators:
            sub = work[(work[persona_col] == persona) & (work[simulator_col] == sim)]
            for axis_label, rec_col in axis_mapping:
                diffs = sub[rec_col].dropna() - ref[rec_col]
                if diffs.empty:
                    raise ValueError(
                        f"No reconstructed data for persona={persona!r} "
                        f"simulator={sim!r} axis={axis_label}"
                    )
                mean = float(diffs.mean())
                std = float(diffs.std()) if len(diffs) > 1 else 0.0
                records.append({persona_col: persona, simulator_col: sim,
                                "axis": axis_label, "mean": mean, "std": std})
    stats = pd.DataFrame(records)

    y_abs = (stats["mean"].abs() + stats["std"].fillna(0)).max() or 1.0
    y_lim = y_abs * 1.15

    nrows = math.ceil(len(personas) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size, sharey=True)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    x = np.arange(len(axis_labels))
    width = 0.8 / len(simulators)
    sim_styles = {sim: {"hatch": "" if i == 0 else "///", "alpha": 0.9 if i == 0 else 0.55}
                  for i, sim in enumerate(simulators)}

    for ax, persona in zip(axes, personas):
        color = palette.get(persona, "gray")
        for i, sim in enumerate(simulators):
            sub = (stats[(stats[persona_col] == persona) & (stats[simulator_col] == sim)]
                   .set_index("axis").reindex(axis_labels))
            offset = (i - (len(simulators) - 1) / 2) * width
            style = sim_styles[sim]
            ax.bar(x + offset, sub["mean"], width, yerr=sub["std"].fillna(0),
                   color=color, alpha=style["alpha"], hatch=style["hatch"],
                   edgecolor="black", linewidth=0.7, capsize=3,
                   label=sim if ax is axes[0] else None)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(axis_labels)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(personas):]:
        ax.set_visible(False)

    for i in range(nrows):
        axes[i * ncols].set_ylabel("Deviation\n(reconstructed - real)")

    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=len(simulators),
                   bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout()
    plt.show()

    return stats


def plot_hexaco_reconstructed_grouped_bar(
    df,
    category="patient_type",
    axis_cols=None,
    title="Average Reconstructed HEXACO Value by Patient Type",
    y_label="Average Reconstructed Value",
    figure_size=(12, 6),
    save_as=None,
):
    if axis_cols is None:
        axis_cols = [
            "personality_reconstructed_H",
            "personality_reconstructed_E",
            "personality_reconstructed_X",
            "personality_reconstructed_A",
            "personality_reconstructed_C",
            "personality_reconstructed_O",
        ]

    required_cols = [category, *axis_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    plot_df = df[required_cols].copy()
    for col in axis_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    summary = plot_df.groupby(category, as_index=False)[axis_cols].mean()
    long_df = summary.melt(
        id_vars=category,
        var_name="axis_col",
        value_name="mean_reconstructed_value",
    )

    axis_labels = [col.rsplit("_", 1)[-1] for col in axis_cols]
    axis_label_map = dict(zip(axis_cols, axis_labels))
    long_df["hexaco_axis"] = pd.Categorical(
        long_df["axis_col"].map(axis_label_map),
        categories=axis_labels,
        ordered=True,
    )

    category_order = _sort_patient_types(long_df[category].dropna().unique().tolist())
    palette = {
        item: PATIENT_TYPE_PALETTE.get(str(item).split("_")[0].strip(), "gray")
        for item in category_order
    }

    fig, ax = plt.subplots(figsize=figure_size)
    sns.barplot(
        data=long_df,
        x="hexaco_axis",
        y="mean_reconstructed_value",
        hue=category,
        hue_order=category_order,
        palette=palette,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("HEXACO Axis")
    ax.set_ylabel(y_label)
    ax.legend(title=category.replace("_", " ").title(), bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return summary


JUDGE_HATCHES = ["", "///", "...", "xxx", "\\\\\\", "++"]


def _parse_distribution(raw):
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError, TypeError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _hatch_for_groups(groups):
    return {g: JUDGE_HATCHES[i % len(JUDGE_HATCHES)] for i, g in enumerate(groups)}


def _group_legend(fig, groups, hatches, title, ncol=None):
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor="white", edgecolor="black", hatch=hatches[g], label=g)
        for g in groups
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol or len(groups),
        bbox_to_anchor=(0.5, 1.03),
        frameon=False,
        title=title,
    )


def _grouped_barh(ax, categories, groups, value_of, err_of, palette, hatches, group_gap=0.8):
    y = np.arange(len(categories))
    height = group_gap / len(groups)
    for gi, g in enumerate(groups):
        offset = ((len(groups) - 1) / 2 - gi) * height
        values = [value_of(c, g) for c in categories]
        errors = [err_of(c, g) for c in categories] if err_of is not None else None
        ax.barh(
            y + offset,
            values,
            height,
            xerr=errors,
            capsize=3,
            color=[palette[c] for c in categories],
            hatch=hatches[g],
            edgecolor="black",
            linewidth=0.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    return y


def plot_category_fraction_by_judge(
    df,
    judge_col,
    value_col,
    target_value,
    category="patient_disp_name",
    judge_order=None,
    title=None,
    x_label="Proportion",
    figure_size=(7, 5),
    save_as=None,
):
    """Horizontal grouped bars of a category fraction, one bar per judge (hatch), colored by patient type."""
    data = df.dropna(subset=[value_col])
    judges = judge_order or list(data[judge_col].dropna().unique())
    frac = (
        data.assign(_hit=data[value_col] == target_value)
        .groupby([category, judge_col], as_index=False)["_hit"]
        .mean()
        .rename(columns={"_hit": "fraction"})
    )

    categories = list(reversed(_sort_patient_types(frac[category].unique().tolist())))
    palette = {
        c: PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    }
    hatches = _hatch_for_groups(judges)
    lut = {(r[category], r[judge_col]): r["fraction"] for _, r in frac.iterrows()}

    fig, ax = plt.subplots(figsize=figure_size)
    _grouped_barh(
        ax,
        categories,
        judges,
        value_of=lambda c, g: lut.get((c, g), 0.0),
        err_of=None,
        palette=palette,
        hatches=hatches,
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(category.replace("_", " ").title())
    ax.set_xlim(0, min(1.0, (max(lut.values()) if lut else 1.0) * 1.05))
    _group_legend(fig, judges, hatches, title="Judge Model")

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return frac


def plot_dist_by_category_by_judge(
    df,
    judge_col,
    distribution_col="relevance_distribution",
    category="patient_disp_name",
    judge_order=None,
    valid_categories=None,
    ncols=5,
    sharey=True,
    save_as=None,
):
    """Per-distribution-value subplots of mean proportion, one bar per judge (hatch), colored by patient type."""
    judges = judge_order or list(df[judge_col].dropna().unique())
    rows = []
    dist_values = set()

    for _, row in df.iterrows():
        dist = _parse_distribution(row[distribution_col])
        if dist is None:
            continue
        dist_values.update(dist.keys())
        total = sum(dist.values()) or 1
        for value, count in dist.items():
            rows.append(
                {
                    category: row[category],
                    judge_col: row[judge_col],
                    "dist_value": value,
                    "proportion": count / total,
                }
            )

    dist_values = sorted(dist_values)
    if valid_categories is not None:
        dist_values = [v for v in dist_values if v in valid_categories]
    if not rows or not dist_values:
        return

    long_df = pd.DataFrame(rows)
    summary = long_df.groupby([category, judge_col, "dist_value"], as_index=False).agg(
        mean=("proportion", "mean"),
        sem=("proportion", lambda x: x.std() / len(x) ** 0.5),
    )

    categories = list(reversed(_sort_patient_types(long_df[category].unique().tolist())))
    palette = {
        c: PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    }
    hatches = _hatch_for_groups(judges)
    mean_lut = {
        (r[category], r[judge_col], r["dist_value"]): r["mean"]
        for _, r in summary.iterrows()
    }
    sem_lut = {
        (r[category], r[judge_col], r["dist_value"]): r["sem"]
        for _, r in summary.iterrows()
    }

    n_plots = len(dist_values)
    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=sharey
    )
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, value in zip(axes, dist_values):
        _grouped_barh(
            ax,
            categories,
            judges,
            value_of=lambda c, g, v=value: mean_lut.get((c, g, v), 0.0),
            err_of=lambda c, g, v=value: sem_lut.get((c, g, v), 0.0),
            palette=palette,
            hatches=hatches,
        )
        ax.set_title(value)
        ax.set_xlabel("Mean Proportion")
        if ax is axes[0]:
            ax.set_ylabel("Patient Type")

    for ax in axes[n_plots:]:
        ax.axis("off")

    _group_legend(fig, judges, hatches, title="Judge Model")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return summary


def plot_hexaco_reconstructed_by_judge(
    df,
    judge_col,
    category="patient_disp_name",
    axis_cols=None,
    judge_order=None,
    ncols=4,
    save_as=None,
):
    """Grid of per-patient-type subplots: mean reconstructed HEXACO per axis, one bar per judge (hatch)."""
    if axis_cols is None:
        axis_cols = [f"personality_reconstructed_{a}" for a in ALL_AXES]

    judges = judge_order or list(df[judge_col].dropna().unique())
    axis_labels = [col.rsplit("_", 1)[-1] for col in axis_cols]

    work = df[[category, judge_col, *axis_cols]].copy()
    for col in axis_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    summary = work.groupby([category, judge_col])[axis_cols].mean()

    categories = _sort_patient_types(
        work[category].dropna().unique().tolist()
    )
    palette = {
        c: PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    }
    hatches = _hatch_for_groups(judges)

    x = np.arange(len(axis_labels))
    width = 0.8 / len(judges)

    nrows = math.ceil(len(categories) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharey=True
    )
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    for ax, cat in zip(axes, categories):
        color = palette[cat]
        for ji, judge in enumerate(judges):
            if (cat, judge) not in summary.index:
                continue
            values = summary.loc[(cat, judge), axis_cols].values.astype(float)
            offset = (ji - (len(judges) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width,
                color=color,
                hatch=hatches[judge],
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_title(cat)
        ax.set_xticks(x)
        ax.set_xticklabels(axis_labels)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(categories):]:
        ax.set_visible(False)
    for i in range(nrows):
        axes[i * ncols].set_ylabel("Mean Reconstructed Value")

    _group_legend(fig, judges, hatches, title="Judge Model")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return summary


def plot_confidence_stacked_bar(
    df,
    category="patient_type",
    confidence_col="confidence",
    levels=None,
    title=None,
    x_label="Proportion",
    figure_size=(8, 4),
    save_as=None,
):
    if levels is None:
        levels = sorted(df[confidence_col].dropna().unique())

    categories = list(reversed(_sort_patient_types(df[category].dropna().unique().tolist())))

    pivot = (
        df.groupby([category, confidence_col])
        .size()
        .unstack(confidence_col, fill_value=0)
        .reindex(categories)
    )
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    colors = sns.color_palette("Blues", len(levels))

    fig, ax = plt.subplots(figsize=figure_size)
    left = np.zeros(len(categories))
    for level, color in zip(levels, colors):
        if level not in pivot.columns:
            continue
        vals = pivot[level].values
        ax.barh(categories, vals, left=left, color=color, label=str(int(level)), edgecolor="white", linewidth=0.4)
        left += vals

    ax.set_xlabel(x_label)
    ax.set_ylabel(category.replace("_", " ").title())
    ax.set_xlim(0, 1)
    ax.set_title(title or f"Confidence Distribution by {category.replace('_', ' ').title()}")
    ax.legend(title="Confidence", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return pivot


def _sdt_metrics(truth, said_real, evidence):
    from scipy.stats import norm
    from sklearn.metrics import roc_auc_score

    real, sim = said_real[truth], said_real[~truth]
    n_real, n_sim = len(real), len(sim)
    hit, fa = real.mean(), sim.mean()
    balanced_accuracy = 0.5 * (hit + (1 - fa))
    hit_c = (real.sum() + 0.5) / (n_real + 1)
    fa_c = (sim.sum() + 0.5) / (n_sim + 1)
    dprime = norm.ppf(hit_c) - norm.ppf(fa_c)
    auc = roc_auc_score(truth, evidence) if 0 < truth.sum() < len(truth) else np.nan
    return {
        "n": len(truth), "n_real": n_real, "n_sim": n_sim,
        "hit_rate": hit, "false_alarm_rate": fa,
        "balanced_accuracy": balanced_accuracy, "dprime": dprime, "auc": auc,
    }


def _bootstrap_cis(truth, said_real, evidence, metrics, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    real_idx = np.flatnonzero(truth)
    sim_idx = np.flatnonzero(~truth)
    samples = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(real_idx, len(real_idx), replace=True),
            rng.choice(sim_idx, len(sim_idx), replace=True),
        ])
        values = _sdt_metrics(truth[idx], said_real[idx], evidence[idx])
        for m in metrics:
            if np.isfinite(values[m]):
                samples[m].append(values[m])
    return {m: np.percentile(samples[m], [2.5, 97.5]) for m in metrics}


DISCRIMINATION_SPECS = {
    "balanced_accuracy": {"chance": 0.5, "label": "Balanced accuracy", "lim": (0, 1)},
    "auc": {"chance": 0.5, "label": "AUC (real vs. simulated)", "lim": (0, 1)},
    "dprime": {"chance": 0.0, "label": "Sensitivity $d'$", "lim": None},
}


def rater_discrimination_table(
    df,
    user_col="user",
    truth_col="is_real",
    decision_col="classification",
    confidence_col="confidence",
    real_label="real",
    n_boot=2000,
):
    truth = df[truth_col].to_numpy(dtype=bool)
    said_real = (df[decision_col] == real_label).to_numpy()
    conf = df[confidence_col].to_numpy(dtype=float)
    evidence = np.where(said_real, conf, -conf)
    df = df.assign(_truth=truth, _said_real=said_real, _evidence=evidence)

    metrics = list(DISCRIMINATION_SPECS)
    rows = []
    for rater, group in list(df.groupby(user_col)) + [("Pooled", df)]:
        t, s, e = group["_truth"].to_numpy(), group["_said_real"].to_numpy(), group["_evidence"].to_numpy()
        row = {"rater": rater, **_sdt_metrics(t, s, e)}
        cis = _bootstrap_cis(t, s, e, metrics, n_boot=n_boot)
        for metric in metrics:
            row[f"{metric}_lo"], row[f"{metric}_hi"] = cis[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_rater_discrimination(
    df,
    metric="auc",
    user_col="user",
    truth_col="is_real",
    decision_col="classification",
    confidence_col="confidence",
    real_label="real",
    n_boot=2000,
    figure_size=(6, 4),
    save_as=None,
):
    if metric not in DISCRIMINATION_SPECS:
        raise ValueError(f"metric must be one of {list(DISCRIMINATION_SPECS)}, got {metric!r}")

    table = rater_discrimination_table(
        df, user_col, truth_col, decision_col, confidence_col, real_label, n_boot
    )
    spec = DISCRIMINATION_SPECS[metric]
    chance = spec["chance"]

    raters = table[table["rater"] != "Pooled"].sort_values(metric)
    pooled = table[table["rater"] == "Pooled"]
    ordered = pd.concat([pooled, raters], ignore_index=True)
    y = np.arange(len(ordered))

    def _color(lo, hi):
        if lo > chance:
            return "#1f4e79"
        if hi < chance:
            return "#a11c1c"
        return "#999999"

    fig, ax = plt.subplots(figsize=figure_size)
    ax.axvline(chance, color="black", linestyle="--", linewidth=1, zorder=0)
    for yi, (_, r) in zip(y, ordered.iterrows()):
        est, lo, hi = r[metric], r[f"{metric}_lo"], r[f"{metric}_hi"]
        color = _color(lo, hi)
        ax.errorbar(
            est, yi, xerr=[[est - lo], [hi - est]], fmt="o", color=color,
            ecolor=color, elinewidth=1.8, capsize=3, markersize=7, zorder=2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(ordered["rater"])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel(spec["label"])
    if spec["lim"]:
        ax.set_xlim(*spec["lim"])
    ax.text(chance, len(ordered) - 0.4, "chance", ha="center", va="bottom", fontsize=9, color="black")
    ax.invert_yaxis()

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return table


def _realism_frame(df, categories, decision_col, real_label, factor):
    data = df[df[factor].isin(categories)].copy()
    data["_y"] = (data[decision_col] == real_label).astype(int)
    return data


def realism_glmm_table(
    df,
    categories,
    reference,
    decision_col="classification",
    real_label="real",
    factor="patient_type",
    group_cols=("user", "conv_id"),
):
    from scipy.special import expit
    from scipy.stats import norm
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    data = _realism_frame(df, categories, decision_col, real_label, factor)
    ordered = [reference] + [c for c in categories if c != reference]
    data["_f"] = pd.Categorical(data[factor], categories=ordered)
    vc = {group: f"0+C({group})" for group in group_cols}
    res = BinomialBayesMixedGLM.from_formula("_y ~ _f", vc, data).fit_vb()

    pass_rate = data.groupby("_f", observed=True)["_y"].mean()
    fe_mean = dict(zip(res.model.exog_names, res.fe_mean))
    fe_sd = dict(zip(res.model.exog_names, res.fe_sd))

    def fitted(mu=0.0, sd=0.0):
        """Model pass rate for a level, at the mean of both random effects."""
        logit, error = fe_mean["Intercept"] + mu, np.hypot(fe_sd["Intercept"], sd)
        lo, hi = logit - 1.96 * error, logit + 1.96 * error
        return {"fit_rate": expit(logit), "fit_lo": expit(lo), "fit_hi": expit(hi)}

    rows = [{
        "level": reference, "pass_rate": pass_rate[reference], "odds_ratio": 1.0,
        "or_lo": np.nan, "or_hi": np.nan, "z": np.nan, "p_value": np.nan, **fitted(),
    }]
    for name, mu, sd in zip(res.model.exog_names, res.fe_mean, res.fe_sd):
        if not name.startswith("_f[T."):
            continue
        level = name[len("_f[T."):-1]
        z = mu / sd
        rows.append({
            "level": level, "pass_rate": pass_rate[level], "odds_ratio": np.exp(mu),
            "or_lo": np.exp(mu - 1.96 * sd), "or_hi": np.exp(mu + 1.96 * sd),
            "z": z, "p_value": 2 * norm.sf(abs(z)), **fitted(mu, sd),
        })
    variance = {name: float(np.exp(mu)) for name, mu in zip(res.model.vcp_names, res.vcp_mean)}
    table = pd.DataFrame(rows)
    table.attrs["random_effect_sd"] = variance
    return table


def plot_realism_pass_rate(
    df,
    categories,
    reference,
    decision_col="classification",
    real_label="real",
    factor="patient_type",
    group_cols=("user", "conv_id"),
    kind="point",
    x_label="Fraction judged 'real'",
    figure_size=(6, 4),
    save_as=None,
):
    """Pass rate per arm as the GLMM fits it, with the model's own 95% intervals.

    Estimate and interval come from the same fit as the contrasts in
    ``realism_glmm_table``, so the figure and the reported significances agree.
    """
    if kind not in ("point", "bar"):
        raise ValueError(f"kind must be 'point' or 'bar', got {kind!r}")

    table = realism_glmm_table(
        df, categories, reference, decision_col, real_label, factor, group_cols
    )
    order = list(reversed(_sort_patient_types(categories)))
    table = table.set_index("level").reindex(order)
    colors = [PATIENT_TYPE_PALETTE.get(level, "gray") for level in order]
    rates = table["fit_rate"].to_numpy()
    err = np.vstack([rates - table["fit_lo"], table["fit_hi"] - rates])

    fig, ax = plt.subplots(figsize=figure_size)
    if kind == "bar":
        ax.barh(order, rates, xerr=err, capsize=4, color=colors, ecolor="#404040", zorder=2)
    else:
        for yi, level in enumerate(order):
            ax.errorbar(
                rates[yi], yi, xerr=err[:, yi : yi + 1], fmt="o", color=colors[yi],
                ecolor=colors[yi], elinewidth=1.8, capsize=3, markersize=7,
            )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)

    ax.set_xlabel(x_label)
    ax.set_xlim(0, 1)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return table.reset_index()


def realism_flag_rate_table(
    df,
    categories,
    reasons,
    decision_col="classification",
    real_label="real",
    reason_col="reasons",
    factor="patient_type",
):
    """Pass rate and per-reason citation rates for each level of ``factor``."""
    data = _realism_frame(df, categories, decision_col, real_label, factor)
    rows = []
    for level, group in data.groupby(factor, observed=True):
        rows.append({
            "level": level,
            "n_columns": len(group),
            "pass_rate": group["_y"].mean(),
            **{r: group[reason_col].apply(lambda rs, r=r: r in rs).mean() for r in reasons},
        })
    return pd.DataFrame(rows).set_index("level").reindex(categories)


_LABEL_CLEAR = 1.0
_LEADER_WIDTH = 6.0


def _overlap_area(a, b):
    return max(min(a.x1, b.x1) - max(a.x0, b.x0), 0) * max(min(a.y1, b.y1) - max(a.y0, b.y0), 0)


def _segment_box_length(start, end, box):
    """Length of the segment running inside the box, scored like an overlap."""
    delta = end - start
    lo, hi = 0.0, 1.0
    for axis, (edge_lo, edge_hi) in enumerate(((box.x0, box.x1), (box.y0, box.y1))):
        if delta[axis] == 0:
            if not edge_lo <= start[axis] <= edge_hi:
                return 0.0
            continue
        cuts = sorted(((edge - start[axis]) / delta[axis] for edge in (edge_lo, edge_hi)))
        lo, hi = max(lo, cuts[0]), min(hi, cuts[1])
        if lo >= hi:
            return 0.0
    return (hi - lo) * np.hypot(*delta)


def _leader(box, center, radius, minimum):
    """Segment from the box corner facing the marker to the marker's edge, if worth drawing."""
    anchor = np.clip(center, [box.x0, box.y0], [box.x1, box.y1])
    direction = center - anchor
    reach = np.hypot(*direction)
    if reach - radius < minimum:
        return None
    return anchor, center - direction / reach * radius


def _label_candidates(rings, shifts):
    """Straight above first, then its mirror below, then sliding left, then a step further out."""
    for ring in range(rings):
        for shift in range(shifts):
            for side in (1, -1):
                yield ring, shift, side


def _label_origin_once(ax):
    """Where both axes start on the same value, print that value once, on the x axis."""
    x_origin, y_origin = (
        min(t for t in ticks if t >= lim[0])
        for ticks, lim in ((ax.get_xticks(), ax.get_xlim()), (ax.get_yticks(), ax.get_ylim()))
    )
    if x_origin == y_origin:
        formatter = ax.yaxis.get_major_formatter()
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos: "" if value == y_origin else formatter(value, pos))
        )


def _place_point_labels(
    ax, xs, ys, labels, marker_sizes, avoid=(),
    gap=4.0, step=16.0, rings=4, shifts=14, pad=2.5, fontsize=9.5,
):
    """Label markers centred above them, dropping below or sliding left when crowded.

    The first candidate that clears the markers, the labels and leaders already
    placed and the axes edge wins, and a label that ends up off its marker gets
    a connector back to it.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_data = ax.transData.inverted().transform
    centers = ax.transData.transform(np.column_stack([xs, ys]))
    radii = np.sqrt(np.asarray(marker_sizes, dtype=float)) / 2 * fig.dpi / 72
    axes_box = ax.get_window_extent(renderer)
    markers = [
        Bbox.from_bounds(cx - r, cy - r, 2 * r, 2 * r) for (cx, cy), r in zip(centers, radii)
    ]
    blockers = [artist.get_window_extent(renderer) for artist in avoid] + markers
    leaders = []

    for marker, center, radius, label in zip(markers, centers, radii, labels):
        text = ax.text(0, 0, label, ha="center", va="center", fontsize=fontsize, zorder=5)
        extent = text.get_window_extent(renderer)
        half = np.array([extent.width, extent.height]) / 2 + pad
        crossable = [box for box in blockers if box is not marker]
        best = None
        for ring, shift, side in _label_candidates(rings, shifts):
            reach = np.array([-shift * step, side * (radius + gap + ring * step + half[1])])
            box = Bbox.from_bounds(*(center + reach - half), *(2 * half))
            leader = _leader(box, center, radius, step / 2)
            cost = sum(_overlap_area(box, other) for other in blockers)
            cost += 4 * (box.width * box.height - _overlap_area(box, axes_box))
            cost += _LEADER_WIDTH * sum(_segment_box_length(*drawn, box) for drawn in leaders)
            if leader:
                cost += _LEADER_WIDTH * sum(
                    _segment_box_length(*leader, other) for other in crossable
                )
            if best is None or cost < best[0]:
                best = (cost, box, leader)
            if cost < _LABEL_CLEAR:
                break

        _, box, leader = best
        text.set_position(to_data(box.get_points().mean(axis=0)))
        blockers.append(box)
        if leader:
            leaders.append(leader)
            ax.plot(*to_data(np.array(leader)).T, color="0.5", linewidth=0.7, zorder=2)


_GAUGE_TRACK_ALPHA = 0.22
_GAUGE_RING_RATIO = 0.36
_GAUGE_DOT_RATIO = 0.17


def _gauge_patches(center, radius, fraction, color):
    """A dial reading ``fraction`` of a full turn: filled arc over a faint track."""
    ring = radius * _GAUGE_RING_RATIO
    return [
        Circle(center, radius, facecolor="white", edgecolor="none", zorder=3),
        Wedge(center, radius, 0, 360, width=ring, facecolor=color,
              alpha=_GAUGE_TRACK_ALPHA, edgecolor="none", zorder=3.1),
        Wedge(center, radius, 90 - 360 * fraction, 90, width=ring, facecolor=color,
              edgecolor="none", zorder=3.2),
        Circle(center, radius * _GAUGE_DOT_RATIO, facecolor=color, edgecolor="none", zorder=3.3),
    ]


def _draw_gauge(ax, x, y, radius, fraction, color):
    """Draw a gauge of fixed physical size, centred on the data point ``(x, y)``."""
    transform = ax.figure.dpi_scale_trans + ScaledTranslation(x, y, ax.transData)
    for patch in _gauge_patches((0, 0), radius, fraction, color):
        patch.set_transform(transform)
        ax.add_patch(patch)


class _GaugeLegendHandler(HandlerBase):
    """Draw the same dial inside a legend entry, sized to the handle box."""

    def __init__(self, fraction, color):
        super().__init__()
        self.fraction, self.color = fraction, color

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        center = (width / 2 - xdescent, height / 2 - ydescent)
        patches = _gauge_patches(center, min(width, height) / 2, self.fraction, self.color)
        for patch in patches:
            patch.set_transform(trans)
        return patches


def plot_realism_reason_scatter(
    df,
    categories,
    x_reason,
    y_reason,
    x_label,
    y_label,
    gauge_label,
    gauge_ticks=(0.25, 0.75),
    gauge_radius=0.115,
    decision_col="classification",
    real_label="real",
    reason_col="reasons",
    factor="patient_type",
    lim=(0.0, 1.08),
    figure_size=(4.4, 3.7),
    save_as=None,
):
    """Two flag-reason axes with the pass rate as a dial inside each marker.

    Both axes show ``1 - `` a citation rate, so the arm that gives itself away
    least on both counts sits top right, and an arm that trades one tell for the
    other lands off the diagonal. Every marker is the same size: the pass rate is
    the filled arc, read against the full turn of the track behind it.
    """
    table = realism_flag_rate_table(
        df, categories, [x_reason, y_reason], decision_col, real_label, reason_col, factor
    )
    frame = table.reset_index().rename(columns={"level": factor})
    frame = frame.assign(_x=1 - frame[x_reason], _y=1 - frame[y_reason])

    fig, ax = plt.subplots(figsize=figure_size)
    for level in _sort_patient_types(categories):
        point = frame[frame[factor] == level].iloc[0]
        _draw_gauge(ax, point["_x"], point["_y"], gauge_radius, point["pass_rate"],
                    PATIENT_TYPE_PALETTE.get(level, "gray"))

    handles = [ax.plot([], [], linestyle="none")[0] for _ in gauge_ticks]
    legend = ax.legend(
        handles, [f"{tick:.0%}" for tick in gauge_ticks], title=gauge_label,
        handler_map={h: _GaugeLegendHandler(tick, "0.55") for h, tick in zip(handles, gauge_ticks)},
        loc="lower right", handleheight=1.8, handlelength=1.8, handletextpad=0.6,
        labelspacing=0.5, borderpad=0.5, borderaxespad=0.2, alignment="left",
        frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.95,
    )
    legend.get_frame().set_linewidth(0.6)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    marker_sizes = np.full(len(frame), (2 * gauge_radius * 72) ** 2)
    _place_point_labels(ax, frame["_x"], frame["_y"], frame[factor], marker_sizes, avoid=[legend])
    _label_origin_once(ax)
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return frame.drop(columns=["_x", "_y"])


def paired_contrast_table(
    df,
    factor,
    reference,
    contrasts,
    value_col,
    target_value,
    user_col="user",
    test="ttest",
):
    """Per-rater pass-rate contrasts against a reference level.

    For each level in ``contrasts``, pairs raters who judged both that level and
    ``reference`` on ``value_col == target_value`` and runs a paired t-test (or
    Wilcoxon signed-rank) on the per-rater pass-rate difference.
    """
    if test not in ("ttest", "wilcoxon"):
        raise ValueError(f"test must be 'ttest' or 'wilcoxon', got {test!r}")
    from scipy import stats

    pass_rate = (
        df.assign(_pass=(df[value_col] == target_value).astype(float))
        .groupby([user_col, factor])["_pass"]
        .mean()
        .unstack(factor)
    )

    rows = []
    for level in contrasts:
        paired = pass_rate[[reference, level]].dropna()
        diff = paired[level] - paired[reference]
        if test == "ttest":
            statistic, p_value = stats.ttest_rel(paired[level], paired[reference])
            se = diff.std(ddof=1) / len(diff) ** 0.5
            t_crit = stats.t.ppf(0.975, len(diff) - 1)
            ci_lo, ci_hi = diff.mean() - t_crit * se, diff.mean() + t_crit * se
        else:
            statistic, p_value = stats.wilcoxon(paired[level], paired[reference])
            ci_lo, ci_hi = np.nan, np.nan
        rows.append({
            "level": level, "n_raters": len(paired), "mean_diff": diff.mean(),
            "ci_lo": ci_lo, "ci_hi": ci_hi, "statistic": statistic, "p_value": p_value,
        })
    return pd.DataFrame(rows)


def paired_metric_table(df, factor, reference, contrasts, metrics, pair_col):
    """Per-metric difference of each level to ``reference`` over shared ``pair_col`` units.

    One row per (level, metric): both means, the mean paired difference and a Wilcoxon
    signed-rank p-value over the units (e.g. conversations) both levels cover.
    """
    from scipy import stats

    wide = df.set_index([factor, pair_col])[metrics]
    rows = []
    for level in contrasts:
        for metric in metrics:
            paired = pd.concat(
                {side: wide.loc[name, metric] for side, name in (("level", level), ("reference", reference))},
                axis=1,
            ).dropna()
            rows.append({
                "level": level, "metric": metric, "n": len(paired),
                "reference_mean": paired["reference"].mean(), "level_mean": paired["level"].mean(),
                "mean_diff": (paired["level"] - paired["reference"]).mean(),
                "p_value": stats.wilcoxon(paired["level"], paired["reference"]).pvalue,
            })
    return pd.DataFrame(rows)


def delta_to_reference_table(df, factor, reference, sim_real_metrics, paired_metrics, pair_col):
    """Mean and SEM of each level's per-conversation difference to the recorded encounter.

    ``sim_real_metrics`` store both sides on every row as ``{metric}_sim``/``{metric}_real``;
    ``paired_metrics`` exist once per row and are differenced against the ``reference``
    level's row of the same ``pair_col`` unit.
    """
    ref = df[df[factor] == reference].set_index(pair_col)
    rest = df[df[factor] != reference]
    deltas = pd.DataFrame(
        {
            **{m: rest[f"{m}_sim"] - rest[f"{m}_real"] for m in sim_real_metrics},
            **{m: rest[m] - rest[pair_col].map(ref[m]) for m in paired_metrics},
            factor: rest[factor],
        }
    )
    grouped = deltas.groupby(factor)
    return pd.concat({"mean": grouped.mean(), "sem": grouped.sem()}, axis=1).swaplevel(axis=1).sort_index(axis=1)

def matched_pair_table(df, factor, reference, contrasts, value_col, target_value, pair_col):
    """Exact McNemar contrasts against a reference level, matched within ``pair_col``.

    Each row of ``contrasts`` must share a ``pair_col`` group with exactly one
    ``reference`` row (e.g. a screen showing both arms to the same rater), so the
    comparison holds rater and conversation fixed.
    """
    from scipy import stats

    hit = df[value_col] == target_value
    ref_hit = df[df[factor] == reference].set_index(pair_col)[value_col].eq(target_value)
    if not ref_hit.index.is_unique:
        raise ValueError(f"{reference!r} appears more than once in some {pair_col!r} group")

    rows = []
    for level in contrasts:
        sub = df[df[factor] == level]
        paired = pd.DataFrame({"level": hit[sub.index].values, "ref": sub[pair_col].map(ref_hit).values})
        paired = paired.dropna()
        ref_only = int((paired["ref"] & ~paired["level"]).sum())
        level_only = int((~paired["ref"] & paired["level"]).sum())
        discordant = ref_only + level_only
        rows.append({
            "level": level, "n_pairs": len(paired),
            f"{reference}_only": ref_only, "level_only": level_only,
            "concordant": len(paired) - discordant,
            "p_value": stats.binomtest(ref_only, discordant, 0.5).pvalue if discordant else np.nan,
        })
    return pd.DataFrame(rows)


def cohen_kappa_matrix(df, user_col, value_col, conv_id_col):
    pivot = df.groupby([conv_id_col, user_col])[value_col].first().unstack(user_col)
    users = pivot.columns.tolist()

    def _cohen_kappa(a, b):
        mask = a.notna() & b.notna()
        a, b = a[mask], b[mask]
        if len(a) < 2:
            return np.nan
        cats = set(a) | set(b)
        po = (a == b).mean()
        pe = sum((a == c).mean() * (b == c).mean() for c in cats)
        return (po - pe) / (1 - pe) if pe < 1 else np.nan

    return pd.DataFrame(
        {u2: {u1: _cohen_kappa(pivot[u1], pivot[u2]) for u1 in users} for u2 in users}
    )


def plot_kappa_heatmap(
    df,
    user_col="user",
    value_col="classification",
    conv_id_col="conv_id",
    figure_size=(5, 4),
    save_as=None,
):
    kappa_matrix = cohen_kappa_matrix(df, user_col, value_col, conv_id_col)
    fig, ax = plt.subplots(figsize=figure_size)
    mask = np.tril(np.ones_like(kappa_matrix, dtype=bool), k=-1)
    ax.set_facecolor("lightgrey")
    sns.heatmap(
        kappa_matrix.astype(float), annot=True, fmt=".2f", cmap="viridis",
        vmin=-1, vmax=1, ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8},
        mask=mask,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return kappa_matrix


def plot_majority_agreement(
    df,
    user_col="user",
    value_col="classification",
    category="patient_type",
    figure_size=(5, 4),
    save_as=None,
):
    def _majority_agreement(labels):
        return (labels == labels.mode().iloc[0]).mean()

    per_cat = df.groupby(category)[value_col].apply(_majority_agreement)
    categories = list(reversed(_sort_patient_types(per_cat.index.tolist())))
    per_cat = per_cat.reindex(categories)

    fig, ax = plt.subplots(figsize=figure_size)
    bar_colors = [PATIENT_TYPE_PALETTE.get(c, "gray") for c in categories]
    ax.barh(categories, per_cat.values, color=bar_colors)
    ax.set_xlabel("Fraction of labels matching majority")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return per_cat


def realism_agreement(df, item_col, value_col, user_col="user", categories=None):
    """Percent agreement and Gwet's AC1 over items rated by 2+ raters.

    ``item_col`` identifies the thing being judged (e.g. a conv_id/arm pair);
    items with fewer than two raters are excluded, since agreement is undefined
    for them.
    """
    pivot = df.groupby([item_col, user_col])[value_col].first().unstack(user_col)
    if categories is None:
        categories = sorted(pd.unique(df[value_col].dropna()))
    if len(categories) < 2:
        raise ValueError(f"Need at least 2 categories, got {categories}")

    n_raters = pivot.notna().sum(axis=1)
    multi = pivot[n_raters >= 2]
    if multi.empty:
        raise ValueError(f"No items in '{item_col}' were rated by 2+ raters")

    def _pct_agree(row):
        vals = row.dropna()
        return np.mean([a == b for a, b in combinations(vals, 2)])

    percent_agreement = multi.apply(_pct_agree, axis=1).mean()

    n = multi.notna().sum(axis=1)
    counts = pd.DataFrame({cat: (multi == cat).sum(axis=1) for cat in categories})
    po = ((counts * (counts - 1)).sum(axis=1) / (n * (n - 1))).mean()
    pi = counts.div(n, axis=0).mean()
    pe = sum(p * (1 - p) for p in pi) / (len(categories) - 1)
    ac1 = (po - pe) / (1 - pe)

    return {
        "n_items": len(multi), "percent_agreement": percent_agreement,
        "gwet_ac1": ac1, "po": po, "pe": pe,
    }


AGREEMENT_METRICS = {"gwet_ac1": "Gwet's AC1", "percent_agreement": "Percent agreement"}


def realism_agreement_table(df, item_col, value_col, user_col="user", factor="patient_type"):
    """``realism_agreement`` overall and within each level of ``factor``.

    The verdict categories are taken from the whole frame, so a level on which the
    panel never disagrees still scores against the same two-category scale.
    """
    categories = sorted(pd.unique(df[value_col].dropna()))
    rows = [
        {"level": "Overall", **realism_agreement(df, item_col, value_col, user_col, categories)}
    ]
    for level in _sort_patient_types(df[factor].dropna().unique().tolist()):
        rows.append({
            "level": level,
            **realism_agreement(df[df[factor] == level], item_col, value_col, user_col, categories),
        })
    return pd.DataFrame(rows).set_index("level")


def ordinal_krippendorff_alpha(units, levels):
    """Krippendorff's alpha with the ordinal distance metric.

    ``units`` is a list of rating lists, one per rated unit, each with 2+ ratings;
    ``levels`` is the ordered rating scale.
    """
    index = {v: i for i, v in enumerate(levels)}
    coincidences = np.zeros((len(levels), len(levels)))
    for ratings in units:
        for a, b in permutations(ratings, 2):
            coincidences[index[a], index[b]] += 1 / (len(ratings) - 1)
    n_c = coincidences.sum(axis=1)
    delta = np.array([
        [(n_c[min(c, k):max(c, k) + 1].sum() - (n_c[c] + n_c[k]) / 2) ** 2 for k in range(len(levels))]
        for c in range(len(levels))
    ])
    observed = (coincidences * delta).sum()
    expected = (np.outer(n_c, n_c) * delta).sum() / (n_c.sum() - 1)
    return 1 - observed / expected


def ordinal_rating_agreement(df, unit_cols, rater_col, rating_cols, levels, n_boot, seed):
    """Inter-rater agreement on ordinal ratings, with a cluster-bootstrap CI for alpha.

    A unit is one rating column of one ``unit_cols`` group (e.g. one HEXACO axis of
    one rated conversation) with 2+ non-missing ratings. The bootstrap resamples
    whole ``unit_cols`` groups, since all rating columns of a group come from the
    same raters.
    """
    long = df.melt(
        id_vars=[*unit_cols, rater_col], value_vars=rating_cols, var_name="item", value_name="rating"
    ).dropna(subset=["rating"])
    units = long.groupby([*unit_cols, "item"])["rating"].apply(lambda s: s.astype(int).tolist())
    units = units[units.apply(len) >= 2]
    clusters = [group.tolist() for _, group in units.groupby(level=list(unit_cols))]
    rng = np.random.default_rng(seed)
    draws = [
        ordinal_krippendorff_alpha(
            [u for i in rng.integers(0, len(clusters), len(clusters)) for u in clusters[i]], levels
        )
        for _ in range(n_boot)
    ]
    gaps = np.array([abs(a - b) for ratings in units for a, b in combinations(ratings, 2)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_clusters": len(clusters),
        "n_units": len(units),
        "n_pairs": len(gaps),
        "alpha": ordinal_krippendorff_alpha(units.tolist(), levels),
        "alpha_ci_low": lo,
        "alpha_ci_high": hi,
        "exact": (gaps == 0).mean(),
        "within_one": (gaps <= 1).mean(),
        "opposite_ends": (gaps == levels[-1] - levels[0]).mean(),
    }

def plot_realism_agreement(
    df,
    item_col,
    value_col,
    user_col="user",
    factor="patient_type",
    metrics=AGREEMENT_METRICS,
    x_label="Inter-rater agreement",
    figure_size=(6, 4),
    save_as=None,
):
    table = realism_agreement_table(df, item_col, value_col, user_col, factor)
    order = list(reversed(table.index.tolist()))
    palette = {level: PATIENT_TYPE_PALETTE.get(level, "#404040") for level in order}
    labels = list(metrics.values())
    column_of = {label: column for column, label in metrics.items()}
    hatches = _hatch_for_groups(labels)

    fig, ax = plt.subplots(figsize=figure_size)
    _grouped_barh(
        ax, order, labels, lambda c, m: table.loc[c, column_of[m]], None, palette, hatches
    )
    ax.axvline(0, color="0.15", linewidth=0.7)
    ax.set_xlabel(x_label)
    ax.set_xlim(min(0, table[list(metrics)].min().min() * 1.1), 1)
    ax.grid(axis="x", alpha=0.3)
    _group_legend(fig, labels, hatches, None, ncol=len(labels))

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return table.reset_index()


def _format_p_value(p):
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.1e}"


def plot_contrast_forest(
    table,
    reference,
    effect_col="odds_ratio",
    lo_col="or_lo",
    hi_col="or_hi",
    p_col="p_value",
    x_label="Odds of being judged realistic",
    figure_size=(6, 3.6),
    save_as=None,
):
    """Forest plot of a contrast table against ``reference``, on a log effect axis.

    Reads the frame ``realism_glmm_table`` returns: the reference row carries no
    interval and becomes the dashed line at 1.
    """
    contrasts = table[table["level"] != reference].set_index("level")
    order = list(reversed(_sort_patient_types(contrasts.index.tolist())))
    contrasts = contrasts.reindex(order)
    effect = contrasts[effect_col].to_numpy()
    err = np.vstack([effect - contrasts[lo_col], contrasts[hi_col] - effect])

    fig, ax = plt.subplots(figsize=figure_size)
    ax.axvline(1, color="0.55", linestyle="--", linewidth=1, zorder=0)
    for yi, level in enumerate(order):
        color = PATIENT_TYPE_PALETTE.get(level, "gray")
        ax.errorbar(
            effect[yi], yi, xerr=err[:, yi : yi + 1], fmt="o", color=color,
            ecolor=color, elinewidth=1.8, capsize=3, markersize=7, zorder=3,
        )
        ax.annotate(
            _format_p_value(contrasts.loc[level, p_col]),
            (contrasts.loc[level, hi_col], yi), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color="0.3",
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(f"{x_label}\nrelative to {reference}")
    ax.set_xlim(contrasts[lo_col].min() * 0.6, 1.8)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return contrasts.reset_index()


def _icc_two_raters(a, b):
    n = len(a)
    ratings = np.stack([a, b], axis=1)
    grand_mean = ratings.mean()
    ss_r = 2 * np.sum((ratings.mean(axis=1) - grand_mean) ** 2)
    ss_c = n * np.sum((ratings.mean(axis=0) - grand_mean) ** 2)
    ss_e = np.sum((ratings - grand_mean) ** 2) - ss_r - ss_c
    ms_r = ss_r / (n - 1)
    ms_c = ss_c
    ms_e = ss_e / (n - 1)
    return (ms_r - ms_e) / (ms_r + ms_e + 2 / n * (ms_c - ms_e))


def plot_hexaco_rater_alignment(
    df,
    patient_type_col="patient_type",
    axis_col="hexaco_axis",
    rater_col="rater",
    patient_types=None,
    exclude_axes=None,
    title=None,
    figure_size=(7, 5),
    save_as=None,
):
    AXES = [ax for ax in ALL_AXES if ax not in (exclude_axes or [])]
    rec_cols = {ax: f"personality_reconstructed_{ax}" for ax in AXES}

    if patient_types is None:
        patient_types = _sort_patient_types(df[patient_type_col].dropna().unique().tolist())

    rows = []
    for extreme in ALL_AXES:
        for pt in patient_types:
            sub = df[(df[axis_col] == extreme) & (df[patient_type_col] == pt)]
            for ev in AXES:
                human_mean = sub[sub[rater_col] == "Human"][rec_cols[ev]].mean()
                auto_mean = sub[sub[rater_col] == "Autorater"][rec_cols[ev]].mean()
                rows.append({"patient_type": pt, "human": human_mean, "autorater": auto_mean})
    agg = pd.DataFrame(rows).dropna()

    r = np.corrcoef(agg["human"], agg["autorater"])[0, 1]
    icc = _icc_two_raters(agg["human"].values, agg["autorater"].values)

    fig, ax = plt.subplots(figsize=figure_size)
    for pt in patient_types:
        sub = agg[agg["patient_type"] == pt]
        ax.scatter(sub["human"], sub["autorater"], color=PATIENT_TYPE_PALETTE.get(pt, "gray"),
                   label=pt, alpha=0.8, edgecolors="white", linewidth=0.5, s=120)

    lo = agg[["human", "autorater"]].min().min() - 0.05
    hi = agg[["human", "autorater"]].max().max() + 0.05
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1, linestyle="--", zorder=0)

    ax.set_xlabel("Human mean reconstruction")
    ax.set_ylabel("Autorater mean reconstruction")
    # ax.set_title(title or "Human vs Autorater HEXACO Alignment")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return r, icc


def hexaco_recovery_table(df, patient_type_col, axis_col, rater_col):
    """Per configured axis: rating on that axis (target, ideal 3) and mean over the other five (leakage, ideal 1)."""
    rec_cols = {ax: f"personality_reconstructed_{ax}" for ax in ALL_AXES}
    rows = []
    for axis in ALL_AXES:
        for (pt, rater), sub in df[df[axis_col] == axis].groupby([patient_type_col, rater_col]):
            rows.append({
                "axis": axis,
                patient_type_col: pt,
                rater_col: rater,
                "target": sub[rec_cols[axis]].mean(),
                "leakage": np.mean([sub[rec_cols[o]].mean() for o in ALL_AXES if o != axis]),
            })
    return pd.DataFrame(rows)


def plot_hexaco_target_recovery(
    df,
    patient_type_col="patient_type",
    axis_col="hexaco_axis",
    rater_col="rater",
    patient_types=None,
    raters=("Human", "Autorater"),
    figure_size=(13, 4),
    save_as=None,
):
    """Configured-axis recovery (ideal 3) beside off-target leakage (ideal 1), grouped by patient type and rater."""
    table = hexaco_recovery_table(df, patient_type_col, axis_col, rater_col)
    patient_types = patient_types or _sort_patient_types(table[patient_type_col].unique().tolist())
    groups = [(pt, r) for pt in patient_types for r in raters]
    hatches = _hatch_for_groups(raters)

    x = np.arange(len(ALL_AXES))
    width = 0.8 / len(groups)
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * width

    fig, axes = plt.subplots(1, 2, figsize=figure_size, sharex=True)
    for ax, (col, ideal, label) in zip(
        axes,
        [("target", 3, "Rating on the configured axis"), ("leakage", 1, "Mean rating on the other five axes")],
    ):
        for offset, (pt, rater) in zip(offsets, groups):
            sub = table[(table[patient_type_col] == pt) & (table[rater_col] == rater)].set_index("axis")
            ax.bar(
                x + offset, sub.reindex(ALL_AXES)[col], width,
                color=PATIENT_TYPE_PALETTE.get(pt, "gray"),
                alpha=0.9 if rater == raters[0] else 0.55,
                hatch=hatches[rater], edgecolor="black", linewidth=0.6,
                label=f"{pt} ({rater})" if ax is axes[0] else None,
            )
        ax.axhline(ideal, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_AXES)
        ax.set_xlabel("Configured extreme axis")
        ax.set_ylabel(label)
        ax.set_ylim(0.9, 3.2)
        ax.grid(axis="y", alpha=0.3)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(groups),
               bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return table


def plot_hexaco_human_autorater_deviation(
    df,
    patient_type_col="patient_type",
    rater_col="rater",
    patient_types=None,
    show_sd=False,
    title=None,
    figure_size=(10, 4),
    save_as=None,
):
    AXES = ["H", "E", "X", "A", "C", "O"]
    rec_cols = {ax: f"personality_reconstructed_{ax}" for ax in AXES}

    if patient_types is None:
        patient_types = _sort_patient_types(df[patient_type_col].dropna().unique().tolist())

    x = np.arange(len(AXES))
    n = len(patient_types)
    width = 0.35
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=figure_size)

    for i, pt in enumerate(patient_types):
        sub = df[df[patient_type_col] == pt]
        human = sub[sub[rater_col] == "Human"]
        autorater = sub[sub[rater_col] == "Autorater"]

        devs = [abs(human[rec_cols[axis]].mean() - autorater[rec_cols[axis]].mean()) for axis in AXES]
        sds = [
            np.sqrt(human[rec_cols[axis]].std() ** 2 + autorater[rec_cols[axis]].std() ** 2)
            for axis in AXES
        ] if show_sd else None

        color = PATIENT_TYPE_PALETTE.get(pt, "gray")
        ax.bar(x + offsets[i], devs, width, yerr=sds, capsize=4, color=color, label=pt, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(AXES)
    ax.set_xlabel("HEXACO Axis")
    ax.set_ylabel("|Human mean − Autorater mean|")
    ax.set_title(title or "Human vs Autorater HEXACO Reconstruction: Absolute Deviation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


def plot_personality_extreme_difference(
    df,
    rater="Human",
    patient_type_col="patient_type",
    axis_col="hexaco_axis",
    rater_col="rater",
    patient_types=None,
    exclude_axes=None,
    show_sd=False,
    title=None,
    figure_size=(9, 4),
    save_as=None,
):
    AXES = [ax for ax in ["H", "E", "X", "A", "C", "O"] if ax not in (exclude_axes or [])]
    rec_cols = {ax: f"personality_reconstructed_{ax}" for ax in AXES}

    sub = df[df[rater_col] == rater]
    if patient_types is None:
        patient_types = _sort_patient_types(sub[patient_type_col].dropna().unique().tolist())

    diffs = {pt: [] for pt in patient_types}
    sds = {pt: [] for pt in patient_types}
    for ax in AXES:
        for pt in patient_types:
            pt_sub = sub[sub[patient_type_col] == pt]
            extreme_vals = pt_sub[pt_sub[axis_col] == ax][rec_cols[ax]]
            low_vals = pt_sub[pt_sub[axis_col] != ax][rec_cols[ax]]
            diffs[pt].append(extreme_vals.mean() - low_vals.mean())
            sds[pt].append(np.sqrt(extreme_vals.std() ** 2 + low_vals.std() ** 2))

    x = np.arange(len(AXES))
    n = len(patient_types)
    width = 0.35
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=figure_size)
    for i, pt in enumerate(patient_types):
        color = PATIENT_TYPE_PALETTE.get(pt, "gray")
        yerr = sds[pt] if show_sd else None
        ax.bar(x + offsets[i], diffs[pt], width, yerr=yerr, capsize=4, color=color, label=pt, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(AXES)
    ax.set_xlabel("HEXACO Axis")
    ax.set_ylabel("Extreme − Low mean reconstruction")
    # ax.set_title(title or f"Personality Extreme Difference ({rater})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


def plot_personality_recovery_ablation(
    axis_stats,
    variant_labels,
    window_labels,
    figure_size=(11, 4),
    save_as=None,
):
    """Per-axis target recovery (level 3 − level 1) for each judge prompt and turn window.

    ``axis_stats`` is the output of ``personality_ablation.axis_recovery``: one row
    per (prompt_variant, window, axis) with ``delta`` and its standard error ``se``.
    One subplot per judge prompt, one bar per turn window.
    """
    windows = list(window_labels)
    variants = list(variant_labels)
    colors = dict(zip(windows, sns.color_palette("Dark2", len(windows))))

    x = np.arange(len(ALL_AXES))
    width = 0.8 / len(windows)
    offsets = np.linspace(-(len(windows) - 1) / 2, (len(windows) - 1) / 2, len(windows))

    fig, axes = plt.subplots(1, len(variants), figsize=figure_size, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, variant in zip(axes, variants):
        for i, window in enumerate(windows):
            sub = (
                axis_stats[
                    (axis_stats["prompt_variant"] == variant)
                    & (axis_stats["window"] == window)
                ]
                .set_index("axis")
                .reindex(ALL_AXES)
            )
            ax.bar(
                x + offsets[i] * width,
                sub["delta"],
                width,
                yerr=sub["se"],
                capsize=3,
                color=colors[window],
                edgecolor="black",
                linewidth=0.6,
                label=window_labels[window] if ax is axes[0] else None,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_AXES)
        ax.set_xlabel("HEXACO Axis")
        ax.set_title(variant_labels[variant])
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Recovered level 3 − level 1")
    fig.legend(frameon=False, loc="center left", bbox_to_anchor=(0.99, 0.5))

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def _hexaco_long(df, patient_type_col, axis_col, rater_col, extreme_value, low_value, keep=()):
    """One row per (input row x rated axis), with the rating and its distance to truth.

    Ground truth is ``extreme_value`` on the row's configured axis and
    ``low_value`` on the other five; unrated axes are dropped.
    """
    rows = []
    for _, row in df.iterrows():
        for axis in ALL_AXES:
            rating = row[f"personality_reconstructed_{axis}"]
            if pd.isna(rating):
                continue
            truth = extreme_value if axis == row[axis_col] else low_value
            rows.append({
                "rater": row[rater_col],
                "patient_type": row[patient_type_col],
                "configured_axis": row[axis_col],
                "axis": axis,
                "rating": rating,
                "abs_dist": abs(rating - truth),
                **{col: row[col] for col in keep},
            })
    return pd.DataFrame(rows)


def hexaco_distance_to_truth(
    df,
    patient_type_col="patient_type",
    axis_col="hexaco_axis",
    rater_col="rater",
    extreme_value=3,
    low_value=1,
):
    """Mean absolute distance between reconstructed and ground-truth HEXACO values.

    Ground truth: the row's ``axis_col`` axis is set to ``extreme_value``, all
    other axes to ``low_value``. Returns a wide DataFrame indexed by
    (rater, patient_type) with one column per axis plus a ``mean`` column
    averaging across axes.
    """
    long = _hexaco_long(df, patient_type_col, axis_col, rater_col, extreme_value, low_value)
    wide = (
        long.groupby(["rater", "patient_type", "axis"])["abs_dist"]
        .mean()
        .unstack("axis")
        .reindex(columns=ALL_AXES)
    )
    wide["mean"] = wide.mean(axis=1)
    return wide


def personality_rater_agreement(
    human_df,
    auto_df,
    cell_cols=("conv_id", "simulator"),
    patient_type_col="patient_type",
    axis_col="hexaco_axis",
    rater_col="rater",
    extreme_value=3,
    low_value=1,
):
    """Human vs. autorater agreement on the HEXACO study, at three aggregations.

    Restricted to the cells both sides scored. Each row asks a different
    question: whether the two raters give the same individual rating, whether
    they produce the same per-axis distances (the body of the reported table),
    and whether they produce the same pooled distance per arm (the number the
    paper quotes). Agreement is not monotone in aggregation, so all three belong
    next to each other.
    """
    from scipy.stats import pearsonr

    keys = list(cell_cols) + ["patient_type", "axis"]
    sides = {
        side: _hexaco_long(
            frame, patient_type_col, axis_col, rater_col, extreme_value, low_value, cell_cols
        ).groupby(keys)[["rating", "abs_dist"]].mean()
        for side, frame in (("human", human_df), ("auto", auto_df))
    }
    merged = sides["human"].join(sides["auto"], how="inner", lsuffix="_human", rsuffix="_auto")

    levels = {
        "single rating": (merged, "rating"),
        "arm x axis distance": (merged.groupby(["patient_type", "axis"]).mean(), "abs_dist"),
        "pooled distance": (merged.groupby("patient_type").mean(), "abs_dist"),
    }
    rows = []
    for level, (frame, column) in levels.items():
        human, auto = frame[f"{column}_human"], frame[f"{column}_auto"]
        comparable = len(frame) > 2
        rows.append({
            "level": level,
            "n": len(frame),
            "pearson_r": pearsonr(human, auto)[0] if comparable else np.nan,
            "icc": _icc_two_raters(human.to_numpy(), auto.to_numpy()) if comparable else np.nan,
            "mean_abs_diff": (human - auto).abs().mean(),
            "max_abs_diff": (human - auto).abs().max(),
        })
    return pd.DataFrame(rows).set_index("level")


def hexaco_distance_latex(
    distances,
    simulators,
    simulator_macros,
    caption,
    label,
    raters=("Autorater", "Human"),
):
    """LaTeX table of ``hexaco_distance_to_truth``, one block per simulator.

    Each block lists the raters and their mean; the per-column winner between the
    two simulator means is bolded, and a final row gives the second simulator
    minus the first.
    """
    columns = list(ALL_AXES) + ["mean"]
    blocks = {}
    for simulator in simulators:
        block = distances.xs(simulator, level="patient_type")[columns].reindex(list(raters))
        blocks[simulator] = pd.concat([block, block.mean().to_frame("Mean").T])
    best = {c: min(simulators, key=lambda s: blocks[s].loc["Mean", c]) for c in columns}

    def row(rater, values, bold_of=None):
        cells = [
            rf"\textbf{{{v:.2f}}}" if bold_of and best[c] == bold_of else f"{v:.2f}"
            for c, v in zip(columns, values)
        ]
        return f"  & {rater:9} & " + " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table*}[t]",
        r"\small",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{ll" + "c" * len(columns) + "}",
        r"\toprule",
        "Patient Simulator & Rater & " + " & ".join(list(ALL_AXES) + ["Mean"]) + r" \\",
        r"\midrule",
    ]
    for i, simulator in enumerate(simulators):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multirow{{{len(raters) + 1}}}{{*}}{{{simulator_macros[simulator]}}}")
        for rater in list(raters) + ["Mean"]:
            lines.append(
                row(rater, blocks[simulator].loc[rater],
                    bold_of=simulator if rater == "Mean" else None)
            )
    lines.append(r"\midrule")
    difference = blocks[simulators[1]].loc["Mean"] - blocks[simulators[0]].loc["Mean"]
    lines.append(
        f"{simulator_macros[simulators[1]]} $-$ {simulator_macros[simulators[0]]} & Mean & "
        + " & ".join(f"${v:+.2f}$" for v in difference)
        + r" \\"
    )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def plot_countplot(
    df,
    x,
    palette=None,
    hue=None,
    title=None,
    x_label=None,
    y_label="Count",
    figure_size=(6, 4),
    save_as=None,
):
    """Styled countplot using PATIENT_TYPE_PALETTE, with optional hue grouping."""
    if palette is None:
        palette = PATIENT_TYPE_PALETTE
    hue_col = hue if hue is not None else x
    order = sorted(df[x].dropna().unique())
    hue_order = sorted(df[hue_col].dropna().unique())
    bar_palette = {v: palette.get(str(v).split("_")[0].strip(), "gray") for v in hue_order}

    fig, ax = plt.subplots(figsize=figure_size)
    sns.countplot(
        data=df, x=x, order=order, hue=hue_col, hue_order=hue_order,
        palette=bar_palette, ax=ax, legend=(hue is not None),
    )
    ax.set_title(title or f"Count by {x.replace('_', ' ').title()}")
    ax.set_xlabel(x_label or x.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


def plot_flag_reasons(
    df,
    category="patient_type",
    flag_col="flagged_turns",
    min_count=0,
    reason_order=None,
    total_df=None,
    title=None,
    x_label="Fraction of Conversations",
    figure_size=(6, 4),
    save_as=None,
):
    """Fraction of conversations flagged for each reason, per category.

    ``total_df`` supplies the denominator when ``df`` is a filtered subset (e.g.
    only the "not realistic" verdicts): pass the unfiltered frame so fractions are
    out of every conversation shown for that category, not only the flagged ones.
    """
    rows = []
    for _, row in df.iterrows():
        reasons_in_conv = set()
        for flag in (row.get(flag_col) or []):
            for reason in (flag.get("reasons") or ["unspecified"]):
                normalized = reason.lower().strip() if reason else "unspecified"
                reasons_in_conv.add("unspecified" if normalized in ("unspecified", "other") else normalized)
        for reason in reasons_in_conv:
            rows.append({category: row[category], "reason": reason})

    if not rows:
        print("No flagged turns found.")
        return

    flag_df = pd.DataFrame(rows)
    counts = flag_df.groupby([category, "reason"]).size().reset_index(name="count")

    total_per_reason = counts.groupby("reason")["count"].sum()
    if reason_order is not None:
        normalized_order = [r.lower().strip().replace(" ", "_") for r in reason_order]
        reasons_filtered = [
            r for r in reversed(normalized_order)
            if r in total_per_reason.index and total_per_reason[r] >= min_count
        ]
    else:
        reasons_filtered = total_per_reason[total_per_reason >= min_count].sort_values(ascending=True).index.tolist()

    if not reasons_filtered:
        print(f"No reasons with count >= {min_count}.")
        return

    counts = counts[counts["reason"].isin(reasons_filtered)]
    total_per_cat = (df if total_df is None else total_df).groupby(category).size()
    counts["relative"] = counts.apply(lambda r: r["count"] / total_per_cat[r[category]], axis=1)

    patient_types = _sort_patient_types(counts[category].unique().tolist())
    bar_palette = {pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray") for pt in patient_types}

    y = np.arange(len(reasons_filtered))
    n = len(patient_types)
    height = 0.8 / n

    fig, ax = plt.subplots(figsize=figure_size)
    for i, pt in enumerate(reversed(patient_types)):
        sub = counts[counts[category] == pt].set_index("reason").reindex(reasons_filtered, fill_value=0)["relative"]
        offset = (i - (n - 1) / 2) * height
        ax.barh(y + offset, sub.values, height * 0.9, color=bar_palette[pt])

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=bar_palette[pt]) for pt in patient_types]

    def _wrap(label):
        words = label.replace("_", " ").title().split()
        if len(words) <= 1:
            return words[0] if words else label
        mid = (len(words) + 1) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

    reasons_filtered = [_wrap(reason) for reason in reasons_filtered]

    ax.set_yticks(y)
    ax.set_yticklabels(reasons_filtered)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Flag Reason")
    ax.legend(
        legend_handles,
        patient_types,
        # frameon=False,
        # loc="upper center",
        bbox_to_anchor=(0.45, 0.3),
        # ncol=len(patient_types),
        borderaxespad=0,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


def plot_rater_comparison_fraction(
    df,
    category,
    human_col,
    judge_col,
    target_value,
    title=None,
    x_label="Proportion",
    figure_size=(8, 4),
    save_as=None,
):
    """Compare per-category classification fraction between human and LLM judge."""
    import matplotlib.patches as mpatches
    categories = list(reversed(_sort_patient_types(df[category].dropna().unique().tolist())))

    def frac(col):
        return df.groupby(category)[col].apply(lambda s: (s == target_value).mean()).reindex(categories)

    human_frac = frac(human_col)
    judge_frac = frac(judge_col)

    y = np.arange(len(categories))
    width = 0.35
    bar_colors = [PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray") for c in categories]

    fig, ax = plt.subplots(figsize=figure_size)
    for i, (cat, color) in enumerate(zip(categories, bar_colors)):
        ax.barh(y[i] - width / 2, human_frac[cat], width, color=color, edgecolor="black", linewidth=0.6)
        ax.barh(y[i] + width / 2, judge_frac[cat], width, color=color, hatch="///",
                edgecolor="black", linewidth=0.6, alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel(x_label)
    ax.set_xlim(0, 1)
    ax.set_title(title or f"Fraction '{target_value}' by {category}: Human vs LLM Judge")

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.6, label="Human"),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.6, hatch="///", label="LLM Judge"),
    ]
    ax.legend(handles=legend_handles, frameon=False)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


def plot_rater_comparison_scores(
    df,
    category,
    human_metrics,
    judge_metrics,
    metric_labels,
    user_col=None,
    title=None,
    score_range=(1, 5),
    figure_size=(12, 4),
    save_as=None,
):
    """Side-by-side horizontal bar charts comparing human and LLM judge metric means per category."""
    import matplotlib.patches as mpatches
    categories = list(reversed(_sort_patient_types(df[category].dropna().unique().tolist())))
    y = np.arange(len(categories))
    width = 0.35
    bar_colors = [PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray") for c in categories]

    n = len(human_metrics)
    fig, axes = plt.subplots(1, n, figsize=figure_size, sharey=True)
    axes = axes if n > 1 else [axes]

    for ax, h_col, j_col, label in zip(axes, human_metrics, judge_metrics, metric_labels):
        h_mean = df.groupby(category)[h_col].mean().reindex(categories)
        j_mean = df.groupby(category)[j_col].mean().reindex(categories)
        if user_col and user_col in df.columns:
            h_sem = df.groupby([category, user_col])[h_col].mean().groupby(category).sem().reindex(categories).fillna(0)
            j_sem = df.groupby([category, user_col])[j_col].mean().groupby(category).sem().reindex(categories).fillna(0)
        else:
            h_sem = df.groupby(category)[h_col].sem().reindex(categories).fillna(0)
            j_sem = df.groupby(category)[j_col].sem().reindex(categories).fillna(0)

        for i, (cat, color) in enumerate(zip(categories, bar_colors)):
            ax.barh(y[i] - width / 2, h_mean[cat], width, xerr=h_sem[cat], capsize=3,
                    color=color, edgecolor="black", linewidth=0.6)
            ax.barh(y[i] + width / 2, j_mean[cat], width, xerr=j_sem[cat], capsize=3,
                    color=color, hatch="///", edgecolor="black", linewidth=0.6, alpha=0.6)
        ax.set_title(label)
        ax.set_xlabel("Mean score")
        ax.set_xlim(score_range[0] - 0.5, score_range[1] + 0.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(categories)
    axes[0].set_ylabel(category.replace("_", " ").title())

    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.6, label="Human"),
        mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.6, hatch="///", label="LLM Judge"),
    ]
    axes[-1].legend(handles=legend_handles, loc="upper right", frameon=False)
    if title:
        fig.suptitle(title, y=1.06)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


def plot_combined_metric_boxplots(
    df,
    metrics,
    category="patient_type",
    metric_suffix="_diff",
    metric_display_names=None,
    figure_size=None,
    save_as=None,
):
    n = len(metrics)
    if n == 0:
        raise ValueError("metrics must contain at least one metric")

    patient_types = _sort_patient_types(df[category].dropna().unique().tolist())
    palette = {pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray") for pt in patient_types}

    plot_df = df.copy()
    plot_df[category] = pd.Categorical(plot_df[category], categories=patient_types, ordered=True)

    if figure_size is None:
        figure_size = (3 * n, 3)

    fig, axes = plt.subplots(1, n, figsize=figure_size, sharey=True)
    if n == 1:
        axes = [axes]

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        col = f"{metric}{metric_suffix}" if metric_suffix else metric
        col_title = (
            metric_display_names.get(metric, metric.replace("_", " ").title())
            if metric_display_names
            else metric.replace("_", " ").title()
        )
        sub = plot_df[[category, col]].dropna()
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=0)
        sns.boxplot(data=sub, x=col, y=category, hue=category, palette=palette, ax=ax, legend=False)
        ax.set_xlabel(f"$\\Delta$ {col_title}")
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Patient Type")

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_difference_hist(
    df,
    axis1,
    axis2,
):
    """Plot histogram of differences between two axes."""
    if axis1 not in df.columns or axis2 not in df.columns:
        print(f"One or both columns '{axis1}' and '{axis2}' not found in DataFrame.")
        return

    diffs = pd.to_numeric(df[axis1], errors="coerce") - pd.to_numeric(df[axis2], errors="coerce")
    diffs = diffs.dropna()

    min_val = int(math.floor(diffs.min()))
    max_val = int(math.ceil(diffs.max()))
    bins = range(min_val, max_val + 2)

    plt.figure(figsize=(8, 5))
    sns.histplot(diffs, bins=bins)
    plt.title(f"Distribution of Differences: {axis1} - {axis2}")
    plt.xlabel("Difference")
    plt.ylabel("Frequency")
    plt.axvline(0, color="black", linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_benchmark_target_performance(
    df,
    target_cols,
    model_col="assistant_model",
    patient_col="patient_name",
    patient_order=None,
    model_order=None,
    target_display_names=None,
    model_display_names=None,
    patient_display_names=None,
    palette=None,
    title=None,
    y_label="Accuracy",
    x_label=None,
    figure_size=None,
    errtype="sd",
    outlier_group_cols=None,
    group_cols=("scenario_id", "run"),
    plot_type="bar",
    subplot_by="patient",
    orientation="horizontal",
    color_by="model",
    xtick_rotation=45,
    show_outliers=False,
    save_as=None,
):
    if plot_type not in ("bar", "violin", "scatter_box"):
        raise ValueError(f"plot_type must be 'bar', 'violin', or 'scatter_box', got {plot_type!r}")
    if subplot_by not in ("patient", "target"):
        raise ValueError(f"subplot_by must be 'patient' or 'target', got {subplot_by!r}")
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")
    if color_by not in ("model", "x"):
        raise ValueError(f"color_by must be 'model' or 'x', got {color_by!r}")
    if patient_order is not None:
        existing = set(df[patient_col].dropna().unique())
        patient_names = [p for p in patient_order if p in existing]
    else:
        patient_names = sorted(df[patient_col].dropna().unique().tolist())
    if model_order is not None:
        existing_models = set(df[model_col].dropna().unique())
        models = [m for m in model_order if m in existing_models]
    else:
        models = sorted(df[model_col].dropna().unique().tolist())
    n_models = len(models)

    if palette is None:
        colors = sns.color_palette("Set2", n_colors=max(n_models, 1))
        palette = dict(zip(models, colors))

    target_labels = (
        [target_display_names.get(t, t) for t in target_cols]
        if target_display_names else list(target_cols)
    )
    patient_labels_full = (
        [patient_display_names.get(p, p) for p in patient_names]
        if patient_display_names else list(patient_names)
    )

    cells = {}
    for patient in patient_names:
        for model in models:
            model_sub = df[(df[patient_col] == patient) & (df[model_col] == model)]
            for target in target_cols:
                vals = pd.to_numeric(model_sub[target], errors="coerce").dropna().astype(float)
                mean = float(vals.mean()) if len(vals) else 0.0

                def _agg(cols):
                    if not cols:
                        return vals
                    return (
                        model_sub.assign(_t=pd.to_numeric(model_sub[target], errors="coerce"))
                        .dropna(subset=["_t"])
                        .groupby(list(cols))["_t"].mean()
                    )

                err_vals = _agg(group_cols)
                if errtype == "sd":
                    err = float(err_vals.std()) if len(err_vals) > 1 else 0.0
                elif errtype == "sem":
                    err = float(err_vals.std() / len(err_vals) ** 0.5) if len(err_vals) > 1 else 0.0
                else:
                    err = 0.0

                outlier_data = _agg(outlier_group_cols)
                if len(outlier_data) >= 4:
                    q1, q3 = outlier_data.quantile(0.25), outlier_data.quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = outlier_data[(outlier_data < lo) | (outlier_data > hi)].tolist()
                else:
                    outliers = []
                cells[(patient, model, target)] = (mean, err, outliers, _agg(group_cols).tolist())

    if subplot_by == "patient":
        subplot_keys, subplot_labels = patient_names, patient_labels_full
        x_keys, x_labels = target_cols, target_labels
    else:
        subplot_keys, subplot_labels = target_cols, target_labels
        x_keys, x_labels = patient_names, patient_labels_full

    n_subplots, n_x = len(subplot_keys), len(x_keys)

    if figure_size is None:
        if orientation == "horizontal":
            figure_size = (max(4.5, 1.3 * n_x) * n_subplots, 4)
        else:
            figure_size = (max(4.5, 1.3 * n_x), 3 * n_subplots)

    per_subplot_ylabel = isinstance(y_label, dict)
    if orientation == "horizontal":
        fig, axes = plt.subplots(
            1, n_subplots, figsize=figure_size, sharey=not per_subplot_ylabel
        )
    else:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figure_size, sharex=True)
    if n_subplots == 1:
        axes = [axes]

    x = np.arange(n_x)
    width = 0.8 / max(n_models, 1)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for ax, sk, slabel in zip(axes, subplot_keys, subplot_labels):
        for j, model in enumerate(models):
            means, errs, outliers, dists = [], [], [], []
            for xk in x_keys:
                key = (sk, model, xk) if subplot_by == "patient" else (xk, model, sk)
                m, e, o, d = cells[key]
                means.append(m)
                errs.append(e)
                outliers.append(o)
                dists.append(d)
            label = (
                model_display_names.get(model, model)
                if model_display_names else model
            )
            if color_by == "model":
                color = palette.get(model, "gray")
                bar_label = label
            else:
                color = [palette.get(xk, "gray") for xk in x_keys]
                bar_label = None
            if plot_type == "bar":
                ax.bar(
                    x + offsets[j], means, width, yerr=errs, capsize=3,
                    color=color, label=bar_label,
                    edgecolor="black", linewidth=0.5,
                )
            elif plot_type == "violin":
                positions = [x[k] + offsets[j] for k, d in enumerate(dists) if len(d) >= 2]
                data = [d for d in dists if len(d) >= 2]
                if data:
                    body_colors = (
                        [color[k] for k, d in enumerate(dists) if len(d) >= 2]
                        if color_by == "x" else [color] * len(data)
                    )
                    vp = ax.violinplot(
                        data, positions=positions, widths=width * 0.9,
                        showmeans=True, showextrema=False,
                    )
                    for body, body_color in zip(vp["bodies"], body_colors):
                        body.set_facecolor(body_color)
                        body.set_edgecolor("black")
                        body.set_linewidth(0.5)
                        body.set_alpha(0.7)
                    if "cmeans" in vp:
                        vp["cmeans"].set_color("black")
                if color_by == "model":
                    ax.bar(np.nan, np.nan, color=color, label=bar_label,
                           edgecolor="black", linewidth=0.5)
            else:
                positions = [x[k] + offsets[j] for k, d in enumerate(dists) if len(d) >= 1]
                data = [d for d in dists if len(d) >= 1]
                if data:
                    box_colors = (
                        [color[k] for k, d in enumerate(dists) if len(d) >= 1]
                        if color_by == "x" else [color] * len(data)
                    )
                    bp = ax.boxplot(
                        data, positions=positions, widths=width * 0.7,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        boxprops=dict(edgecolor="black", linewidth=0.5),
                        whiskerprops=dict(color="black", linewidth=0.5),
                        capprops=dict(color="black", linewidth=0.5),
                    )
                    for patch, box_color in zip(bp["boxes"], box_colors):
                        patch.set_facecolor(box_color)
                        patch.set_alpha(0.6)
                    for pos, d, box_color in zip(positions, data, box_colors):
                        jitter = (np.random.RandomState(0).rand(len(d)) - 0.5) * width * 0.5
                        ax.scatter(
                            np.full(len(d), pos) + jitter, d,
                            s=10, color=box_color, edgecolor="black", linewidth=0.3,
                            alpha=0.7, zorder=3,
                        )
                if color_by == "model":
                    ax.bar(np.nan, np.nan, color=color, label=bar_label,
                           edgecolor="black", linewidth=0.5)
            if show_outliers and plot_type != "scatter_box":
                for k, pts in enumerate(outliers):
                    if pts:
                        ax.scatter(
                            np.full(len(pts), x[k] + offsets[j]), pts,
                            s=14, color="black", alpha=0.6, zorder=3,
                        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            x_label if x_label else x_labels,
            rotation=xtick_rotation,
            ha="right" if xtick_rotation else "center",
            rotation_mode="anchor" if xtick_rotation else "default",
        )
        ax.set_title(slabel)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    def _ylabel(sk):
        return y_label.get(sk, "") if per_subplot_ylabel else y_label

    if orientation == "horizontal" and not per_subplot_ylabel:
        axes[0].set_ylabel(y_label)
    else:
        for ax, sk in zip(axes, subplot_keys):
            ax.set_ylabel(_ylabel(sk))
    handles, labels_ = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, -0.05), ncol=n_models)

    if title:
        fig.suptitle(title, y=1.03)
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def _metric_stat(sub, metric, group_cols, errtype):
    """Mean of ``metric`` over ``sub`` plus an ``errtype`` (sem/sd) error, where the
    error is computed across the means of ``group_cols`` groups (e.g. per case/run)."""
    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
    if vals.empty:
        return np.nan, 0.0
    grouped = vals.groupby([sub.loc[vals.index, c] for c in group_cols]).mean()
    sd = grouped.std()
    if pd.isna(sd):
        return float(vals.mean()), 0.0
    err = sd / np.sqrt(len(grouped)) if errtype == "sem" else sd
    return float(vals.mean()), float(err)


def _resolve_patients_models(df, patient_col, patient_order, model_col, model_order):
    existing_p = set(df[patient_col].dropna().unique())
    patients = [p for p in patient_order if p in existing_p]
    existing_m = set(df[model_col].dropna().unique())
    models = (
        [m for m in model_order if m in existing_m]
        if model_order is not None
        else sorted(existing_m)
    )
    if not patients or not models:
        raise ValueError("no patients or models present in df for the given orders")
    return patients, models


def plot_metric_dotplot(
    df,
    metric_cols,
    patient_col,
    patient_order,
    model_col="assistant_model",
    model_order=None,
    metric_display_names=None,
    model_display_names=None,
    palette=None,
    group_cols=("case_id", "run"),
    errtype="sem",
    x_pad=0.06,
    value_fmt="{:.2f}",
    figure_size=None,
    save_as=None,
):
    """Cleveland dot plot of one or more metrics per (patient x model).

    Zoomed per-metric x-axis (position encoding, so a non-zero baseline is honest,
    unlike a truncated bar). One subplot per metric, patients down the y-axis, a dot
    per model with an ``errtype`` whisker and a direct value label.
    """
    patients, models = _resolve_patients_models(
        df, patient_col, patient_order, model_col, model_order
    )

    mdisp = model_display_names or {}
    tdisp = metric_display_names or {}
    if palette is None:
        colors = sns.color_palette("Set2", n_colors=len(models))
        palette = dict(zip(models, colors))

    def _stat(sub, metric):
        return _metric_stat(sub, metric, group_cols, errtype)

    n = len(metric_cols)
    if figure_size is None:
        figure_size = (5.2 * n, max(3.0, 0.42 * len(patients) + 1.2))
    fig, axes = plt.subplots(1, n, figsize=figure_size, sharey=True)
    if n == 1:
        axes = [axes]

    y = np.arange(len(patients))[::-1]
    dodge = np.linspace(-0.24, 0.24, len(models)) if len(models) > 1 else [0.0]

    for ax, metric in zip(axes, metric_cols):
        lo, hi = np.inf, -np.inf
        for j, model in enumerate(models):
            xs, es = [], []
            for patient in patients:
                sub = df[(df[patient_col] == patient) & (df[model_col] == model)]
                m, e = _stat(sub, metric)
                xs.append(m)
                es.append(e)
            xs, es = np.array(xs, float), np.array(es, float)
            ax.errorbar(
                xs, y + dodge[j], xerr=es, fmt="o", ms=7, color=palette[model],
                mec="black", mew=0.6, ecolor=palette[model], elinewidth=1.4,
                capsize=2.5, zorder=3, label=mdisp.get(model, model),
            )
            for xi, yi, ei in zip(xs, y + dodge[j], es):
                if np.isnan(xi):
                    continue
                ax.annotate(
                    value_fmt.format(xi), (xi, yi), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color="0.25",
                )
                lo, hi = min(lo, xi - ei), max(hi, xi + ei)
        span = max(hi - lo, 1e-3)
        ax.set_xlim(lo - x_pad * span, hi + x_pad * span)
        ax.set_title(tdisp.get(metric, metric))
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", alpha=0.12)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(patients)
    if len(models) > 1:
        handles, labels_ = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles, labels_, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, -0.02), ncol=len(models),
        )
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_metric_barplot(
    df,
    metric_cols,
    patient_col,
    patient_order,
    model_col="assistant_model",
    model_order=None,
    metric_display_names=None,
    model_display_names=None,
    palette=None,
    group_cols=("case_id", "run"),
    errtype="sem",
    value_fmt="{:.2f}",
    ylim=(0.0, 1.0),
    figure_size=None,
    save_as=None,
):
    """Vertical grouped bar chart of one or more metrics per (patient x model).

    Patients along the x-axis, one bar per model (dodged) with an ``errtype`` whisker
    and a value label. One stacked subplot per metric sharing the x-axis. Bars start at
    the ``ylim`` floor (0 by default) so bar length is an honest encoding.
    """
    patients, models = _resolve_patients_models(
        df, patient_col, patient_order, model_col, model_order
    )
    mdisp = model_display_names or {}
    tdisp = metric_display_names or {}
    if palette is None:
        colors = sns.color_palette("Set2", n_colors=len(models))
        palette = dict(zip(models, colors))

    n = len(metric_cols)
    if figure_size is None:
        figure_size = (max(6.0, 0.7 * len(patients) + 1.5), 3.4 * n)
    fig, axes = plt.subplots(n, 1, figsize=figure_size, sharex=True)
    if n == 1:
        axes = [axes]

    x = np.arange(len(patients))
    width = 0.8 / len(models)

    for ax, metric in zip(axes, metric_cols):
        for j, model in enumerate(models):
            means, errs = [], []
            for patient in patients:
                sub = df[(df[patient_col] == patient) & (df[model_col] == model)]
                m, e = _metric_stat(sub, metric, group_cols, errtype)
                means.append(m)
                errs.append(e)
            offset = (j - (len(models) - 1) / 2) * width
            ax.bar(
                x + offset, means, width, yerr=errs, color=palette[model],
                edgecolor="black", linewidth=0.6, capsize=2.5,
                error_kw=dict(elinewidth=1.2), label=mdisp.get(model, model), zorder=3,
            )
            for xi, mi in zip(x + offset, means):
                if np.isnan(mi):
                    continue
                ax.annotate(
                    value_fmt.format(mi), (xi, 0), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color="white", fontweight="bold",
                )
        ax.set_ylabel(tdisp.get(metric, metric))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(patients, rotation=30, ha="right")
    if len(models) > 1:
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels_, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, -0.02), ncol=len(models),
        )
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_coverage_accuracy_scatter(
    df,
    x_metric="coverage",
    y_metric="correct_primary",
    patient_col="patient_disp_name",
    patient_order=None,
    marker_col=None,
    model_col="assistant_model",
    model_order=None,
    metric_display_names=None,
    model_display_names=None,
    patient_palette=None,
    group_cols=("case_id", "run"),
    figure_size=(7.5, 6.0),
    save_as=None,
):
    """Scatter relating information coverage (x) to accuracy (y).

    One point per (patient type x model): colour encodes patient type (``patient_col``);
    marker shape encodes ``marker_col`` (defaults to ``patient_col``), so passing a family
    column gives one shape per family (all its variants share it); fill encodes model
    (filled = first model, open = the rest). Rows with a missing x/y metric (e.g. the
    no-dialogue ``direct`` arm has null coverage) are dropped. Annotates the Pearson
    correlation across the points.
    """
    work = df.dropna(subset=[x_metric, y_metric])
    order = patient_order or sorted(work[patient_col].dropna().unique())
    patients, models = _resolve_patients_models(
        work, patient_col, order, model_col, model_order
    )
    mdisp = model_display_names or {}
    tdisp = metric_display_names or {}
    if patient_palette is None:
        colors = sns.color_palette("husl", n_colors=len(patients))
        patient_palette = dict(zip(patients, colors))
    markers = ["o", "s", "^", "v", "D", "p", "P", "X", "*", "<", ">", "h", "H", "d", "8"]
    marker_col = marker_col or patient_col
    patient_family = work.groupby(patient_col)[marker_col].first()
    family_order = list(dict.fromkeys(patient_family.get(p, p) for p in patients))
    family_marker = {g: markers[i % len(markers)] for i, g in enumerate(family_order)}
    patient_marker = {p: family_marker[patient_family.get(p, p)] for p in patients}

    fig, ax = plt.subplots(figsize=figure_size)
    xs_all, ys_all = [], []
    for j, model in enumerate(models):
        filled = j == 0
        for patient in patients:
            sub = work[(work[patient_col] == patient) & (work[model_col] == model)]
            if sub.empty:
                continue
            xm, _ = _metric_stat(sub, x_metric, group_cols, "sem")
            ym, _ = _metric_stat(sub, y_metric, group_cols, "sem")
            if np.isnan(xm) or np.isnan(ym):
                continue
            color = patient_palette[patient]
            ax.scatter(
                xm, ym, s=110, marker=patient_marker[patient],
                facecolor=color if filled else "none",
                edgecolor="black" if filled else color, linewidth=0.8, zorder=3,
            )
            xs_all.append(xm)
            ys_all.append(ym)

    xs_all, ys_all = np.array(xs_all), np.array(ys_all)
    if len(xs_all) >= 2:
        r = np.corrcoef(xs_all, ys_all)[0, 1]
        slope, intercept = np.polyfit(xs_all, ys_all, 1)
        xline = np.array([xs_all.min(), xs_all.max()])
        ax.plot(xline, intercept + slope * xline, color="0.4", ls="--", lw=1.2, zorder=2)
        ax.annotate(
            f"Pearson r = {r:.2f}", (0.03, 0.97), xycoords="axes fraction",
            ha="left", va="top", fontsize=9, color="0.3",
        )

    ax.set_xlabel(tdisp.get(x_metric, x_metric))
    ax.set_ylabel(tdisp.get(y_metric, y_metric))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    if len(models) > 1:
        model_handles = [
            plt.Line2D([0], [0], marker="o", ls="", ms=8, mew=0.8,
                       mfc="0.4" if j == 0 else "none",
                       mec="black" if j == 0 else "0.4", label=mdisp.get(m, m))
            for j, m in enumerate(models)
        ]
        ax.add_artist(ax.legend(
            handles=model_handles, title="Model", frameon=False,
            loc="lower right", fontsize=8, title_fontsize=9,
        ))
    patient_handles = [
        plt.Line2D([0], [0], marker=patient_marker[p], ls="", mfc=patient_palette[p],
                   mec="black", mew=0.6, ms=8, label=p)
        for p in patients
    ]
    ax.legend(
        handles=patient_handles, title="Patient type", frameon=False,
        loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title_fontsize=9,
    )
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_stacked_proportion_bar(
    df,
    label_col,
    group_col,
    label_order,
    label_colors,
    label_display_names=None,
    group_order=None,
    group_display_names=None,
    ylabel="Proportion",
    title=None,
    min_annot=0.04,
    xtick_rotation=0,
    figure_size=None,
    save_as=None,
):
    """Vertical stacked bars of each ``label_col`` value's proportion within each
    ``group_col`` (segments sum to 1 per group). Segments >= ``min_annot`` are labelled."""
    present = set(df[group_col].dropna().unique())
    groups = [g for g in (group_order or sorted(present)) if g in present]
    if not groups:
        raise ValueError("no groups present in df for group_col")
    ldisp = label_display_names or {}
    gdisp = group_display_names or {}
    counts = (
        df.groupby([group_col, label_col])
        .size()
        .unstack(label_col)
        .reindex(index=groups, columns=label_order)
        .fillna(0.0)
    )
    props = counts.div(counts.sum(axis=1), axis=0)

    if figure_size is None:
        figure_size = (max(4.0, 1.4 * len(groups) + 1.6), 5.0)
    fig, ax = plt.subplots(figsize=figure_size)
    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    for lab in label_order:
        vals = props[lab].to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.6, color=label_colors[lab],
               edgecolor="white", linewidth=0.8, label=ldisp.get(lab, lab), zorder=3)
        for xi, v, b in zip(x, vals, bottom):
            if v >= min_annot:
                ax.annotate(f"{v:.0%}", (xi, b + v / 2), ha="center", va="center",
                            fontsize=8, color="black")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(
        [gdisp.get(g, g) for g in groups],
        rotation=xtick_rotation, ha="right" if xtick_rotation else "center",
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_disclosure_curve(
    df,
    group_col="patient_type",
    ttf_col="ttf_per_field",
    max_turns=None,
    show_sd=False,
    title=None,
    palette=None,
    figure_size=(5, 3),
    save_as=None,
):
    if palette is None:
        palette = PATIENT_TYPE_PALETTE

    def _parse(raw):
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)

    conv_data = []
    for _, row in df.iterrows():
        raw = row[ttf_col]
        if pd.isna(raw):
            continue
        ttfs = _parse(raw)
        n_fields = len(ttfs) if ttfs else 0
        if not n_fields:
            continue
        disclosed_ttfs = [t for t in ttfs.values() if t is not None]
        conv_data.append({"group": row[group_col], "n_fields": n_fields, "disclosed_ttfs": disclosed_ttfs})

    if not conv_data:
        raise ValueError("No disclosure data available to plot")

    global_max = max_turns or max(
        (max(c["disclosed_ttfs"]) for c in conv_data if c["disclosed_ttfs"]),
        default=1,
    )

    rows = []
    for c in conv_data:
        for t in range(1, global_max + 1):
            fraction = sum(1 for v in c["disclosed_ttfs"] if v <= t) / c["n_fields"]
            rows.append({"turn": t, "fraction": fraction, "group": c["group"]})

    curve_df = pd.DataFrame(rows)
    if curve_df.empty:
        raise ValueError("No disclosure data available to plot")

    groups = _sort_patient_types(curve_df["group"].unique().tolist())
    bar_palette = {g: palette.get(str(g).split("_")[0].strip(), "gray") for g in groups}

    stats = curve_df.groupby(["turn", "group"])["fraction"].agg(mean="mean", std="std").reset_index()
    if show_sd:
        y_max = (stats["mean"] + stats["std"].fillna(0)).max() * 1.05
    else:
        y_max = stats["mean"].max() * 1.05

    fig, ax = plt.subplots(figsize=figure_size)
    sns.lineplot(
        data=curve_df,
        x="turn",
        y="fraction",
        hue="group",
        hue_order=groups,
        palette=bar_palette,
        errorbar="sd" if show_sd else None,
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Turn")
    ax.set_ylabel("Fraction of fields disclosed")
    ax.set_xlim(0, global_max)
    ax.set_ylim(0, y_max)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="0.7", zorder=0)
    ax.set_axisbelow(True)
    ax.legend()#bbox_to_anchor=(1.02, 0.7), loc="upper left")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig



def plot_turn_distribution(
    turns,
    metric,
    x_label,
    category="patient_display_name",
    ascending=True,
    figure_size=(9, 5),
    save_as=None,
):
    """Per-turn distribution of `metric` for each simulator, ordered by median."""
    order = (
        turns.groupby(category)[metric]
        .median()
        .sort_values(ascending=ascending)
        .index.tolist()
    )
    palette = {
        c: PATIENT_TYPE_PALETTE.get(c.split("\n")[0].strip(), "gray") for c in order
    }

    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(
        data=turns,
        x=metric,
        y=category,
        order=order,
        hue=category,
        hue_order=order,
        palette=palette,
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def _pacing_frame(df, reference, group_col, ttf_col, max_turn):
    turns = np.arange(1, max_turn + 1)
    curves = {}
    for _, row in df.iterrows():
        ttfs = _parse_distribution(row[ttf_col])
        disclosed = [t for t in ttfs.values() if t is not None] if ttfs else []
        if not disclosed:
            continue
        curves[(row[group_col], row["conversation_name"])] = np.array(
            [sum(1 for v in disclosed if v <= t) / len(disclosed) for t in turns]
        )
    groups = _sort_patient_types(df[group_col].unique().tolist())
    conversations = sorted(
        set.intersection(*[{c for a, c in curves if a == g} for g in groups])
    )
    arms = [g for g in groups if g != reference]
    return turns, curves, conversations, arms


def _pacing_gap(curves, arms, reference, conversations):
    ref = np.mean([curves[(reference, c)] for c in conversations], axis=0)
    return {
        a: np.abs(np.mean([curves[(a, c)] for c in conversations], axis=0) - ref).mean()
        for a in arms
    }


def disclosure_pacing_table(
    df,
    reference,
    group_col="patient_disp_name",
    ttf_col="ttf_per_field",
    max_turn=30,
    n_boot=5000,
    seed=0,
):
    _, curves, conversations, arms = _pacing_frame(df, reference, group_col, ttf_col, max_turn)
    observed = _pacing_gap(curves, arms, reference, conversations)
    rng = np.random.default_rng(seed)
    draws = {a: [] for a in arms}
    for _ in range(n_boot):
        sample = list(rng.choice(conversations, len(conversations), replace=True))
        gaps = _pacing_gap(curves, arms, reference, sample)
        for a in arms:
            draws[a].append(gaps[a])
    return pd.DataFrame(
        [
            {
                "arm": a,
                "gap": observed[a],
                "ci_lo": np.percentile(draws[a], 2.5),
                "ci_hi": np.percentile(draws[a], 97.5),
            }
            for a in arms
        ]
    )


def plot_disclosure_pacing(
    df,
    reference,
    group_col="patient_disp_name",
    ttf_col="ttf_per_field",
    max_turn=30,
    figure_size=(5, 3),
    save_as=None,
):
    turns, curves, conversations, arms = _pacing_frame(df, reference, group_col, ttf_col, max_turn)
    palette = {g: PATIENT_TYPE_PALETTE.get(g, "gray") for g in arms + [reference]}

    fig, ax = plt.subplots(figsize=figure_size)
    ax.plot(
        turns,
        np.mean([curves[(reference, c)] for c in conversations], axis=0),
        color=palette[reference],
        linewidth=3.0,
        linestyle=(0, (4, 1.5)),
        label=reference,
        zorder=5,
    )
    for arm in arms:
        ax.plot(
            turns,
            np.mean([curves[(arm, c)] for c in conversations], axis=0),
            color=palette[arm],
            linewidth=2.0,
            label=arm,
        )
    ax.set_xlabel("Turn")
    ax.set_ylabel("Share of disclosed fields")
    ax.set_xlim(1, max_turn)
    ax.set_ylim(0, None)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="0.85")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig
