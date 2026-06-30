"""Plotting utilities for patient simulator evaluation."""

import ast
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    "Human Actor",
]

ALL_AXES = ["H", "E", "X", "A", "C", "O"]


def _sort_patient_types(types):
    order = {pt: i for i, pt in enumerate(PATIENT_TYPE_ORDER)}
    return sorted(types, key=lambda x: (order.get(x, len(PATIENT_TYPE_ORDER)), x))


sort_patient_types = _sort_patient_types


def save_to_figures(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = _FIGURES_DIR / name if Path(name).suffix else _FIGURES_DIR / f"{name}.pdf"
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
    "Human Actor": "#808080",
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
            plot_df[all_cols].apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
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
    fig, axes = plt.subplots(
        1, n, figsize=(3.8 * n, figure_height), sharey=True, sharex=True
    )
    if n == 1:
        axes = [axes]

    for i, (ax, metric_col) in enumerate(zip(axes, all_cols)):
        summary = plot_df.groupby(
            [model_col, display_name_col, model_size_col], as_index=False
        )[metric_col].agg(mean="mean", std="std")
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

        col_title = (
            metric_display_names.get(metric_col, metric_col)
            if metric_display_names
            else metric_col
        )
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

    palette = {
        pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray")
        for pt in patient_types
    }
    plot_df[category] = pd.Categorical(
        plot_df[category], categories=list(reversed(patient_types)), ordered=True
    )
    plot_df = plot_df.sort_values(by=category)

    fig, ax = plt.subplots(figsize=figure_size)
    sns.boxplot(
        data=plot_df,
        x=metric,
        y=category,
        hue=category,
        palette=palette,
        ax=ax,
        legend=False,
    )

    if show_title:
        ax.set_title(
            f"{label} by Patient Type" if label else f"{metric} by Patient Type"
        )
    ax.set_xlabel(label if label else metric)
    ax.set_ylabel("Patient Type")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig


def plot_category_fraction(
    df,
    category,
    value_col,
    target_value,
    user_col="user",
    fraction_col="fraction",
    error_type=None,
    title=None,
    x_label="Proportion",
    y_label=None,
    figure_size=(6, 4),
    save_as=None,
):
    if error_type not in (None, "std", "sem", "mad"):
        raise ValueError(
            f"error_type must be None, 'std', 'sem', or 'mad', got {error_type!r}"
        )

    missing_cols = [col for col in [category, value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    grouped = df.groupby(category, as_index=False).agg(
        **{fraction_col: (value_col, lambda s: (s == target_value).mean())}
    )

    if user_col in df.columns and error_type is not None:
        per_user = (
            df.groupby([category, user_col])[value_col]
            .apply(lambda s: (s == target_value).mean())
            .reset_index(name=fraction_col)
        )

        def compute_error(x):
            if error_type == "std":
                return x.std()
            if error_type == "sem":
                return x.std() / len(x) ** 0.5
            if error_type == "mad":
                from scipy.stats import median_abs_deviation

                return median_abs_deviation(x, nan_policy="omit")

        errors = (
            per_user.groupby(category)[fraction_col]
            .agg(error=compute_error)
            .reset_index()
        )
        grouped = grouped.merge(errors, on=category)
    else:
        grouped["error"] = 0.0

    categories = list(reversed(_sort_patient_types(grouped[category].tolist())))
    grouped[category] = pd.Categorical(
        grouped[category], categories=categories, ordered=True
    )
    grouped = grouped.sort_values(category)

    palette = {
        c: PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    }

    fig, ax = plt.subplots(figsize=figure_size)
    ax.barh(
        grouped[category].tolist(),
        grouped[fraction_col],
        xerr=grouped["error"] if error_type is not None else None,
        capsize=4 if error_type is not None else 0,
        color=[palette[c] for c in grouped[category]],
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label or category.replace("_", " ").title())
    x_max = (grouped[fraction_col] + grouped["error"]).max()
    ax.set_xlim(0, min(1.0, x_max * 1.05))

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
    patient_types = list(
        reversed(_sort_patient_types(df[category].dropna().unique().tolist()))
    )

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
            rows.append(
                {
                    category: row[category],
                    "dist_value": value,
                    "proportion": count / total,
                }
            )

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
    palette = {
        pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray")
        for pt in patient_types
    }

    n_plots = len(dist_values)
    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=sharey
    )
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, value in zip(axes, dist_values):
        sub = summary[summary["dist_value"] == value]
        means = [
            sub.loc[sub[category] == pt, "mean"].iloc[0]
            if (sub[category] == pt).any()
            else 0
            for pt in patient_types
        ]
        sems = [
            sub.loc[sub[category] == pt, "sem"].iloc[0]
            if (sub[category] == pt).any()
            else 0
            for pt in patient_types
        ]

        ax.barh(
            patient_types,
            means,
            xerr=sems,
            capsize=4,
            color=[palette[pt] for pt in patient_types],
        )
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
        id_vars=[category],
        value_vars=value_cols,
        var_name="category",
        value_name="value",
    )

    canonical_order = list(
        reversed(_sort_patient_types(plot_df[category].unique().tolist()))
    )
    plot_df[category] = pd.Categorical(
        plot_df[category], categories=canonical_order, ordered=True
    )
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
    sns.lineplot(
        data=df, x=category, y=metric, hue=hue_col, palette=palette, ax=ax, linewidth=3
    )

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
                diffs = reconstructed_vals[valid_mask].astype(int) - real_vals[
                    valid_mask
                ].astype(int)
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
    ax.barh(
        labels,
        deviations,
        xerr=std_devs,
        capsize=5,
        color=colors,
        ecolor="black",
        alpha=0.8,
    )
    ax.axvline(0, color="black", linewidth=1)

    max_abs = max(
        (abs(dev) + std for dev, std in zip(deviations, std_devs)), default=1.0
    )
    if max_abs == 0:
        max_abs = 1.0
    ax.set_xlim(-max_abs, max_abs)

    ax.set_title(title)
    ax.set_xlabel("Deviation (Reconstructed - Real)")
    ax.set_ylabel("HEXACO Axis")

    fig.tight_layout()
    plt.show()

    return pd.DataFrame(
        {"axis": labels, "mean_deviation": deviations, "std_deviation": std_devs}
    )


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
    work = pd.concat(
        [
            df[[persona_col, simulator_col]].reset_index(drop=True),
            num.reset_index(drop=True),
        ],
        axis=1,
    )

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
                records.append(
                    {
                        persona_col: persona,
                        simulator_col: sim,
                        "axis": axis_label,
                        "mean": mean,
                        "std": std,
                    }
                )
    stats = pd.DataFrame(records)

    y_abs = (stats["mean"].abs() + stats["std"].fillna(0)).max() or 1.0
    y_lim = y_abs * 1.15

    nrows = math.ceil(len(personas) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size, sharey=True)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    x = np.arange(len(axis_labels))
    width = 0.8 / len(simulators)
    sim_styles = {
        sim: {"hatch": "" if i == 0 else "///", "alpha": 0.9 if i == 0 else 0.55}
        for i, sim in enumerate(simulators)
    }

    for ax, persona in zip(axes, personas):
        color = palette.get(persona, "gray")
        for i, sim in enumerate(simulators):
            sub = (
                stats[(stats[persona_col] == persona) & (stats[simulator_col] == sim)]
                .set_index("axis")
                .reindex(axis_labels)
            )
            offset = (i - (len(simulators) - 1) / 2) * width
            style = sim_styles[sim]
            ax.bar(
                x + offset,
                sub["mean"],
                width,
                yerr=sub["std"].fillna(0),
                color=color,
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.7,
                capsize=3,
                label=sim if ax is axes[0] else None,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(axis_labels)
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(axis="y", alpha=0.25)

    for ax in axes[len(personas) :]:
        ax.set_visible(False)

    for i in range(nrows):
        axes[i * ncols].set_ylabel("Deviation\n(reconstructed - real)")

    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_,
            loc="upper center",
            ncol=len(simulators),
            bbox_to_anchor=(0.5, 1.02),
            frameon=False,
        )
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
    ax.legend(
        title=category.replace("_", " ").title(),
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

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

    categories = list(
        reversed(_sort_patient_types(df[category].dropna().unique().tolist()))
    )

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
        ax.barh(
            categories,
            vals,
            left=left,
            color=color,
            label=str(int(level)),
            edgecolor="white",
            linewidth=0.4,
        )
        left += vals

    ax.set_xlabel(x_label)
    ax.set_ylabel(category.replace("_", " ").title())
    ax.set_xlim(0, 1)
    ax.set_title(
        title or f"Confidence Distribution by {category.replace('_', ' ').title()}"
    )
    ax.legend(
        title="Confidence", bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False
    )

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return pivot


def _cohen_kappa_matrix(df, user_col, value_col, conv_id_col):
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
    kappa_matrix = _cohen_kappa_matrix(df, user_col, value_col, conv_id_col)
    fig, ax = plt.subplots(figsize=figure_size)
    mask = np.tril(np.ones_like(kappa_matrix, dtype=bool), k=-1)
    ax.set_facecolor("lightgrey")
    sns.heatmap(
        kappa_matrix.astype(float),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
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
        patient_types = _sort_patient_types(
            df[patient_type_col].dropna().unique().tolist()
        )

    rows = []
    for extreme in ALL_AXES:
        for pt in patient_types:
            sub = df[(df[axis_col] == extreme) & (df[patient_type_col] == pt)]
            for ev in AXES:
                human_mean = sub[sub[rater_col] == "Human"][rec_cols[ev]].mean()
                auto_mean = sub[sub[rater_col] == "Autorater"][rec_cols[ev]].mean()
                rows.append(
                    {"patient_type": pt, "human": human_mean, "autorater": auto_mean}
                )
    agg = pd.DataFrame(rows).dropna()

    r = np.corrcoef(agg["human"], agg["autorater"])[0, 1]
    icc = _icc_two_raters(agg["human"].values, agg["autorater"].values)

    fig, ax = plt.subplots(figsize=figure_size)
    for pt in patient_types:
        sub = agg[agg["patient_type"] == pt]
        ax.scatter(
            sub["human"],
            sub["autorater"],
            color=PATIENT_TYPE_PALETTE.get(pt, "gray"),
            label=pt,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.5,
            s=120,
        )

    lo = agg[["human", "autorater"]].min().min() - 0.05
    hi = agg[["human", "autorater"]].max().max() + 0.05
    ax.plot([lo, hi], [lo, hi], color="gray", linewidth=1, linestyle="--", zorder=0)

    ax.set_xlabel("Human mean reconstruction")
    ax.set_ylabel("Autorater mean reconstruction")
    # ax.set_title(title or "Human vs Autorater HEXACO Alignment")
    ax.legend(
        frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return r, icc


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
        patient_types = _sort_patient_types(
            df[patient_type_col].dropna().unique().tolist()
        )

    x = np.arange(len(AXES))
    n = len(patient_types)
    width = 0.35
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    fig, ax = plt.subplots(figsize=figure_size)

    for i, pt in enumerate(patient_types):
        sub = df[df[patient_type_col] == pt]
        human = sub[sub[rater_col] == "Human"]
        autorater = sub[sub[rater_col] == "Autorater"]

        devs = [
            abs(human[rec_cols[axis]].mean() - autorater[rec_cols[axis]].mean())
            for axis in AXES
        ]
        sds = (
            [
                np.sqrt(
                    human[rec_cols[axis]].std() ** 2
                    + autorater[rec_cols[axis]].std() ** 2
                )
                for axis in AXES
            ]
            if show_sd
            else None
        )

        color = PATIENT_TYPE_PALETTE.get(pt, "gray")
        ax.bar(
            x + offsets[i],
            devs,
            width,
            yerr=sds,
            capsize=4,
            color=color,
            label=pt,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(AXES)
    ax.set_xlabel("HEXACO Axis")
    ax.set_ylabel("|Human mean − Autorater mean|")
    ax.set_title(
        title or "Human vs Autorater HEXACO Reconstruction: Absolute Deviation"
    )
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
    AXES = [
        ax for ax in ["H", "E", "X", "A", "C", "O"] if ax not in (exclude_axes or [])
    ]
    rec_cols = {ax: f"personality_reconstructed_{ax}" for ax in AXES}

    sub = df[df[rater_col] == rater]
    if patient_types is None:
        patient_types = _sort_patient_types(
            sub[patient_type_col].dropna().unique().tolist()
        )

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
        ax.bar(
            x + offsets[i],
            diffs[pt],
            width,
            yerr=yerr,
            capsize=4,
            color=color,
            label=pt,
            edgecolor="black",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(AXES)
    ax.set_xlabel("HEXACO Axis")
    ax.set_ylabel("Extreme − Low mean reconstruction")
    # ax.set_title(title or f"Personality Extreme Difference ({rater})")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(
        frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0
    )
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()


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
    rec_cols = {ax: f"personality_reconstructed_{ax}" for ax in ALL_AXES}

    rows = []
    for _, r in df.iterrows():
        truth = {
            ax: extreme_value if ax == r[axis_col] else low_value for ax in ALL_AXES
        }
        for ax in ALL_AXES:
            val = r[rec_cols[ax]]
            if pd.notna(val):
                rows.append(
                    {
                        "rater": r[rater_col],
                        "patient_type": r[patient_type_col],
                        "axis": ax,
                        "abs_dist": abs(val - truth[ax]),
                    }
                )

    long = pd.DataFrame(rows)
    wide = (
        long.groupby(["rater", "patient_type", "axis"])["abs_dist"]
        .mean()
        .unstack("axis")
        .reindex(columns=ALL_AXES)
    )
    wide["mean"] = wide.mean(axis=1)
    return wide


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
    bar_palette = {
        v: palette.get(str(v).split("_")[0].strip(), "gray") for v in hue_order
    }

    fig, ax = plt.subplots(figsize=figure_size)
    sns.countplot(
        data=df,
        x=x,
        order=order,
        hue=hue_col,
        hue_order=hue_order,
        palette=bar_palette,
        ax=ax,
        legend=(hue is not None),
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
    title=None,
    figure_size=(6, 4),
    save_as=None,
):
    rows = []
    for _, row in df.iterrows():
        reasons_in_conv = set()
        for flag in row.get(flag_col) or []:
            for reason in flag.get("reasons") or ["unspecified"]:
                normalized = reason.lower().strip() if reason else "unspecified"
                reasons_in_conv.add(
                    "unspecified"
                    if normalized in ("unspecified", "other")
                    else normalized
                )
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
            r
            for r in reversed(normalized_order)
            if r in total_per_reason.index and total_per_reason[r] >= min_count
        ]
    else:
        reasons_filtered = (
            total_per_reason[total_per_reason >= min_count]
            .sort_values(ascending=True)
            .index.tolist()
        )

    if not reasons_filtered:
        print(f"No reasons with count >= {min_count}.")
        return

    counts = counts[counts["reason"].isin(reasons_filtered)]
    total_per_cat = df.groupby(category).size()
    counts["relative"] = counts.apply(
        lambda r: r["count"] / total_per_cat[r[category]], axis=1
    )

    patient_types = _sort_patient_types(counts[category].unique().tolist())
    bar_palette = {
        pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray")
        for pt in patient_types
    }

    y = np.arange(len(reasons_filtered))
    n = len(patient_types)
    height = 0.8 / n

    fig, ax = plt.subplots(figsize=figure_size)
    for i, pt in enumerate(reversed(patient_types)):
        sub = (
            counts[counts[category] == pt]
            .set_index("reason")
            .reindex(reasons_filtered, fill_value=0)["relative"]
        )
        offset = (i - (n - 1) / 2) * height
        ax.barh(y + offset, sub.values, height * 0.9, color=bar_palette[pt])

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_palette[pt]) for pt in patient_types
    ]

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
    ax.set_xlabel("Fraction of Conversations")
    ax.set_ylabel("Flag Reason")
    ax.legend(
        legend_handles,
        patient_types,
        # frameon=False,
        # loc="upper center",
        bbox_to_anchor=(0.6, 0.35),
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

    categories = list(
        reversed(_sort_patient_types(df[category].dropna().unique().tolist()))
    )

    def frac(col):
        return (
            df.groupby(category)[col]
            .apply(lambda s: (s == target_value).mean())
            .reindex(categories)
        )

    human_frac = frac(human_col)
    judge_frac = frac(judge_col)

    y = np.arange(len(categories))
    width = 0.35
    bar_colors = [
        PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    ]

    fig, ax = plt.subplots(figsize=figure_size)
    for i, (cat, color) in enumerate(zip(categories, bar_colors)):
        ax.barh(
            y[i] - width / 2,
            human_frac[cat],
            width,
            color=color,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.barh(
            y[i] + width / 2,
            judge_frac[cat],
            width,
            color=color,
            hatch="///",
            edgecolor="black",
            linewidth=0.6,
            alpha=0.6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel(x_label)
    ax.set_xlim(0, 1)
    ax.set_title(
        title or f"Fraction '{target_value}' by {category}: Human vs LLM Judge"
    )

    legend_handles = [
        mpatches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.6, label="Human"
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
            label="LLM Judge",
        ),
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

    categories = list(
        reversed(_sort_patient_types(df[category].dropna().unique().tolist()))
    )
    y = np.arange(len(categories))
    width = 0.35
    bar_colors = [
        PATIENT_TYPE_PALETTE.get(str(c).split("_")[0].strip(), "gray")
        for c in categories
    ]

    n = len(human_metrics)
    fig, axes = plt.subplots(1, n, figsize=figure_size, sharey=True)
    axes = axes if n > 1 else [axes]

    for ax, h_col, j_col, label in zip(
        axes, human_metrics, judge_metrics, metric_labels
    ):
        h_mean = df.groupby(category)[h_col].mean().reindex(categories)
        j_mean = df.groupby(category)[j_col].mean().reindex(categories)
        if user_col and user_col in df.columns:
            h_sem = (
                df.groupby([category, user_col])[h_col]
                .mean()
                .groupby(category)
                .sem()
                .reindex(categories)
                .fillna(0)
            )
            j_sem = (
                df.groupby([category, user_col])[j_col]
                .mean()
                .groupby(category)
                .sem()
                .reindex(categories)
                .fillna(0)
            )
        else:
            h_sem = df.groupby(category)[h_col].sem().reindex(categories).fillna(0)
            j_sem = df.groupby(category)[j_col].sem().reindex(categories).fillna(0)

        for i, (cat, color) in enumerate(zip(categories, bar_colors)):
            ax.barh(
                y[i] - width / 2,
                h_mean[cat],
                width,
                xerr=h_sem[cat],
                capsize=3,
                color=color,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.barh(
                y[i] + width / 2,
                j_mean[cat],
                width,
                xerr=j_sem[cat],
                capsize=3,
                color=color,
                hatch="///",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.6,
            )
        ax.set_title(label)
        ax.set_xlabel("Mean score")
        ax.set_xlim(score_range[0] - 0.5, score_range[1] + 0.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(categories)
    axes[0].set_ylabel(category.replace("_", " ").title())

    legend_handles = [
        mpatches.Patch(
            facecolor="white", edgecolor="black", linewidth=0.6, label="Human"
        ),
        mpatches.Patch(
            facecolor="white",
            edgecolor="black",
            linewidth=0.6,
            hatch="///",
            label="LLM Judge",
        ),
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
    palette = {
        pt: PATIENT_TYPE_PALETTE.get(pt.split("_")[0].strip(), "gray")
        for pt in patient_types
    }

    plot_df = df.copy()
    plot_df[category] = pd.Categorical(
        plot_df[category], categories=patient_types, ordered=True
    )

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
        sns.boxplot(
            data=sub,
            x=col,
            y=category,
            hue=category,
            palette=palette,
            ax=ax,
            legend=False,
        )
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

    diffs = pd.to_numeric(df[axis1], errors="coerce") - pd.to_numeric(
        df[axis2], errors="coerce"
    )
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
        conv_data.append(
            {
                "group": row[group_col],
                "n_fields": n_fields,
                "disclosed_ttfs": disclosed_ttfs,
            }
        )

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

    stats = (
        curve_df.groupby(["turn", "group"])["fraction"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
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
    ax.legend()  # bbox_to_anchor=(1.02, 0.7), loc="upper left")
    fig.tight_layout()
    if save_as:
        save_to_figures(fig, save_as)
    plt.show()
    return fig
