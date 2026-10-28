"""Conversation-embedding diversity analysis utilities."""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patheffects import withStroke
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer


def conversation_texts(df, text_col="simulated_response", turns_filename="turns.csv"):
    """Concatenated simulated-patient turns of each conversation in ``df``."""
    if "path" not in df.columns:
        raise ValueError("df must contain a 'path' column pointing to conversation dirs")

    texts = []
    for path in df["path"]:
        turns_path = os.path.join(path, turns_filename)
        if not os.path.isfile(turns_path):
            raise FileNotFoundError(f"Missing turns file: {turns_path}")
        turns = pd.read_csv(turns_path)
        if text_col not in turns.columns:
            raise ValueError(f"Column '{text_col}' not in {turns_path}")
        texts.append("\n".join(turns[text_col].dropna().astype(str).tolist()))
    return texts


def _context_windows(text, tokenizer, window_size):
    """``text`` split into consecutive token windows of at most ``window_size``."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(ids[i:i + window_size]) for i in range(0, max(len(ids), 1), window_size)]


def embed_conversations(
    df,
    text_col="simulated_response",
    turns_filename="turns.csv",
    model_name="Qwen/Qwen3-Embedding-0.6B",
    device=None,
    batch_size=1,
):
    """Embed each conversation in ``df`` as a single vector.

    For every row, read ``{row.path}/{turns_filename}`` and concatenate all
    ``text_col`` turns into one string. The default encoder takes 32k tokens, so a
    conversation is encoded in one pass; anything longer than the context window is
    split into windows that fit it, encoded, and mean-pooled back into one unit
    vector. Encoding the concatenation directly under a short-context encoder would
    instead silently keep only its opening turns, over a window whose size depends
    on how verbose the arm being measured is.

    ``batch_size`` stays at 1 because attention over a full-length conversation
    allocates several GiB per sample.
    """
    texts = conversation_texts(df, text_col=text_col, turns_filename=turns_filename)
    model = SentenceTransformer(model_name, device=device)
    window_size = model.max_seq_length - model.tokenizer.num_special_tokens_to_add()
    windows = [_context_windows(text, model.tokenizer, window_size) for text in texts]

    encoded = model.encode(
        [window for conversation in windows for window in conversation],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    splits = np.cumsum([len(conversation) for conversation in windows])[:-1]
    pooled = np.vstack([part.mean(axis=0) for part in np.split(encoded, splits)])
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


def within_case_spread(embeddings, df, group_col, case_col):
    """Mean pairwise embedding distance among conversations sharing a case, per group.

    Holding the clinical case fixed isolates persona-driven behavioural variation
    from case/topic variation, which otherwise dominates the embedding space.
    Returns a long DataFrame with one row per (group, case).
    """
    if len(df) != len(embeddings):
        raise ValueError("df and embeddings must have the same length")

    frame = df.reset_index(drop=True)
    rows = []
    for (group, case), idx in frame.groupby([group_col, case_col], observed=True).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        rows.append({group_col: group, case_col: case, "spread": float(pdist(embeddings[idx]).mean())})
    return pd.DataFrame(rows)


def paired_group_comparison(long_df, group_col, case_col, value_col, groups, n_boot=10000, seed=0):
    """Paired comparison of two groups measured on the same cases.

    Returns per-group means with SEM plus the paired difference, its bootstrap CI,
    the paired t / Wilcoxon p-values and the fraction of cases won by ``groups[0]``.
    """
    from scipy.stats import ttest_rel, wilcoxon

    if len(groups) != 2:
        raise ValueError("groups must name exactly two groups to pair")

    wide = long_df.pivot(index=case_col, columns=group_col, values=value_col)
    missing = [g for g in groups if g not in wide.columns]
    if missing:
        raise ValueError(f"groups not present in {group_col}: {missing}")
    wide = wide[list(groups)].dropna()
    if wide.empty:
        raise ValueError("no cases shared by both groups")

    a, b = wide[groups[0]].to_numpy(), wide[groups[1]].to_numpy()
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)

    return pd.Series({
        "n_cases": len(diff),
        f"mean_{groups[0]}": a.mean(),
        f"sem_{groups[0]}": a.std(ddof=1) / np.sqrt(len(a)),
        f"mean_{groups[1]}": b.mean(),
        f"sem_{groups[1]}": b.std(ddof=1) / np.sqrt(len(b)),
        "mean_diff": diff.mean(),
        "ci_lo": float(np.percentile(boot, 2.5)),
        "ci_hi": float(np.percentile(boot, 97.5)),
        "t_stat": float(ttest_rel(a, b).statistic),
        "p_ttest": float(ttest_rel(a, b).pvalue),
        "p_wilcoxon": float(wilcoxon(a, b).pvalue),
        "win_rate": float((diff > 0).mean()),
    })


def persona_displacement(embeddings, df, group_col, persona_col, case_col, reference_persona):
    """Within-case cosine distance from each persona to ``reference_persona``, per group.

    Measures how far a persona parametrization actually moves conversational
    behaviour away from the neutral persona on the same clinical case.
    """
    if len(df) != len(embeddings):
        raise ValueError("df and embeddings must have the same length")

    frame = df.reset_index(drop=True)
    if reference_persona not in set(frame[persona_col]):
        raise ValueError(f"reference persona '{reference_persona}' not found in {persona_col}")

    rows = []
    for (group, case), sub in frame.groupby([group_col, case_col], observed=True):
        ref_idx = sub.index[sub[persona_col] == reference_persona]
        if len(ref_idx) != 1:
            continue
        ref = embeddings[ref_idx[0]]
        for i, persona in zip(sub.index, sub[persona_col]):
            if persona == reference_persona:
                continue
            rows.append({
                group_col: group,
                case_col: case,
                persona_col: persona,
                "displacement": float(1.0 - embeddings[i] @ ref),
            })
    return pd.DataFrame(rows)


def cross_case_ngram_overlap(
    df,
    group_col,
    persona_col,
    case_col,
    n,
    text_col="simulated_response",
    turns_filename="turns.csv",
):
    """Mean n-gram overlap between same-persona conversations on *different* cases.

    A persona rendered as a fixed template reuses literal phrasing regardless of the
    clinical case, so high overlap indicates canned persona behaviour rather than
    behaviour that adapts to the encounter. Overlap is |A n B| / min(|A|, |B|).
    """
    frame = df.reset_index(drop=True)
    tokens = [re.findall(r"[a-z']+", text.lower()) for text in
              conversation_texts(frame, text_col=text_col, turns_filename=turns_filename)]
    grams = [set(zip(*[t[i:] for i in range(n)])) for t in tokens]

    rows = []
    for (group, persona), sub in frame.groupby([group_col, persona_col], observed=True):
        idx = list(sub.index)
        cases = frame.loc[idx, case_col].to_numpy()
        overlaps = [
            len(grams[i] & grams[j]) / min(len(grams[i]), len(grams[j]))
            for a, i in enumerate(idx)
            for b, j in enumerate(idx)
            if b > a and cases[a] != cases[b] and min(len(grams[i]), len(grams[j])) > 0
        ]
        if overlaps:
            rows.append({group_col: group, persona_col: persona, "ngram_overlap": float(np.mean(overlaps))})
    return pd.DataFrame(rows)


def repeated_ngrams(
    df,
    group_col,
    persona_col,
    case_col,
    n,
    top_k,
    text_col="simulated_response",
    turns_filename="turns.csv",
):
    """Most frequent n-grams per (group, persona), counted once per case.

    Surfaces the literal boilerplate behind a high ``cross_case_ngram_overlap``.
    """
    from collections import Counter

    frame = df.reset_index(drop=True)
    tokens = [re.findall(r"[a-z']+", text.lower()) for text in
              conversation_texts(frame, text_col=text_col, turns_filename=turns_filename)]

    rows = []
    for (group, persona), sub in frame.groupby([group_col, persona_col], observed=True):
        counts = Counter()
        for i in sub.index:
            t = tokens[i]
            counts.update(set(zip(*[t[k:] for k in range(n)])))
        n_cases = sub[case_col].nunique()
        for gram, count in counts.most_common(top_k):
            rows.append({
                group_col: group,
                persona_col: persona,
                "ngram": " ".join(gram),
                "n_cases": count,
                "case_fraction": count / n_cases,
            })
    return pd.DataFrame(rows)


_COLUMN_RC = {
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 7,
}


def _simulator_colors(simulator_order):
    from patient_simulator.misc.plotting import PATIENT_TYPE_PALETTE

    return {sim: PATIENT_TYPE_PALETTE.get(sim, "#888888") for sim in simulator_order}


def _spread_scatter(ax, spread_df, group_col, case_col, simulator_order, point_color, marker_size,
                    equal_aspect, line_style):
    """Per-case spread of the first simulator against the second, on the identity line."""
    wide = spread_df.pivot(index=case_col, columns=group_col, values="spread")[
        list(simulator_order)
    ].dropna()
    x, y = wide[simulator_order[1]], wide[simulator_order[0]]
    lims = (min(x.min(), y.min()), max(x.max(), y.max()))
    pad = 0.05 * (lims[1] - lims[0])
    lims = (lims[0] - pad, lims[1] + pad)

    ax.plot(lims, lims, color="black", linewidth=1.0, linestyle=line_style, zorder=1)
    ax.scatter(x, y, s=marker_size, color=point_color, edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    if equal_aspect:
        ax.set_aspect("equal")
    return wide


def _persona_means(long_df, group_col, persona_col, value_col, simulator_order, persona_order):
    """Mean ``value_col`` per persona and simulator, as a persona x simulator frame."""
    return (
        long_df.groupby([persona_col, group_col], observed=True)[value_col]
        .mean()
        .unstack(group_col)
        .reindex(index=persona_order, columns=list(simulator_order))
    )


def _persona_dumbbells(ax, means, simulator_order, colors, marker_size):
    """One row per persona, a dot per simulator joined by a connector."""
    positions = np.arange(len(means))[::-1]
    ax.hlines(positions, means[simulator_order[1]], means[simulator_order[0]],
              color="0.6", linewidth=1.0, zorder=1)
    for sim in simulator_order:
        ax.scatter(means[sim], positions, s=marker_size, color=colors[sim], edgecolor="black",
                   linewidth=0.6, zorder=3, label=sim)
    ax.set_yticks(positions)
    ax.set_yticklabels(means.index)
    ax.set_ylim(positions.min() - 0.6, positions.max() + 0.6)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
    ax.grid(axis="x", alpha=0.3)


def plot_persona_diversity(
    spread_df,
    displacement_df,
    overlap_df,
    group_col,
    persona_col,
    case_col,
    simulator_order,
    persona_order,
    figure_size=(15, 4.2),
    save_as=None,
):
    """Three-panel case-controlled persona diversity summary.

    Left: per-case within-case persona spread, one point per case, against the
    identity line — points above it are cases where the first simulator spreads wider.
    Middle: displacement from the reference persona, per persona.
    Right: cross-case n-gram overlap within a persona (lower = less templated).
    """
    from patient_simulator.misc.plotting import save_to_figures

    colors = _simulator_colors(simulator_order)
    fig, axes = plt.subplots(1, 3, figsize=figure_size)

    wide = _spread_scatter(
        axes[0], spread_df, group_col, case_col, simulator_order,
        point_color=colors[simulator_order[0]], marker_size=45, equal_aspect=True,
        line_style="--",
    )
    wins = (wide[simulator_order[0]] > wide[simulator_order[1]]).sum()
    axes[0].set_xlabel(f"{simulator_order[1]} spread")
    axes[0].set_ylabel(f"{simulator_order[0]} spread")
    axes[0].set_title("Within-case persona spread")
    axes[0].text(0.04, 0.94, f"{wins / len(wide):.0%} of cases above the line",
                 transform=axes[0].transAxes, va="top", fontsize=9)

    idx = np.arange(len(persona_order))
    width = 0.38
    for i, sim in enumerate(simulator_order):
        stats = (
            displacement_df[displacement_df[group_col] == sim]
            .groupby(persona_col, observed=True)["displacement"]
            .agg(["mean", "sem"])
            .reindex(persona_order)
        )
        axes[1].bar(idx + (i - 0.5) * width, stats["mean"], width, yerr=stats["sem"],
                    capsize=3, color=colors[sim], edgecolor="black", linewidth=0.7, label=sim)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(persona_order, rotation=25, ha="right")
    axes[1].set_ylabel("Cosine distance from Default")
    axes[1].set_title("Displacement from the neutral persona")
    axes[1].grid(axis="y", alpha=0.3)

    stats = (
        overlap_df.groupby(group_col, observed=True)["ngram_overlap"]
        .agg(["mean", "sem"])
        .reindex(simulator_order)
    )
    axes[2].bar(range(len(simulator_order)), stats["mean"], 0.55, yerr=stats["sem"], capsize=4,
                color=[colors[s] for s in simulator_order], edgecolor="black", linewidth=0.7)
    axes[2].set_xticks(range(len(simulator_order)))
    axes[2].set_xticklabels(simulator_order, rotation=15, ha="right")
    axes[2].set_ylabel("Cross-case n-gram overlap")
    axes[2].set_title("Persona templatedness")
    axes[2].grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(facecolor=colors[s], edgecolor="black", linewidth=0.5, label=s)
               for s in simulator_order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncols=len(simulator_order), frameon=False)
    plt.tight_layout()

    if save_as:
        save_to_figures(fig, save_as)

    plt.show()
    return fig


def _diagonal_regions(ax, colors, simulator_order, simulator_labels):
    """Tint each side of the identity line and label it with an arrow off the diagonal."""
    lo, hi = ax.get_xlim()
    span = hi - lo
    above, below = simulator_order
    ax.fill_between((lo, hi), (lo, hi), hi, color=colors[above], alpha=0.12, linewidth=0, zorder=0)
    ax.fill_between((lo, hi), lo, (lo, hi), color=colors[below], alpha=0.12, linewidth=0, zorder=0)

    anchor = lo + 0.75 * span
    halo = [withStroke(linewidth=2.0, foreground="white")]
    for sim, direction, corner in ((above, 1, (0.03, 0.97)), (below, -1, (0.97, 0.03))):
        arrow = direction * 0.09 * span
        ax.annotate(
            "", xy=(anchor - arrow, anchor + arrow), xytext=(anchor, anchor),
            arrowprops=dict(arrowstyle="->", color=colors[sim], linewidth=0.9, shrinkA=1,
                            shrinkB=0, path_effects=halo),
            zorder=4,
        )
        ax.text(
            *corner, f"{simulator_labels[sim]} is more diverse", transform=ax.transAxes,
            color=colors[sim], ha="left" if direction > 0 else "right",
            va="top" if direction > 0 else "bottom",
            fontsize=6, zorder=4, path_effects=halo,
        )


def plot_case_spread(
    spread_df,
    group_col,
    case_col,
    simulator_order,
    simulator_labels,
    figure_size=(2.6, 2.6),
    save_as=None,
):
    """The within-case persona spread scatter alone, square and stripped to its axes.

    One point per case, the first simulator against the second. Each side of the dotted
    identity line is tinted in that simulator's colour and labelled with an arrow off
    the diagonal, so the direction of the comparison reads without a caption.
    """
    from patient_simulator.misc.plotting import save_to_figures

    colors = _simulator_colors(simulator_order)

    with plt.rc_context(_COLUMN_RC):
        fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
        _spread_scatter(
            ax, spread_df, group_col, case_col, simulator_order, point_color="0.65",
            marker_size=18, equal_aspect=True, line_style=":",
        )
        _diagonal_regions(ax, colors, simulator_order, simulator_labels)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.set_xlabel(f"{simulator_labels[simulator_order[1]]} persona spread")
        ax.set_ylabel(f"{simulator_labels[simulator_order[0]]} persona spread")

        if save_as:
            save_to_figures(fig, save_as)

    plt.show()
    return fig


def plot_persona_diversity_compact(
    spread_df,
    displacement_df,
    overlap_df,
    group_col,
    persona_col,
    case_col,
    simulator_order,
    persona_order,
    simulator_labels,
    n_gram,
    figure_size=(3.25, 3.9),
    save_as=None,
):
    r"""The same three measurements laid out for one column of a two-column page.

    Written at final print width, so it is included at ``\columnwidth`` without scaling.
    Top: per-case within-case persona spread against the identity line. Bottom left:
    displacement from the neutral persona. Bottom right: cross-case n-gram overlap,
    the templatedness control on the two panels above it.
    """
    from patient_simulator.misc.plotting import save_to_figures

    colors = _simulator_colors(simulator_order)

    with plt.rc_context(_COLUMN_RC):
        fig = plt.figure(figsize=figure_size, layout="constrained")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1])
        ax_spread = fig.add_subplot(grid[0, :])
        ax_displacement = fig.add_subplot(grid[1, 0])
        ax_overlap = fig.add_subplot(grid[1, 1], sharey=ax_displacement)

        wide = _spread_scatter(
            ax_spread, spread_df, group_col, case_col, simulator_order,
            point_color=colors[simulator_order[0]], marker_size=14, equal_aspect=False,
            line_style="--",
        )
        primary, comparison = (wide[sim] for sim in simulator_order)
        ax_spread.set_xlabel(f"{simulator_labels[simulator_order[1]]} spread")
        ax_spread.set_ylabel(f"{simulator_labels[simulator_order[0]]} spread")
        ax_spread.set_title("(a) Within-case persona spread")
        ax_spread.text(
            0.97, 0.06,
            f"mean {primary.mean():.3f} vs {comparison.mean():.3f}\n"
            f"{(primary > comparison).sum()} of {len(wide)} cases above the line",
            transform=ax_spread.transAxes, ha="right", va="bottom", linespacing=1.4,
        )

        _persona_dumbbells(
            ax_displacement,
            _persona_means(displacement_df, group_col, persona_col, "displacement",
                           simulator_order, persona_order),
            simulator_order, colors, marker_size=14,
        )
        ax_displacement.set_xlabel("Distance from Default")
        ax_displacement.set_title("(b) Persona displacement")

        _persona_dumbbells(
            ax_overlap,
            _persona_means(overlap_df, group_col, persona_col, "ngram_overlap",
                           simulator_order, persona_order),
            simulator_order, colors, marker_size=14,
        )
        ax_overlap.set_xlabel(f"Cross-case {n_gram}-gram overlap")
        ax_overlap.set_title("(c) Templatedness")
        ax_overlap.tick_params(labelleft=False)

        handles = [mpatches.Patch(facecolor=colors[s], edgecolor="black", linewidth=0.5,
                                  label=simulator_labels[s]) for s in simulator_order]
        fig.legend(handles=handles, loc="outside upper center", ncols=len(simulator_order),
                   frameon=False, handlelength=1.2, columnspacing=1.5, handletextpad=0.5)

        if save_as:
            save_to_figures(fig, save_as)

    plt.show()
    return fig
