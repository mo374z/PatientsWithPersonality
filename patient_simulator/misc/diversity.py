"""Conversation-embedding diversity analysis utilities."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


def embed_conversations(
    df,
    text_col="simulated_response",
    turns_filename="turns.csv",
    model_name="BAAI/bge-large-en-v1.5",
    device=None,
    batch_size=8,
):
    """Embed each conversation in ``df`` as a single vector.

    For every row, read ``{row.path}/{turns_filename}``, concatenate all
    ``text_col`` turns into one string, and encode with a SentenceTransformer.
    """
    if "path" not in df.columns:
        raise ValueError(
            "df must contain a 'path' column pointing to conversation dirs"
        )

    texts = []
    for path in df["path"]:
        turns_path = os.path.join(path, turns_filename)
        if not os.path.isfile(turns_path):
            raise FileNotFoundError(f"Missing turns file: {turns_path}")
        turns = pd.read_csv(turns_path)
        if text_col not in turns.columns:
            raise ValueError(f"Column '{text_col}' not in {turns_path}")
        texts.append("\n".join(turns[text_col].dropna().astype(str).tolist()))

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def pca_nd(embeddings, n_components=2):
    """Standardize then project embeddings to ``n_components`` principal components."""
    scaled = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(scaled)
    return pcs, pca.explained_variance_ratio_


def pca_2d(embeddings):
    return pca_nd(embeddings, n_components=2)


def plot_conversation_pca(
    df,
    pcs,
    hue_col="persona",
    style_col="simulator",
    palette=None,
    hull_col=None,
    hull_palette=None,
    hull_alpha=0.12,
    reduce_to_centroids=False,
    centroid_group_cols=None,
    marker_size=220,
    alpha=0.85,
    figure_size=(11, 7),
    title="Conversation diversity (2-PC projection)",
    ax=None,
):
    """Scatter the 2-PC projection.

    Points are colored by ``hue_col`` and shaped by ``style_col``.
    If ``hull_col`` is set, filled convex hulls show the coverage region of each
    group (e.g. per simulator). Set ``reduce_to_centroids=True`` to plot
    per-(hue, style) means instead of raw points, reducing visual clutter.
    """
    import matplotlib.patches as mpatches

    if len(df) != len(pcs):
        raise ValueError("df and pcs must have the same length")

    plot_df = df.copy().reset_index(drop=True)
    plot_df["pc1"] = pcs[:, 0]
    plot_df["pc2"] = pcs[:, 1]

    if ax is None:
        _, ax = plt.subplots(figsize=figure_size)

    # Draw filled convex hulls below scatter points
    hull_handles, hull_labels = [], []
    if hull_col is not None:
        hull_groups = list(dict.fromkeys(plot_df[hull_col]))
        if hull_palette is None:
            hull_palette = dict(
                zip(hull_groups, sns.color_palette("muted", n_colors=len(hull_groups)))
            )
        elif not isinstance(hull_palette, dict):
            hull_palette = dict(zip(hull_groups, hull_palette))
        for group in hull_groups:
            pts = plot_df.loc[plot_df[hull_col] == group, ["pc1", "pc2"]].to_numpy()
            if len(pts) < 3:
                continue
            hull = ConvexHull(pts)
            color = hull_palette[group]
            poly = mpatches.Polygon(
                pts[hull.vertices],
                closed=True,
                facecolor=color,
                alpha=hull_alpha,
                edgecolor=color,
                linewidth=1.8,
                linestyle="--",
            )
            ax.add_patch(poly)
            hull_handles.append(
                mpatches.Patch(
                    facecolor=color,
                    alpha=0.35,
                    edgecolor=color,
                    linewidth=1.5,
                    linestyle="--",
                )
            )
            hull_labels.append(group)

    # Optionally aggregate to per-group centroids before scattering
    if reduce_to_centroids:
        group_cols = centroid_group_cols or [
            c for c in [hue_col, style_col] if c is not None
        ]
        scatter_df = (
            plot_df.groupby(group_cols, observed=True)[["pc1", "pc2"]]
            .mean()
            .reset_index()
        )
    else:
        scatter_df = plot_df

    sns.scatterplot(
        data=scatter_df,
        x="pc1",
        y="pc2",
        hue=hue_col,
        style=style_col,
        palette=palette,
        s=marker_size,
        alpha=alpha,
        ax=ax,
        edgecolor="black",
        linewidth=0.6,
    )

    scatter_handles, scatter_labels = ax.get_legend_handles_labels()
    ax.legend(
        scatter_handles + hull_handles,
        scatter_labels + hull_labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=9,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.show()


def plot_conversation_pca_3d(
    df,
    pcs,
    hue_col="persona",
    style_col="simulator",
    palette=None,
    marker_size=160,
    alpha=0.55,
    figure_size=(12, 9),
    elev=22,
    azim=-60,
    title="Conversation diversity (3-PC projection)",
):
    """3D scatter of the first three principal components."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

    if pcs.shape[1] < 3:
        raise ValueError(f"pcs must have >= 3 columns, got {pcs.shape[1]}")
    if len(df) != len(pcs):
        raise ValueError("df and pcs must have the same length")

    plot_df = df.copy().reset_index(drop=True)
    plot_df["pc1"], plot_df["pc2"], plot_df["pc3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2]

    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection="3d")

    hue_vals = (
        list(plot_df[hue_col].cat.categories)
        if hasattr(plot_df[hue_col], "cat")
        else list(dict.fromkeys(plot_df[hue_col]))
    )
    style_vals = list(dict.fromkeys(plot_df[style_col]))
    marker_shapes = ["o", "^", "s", "D", "v", "P", "X"]
    marker_map = {
        v: marker_shapes[i % len(marker_shapes)] for i, v in enumerate(style_vals)
    }

    for h in hue_vals:
        color = palette[h] if palette else None
        for s in style_vals:
            sub = plot_df[(plot_df[hue_col] == h) & (plot_df[style_col] == s)]
            if sub.empty:
                continue
            ax.scatter(
                sub["pc1"],
                sub["pc2"],
                sub["pc3"],
                c=[color] * len(sub) if color else None,
                marker=marker_map[s],
                s=marker_size,
                alpha=alpha,
                edgecolors="black",
                linewidths=0.3,
                depthshade=False,
            )

    hue_handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=11,
            markerfacecolor=(palette[h] if palette else "gray"),
            markeredgecolor="black",
            label=h,
        )
        for h in hue_vals
    ]
    style_handles = [
        plt.Line2D(
            [],
            [],
            marker=marker_map[s],
            linestyle="",
            markersize=11,
            markerfacecolor="lightgray",
            markeredgecolor="black",
            label=s,
        )
        for s in style_vals
    ]
    ax.legend(
        handles=hue_handles + style_handles,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=9,
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()


def plot_diversity_overview(
    df,
    pcs,
    persona_col="persona",
    simulator_col="simulator",
    palette=None,
    simulator_order=None,
    persona_order=None,
    figure_size=(10, 4),
    alpha_raw=0.25,
    alpha_hull=0.13,
    save_as=None,
):
    """Side-by-side PCA scatter for each simulator.

    Points are colored by persona; convex hulls use PATIENT_TYPE_PALETTE colors
    keyed by simulator name. Both panels share axis limits. Persona legend at the bottom.
    """
    import matplotlib.patches as mpatches
    from patient_simulator.misc.plotting import PATIENT_TYPE_PALETTE, save_to_figures

    if len(df) != len(pcs):
        raise ValueError("df and pcs must have the same length")

    plot_df = df.copy().reset_index(drop=True)
    plot_df["_pc1"] = pcs[:, 0]
    plot_df["_pc2"] = pcs[:, 1]

    if simulator_order is None:
        simulator_order = list(dict.fromkeys(plot_df[simulator_col]))
    if persona_order is None:
        persona_order = (
            list(plot_df[persona_col].cat.categories)
            if hasattr(plot_df[persona_col], "cat")
            else list(dict.fromkeys(plot_df[persona_col]))
        )
    if palette is None:
        palette = dict(
            zip(persona_order, sns.color_palette("tab10", len(persona_order)))
        )

    fig, axes = plt.subplots(1, len(simulator_order), figsize=figure_size)
    if len(simulator_order) == 1:
        axes = [axes]

    for col_i, sim in enumerate(simulator_order):
        ax = axes[col_i]
        sim_df = plot_df[plot_df[simulator_col] == sim]

        for persona in persona_order:
            pts = sim_df.loc[
                sim_df[persona_col] == persona, ["_pc1", "_pc2"]
            ].to_numpy()
            if len(pts) == 0:
                continue
            color = palette[persona]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=[color] * len(pts),
                s=15,
                alpha=alpha_raw,
                zorder=2,
                edgecolor="none",
            )
            centroid = pts.mean(axis=0)
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                c=[color],
                s=120,
                alpha=0.95,
                zorder=4,
                edgecolor="black",
                linewidth=0.8,
            )

        all_pts = sim_df[["_pc1", "_pc2"]].to_numpy()
        if len(all_pts) >= 3:
            hull = ConvexHull(all_pts)
            color = PATIENT_TYPE_PALETTE.get(sim, "#888888")
            poly = mpatches.Polygon(
                all_pts[hull.vertices],
                closed=True,
                facecolor=color,
                alpha=alpha_hull,
                edgecolor=color,
                linewidth=1.5,
                linestyle="--",
            )
            ax.add_patch(poly)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if col_i == 0 else "")

    xlims = [ax.get_xlim() for ax in axes]
    ylims = [ax.get_ylim() for ax in axes]
    for ax in axes:
        ax.set_xlim(min(lim[0] for lim in xlims), max(lim[1] for lim in xlims))
        ax.set_ylim(min(lim[0] for lim in ylims), max(lim[1] for lim in ylims))

    persona_handles = [
        mpatches.Patch(facecolor=palette[p], label=p, edgecolor="black", linewidth=0.4)
        for p in persona_order
    ]
    fig.legend(
        handles=persona_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncols=len(persona_order),
        frameon=False,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)

    if save_as:
        save_to_figures(fig, save_as)

    plt.show()
    return fig


def plot_intra_persona_diversity(
    df,
    pcs,
    persona_col="persona",
    simulator_col="simulator",
    palette=None,
    simulator_order=None,
    persona_order=None,
    figure_size=(11, 4),
    save_as=None,
):
    """Grouped bar chart of mean intra-group pairwise distance per (persona × simulator).

    Uses PATIENT_TYPE_PALETTE for simulator bar colors. No title; legend to the right.
    """
    from patient_simulator.misc.plotting import PATIENT_TYPE_PALETTE, save_to_figures

    if len(df) != len(pcs):
        raise ValueError("df and pcs must have the same length")

    plot_df = df.copy().reset_index(drop=True)
    plot_df["_pc1"] = pcs[:, 0]
    plot_df["_pc2"] = pcs[:, 1]

    if simulator_order is None:
        simulator_order = list(dict.fromkeys(plot_df[simulator_col]))
    if persona_order is None:
        persona_order = (
            list(plot_df[persona_col].cat.categories)
            if hasattr(plot_df[persona_col], "cat")
            else list(dict.fromkeys(plot_df[persona_col]))
        )
    if palette is None:
        palette = dict(
            zip(persona_order, sns.color_palette("tab10", len(persona_order)))
        )

    rows = []
    for sim in simulator_order:
        for persona in persona_order:
            mask = (plot_df[simulator_col] == sim) & (plot_df[persona_col] == persona)
            pts = plot_df.loc[mask, ["_pc1", "_pc2"]].to_numpy()
            if len(pts) < 2:
                continue
            rows.append(
                {
                    "Persona": persona,
                    "Simulator": sim,
                    "dist": float(pdist(pts).mean()),
                }
            )
    stats_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figure_size)
    x = np.arange(len(persona_order))
    width = 0.38

    for i, sim in enumerate(simulator_order):
        sub = (
            stats_df[stats_df["Simulator"] == sim]
            .set_index("Persona")
            .reindex(persona_order)
        )
        color = PATIENT_TYPE_PALETTE.get(sim, "#888888")
        ax.bar(
            x + (i - 0.5) * width,
            sub["dist"].values,
            width,
            color=color,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.7,
            label=sim,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(persona_order, rotation=20, ha="right")
    for label, persona in zip(ax.get_xticklabels(), persona_order):
        label.set_color(palette[persona])
        label.set_fontweight("bold")
    ax.set_ylabel("Mean pairwise distance")
    ax.set_ylim(0, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_as:
        save_to_figures(fig, save_as)

    plt.show()
    return fig


def vendi_score(embeddings):
    """Vendi score for a set of embeddings.

    Expects L2-normalised vectors (cosine similarity used as kernel).
    Score = exp(Shannon entropy of eigenspectrum of the normalised kernel matrix).
    Higher means more diverse.
    """
    n = len(embeddings)
    K = (embeddings @ embeddings.T) / n
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def compute_diversity_stats(pcs, labels, embeddings=None):
    """Per-label mean pairwise Euclidean distance, std, convex hull area, and Vendi score.

    ``pcs`` drives the geometric metrics (pairwise distance, hull area).
    ``embeddings``, when provided, must be L2-normalised and row-aligned with ``pcs``;
    the Vendi score is then computed on the full embedding vectors instead of the PCs.

    ``labels`` may be a 1-D Series/array (single grouping, output column: "label") or a
    DataFrame with multiple columns (multi-level grouping, output columns match DataFrame).
    """
    if isinstance(labels, pd.DataFrame):
        label_df = labels.reset_index(drop=True)
        group_cols = list(label_df.columns)
        groups = label_df.groupby(group_cols, observed=True).groups

        def _row_prefix(key):
            return dict(zip(group_cols, key if isinstance(key, tuple) else [key]))
    else:
        label_series = pd.Series(labels).reset_index(drop=True)
        group_cols = ["label"]
        groups = label_series.groupby(label_series, observed=True).groups

        def _row_prefix(key):
            return {"label": key}

    rows = []
    for key, idx in groups.items():
        idx = list(idx)
        pts = pcs[idx]
        n = len(pts)
        row = _row_prefix(key)
        if n < 2:
            row.update(
                n=n,
                mean_pairwise_dist=np.nan,
                std_pairwise_dist=np.nan,
                hull_area=np.nan,
                vendi_score=np.nan,
            )
        else:
            dists = pdist(pts)
            vs = vendi_score(embeddings[idx]) if embeddings is not None else np.nan
            row.update(
                n=n,
                mean_pairwise_dist=float(dists.mean()),
                std_pairwise_dist=float(dists.std()),
                hull_area=float(ConvexHull(pts).volume) if n >= 3 else np.nan,
                vendi_score=vs,
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("mean_pairwise_dist", ascending=False)
        .reset_index(drop=True)
    )
