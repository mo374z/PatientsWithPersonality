from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).parent.parent
BASE_RESULTS_DIR = PROJECT_ROOT / "results"
TURN_METRIC_COLUMNS = {
    "relevance_category",
    "realism_content_category",
    "realism_style_category",
    "token_ratio_patient_real_sim",
    "sentiment_neg_real",
    "sentiment_neg_sim",
}


def _available_results_dirs() -> list[Path]:
    if not BASE_RESULTS_DIR.exists():
        return []

    dirs = [
        d
        for d in sorted(BASE_RESULTS_DIR.iterdir())
        if d.is_dir() and (d / "all_conversations.csv").exists()
    ]

    if (BASE_RESULTS_DIR / "all_conversations.csv").exists():
        dirs = [BASE_RESULTS_DIR] + dirs

    return dirs


def _ensure_path_metadata(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["path"].astype(str).str.split("/")

    if "patient_name" not in df.columns:
        df["patient_name"] = parts.str[-3]
    if "model_name" not in df.columns:
        df["model_name"] = parts.str[-2]
    if "conversation_name" not in df.columns:
        df["conversation_name"] = parts.str[-1]
    if "patient_type" not in df.columns:
        df["patient_type"] = df["patient_name"].astype(str).str.split("_").str[0]

    return df


def load_all_conversations(results_dir: Path):
    csv_path = results_dir / "all_conversations.csv"
    if not csv_path.exists():
        return None
    return _ensure_path_metadata(pd.read_csv(csv_path))


def load_conversation_turns(path: str):
    csv_path = PROJECT_ROOT / path / "turns.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _run_label(row: pd.Series) -> str:
    patient_name = row.get("patient_name", row.get("patient_type", "Unknown"))
    model_name = row.get("model_name", "Unknown")
    return f"{patient_name} | {model_name}"


def _has_turn_metrics(turn_data: pd.Series) -> bool:
    return TURN_METRIC_COLUMNS.issubset(set(turn_data.index))


def _has_turn_field(turn_data: pd.Series, key: str) -> bool:
    return key in turn_data.index and pd.notna(turn_data[key])


def display_turn_metrics(turn_data: pd.Series):
    st.markdown(
        """
            <style>
            div[data-testid="stMetricValue"] {
                font-size: 1rem !important;
            }
            div[data-testid="stMetricLabel"] {
                font-size: 0.75rem !important;
            }
            div[data-testid="stMetricDelta"] {
                font-size: 0.7rem !important;
            }
            </style>
            """,
        unsafe_allow_html=True,
    )

    if _has_turn_field(turn_data, "relevance_category"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Relevance", turn_data["relevance_category"])
        with col2:
            if _has_turn_field(turn_data, "relevance_explanation"):
                with st.expander("Explanation"):
                    st.caption(turn_data["relevance_explanation"])

    if _has_turn_field(turn_data, "realism_content_category"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Realism (Content)", turn_data["realism_content_category"])
        with col2:
            if _has_turn_field(turn_data, "realism_content_explanation"):
                with st.expander("Explanation"):
                    st.caption(turn_data["realism_content_explanation"])

    if _has_turn_field(turn_data, "realism_style_category"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Realism (Style)", turn_data["realism_style_category"])
        with col2:
            if _has_turn_field(turn_data, "realism_style_explanation"):
                with st.expander("Explanation"):
                    st.caption(turn_data["realism_style_explanation"])

    if _has_turn_field(turn_data, "token_ratio_patient_real_sim"):
        st.markdown("<small><b>Token Ratios:</b></small>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Patient Real/Sim", f"{turn_data['token_ratio_patient_real_sim']:.3f}"
            )
        with cols[1]:
            st.metric(
                "Doctor/Patient (Real)",
                f"{turn_data['token_ratio_doctor_patient_real']:.3f}",
            )
        with cols[2]:
            st.metric(
                "Doctor/Patient (Sim)",
                f"{turn_data['token_ratio_doctor_patient_sim']:.3f}",
            )

    if _has_turn_field(turn_data, "sentiment_neg_real"):
        st.markdown("<small><b>Sentiment Analysis:</b></small>", unsafe_allow_html=True)
        sentiment_cols = st.columns(2)

        with sentiment_cols[0]:
            st.markdown("<small><i>Real Response</i></small>", unsafe_allow_html=True)
            sent_cols = st.columns(3)
            with sent_cols[0]:
                st.metric("Neg", f"{turn_data['sentiment_neg_real']:.3f}")
            with sent_cols[1]:
                st.metric("Neu", f"{turn_data['sentiment_neu_real']:.3f}")
            with sent_cols[2]:
                st.metric("Pos", f"{turn_data['sentiment_pos_real']:.3f}")

        with sentiment_cols[1]:
            st.markdown(
                "<small><i>Simulated Response</i></small>", unsafe_allow_html=True
            )
            sent_cols = st.columns(3)
            with sent_cols[0]:
                st.metric("Neg", f"{turn_data['sentiment_neg_sim']:.3f}")
            with sent_cols[1]:
                st.metric("Neu", f"{turn_data['sentiment_neu_sim']:.3f}")
            with sent_cols[2]:
                st.metric("Pos", f"{turn_data['sentiment_pos_sim']:.3f}")


def display_overall_metrics(metrics: pd.Series):
    if pd.notna(metrics.get("total_steps")):
        st.markdown("**Total Turns:** " + str(int(metrics["total_steps"])))
    if pd.notna(metrics.get("persona_consistency")):
        st.markdown(f"**Persona Consistency:** {metrics['persona_consistency']}")
    if pd.notna(metrics.get("persona_consistency_explanation")):
        with st.expander("Explanation"):
            st.write(metrics["persona_consistency_explanation"])

    if pd.notna(metrics.get("realism_content_similarity")) and pd.notna(
        metrics.get("realism_style_similarity")
    ):
        st.markdown("**Realism:**")
        cols = st.columns(2)
        with cols[0]:
            st.metric(
                "Content Similarity", f"{metrics['realism_content_similarity']:.3f}"
            )
        with cols[1]:
            st.metric("Style Similarity", f"{metrics['realism_style_similarity']:.3f}")

    if pd.notna(metrics.get("mean_token_ratio_patient_real_sim")):
        st.markdown("**Mean Token Ratios:**")
        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Patient Real/Sim",
                f"{metrics['mean_token_ratio_patient_real_sim']:.3f}",
            )
        with cols[1]:
            st.metric(
                "Doctor/Patient (Real)",
                f"{metrics['mean_token_ratio_doctor_patient_real']:.3f}",
            )
        with cols[2]:
            st.metric(
                "Doctor/Patient (Sim)",
                f"{metrics['mean_token_ratio_doctor_patient_sim']:.3f}",
            )

    if pd.notna(metrics.get("domain_term_count_real")):
        st.markdown("**Domain Terms:**")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Real", int(metrics["domain_term_count_real"]))
        with cols[1]:
            st.metric("Sim", int(metrics["domain_term_count_sim"]))
        with cols[2]:
            st.metric("Domain Term Ratio", f"{metrics['domain_term_ratio']:.3f}")

    if pd.notna(metrics.get("mean_sentiment_neg_real")):
        st.markdown("**Mean Sentiment:**")
        st.markdown("*Real*")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Neg", f"{metrics['mean_sentiment_neg_real']:.3f}")
        with cols[1]:
            st.metric("Neu", f"{metrics['mean_sentiment_neu_real']:.3f}")
        with cols[2]:
            st.metric("Pos", f"{metrics['mean_sentiment_pos_real']:.3f}")

        st.markdown("*Simulated*")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Neg", f"{metrics['mean_sentiment_neg_sim']:.3f}")
        with cols[1]:
            st.metric("Neu", f"{metrics['mean_sentiment_neu_sim']:.3f}")
        with cols[2]:
            st.metric("Pos", f"{metrics['mean_sentiment_pos_sim']:.3f}")

    if pd.notna(metrics.get("readability_score_real")):
        st.markdown("**Readability & Diversity:**")
        cols = st.columns(2)
        with cols[0]:
            st.metric("Read. (Real)", f"{metrics['readability_score_real']:.2f}")
            st.metric("Lex. Div. (Real)", f"{metrics['lexical_diversity_real']:.3f}")
        with cols[1]:
            st.metric("Read. (Sim)", f"{metrics['readability_score_sim']:.2f}")
            st.metric("Lex. Div. (Sim)", f"{metrics['lexical_diversity_sim']:.3f}")

    if pd.notna(metrics.get("param_bias")):
        st.markdown("---")
        st.markdown("**Parameters:**")
        st.write(f"Bias: {metrics['param_bias']}")

    if pd.notna(metrics.get("param_cefr_type")):
        st.write(f"CEFR: {metrics['param_cefr_type']}")
        st.write(f"Personality: {metrics['param_personality_type']}")
        st.write(f"Recall: {metrics['param_recall_level_type']}")
        st.write(f"Dazed: {metrics['param_dazed_level_type']}")


def main():
    st.set_page_config(page_title="Evaluation Explorer", page_icon="📊", layout="wide")

    st.title("📊 Evaluation Results Explorer")
    st.markdown("Explore turn-by-turn evaluation results for patient simulations")

    available_dirs = _available_results_dirs()
    if not available_dirs:
        st.error(
            f"No result directories with all_conversations.csv found in {BASE_RESULTS_DIR}"
        )
        return

    dir_labels = [d.relative_to(PROJECT_ROOT).as_posix() for d in available_dirs]
    selected_dir_label = st.sidebar.selectbox(
        "Results Subdirectory",
        dir_labels,
        index=0,
    )
    selected_dir_index = dir_labels.index(selected_dir_label)
    results_dir = available_dirs[selected_dir_index]
    st.caption(f"Results directory: {results_dir.relative_to(PROJECT_ROOT).as_posix()}")

    all_conversations_df = load_all_conversations(results_dir)
    if all_conversations_df is None:
        st.error(f"Could not load all_conversations.csv from {results_dir}")
        return

    st.sidebar.header("Selection")

    conversations = sorted(all_conversations_df["conversation_name"].dropna().unique())
    selected_conversation = st.sidebar.selectbox(
        "Conversation", conversations, index=0 if conversations else None
    )
    if not selected_conversation:
        st.warning("No conversations found")
        return

    conversation_df = all_conversations_df[
        all_conversations_df["conversation_name"] == selected_conversation
    ].copy()
    if conversation_df.empty:
        st.warning("No runs found for selected conversation")
        return

    conversation_df["run_label"] = conversation_df.apply(_run_label, axis=1)
    seen_labels: dict[str, int] = {}
    unique_labels: list[str] = []
    for label in conversation_df["run_label"]:
        seen_labels[label] = seen_labels.get(label, 0) + 1
        if seen_labels[label] == 1:
            unique_labels.append(label)
        else:
            unique_labels.append(f"{label} [{seen_labels[label]}]")
    conversation_df["run_label"] = unique_labels

    run_options = conversation_df["run_label"].tolist()
    default_runs = run_options[: min(2, len(run_options))]
    selected_runs = st.sidebar.multiselect(
        "Generated Answers (up to 3)",
        options=run_options,
        default=default_runs,
        max_selections=3,
    )
    if not selected_runs:
        st.warning("Select at least one generated answer")
        return

    selected_rows = conversation_df[conversation_df["run_label"].isin(selected_runs)]
    selected_rows = (
        selected_rows.set_index("run_label").loc[selected_runs].reset_index()
    )

    turns_by_run: dict[str, pd.DataFrame] = {}
    for _, run_row in selected_rows.iterrows():
        turns_df = load_conversation_turns(run_row["path"])
        if turns_df is None:
            st.error(f"Could not load conversation data for {run_row['run_label']}")
            return
        turns_by_run[run_row["run_label"]] = turns_df

    st.sidebar.header("Overall Metrics")
    for _, run_row in selected_rows.iterrows():
        with st.sidebar.expander(run_row["run_label"], expanded=False):
            display_overall_metrics(run_row)

    st.markdown(f"## {selected_conversation}")
    st.markdown("Compared generated answers:")
    for run_label in selected_runs:
        st.markdown(f"- {run_label}")

    min_turns = min(len(df) for df in turns_by_run.values())

    for idx in range(min_turns):
        cols = st.columns([0.3, 3, 3] + [3] * len(selected_runs))

        with cols[0]:
            st.markdown(f"#### {idx + 1})")

        first_turn = turns_by_run[selected_runs[0]].iloc[idx]
        with cols[1]:
            st.markdown("👨‍⚕️ **Doctor's Question**")
            st.info(first_turn.get("doctor_question", ""))

        with cols[2]:
            st.markdown("🧑 **Real Response**")
            st.success(first_turn.get("real_response", ""))

        for col_idx, run_label in enumerate(selected_runs, start=3):
            turn_data = turns_by_run[run_label].iloc[idx]
            with cols[col_idx]:
                st.markdown(f"🤖 **{run_label}**")
                st.warning(turn_data.get("simulated_response", ""))

        with st.expander("📈 View Turn Metrics", expanded=False):
            if len(selected_runs) == 1:
                turn_data = turns_by_run[selected_runs[0]].iloc[idx]
                if _has_turn_metrics(turn_data):
                    display_turn_metrics(turn_data)
                else:
                    st.info("No turn-level evaluation metrics available for this run.")
            else:
                tabs = st.tabs(selected_runs)
                for tab, run_label in zip(tabs, selected_runs):
                    with tab:
                        turn_data = turns_by_run[run_label].iloc[idx]
                        if _has_turn_metrics(turn_data):
                            display_turn_metrics(turn_data)
                        else:
                            st.info(
                                "No turn-level evaluation metrics available for this run."
                            )


if __name__ == "__main__":
    main()
