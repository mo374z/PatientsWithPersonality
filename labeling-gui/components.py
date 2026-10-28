from dash import html

REASONS = [
    "medically_implausible",
    "non_response",
    "linguistically_unnatural",
    "inconsistent_with_prior_context",
    "too_informative",
    "uninformative",
    "other",
]

SPEAKER_LABELS = {"doctor": "👨‍⚕️ Doctor", "patient": "🧑‍🦱 Patient"}

REASON_LABELS = {
    "medically_implausible": "Medically implausible",
    "non_response": "Non-response",
    "linguistically_unnatural": "Linguistically unnatural",
    "inconsistent_with_prior_context": "Inconsistent with context",
    "too_informative": "Too informative",
    "uninformative": "Uninformative",
    "other": "Other",
}


def render_patient_info(conversation_id: str, summary: str) -> html.Div:
    if not summary:
        return html.Div()
    return html.Div(
        [
            html.Span("🩺 ", style={"fontSize": "16px"}),
            html.Strong(conversation_id, style={"marginRight": "8px"}),
            html.Span(summary, style={"color": "#555"}),
        ],
        className="patient-info-bar",
    )


def render_turn(text: str, speaker: str) -> html.Div:
    return html.Div(
        [
            html.Div(SPEAKER_LABELS[speaker], className="turn-speaker"),
            html.Div(text, className="turn-bubble"),
        ],
        className=f"turn turn-{speaker}",
    )


def render_conversation(turns: list[dict], id_prefix: str = "") -> html.Div:
    return html.Div(
        [
            render_turn(turn[speaker], speaker)
            for turn in turns
            for speaker in SPEAKER_LABELS
        ],
        className="conversation-container",
        id=f"{id_prefix}conversation",
    )


def render_thank_you() -> html.Div:
    return html.Div(
        html.Div(
            [
                html.Div("🎉", style={"fontSize": "56px", "marginBottom": "12px"}),
                html.H2(
                    "Thank you for your time!",
                    style={"margin": "0 0 10px 0", "color": "#2c3e50"},
                ),
                html.P(
                    "Both tasks are complete and your labels have been saved. "
                    "You may now close this window.",
                    style={"fontSize": "15px", "color": "#555", "margin": 0},
                ),
            ],
            className="thank-you-card",
        ),
        className="thank-you-overlay",
    )


def render_task_header(current: int, total: int, task_label: str) -> html.Div:
    return html.Div(
        [
            html.Span(
                f"{task_label} — {current + 1} of {total}", className="task-counter"
            ),
            html.Div(
                [
                    html.Button("← Prev  [←]", id="btn-prev", n_clicks=0),
                    html.Button("Next →  [→]", id="btn-next", n_clicks=0),
                ],
                className="nav-buttons",
            ),
        ],
        className="task-header",
    )


def render_progress_bar(completed: int, total: int) -> html.Div:
    pct = (completed / total * 100) if total > 0 else 0
    return html.Div(
        [
            html.Div(
                style={
                    "width": f"{pct}%",
                    "height": "4px",
                    "background": "#27ae60" if pct == 100 else "#3498db",
                    "borderRadius": "2px",
                    "transition": "width 0.3s",
                }
            ),
        ],
        style={
            "background": "#34495e",
            "borderRadius": "2px",
            "marginTop": "4px",
            "marginBottom": "4px",
        },
    )
