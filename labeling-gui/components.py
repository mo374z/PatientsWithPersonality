from dash import html, dcc

FLAG_REASONS = [
    "medically_implausible",
    "non_response",
    "linguistically_unnatural",
    "inconsistent_with_prior_context",
    "too_informative",
    "uninformative",
    "other",
]

FLAG_REASON_LABELS = {
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


def _flag_annotation(abs_idx: int, reasons: list, other_text: str = "") -> html.Div:
    return html.Div(
        [
            html.Span("⚑ ", style={"color": "#e74c3c"}),
            dcc.Checklist(
                id={"type": "flag-reasons", "index": abs_idx},
                options=[
                    {"label": f" {FLAG_REASON_LABELS[r]}", "value": r}
                    for r in FLAG_REASONS
                ],
                value=reasons,
                inline=True,
                labelStyle={
                    "marginRight": "8px",
                    "fontSize": "11px",
                    "cursor": "pointer",
                },
            ),
            dcc.Input(
                id={"type": "flag-other-text", "index": abs_idx},
                type="text",
                value=other_text,
                placeholder="Specify (optional)...",
                debounce=True,
                style={
                    "display": "inline-block" if "other" in reasons else "none",
                    "marginLeft": "8px",
                    "fontSize": "11px",
                    "padding": "2px 6px",
                    "border": "1px solid #bdc3c7",
                    "borderRadius": "4px",
                    "minWidth": "220px",
                },
            ),
        ],
        className="flag-annotation",
    )


def render_conversation(
    turns: list[dict],
    id_prefix: str = "",
    turn_offset: int = 0,
    flags: dict = None,
    other_texts: dict = None,
) -> html.Div:
    header = html.Div(
        [
            html.Div(
                [html.Span("👨‍⚕️", className="chat-icon"), html.Span("Doctor")],
                className="chat-legend-doctor",
            ),
            html.Div(
                [html.Span("Patient"), html.Span("🧑‍🦱", className="chat-icon")],
                className="chat-legend-patient",
            ),
        ],
        className="chat-legend",
    )

    bubbles = []
    for i, turn in enumerate(turns):
        absolute = i + turn_offset
        display_num = absolute + 1

        bubbles.append(
            html.Div(
                [
                    html.Div(f"{display_num}", className="turn-number"),
                    html.Div(turn["doctor"], className="turn-bubble"),
                ],
                className="turn turn-doctor",
            )
        )

        if flags is not None:
            turn_str = str(absolute)
            is_flagged = turn_str in flags
            reasons = flags.get(turn_str, [])

            patient_children = [
                html.Div(turn["patient"], className="turn-bubble"),
                html.Button(
                    "⚑",
                    id={"type": "flag-btn", "index": absolute},
                    className=f"flag-btn {'flagged' if is_flagged else ''}",
                    n_clicks=0,
                ),
            ]
            if is_flagged:
                patient_children.append(
                    _flag_annotation(
                        absolute, reasons, (other_texts or {}).get(turn_str, "")
                    )
                )

            bubbles.append(
                html.Div(
                    patient_children,
                    className=f"turn turn-patient turn-patient-flaggable {'turn-flagged' if is_flagged else ''}",
                )
            )
        else:
            bubbles.append(
                html.Div(
                    [
                        html.Div(turn["patient"], className="turn-bubble"),
                        html.Div(f"{display_num}", className="turn-number"),
                    ],
                    className="turn turn-patient",
                )
            )

    return html.Div(
        [header, *bubbles],
        className="conversation-container",
        id=f"{id_prefix}conversation",
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
