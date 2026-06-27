from datetime import datetime

from dash import html, dcc, Input, Output, State, callback, no_update, ctx

import data
import storage
import assignments
from components import render_conversation, render_task_header, render_patient_info

_NAV_WARNING = "Please read the conversation and submit your assessment before moving to the next one."
_WARNING_STYLE = {"color": "#e74c3c", "fontSize": "12px", "marginTop": "6px"}


@callback(
    Output("personality-index", "data", allow_duplicate=True),
    Output("personality-submit-trigger", "data", allow_duplicate=True),
    Output("personality-nav-warning", "children", allow_duplicate=True),
    Input("keypress-value", "value"),
    State("url", "pathname"),
    State("personality-index", "data"),
    State("personality-user", "data"),
    State("personality-submit-trigger", "data"),
    prevent_initial_call=True,
)
def keyboard_personality(key_val, pathname, idx, user_id, trigger):
    if pathname != "/personality" or not key_val or user_id is None:
        return no_update, no_update, no_update

    key = key_val.split(":")[0]

    if key == "ArrowLeft":
        return max(0, idx - 1), no_update, ""
    if key == "ArrowRight":
        task_data = assignments.get_personality_assignments(user_id)
        labels = task_data.get("labels", {})
        task_id = task_data["order"][idx]
        if task_id not in labels:
            return no_update, no_update, _NAV_WARNING
        return min(len(task_data["order"]) - 1, idx + 1), no_update, ""
    if key == "Enter":
        return no_update, (trigger or 0) + 1, no_update

    return no_update, no_update, no_update


HEXACO_DIMS = [
    (
        "h",
        "Honesty / Disclosure (H)",
        [
            (1, "Transparent & open"),
            (2, "Mostly truthful but hesitant"),
            (3, "Conceals or distorts"),
        ],
    ),
    (
        "e",
        "Emotional State (E)",
        [
            (1, "Calm & steady"),
            (2, "Anxious, reassurance-seeking"),
            (3, "Marked distress"),
        ],
    ),
    (
        "x",
        "Extraversion (X)",
        [
            (1, "Minimal, terse"),
            (2, "Concise but complete"),
            (3, "Talkative with anecdotes"),
        ],
    ),
    (
        "a",
        "Agreeableness (A)",
        [
            (1, "Cooperative & trusting"),
            (2, "Guarded & cautious"),
            (3, "Frustrated, confrontational"),
        ],
    ),
    (
        "c",
        "Conscientiousness (C)",
        [
            (1, "Precise information recall"),
            (2, "Approximate, fuzzy"),
            (3, "Disorganized, uncertain"),
        ],
    ),
    (
        "o",
        "Openness / Prior Beliefs (O)",
        [
            (1, "Open to explanations"),
            (2, "Mild skepticism"),
            (3, "Dogmatic, dismisses alternatives"),
        ],
    ),
]


def layout(user_id: str) -> html.Div:
    task_data = assignments.get_personality_assignments(user_id)
    order = task_data["order"]
    labels = task_data.get("labels", {})

    first_unlabeled = next((i for i, tid in enumerate(order) if tid not in labels), 0)

    return html.Div(
        [
            dcc.Store(id="personality-user", data=user_id),
            dcc.Store(id="personality-index", data=first_unlabeled),
            dcc.Store(id="personality-submit-trigger", data=0),
            html.Div(id="personality-header"),
            html.Div(id="personality-patient-info"),
            html.Div(id="personality-body"),
            html.Div(id="personality-status"),
            html.Div(id="personality-nav-warning", style=_WARNING_STYLE),
            html.Div(id="personality-complete-overlay"),
        ]
    )


def _thank_you_overlay() -> html.Div:
    return html.Div(
        html.Div(
            [
                html.Div("🎉", style={"fontSize": "56px", "marginBottom": "12px"}),
                html.H2(
                    "Thank you for your time!",
                    style={"margin": "0 0 10px 0", "color": "#2c3e50"},
                ),
                html.P(
                    "Your labels have been saved. You may now close this window.",
                    style={"fontSize": "15px", "color": "#555", "margin": 0},
                ),
            ],
            style={
                "background": "#fff",
                "padding": "40px 48px",
                "borderRadius": "12px",
                "boxShadow": "0 10px 40px rgba(0,0,0,0.25)",
                "textAlign": "center",
                "maxWidth": "420px",
            },
        ),
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "right": 0,
            "bottom": 0,
            "background": "rgba(44, 62, 80, 0.55)",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "zIndex": 1000,
        },
    )


@callback(
    Output("personality-header", "children"),
    Output("personality-patient-info", "children"),
    Output("personality-body", "children"),
    Output("personality-status", "children"),
    Output("personality-nav-warning", "children"),
    Input("personality-index", "data"),
    State("personality-user", "data"),
)
def render_personality(idx, user_id):
    if user_id is None:
        return no_update, no_update, no_update, no_update, no_update

    task_data = assignments.get_personality_assignments(user_id)
    order = task_data["order"]
    labels = task_data.get("labels", {})

    if not order:
        return (
            html.Div("No personality evaluation tasks available."),
            html.Div(),
            html.Div(),
            html.Div(),
            "",
        )

    if idx is None or idx < 0:
        idx = 0
    if idx >= len(order):
        idx = len(order) - 1

    task_id = order[idx]
    assignment = task_data["task_assignments"][task_id]
    conv_id = assignment["conv_id"]
    turns = data.load_conversation(assignment["simulator"], conv_id)
    turns, offset = data.select_turn_window(turns, f"{user_id}:{conv_id}")

    header = render_task_header(idx, len(order), "Personality Evaluation")
    patient_info = render_patient_info(conv_id, data.get_patient_summary(conv_id))
    conversation = render_conversation(turns, id_prefix="pers-", turn_offset=offset)

    existing = labels.get(task_id, {})

    hexaco_items = []
    for dim_key, dim_label, levels in HEXACO_DIMS:
        hexaco_items.append(
            html.Div(
                [
                    html.Label(dim_label),
                    dcc.RadioItems(
                        id=f"personality-{dim_key}",
                        options=[{"label": " Unclear", "value": 0}]
                        + [
                            {"label": f" {val}: {desc}", "value": val}
                            for val, desc in levels
                        ],
                        value=existing.get(dim_key, 0),
                        inline=False,
                        labelStyle={
                            "marginRight": "10px",
                            "cursor": "pointer",
                            "fontSize": "13px",
                            "display": "inline",
                        },
                    ),
                ],
                className="hexaco-item",
            )
        )

    assessment = html.Div(
        [
            html.H4("Recover Personality Parameters"),
            html.P(
                "Estimate the HEXACO parameters used to generate this conversation.",
                style={"fontSize": "13px", "color": "#7f8c8d", "marginBottom": "16px"},
            ),
            html.Div(hexaco_items, className="hexaco-grid"),
            html.Button(
                "Submit & Next  [Enter]",
                id="personality-submit",
                n_clicks=0,
                style={
                    "padding": "8px 20px",
                    "background": "#27ae60",
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "marginTop": "16px",
                    "width": "100%",
                },
            ),
            html.Div(
                id="personality-submit-error",
                style={"color": "#e74c3c", "fontSize": "12px", "marginTop": "6px"},
            ),
        ],
        className="controls-panel",
    )

    body = html.Div(
        [
            html.Div(conversation, className="labeling-chat"),
            html.Div(assessment, className="labeling-assessment"),
        ],
        className="labeling-layout",
    )

    status = ""
    if task_id in labels:
        status = html.Span("✓ Submitted", className="submitted-badge")

    return header, patient_info, body, status, ""


@callback(
    Output("personality-index", "data", allow_duplicate=True),
    Output("personality-nav-warning", "children", allow_duplicate=True),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    State("personality-index", "data"),
    State("personality-user", "data"),
    prevent_initial_call=True,
)
def navigate_personality(prev_clicks, next_clicks, idx, user_id):
    if user_id is None:
        return no_update, no_update

    triggered = ctx.triggered_id
    if triggered == "btn-prev" and not prev_clicks:
        return no_update, no_update
    if triggered == "btn-next" and not next_clicks:
        return no_update, no_update

    task_data = assignments.get_personality_assignments(user_id)
    labels = task_data.get("labels", {})
    total = len(task_data["order"])

    if triggered == "btn-prev":
        return max(0, idx - 1), ""
    if triggered == "btn-next":
        task_id = task_data["order"][idx]
        if task_id not in labels:
            return no_update, _NAV_WARNING
        return min(total - 1, idx + 1), ""
    return no_update, no_update


@callback(
    Output("personality-index", "data", allow_duplicate=True),
    Output("personality-status", "children", allow_duplicate=True),
    Output("personality-submit-error", "children"),
    Output("sidebar-trigger", "data", allow_duplicate=True),
    Output("personality-complete-overlay", "children"),
    Input("personality-submit", "n_clicks"),
    Input("personality-submit-trigger", "data"),
    State("personality-h", "value"),
    State("personality-e", "value"),
    State("personality-x", "value"),
    State("personality-a", "value"),
    State("personality-c", "value"),
    State("personality-o", "value"),
    State("personality-index", "data"),
    State("personality-user", "data"),
    prevent_initial_call=True,
)
def submit_personality(n_clicks, trigger, h, e, x, a, c, o, idx, user_id):
    if ctx.triggered_id == "personality-submit" and not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
    if ctx.triggered_id == "personality-submit-trigger" and not trigger:
        return no_update, no_update, no_update, no_update, no_update
    if user_id is None:
        return no_update, no_update, no_update, no_update, no_update

    params = {"h": h, "e": e, "x": x, "a": a, "c": c, "o": o}
    missing = [k for k, v in params.items() if v is None]
    if missing:
        return (
            no_update,
            no_update,
            "Please fill in all fields before submitting.",
            no_update,
            no_update,
        )

    task_data = assignments.get_personality_assignments(user_id)
    order = task_data["order"]
    task_id = order[idx]

    params["timestamp"] = datetime.now().isoformat()
    task_data.setdefault("labels", {})[task_id] = params
    storage.save_labels(user_id, "personality", task_data)

    is_last = idx == len(order) - 1
    next_idx = min(idx + 1, len(order) - 1)
    return (
        next_idx,
        html.Span("✓ Saved", className="submitted-badge"),
        "",
        datetime.now().isoformat(),
        _thank_you_overlay() if is_last else no_update,
    )
