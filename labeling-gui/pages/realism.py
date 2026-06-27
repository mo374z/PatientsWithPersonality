from datetime import datetime

from dash import html, dcc, Input, Output, State, callback, no_update, ctx, ALL

import data
import storage
import assignments
from components import render_conversation, render_task_header, render_patient_info

_NAV_WARNING = "Please read the conversation and submit your assessment before moving to the next one."
_WARNING_STYLE = {"color": "#e74c3c", "fontSize": "12px", "marginTop": "6px"}

RATING_DIMENSIONS = ["symptom_realism", "information_control", "style_realism"]

RATING_TOOLTIPS = {
    "symptom_realism": (
        "Are the symptoms clinically plausible for the condition, and are they "
        "brought up naturally (as a real patient would report them)?"
    ),
    "information_control": (
        "Does the patient share medical knowledge appropriately? Low scores for "
        "volunteering clinical terminology, textbook explanations, or details not "
        "explicitly asked for. High scores for realistic gaps in understanding "
        "and lay language."
    ),
    "style_realism": (
        "Does the patient sound like a real person? Consider language, sentence "
        "structure and conversational naturalness."
    ),
}

RATING_LABELS = {
    "symptom_realism": "Symptom Description Realism",
    "information_control": "Information Control",
    "style_realism": "Style Realism",
}

RATING_SCALES = {
    "symptom_realism": "1 = implausible / textbook, 5 = clinically plausible & natural",
    "information_control": "1 = volunteers clinical detail, 5 = realistic lay knowledge",
    "style_realism": "1 = artificial phrasing, 5 = sounds like a real person",
}


def layout(user_id: str) -> html.Div:
    task_data = assignments.get_realism_assignments(user_id)
    order = task_data["order"]
    labels = task_data.get("labels", {})

    first_unlabeled = next((i for i, tid in enumerate(order) if tid not in labels), 0)

    return html.Div(
        [
            dcc.Store(id="realism-user", data=user_id),
            dcc.Store(id="realism-index", data=first_unlabeled),
            dcc.Store(id="realism-current-task", data=None),
            dcc.Store(id="realism-submit-trigger", data=0),
            dcc.Store(id="realism-flags", data={}),
            dcc.Store(id="realism-flag-other-texts", data={}),
            dcc.Store(id="realism-turn-offset", data=0),
            html.Div(id="realism-header"),
            html.Div(id="realism-patient-info"),
            html.Div(
                [
                    html.Div(id="realism-conversation", className="labeling-chat"),
                    html.Div(id="realism-assessment", className="labeling-assessment"),
                ],
                className="labeling-layout",
            ),
            html.Div(id="realism-status"),
            html.Div(id="realism-nav-warning", style=_WARNING_STYLE),
        ]
    )


def _render_assessment(existing: dict) -> html.Div:
    classification_value = existing.get("classification")
    confidence_value = existing.get("confidence", 3)

    rating_rows = []
    for dim in RATING_DIMENSIONS:
        rating_rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                RATING_LABELS[dim],
                                style={"fontWeight": "600", "fontSize": "13px"},
                            ),
                            html.Div(
                                RATING_TOOLTIPS[dim],
                                style={
                                    "fontSize": "12px",
                                    "color": "#474f50",
                                    "marginTop": "2px",
                                },
                            ),
                        ],
                        style={"marginBottom": "6px"},
                    ),
                    html.Div(
                        RATING_SCALES[dim],
                        style={
                            "fontSize": "12px",
                            "color": "#7f8c8d",
                            "marginBottom": "6px",
                        },
                    ),
                    dcc.Slider(
                        id=f"realism-{dim}",
                        min=1,
                        max=5,
                        step=1,
                        value=existing.get(dim, 3),
                        marks={i: str(i) for i in range(1, 6)},
                    ),
                ],
                style={"marginBottom": "16px"},
            )
        )

    return html.Div(
        [
            html.H4("Your Assessment"),
            html.Div(
                "Does the patient in this conversation feel real or simulated?",
                style={"marginBottom": "12px", "fontSize": "14px", "color": "#555"},
            ),
            dcc.RadioItems(
                id="realism-classification",
                options=[
                    {"label": " Real  [R]", "value": "real"},
                    {"label": " Simulated  [S]", "value": "simulated"},
                ],
                value=classification_value,
                inline=True,
                labelStyle={
                    "marginRight": "20px",
                    "cursor": "pointer",
                    "fontSize": "14px",
                },
                style={"marginBottom": "20px"},
            ),
            html.Div(
                [
                    html.Span(
                        "Confidence  ",
                        style={"fontWeight": "600", "fontSize": "13px"},
                    ),
                    html.Span("[1–5]", style={"fontSize": "11px", "color": "#95a5a6"}),
                ],
                style={"marginBottom": "4px"},
            ),
            html.Div(
                "1 = guessing, 5 = certain",
                style={"fontSize": "12px", "color": "#7f8c8d", "marginBottom": "10px"},
            ),
            dcc.Slider(
                id="realism-confidence",
                min=1,
                max=5,
                step=1,
                value=confidence_value,
                marks={i: str(i) for i in range(1, 6)},
            ),
            *rating_rows,
            html.Div(
                [
                    html.Span(
                        "⚑",
                        style={
                            "marginRight": "6px",
                            "fontSize": "14px",
                            "color": "#B10000",
                        },
                    ),
                    html.Span(
                        "Flag patient turns that led to your assessment",
                        style={
                            "fontSize": "14px",
                            "color": "#000000",
                            "fontWeight": "bold",
                        },
                    ),
                ],
                style={
                    "textAlign": "right",
                    "paddingBottom": "4px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        "Additional comments (optional)",
                        style={
                            "fontSize": "13px",
                            "fontWeight": "600",
                            "marginBottom": "4px",
                        },
                    ),
                    dcc.Textarea(
                        id="realism-comment",
                        value=existing.get("comment", ""),
                        placeholder="Any overall thoughts on this conversation...",
                        style={
                            "width": "100%",
                            "minHeight": "60px",
                            "padding": "8px",
                            "borderRadius": "6px",
                            "border": "1px solid #bdc3c7",
                            "fontSize": "13px",
                            "fontFamily": "inherit",
                            "boxSizing": "border-box",
                            "resize": "vertical",
                        },
                    ),
                ],
                style={"marginTop": "16px"},
            ),
            html.Button(
                "Submit & Next  [Enter]",
                id="realism-submit",
                n_clicks=0,
                style={
                    "padding": "8px 20px",
                    "background": "#27ae60",
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "marginTop": "12px",
                    "width": "100%",
                },
            ),
            html.Div(
                id="realism-submit-error",
                style={"color": "#e74c3c", "fontSize": "12px", "marginTop": "6px"},
            ),
        ],
        className="controls-panel",
    )


@callback(
    Output("realism-header", "children"),
    Output("realism-patient-info", "children"),
    Output("realism-conversation", "children"),
    Output("realism-assessment", "children"),
    Output("realism-status", "children"),
    Output("realism-current-task", "data"),
    Output("realism-flags", "data"),
    Output("realism-flag-other-texts", "data"),
    Output("realism-turn-offset", "data"),
    Output("realism-nav-warning", "children"),
    Input("realism-index", "data"),
    State("realism-user", "data"),
)
def render_realism(idx, user_id):
    if user_id is None:
        return (no_update,) * 10

    task_data = assignments.get_realism_assignments(user_id)
    order = task_data["order"]
    labels = task_data.get("labels", {})

    if idx is None or idx < 0:
        idx = 0
    if idx >= len(order):
        idx = len(order) - 1

    task_id = order[idx]
    assignment = task_data["task_assignments"][task_id]
    conv_id = assignment["conv_id"]
    turns = data.load_source_conversation(assignment["source"], conv_id)
    turns, offset = data.select_turn_window(turns, f"{user_id}:{conv_id}")

    existing = labels.get(task_id, {})
    flags = {
        str(f["turn_idx"]): f["reasons"] for f in existing.get("flagged_turns", [])
    }
    other_texts = {
        str(f["turn_idx"]): f.get("other_text", "")
        for f in existing.get("flagged_turns", [])
        if f.get("other_text")
    }

    header = render_task_header(idx, len(order), "Conversation Realism")
    patient_info = render_patient_info(conv_id, data.get_patient_summary(conv_id))
    conversation = render_conversation(
        turns,
        id_prefix="real-",
        turn_offset=offset,
        flags=flags,
        other_texts=other_texts,
    )
    assessment = _render_assessment(existing)
    status = (
        html.Span("✓ Submitted", className="submitted-badge")
        if task_id in labels
        else ""
    )

    return (
        header,
        patient_info,
        conversation,
        assessment,
        status,
        task_id,
        flags,
        other_texts,
        offset,
        "",
    )


@callback(
    Output("realism-flags", "data", allow_duplicate=True),
    Output("realism-flag-other-texts", "data", allow_duplicate=True),
    Output("realism-conversation", "children", allow_duplicate=True),
    Input({"type": "flag-btn", "index": ALL}, "n_clicks"),
    State("realism-flags", "data"),
    State("realism-flag-other-texts", "data"),
    State("realism-turn-offset", "data"),
    State("realism-current-task", "data"),
    State("realism-user", "data"),
    prevent_initial_call=True,
)
def toggle_flag(clicks, flags, other_texts, offset, current_task_id, user_id):
    triggered = ctx.triggered_id
    if not triggered or user_id is None or current_task_id is None:
        return no_update, no_update, no_update
    if not ctx.triggered[0]["value"]:
        return no_update, no_update, no_update

    turn_idx = str(triggered["index"])
    new_flags = dict(flags or {})
    new_other = dict(other_texts or {})
    if turn_idx in new_flags:
        del new_flags[turn_idx]
        new_other.pop(turn_idx, None)
    else:
        new_flags[turn_idx] = []

    task_data = assignments.get_realism_assignments(user_id)
    assignment = task_data["task_assignments"][current_task_id]
    turns = data.load_source_conversation(assignment["source"], assignment["conv_id"])
    turns, _ = data.select_turn_window(turns, f"{user_id}:{assignment['conv_id']}")

    return (
        new_flags,
        new_other,
        render_conversation(
            turns,
            id_prefix="real-",
            turn_offset=offset,
            flags=new_flags,
            other_texts=new_other,
        ),
    )


@callback(
    Output("realism-flags", "data", allow_duplicate=True),
    Output("realism-conversation", "children", allow_duplicate=True),
    Input({"type": "flag-reasons", "index": ALL}, "value"),
    State("realism-flags", "data"),
    State("realism-flag-other-texts", "data"),
    State("realism-turn-offset", "data"),
    State("realism-current-task", "data"),
    State("realism-user", "data"),
    prevent_initial_call=True,
)
def update_flag_reasons(reasons, flags, other_texts, offset, current_task_id, user_id):
    triggered = ctx.triggered_id
    if not triggered or user_id is None or current_task_id is None:
        return no_update, no_update

    turn_idx = str(triggered["index"])
    new_flags = dict(flags or {})
    if turn_idx not in new_flags:
        return no_update, no_update

    vals = ctx.triggered[0]["value"]
    new_flags[turn_idx] = vals if vals else []

    task_data = assignments.get_realism_assignments(user_id)
    assignment = task_data["task_assignments"][current_task_id]
    turns = data.load_source_conversation(assignment["source"], assignment["conv_id"])
    turns, _ = data.select_turn_window(turns, f"{user_id}:{assignment['conv_id']}")

    return new_flags, render_conversation(
        turns,
        id_prefix="real-",
        turn_offset=offset,
        flags=new_flags,
        other_texts=other_texts,
    )


@callback(
    Output("realism-flag-other-texts", "data", allow_duplicate=True),
    Input({"type": "flag-other-text", "index": ALL}, "value"),
    State("realism-flag-other-texts", "data"),
    prevent_initial_call=True,
)
def update_flag_other_text(values, other_texts):
    triggered = ctx.triggered_id
    if not triggered:
        return no_update

    turn_idx = str(triggered["index"])
    new_other = dict(other_texts or {})
    val = ctx.triggered[0]["value"] or ""
    if val:
        new_other[turn_idx] = val
    else:
        new_other.pop(turn_idx, None)
    return new_other


@callback(
    Output("realism-index", "data", allow_duplicate=True),
    Output("realism-classification", "value", allow_duplicate=True),
    Output("realism-confidence", "value", allow_duplicate=True),
    Output("realism-submit-trigger", "data", allow_duplicate=True),
    Output("realism-nav-warning", "children", allow_duplicate=True),
    Input("keypress-value", "value"),
    State("url", "pathname"),
    State("realism-index", "data"),
    State("realism-user", "data"),
    State("realism-submit-trigger", "data"),
    State("realism-classification", "value"),
    State("realism-confidence", "value"),
    prevent_initial_call=True,
)
def keyboard_realism(
    key_val, pathname, idx, user_id, trigger, classification, confidence
):
    if pathname != "/realism" or not key_val or user_id is None:
        return no_update, no_update, no_update, no_update, no_update

    key = key_val.split(":")[0]

    if key == "ArrowLeft":
        return max(0, idx - 1), no_update, no_update, no_update, ""
    if key == "ArrowRight":
        task_data = assignments.get_realism_assignments(user_id)
        labels = task_data.get("labels", {})
        task_id = task_data["order"][idx]
        if task_id not in labels:
            return no_update, no_update, no_update, no_update, _NAV_WARNING
        return (
            min(len(task_data["order"]) - 1, idx + 1),
            no_update,
            no_update,
            no_update,
            "",
        )
    if key == "Enter":
        return no_update, no_update, no_update, (trigger or 0) + 1, no_update
    if key.lower() == "r":
        return no_update, "real", no_update, no_update, no_update
    if key.lower() == "s":
        return no_update, "simulated", no_update, no_update, no_update
    if key in "12345":
        return no_update, no_update, int(key), no_update, no_update

    return no_update, no_update, no_update, no_update, no_update


@callback(
    Output("realism-index", "data", allow_duplicate=True),
    Output("realism-nav-warning", "children", allow_duplicate=True),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    State("realism-index", "data"),
    State("realism-user", "data"),
    prevent_initial_call=True,
)
def navigate_realism(prev_clicks, next_clicks, idx, user_id):
    if user_id is None:
        return no_update, no_update

    triggered = ctx.triggered_id
    if triggered == "btn-prev" and not prev_clicks:
        return no_update, no_update
    if triggered == "btn-next" and not next_clicks:
        return no_update, no_update

    task_data = assignments.get_realism_assignments(user_id)
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
    Output("realism-index", "data", allow_duplicate=True),
    Output("realism-status", "children", allow_duplicate=True),
    Output("realism-submit-error", "children"),
    Output("sidebar-trigger", "data", allow_duplicate=True),
    Output("url", "pathname", allow_duplicate=True),
    Input("realism-submit", "n_clicks"),
    Input("realism-submit-trigger", "data"),
    State("realism-classification", "value"),
    State("realism-confidence", "value"),
    *[State(f"realism-{dim}", "value") for dim in RATING_DIMENSIONS],
    State("realism-flags", "data"),
    State("realism-flag-other-texts", "data"),
    State("realism-comment", "value"),
    State("realism-current-task", "data"),
    State("realism-index", "data"),
    State("realism-user", "data"),
    prevent_initial_call=True,
)
def submit_realism(n_clicks, trigger, classification, confidence, *args):
    n = len(RATING_DIMENSIONS)
    rating_vals = args[:n]
    flags, other_texts, comment, current_task_id, idx, user_id = args[n:]

    if ctx.triggered_id == "realism-submit" and not n_clicks:
        return no_update, no_update, no_update, no_update, no_update
    if ctx.triggered_id == "realism-submit-trigger" and not trigger:
        return no_update, no_update, no_update, no_update, no_update
    if not classification or user_id is None:
        return (
            no_update,
            no_update,
            "Please select Real or Simulated.",
            no_update,
            no_update,
        )

    task_data = assignments.get_realism_assignments(user_id)
    order = task_data["order"]
    task_id = current_task_id if current_task_id is not None else order[idx]
    current_idx = order.index(task_id)

    ratings = dict(zip(RATING_DIMENSIONS, rating_vals))
    flagged_turns = []
    for t, r in (flags or {}).items():
        entry = {"turn_idx": int(t), "reasons": r}
        if "other" in r and (other_texts or {}).get(t):
            entry["other_text"] = other_texts[t]
        flagged_turns.append(entry)

    task_data.setdefault("labels", {})[task_id] = {
        "classification": classification,
        "confidence": confidence,
        **ratings,
        "flagged_turns": flagged_turns,
        "comment": (comment or "").strip(),
        "timestamp": datetime.now().isoformat(),
    }
    storage.save_labels(user_id, "realism", task_data)

    is_last = current_idx == len(order) - 1
    next_idx = min(current_idx + 1, len(order) - 1)
    return (
        next_idx,
        html.Span("✓ Saved", className="submitted-badge"),
        "",
        datetime.now().isoformat(),
        "/personality" if is_last else no_update,
    )
