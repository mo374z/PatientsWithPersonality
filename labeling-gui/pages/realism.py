from datetime import datetime

from dash import html, dcc, Input, Output, State, callback, no_update, ctx, ALL, MATCH

import data
import storage
import assignments
from components import (
    REASONS,
    REASON_LABELS,
    SPEAKER_LABELS,
    render_turn,
    render_task_header,
    render_patient_info,
    render_thank_you,
)

_NAV_WARNING = "Please rate every conversation and submit before moving on."
_WARNING_STYLE = {"color": "#e74c3c", "fontSize": "12px", "marginTop": "6px"}

COLUMN_LABELS = "ABCDEFGH"

VERDICT_OPTIONS = [
    {"label": " Real", "value": "realistic"},
    {"label": " Not real", "value": "not_realistic"},
]

INSTRUCTIONS = [
    html.Strong("Rate the patient only. "),
    "Judge each column on its own: does this ",
    html.Strong("patient"),
    " sound like a real person? Ignore the doctor — the doctor's turns differ between "
    "columns and are not being rated. Any number of these conversations can be real, so "
    "do not treat this as picking the real one out of a line-up.",
]


def _cid(kind: str, case, position) -> dict:
    return {"type": f"realism-{kind}", "case": case, "index": position}


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
            dcc.Store(id="realism-open-column", data=None),
            html.Div(id="realism-header"),
            html.Div(id="realism-patient-info"),
            html.Div(INSTRUCTIONS, className="important-notice"),
            html.Div(id="realism-board"),
            html.Div(id="realism-status"),
            html.Div(id="realism-nav-warning", style=_WARNING_STYLE),
            html.Div(id="realism-complete-overlay"),
        ]
    )


def _visibility(shown: bool, display: str = "block") -> dict:
    return {"display": display if shown else "none"}


def _grid_style(n_columns: int) -> dict:
    return {"gridTemplateColumns": f"repeat({n_columns}, minmax(260px, 1fr))"}


def _turn_grid(excerpts: list[list[dict]]) -> html.Div:
    lengths = {len(turns) for turns in excerpts}
    if len(lengths) != 1:
        raise ValueError(f"Arms of one case have differing excerpt lengths: {lengths}")

    headers = [
        html.Div(f"Conversation {COLUMN_LABELS[position]}", className="grid-header")
        for position in range(len(excerpts))
    ]
    cells = [
        html.Div(render_turn(turns[i][speaker], speaker), className="grid-cell")
        for i in range(next(iter(lengths)))
        for speaker in SPEAKER_LABELS
        for turns in excerpts
    ]
    return html.Div(
        headers + cells, className="realism-grid", style=_grid_style(len(excerpts))
    )


def _reason_dialog(case: str, position: int, existing: dict) -> html.Div:
    reasons = existing.get("reasons", [])
    return html.Div(
        [
            html.Div(
                f"What made the patient in conversation {COLUMN_LABELS[position]} "
                "sound not real?",
                className="reason-dialog-title",
            ),
            dcc.Checklist(
                id=_cid("reasons", case, position),
                options=[
                    {"label": f" {REASON_LABELS[r]}", "value": r} for r in REASONS
                ],
                value=reasons,
                className="reason-checklist",
            ),
            dcc.Input(
                id=_cid("reason-other", case, position),
                type="text",
                value=existing.get("other_text", ""),
                placeholder="Specify...",
                debounce=True,
                className="reason-other",
                style=_visibility("other" in reasons),
            ),
            html.Button(
                "Done",
                id=_cid("done", case, position),
                n_clicks=0,
                className="reason-done",
            ),
        ],
        id=_cid("dialog", case, position),
        className="reason-dialog",
        style={"display": "none"},
    )


def _reason_button_label(reasons: list) -> str:
    return f"Reasons — {len(reasons)} selected" if reasons else "Add reasons"


def _controls_cell(case: str, position: int, existing: dict) -> html.Div:
    verdict = existing.get("verdict")
    return html.Div(
        [
            _reason_dialog(case, position, existing),
            html.Div(
                f"Conversation {COLUMN_LABELS[position]} — does the patient sound real?",
                className="controls-question",
            ),
            dcc.RadioItems(
                id=_cid("verdict", case, position),
                options=VERDICT_OPTIONS,
                value=verdict,
                labelStyle={"marginRight": "14px", "cursor": "pointer"},
                inline=True,
            ),
            html.Button(
                _reason_button_label(existing.get("reasons", [])),
                id=_cid("reasons-btn", case, position),
                n_clicks=0,
                className="reason-button",
                style=_visibility(verdict == "not_realistic"),
            ),
        ],
        className="controls-cell",
    )


def _render_board(case: str, assignment: dict, existing: dict) -> html.Div:
    conv_id = assignment["conv_id"]
    columns = existing.get("columns", [])
    excerpts = [
        data.take_excerpt(data.load_conversation(arm, conv_id), conv_id)
        for arm in assignment["arms"]
    ]
    per_column = [
        columns[position] if position < len(columns) else {}
        for position in range(len(excerpts))
    ]

    return html.Div(
        [
            _turn_grid(excerpts),
            html.Div(
                [
                    html.Div(
                        [
                            _controls_cell(case, position, column)
                            for position, column in enumerate(per_column)
                        ],
                        className="realism-controls",
                        style=_grid_style(len(excerpts)),
                    ),
                    html.Button(
                        "Submit & Next  [Enter]",
                        id="realism-submit",
                        n_clicks=0,
                        className="submit-button",
                    ),
                    html.Div(id="realism-submit-error", className="submit-error"),
                    html.Div(
                        id="realism-backdrop",
                        className="modal-backdrop",
                        n_clicks=0,
                        style={"display": "none"},
                    ),
                ],
                className="realism-bar",
            ),
        ],
        className="realism-board",
    )


@callback(
    Output("realism-header", "children"),
    Output("realism-patient-info", "children"),
    Output("realism-board", "children"),
    Output("realism-status", "children"),
    Output("realism-current-task", "data"),
    Output("realism-open-column", "data"),
    Output("realism-nav-warning", "children"),
    Input("realism-index", "data"),
    State("realism-user", "data"),
)
def render_realism(idx, user_id):
    if user_id is None:
        return (no_update,) * 7

    task_data = assignments.get_realism_assignments(user_id)
    order = task_data["order"]
    labels = task_data.get("labels", {})

    idx = min(max(idx or 0, 0), len(order) - 1)
    task_id = order[idx]
    assignment = task_data["task_assignments"][task_id]
    existing = labels.get(task_id, {})

    return (
        render_task_header(idx, len(order), "Conversation Realism"),
        render_patient_info(
            assignment["conv_id"], data.get_patient_summary(assignment["conv_id"])
        ),
        _render_board(task_id, assignment, existing),
        html.Span("✓ Submitted", className="submitted-badge") if existing else "",
        task_id,
        None,
        "",
    )


@callback(
    Output(_cid("dialog", ALL, ALL), "style"),
    Output("realism-backdrop", "style"),
    Input("realism-open-column", "data"),
    State(_cid("dialog", ALL, ALL), "id"),
)
def toggle_dialogs(open_column, dialog_ids):
    return (
        [_visibility(open_column == d["index"]) for d in dialog_ids],
        _visibility(open_column is not None),
    )


@callback(
    Output("realism-open-column", "data", allow_duplicate=True),
    Input(_cid("verdict", ALL, ALL), "value"),
    State(_cid("reasons", ALL, ALL), "value"),
    prevent_initial_call=True,
)
def open_reasons_on_verdict(verdicts, reasons):
    if ctx.triggered_id is None:
        return no_update
    position = ctx.triggered_id["index"]
    if verdicts[position] != "not_realistic":
        return None
    return position if not reasons[position] else no_update


@callback(
    Output("realism-open-column", "data", allow_duplicate=True),
    Input(_cid("reasons-btn", ALL, ALL), "n_clicks"),
    prevent_initial_call=True,
)
def open_reasons_on_click(clicks):
    if not ctx.triggered[0]["value"]:
        return no_update
    return ctx.triggered_id["index"]


@callback(
    Output("realism-open-column", "data", allow_duplicate=True),
    Input(_cid("done", ALL, ALL), "n_clicks"),
    Input("realism-backdrop", "n_clicks"),
    prevent_initial_call=True,
)
def close_reasons(done_clicks, backdrop_clicks):
    if not ctx.triggered[0]["value"]:
        return no_update
    return None


@callback(
    Output(_cid("reasons-btn", MATCH, MATCH), "style"),
    Input(_cid("verdict", MATCH, MATCH), "value"),
)
def toggle_reason_button(verdict):
    return _visibility(verdict == "not_realistic")


@callback(
    Output(_cid("reasons-btn", MATCH, MATCH), "children"),
    Output(_cid("reason-other", MATCH, MATCH), "style"),
    Input(_cid("reasons", MATCH, MATCH), "value"),
)
def sync_reason_widgets(reasons):
    return _reason_button_label(reasons or []), _visibility("other" in (reasons or []))


@callback(
    Output("realism-index", "data", allow_duplicate=True),
    Output("realism-submit-trigger", "data", allow_duplicate=True),
    Output("realism-nav-warning", "children", allow_duplicate=True),
    Input("keypress-value", "value"),
    State("url", "pathname"),
    State("realism-index", "data"),
    State("realism-user", "data"),
    State("realism-submit-trigger", "data"),
    prevent_initial_call=True,
)
def keyboard_realism(key_val, pathname, idx, user_id, trigger):
    if pathname != "/realism" or not key_val or user_id is None:
        return no_update, no_update, no_update

    key = key_val.split(":")[0]
    if key == "Enter":
        return no_update, (trigger or 0) + 1, no_update
    if key == "ArrowLeft":
        return max(0, idx - 1), no_update, ""
    if key == "ArrowRight":
        task_data = assignments.get_realism_assignments(user_id)
        if task_data["order"][idx] not in task_data.get("labels", {}):
            return no_update, no_update, _NAV_WARNING
        return min(len(task_data["order"]) - 1, idx + 1), no_update, ""

    return no_update, no_update, no_update


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
    if user_id is None or not ctx.triggered[0]["value"]:
        return no_update, no_update

    if ctx.triggered_id == "btn-prev":
        return max(0, idx - 1), ""

    task_data = assignments.get_realism_assignments(user_id)
    if task_data["order"][idx] not in task_data.get("labels", {}):
        return no_update, _NAV_WARNING
    return min(len(task_data["order"]) - 1, idx + 1), ""


def _collect_columns(arms, verdicts, reasons, other_texts) -> list[dict]:
    return [
        {
            "arm": arm,
            "position": position,
            "verdict": verdict,
            "reasons": column_reasons if verdict == "not_realistic" else [],
            "other_text": (other or "").strip()
            if verdict == "not_realistic" and "other" in column_reasons
            else "",
        }
        for position, (arm, verdict, column_reasons, other) in enumerate(
            zip(arms, verdicts, reasons, other_texts)
        )
    ]


def _first_incomplete(columns: list[dict]) -> tuple[str, int]:
    for column in columns:
        if not column["verdict"]:
            label = COLUMN_LABELS[column["position"]]
            return f"Please rate conversation {label}.", None
    for column in columns:
        if column["verdict"] == "not_realistic" and not column["reasons"]:
            label = COLUMN_LABELS[column["position"]]
            return (
                f"Please give at least one reason for conversation {label}.",
                column["position"],
            )
    return "", None


@callback(
    Output("realism-index", "data", allow_duplicate=True),
    Output("realism-status", "children", allow_duplicate=True),
    Output("realism-submit-error", "children"),
    Output("realism-open-column", "data", allow_duplicate=True),
    Output("sidebar-trigger", "data", allow_duplicate=True),
    Output("realism-complete-overlay", "children"),
    Output("url", "pathname", allow_duplicate=True),
    Input("realism-submit", "n_clicks"),
    Input("realism-submit-trigger", "data"),
    State(_cid("verdict", ALL, ALL), "value"),
    State(_cid("reasons", ALL, ALL), "value"),
    State(_cid("reason-other", ALL, ALL), "value"),
    State("realism-current-task", "data"),
    State("realism-index", "data"),
    State("realism-user", "data"),
    prevent_initial_call=True,
)
def submit_realism(
    n_clicks,
    trigger,
    verdicts,
    reasons,
    other_texts,
    current_task_id,
    idx,
    user_id,
):
    if user_id is None or not ctx.triggered[0]["value"]:
        return (no_update,) * 7

    task_data = assignments.get_realism_assignments(user_id)
    order = task_data["order"]
    task_id = current_task_id if current_task_id is not None else order[idx]
    assignment = task_data["task_assignments"][task_id]

    columns = _collect_columns(assignment["arms"], verdicts, reasons, other_texts)
    error, reopen = _first_incomplete(columns)
    if error:
        return (
            no_update,
            no_update,
            error,
            reopen if reopen is not None else no_update,
            no_update,
            no_update,
            no_update,
        )

    task_data.setdefault("labels", {})[task_id] = {
        "columns": columns,
        "timestamp": datetime.now().isoformat(),
    }
    storage.save_labels(user_id, "realism", task_data)

    current_idx = order.index(task_id)
    is_last = current_idx == len(order) - 1
    complete = storage.all_tasks_complete(user_id)
    return (
        min(current_idx + 1, len(order) - 1),
        html.Span("✓ Saved", className="submitted-badge"),
        "",
        None,
        datetime.now().isoformat(),
        render_thank_you() if complete else no_update,
        "/personality" if is_last and not complete else no_update,
    )
