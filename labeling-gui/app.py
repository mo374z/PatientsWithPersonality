import argparse

from dash import Dash, html, dcc, Input, Output, State, callback, no_update

import assignments
import data
import storage
from components import render_progress_bar
from pages import realism, personality


app = Dash(
    __name__, suppress_callback_exceptions=True, title="Patient Simulator Labeling"
)
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏥</text></svg>">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="user-store", storage_type="local"),
        dcc.Store(id="sidebar-trigger", data=""),
        dcc.Input(
            id="keypress-value",
            value="",
            type="text",
            style={"display": "none"},
            debounce=False,
        ),
        html.Div(id="sidebar"),
        html.Div(id="page-content", className="main-content"),
    ]
)

_TABS = [
    ("Conversation Realism", "/realism", "realism"),
    ("Patient Personality", "/personality", "personality"),
]


def _build_sidebar(user_id: str, pathname: str) -> html.Div:
    nav_items = []
    for label, href, task_type in _TABS:
        completed, total = storage.get_progress(user_id, task_type)
        active = "active" if pathname == href else ""
        nav_items.append(html.A(label, href=href, className=f"nav-link {active}"))
        if total > 0:
            nav_items.append(render_progress_bar(completed, total))
            nav_items.append(
                html.Div(f"{completed}/{total}", className="progress-text")
            )

    return html.Div(
        [
            html.H3("Patient Simulator Labeling"),
            html.Div(f"User: {user_id}", className="user-id-display"),
            *nav_items,
            html.A(
                "Change User",
                href="/",
                className="nav-link",
                style={"fontSize": "12px", "marginTop": "auto"},
            ),
        ],
        className="sidebar",
        style={"display": "flex", "flexDirection": "column"},
    )


def _welcome_page() -> html.Div:
    intro = html.Div(
        [
            html.P(
                "Some of the patients in these conversations are real people, others are "
                "simulated. The study has two tasks, available from the sidebar once you "
                "log in:",
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Strong("1. Conversation Realism — "),
                    html.Span(
                        "Three versions of the same encounter are shown side by side. For "
                        "each one, say whether the patient sounds real, and if not, what "
                        "gave it away."
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            html.Div(
                [
                    html.Strong("2. Patient Personality — "),
                    html.Span(
                        "Estimate the HEXACO personality traits of the simulated patient. "
                        "The HEXACO personality model measures different character traits."
                    ),
                ],
                style={"marginBottom": "14px"},
            ),
            html.Div(
                [
                    html.Strong("Rate the patient, not the doctor. "),
                    "The doctor's questions differ between the versions and are not part of "
                    "the judgement — only the patient's replies are. Ask yourself whether "
                    "a real person would say this: does it hang together medically, is the "
                    "detail specific, does the wording sound like speech?",
                ],
                className="important-notice",
            ),
            html.P(
                [
                    "Any number of the versions can be real — all three, one, or none — so "
                    "do not try to pick out the real one. Rate each version on its own. Do ",
                    html.Em("not"),
                    f" judge how helpful or cooperative the patient is. Only a "
                    f"{data.get_turn_window_size()}-turn excerpt of each conversation is "
                    "shown; this is intentional.",
                ],
                style={"fontSize": "13px", "color": "#7f8c8d", "marginBottom": "20px"},
            ),
        ],
        style={
            "textAlign": "left",
            "fontSize": "14px",
            "color": "#2c3e50",
            "lineHeight": "1.55",
        },
    )

    return html.Div(
        [
            html.H2("Patient Simulator Labeling"),
            html.P(
                "Pick any anonymous ID (at least 4 letters/digits) and enter it to begin. "
                "Use the same ID every time so your work is picked up where you left off."
            ),
            intro,
            dcc.Input(
                id="user-id-input",
                type="text",
                placeholder="At least 4 letters/digits, e.g. abcd",
                style={
                    "width": "100%",
                    "padding": "10px",
                    "marginBottom": "12px",
                    "borderRadius": "6px",
                    "border": "1px solid #bdc3c7",
                    "fontSize": "15px",
                    "boxSizing": "border-box",
                },
            ),
            html.Button(
                "Start Labeling",
                id="btn-login",
                n_clicks=0,
                style={
                    "padding": "10px 24px",
                    "background": "#3498db",
                    "color": "#fff",
                    "border": "none",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "fontSize": "15px",
                },
            ),
            html.Div(id="login-error", style={"color": "#e74c3c", "marginTop": "8px"}),
        ],
        className="welcome-container",
    )


@callback(
    Output("user-store", "data"),
    Output("url", "pathname", allow_duplicate=True),
    Output("login-error", "children"),
    Input("btn-login", "n_clicks"),
    Input("user-id-input", "n_submit"),
    State("user-id-input", "value"),
    prevent_initial_call=True,
)
def handle_login(n_clicks, n_submit, user_id):
    rater_id = data.normalize_rater_id(user_id)
    if rater_id is None:
        return no_update, no_update, "Please enter at least 4 letters/digits, e.g. abcd."
    for task in storage.TASKS:
        assignments.get_assignments(rater_id, task)
    return rater_id, "/realism", ""


@callback(
    Output("sidebar", "children"),
    Input("url", "pathname"),
    Input("user-store", "data"),
    Input("sidebar-trigger", "data"),
)
def update_sidebar(pathname, user_id, _):
    if not user_id or pathname == "/":
        return html.Div()
    return _build_sidebar(user_id, pathname)


@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("user-store", "data"),
)
def route_page(pathname, user_id):
    if not user_id or pathname == "/":
        return _welcome_page()

    if pathname == "/personality":
        return personality.layout(user_id)
    return realism.layout(user_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--study-config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data.init(args.results_dir, args.data_dir)
    data.init_study_config(args.study_config)
    storage.init(args.labels_dir)

    app.run(debug=args.debug, port=args.port, host="0.0.0.0")
