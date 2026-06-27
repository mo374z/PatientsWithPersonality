import argparse
from pathlib import Path

from dash import Dash, html, dcc, Input, Output, State, callback, no_update

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
                "You will evaluate simulated patients in doctor-patient conversations. "
                "The study has two tasks, available from the sidebar once you log in:",
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Strong("1. Conversation Realism — "),
                    html.Span(
                        [
                            "You are shown one conversation at a time. Decide whether the patient "
                            "is a real human or an AI simulation, rate your confidence, and assess "
                            "symptom description realism, information control, and style realism. "
                            "You can also flag individual patient turns that drove your judgment. ",
                            html.Br(),
                            html.Strong("IMPORTANT: "),
                            "There can be up to two real patient conversations per case, but also none. Your task is ",
                            html.Strong("not "),
                            "to find the real conversation per case, "
                            "but to judge the realism of each conversation independently.",
                        ]
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
            html.P(
                [
                    html.Strong("What to focus on: "),
                    "realism cues — medical plausibility, specificity of detail, hesitation and "
                    "emotional tone, and the naturalness of the phrasing. Do ",
                    html.Em("not"),
                    " judge how helpful, complete, or cooperative the patient is.",
                ],
                style={"marginBottom": "8px"},
            ),
            html.P(
                "Only a 5-turn window of each conversation is shown — this is intentional.",
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
            html.P("Enter your anonymous user ID to begin."),
            intro,
            dcc.Input(
                id="user-id-input",
                type="text",
                placeholder="Enter user ID...",
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
    if not user_id or len(user_id.strip()) < 3:
        return no_update, no_update, "User ID must be at least 3 characters."
    return user_id.strip(), "/realism", ""


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
    parser.add_argument("--labels-dir", default=None)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data.init(args.results_dir, args.data_dir)
    data.init_study_config(args.study_config)
    labels_dir = args.labels_dir or str(Path(__file__).parent / "labels")
    storage.init(labels_dir)

    app.run(debug=args.debug, port=args.port, host="0.0.0.0")
