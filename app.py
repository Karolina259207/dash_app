
from __future__ import annotations

import uuid

import dash
from dash import Dash, Input, Output, dcc, html


NAV_LINKS = [
    ("Príprava a tréning", "/"),
    ("Výsledky", "/results"),
]


def serve_layout():
    """Spoločný layout aplikácie."""
    return html.Div(
        className="app-shell",
        children=[
            dcc.Location(id="url", refresh=False),

            dcc.Store(
                id="session-store",
                data={"session_id": uuid.uuid4().hex},
                storage_type="session",
            ),

            html.Header(
                className="topbar",
                children=[
                    html.Div(
                        className="brand",
                        children=[
                            html.Div("NN", className="brand-mark"),
                            html.Div(
                                [
                                    html.Div(
                                        "NEURAL LAB",
                                        className="brand-eyebrow",
                                    ),
                                    html.Div(
                                        "Tréning neurónových sietí",
                                        className="brand-name",
                                    ),
                                ]
                            ),
                        ],
                    ),

                    html.Nav(
                        className="nav-links",
                        children=[
                            dcc.Link(
                                label,
                                href=href,
                                id=(
                                    "nav-training"
                                    if href == "/"
                                    else "nav-results"
                                ),
                                className="nav-link",
                            )
                            for label, href in NAV_LINKS
                        ],
                    ),
                ],
            ),

            html.Main(
                className="main-content",
                children=dash.page_container,
            ),

            html.Footer(
                className="footer",
                children=(
                    "Zadanie 3 · Plotly Dash + PyTorch + TensorBoard"
                ),
            ),
        ],
    )


app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Neural Lab | Zadanie 3",
    update_title="Neural Lab pracuje…",
)

app.layout = serve_layout


@app.callback(
    Output("nav-training", "className"),
    Output("nav-results", "className"),
    Input("url", "pathname"),
)
def update_active_navigation(pathname):
    """Zvýrazní odkaz zodpovedajúci aktuálnej stránke."""

    training_class = (
        "nav-link active"
        if pathname == "/"
        else "nav-link"
    )

    results_class = (
        "nav-link active"
        if pathname == "/results"
        else "nav-link"
    )

    return training_class, results_class


server = app.server


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )
