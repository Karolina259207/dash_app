from __future__ import annotations

import dash
from datetime import datetime

import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

from backend.results import list_runs, load_run


dash.register_page(__name__, path="/results", name="Výsledky", order=1)


def placeholder(text: str) -> html.Div:
    return html.Div([html.Div("↗", className="empty-icon"), html.H3("Výsledky zatiaľ nie sú dostupné"), html.P(text)], className="empty-state")


def _figure(title: str, y_title: str, series: list[tuple[str, list[dict]]]) -> go.Figure:
    fig = go.Figure()
    for label, points in series:
        if points:
            fig.add_trace(
                go.Scatter(
                    x=[point["step"] for point in points],
                    y=[point["value"] for point in points],
                    mode="lines+markers",
                    name=label,
                    line={"width": 2.5},
                    marker={"size": 5},
                )
            )
    fig.update_layout(
        title={"text": title, "x": 0.03, "xanchor": "left"},
        xaxis_title="Epocha",
        yaxis_title=y_title,
        template="plotly_white",
        margin={"l": 45, "r": 20, "t": 65, "b": 45},
        legend={"orientation": "h", "y": -0.22},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9edf5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e9edf5", zeroline=False)
    return fig


def _runtime(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    hours, remainder = divmod(seconds_int, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"
    return f"{minutes:02d}:{seconds_int:02d}"


layout = html.Div(
    className="page page-results",
    children=[
        html.Div(
            className="page-heading",
            children=[
                html.Div("ANALYTICS / TENSORBOARD", className="eyebrow"),
                html.H1("Výsledky tréningu"),
                html.P("Vyberte dokončený beh. Grafy sa načítajú z TensorBoard event súborov uložených backendom."),
            ],
        ),
        html.Section(
            className="step-card results-toolbar",
            children=[
                html.Div(
                    className="fields-grid fields-grid--two",
                    children=[
                        html.Div(
                            [
                                html.Label("Tréningový beh", className="field-label"),
                                dcc.Dropdown(id="result-run-select", options=[], value=None, placeholder="Vyberte dokončený beh…", clearable=False, className="input-control"),
                            ],
                            className="field",
                        ),
                        html.Div(id="result-list-status", className="readout-card", children="Vyhľadávam dokončené tréningy…"),
                    ],
                ),
                dcc.Interval(id="results-initial-refresh", interval=700, n_intervals=0, max_intervals=1),
            ],
        ),
        html.Div(
            id="result-content",
            children=placeholder("Keď dokončíte tréning na stránke Príprava a tréning, vráťte sa sem a vyberte jeho beh."),
        ),
    ],
)


@callback(
    Output("result-run-select", "options"),
    Output("result-run-select", "value"),
    Output("result-list-status", "children"),
    Input("results-initial-refresh", "n_intervals"),
    State("result-run-select", "value"),
)
def refresh_run_list(_n_intervals, current_value):
    runs = list_runs()
    if not runs:
        return [], None, html.Div("Dokončených behov zatiaľ nie je. Najprv spustite tréning.", className="empty-note")
    options = []
    for run in runs:
        created_at = run.get("created_at", "")
        try:
            parsed = datetime.fromisoformat(created_at)
            created_label = parsed.strftime("%d.%m.%Y %H:%M")
        except (TypeError, ValueError):
            created_label = created_at
        options.append(
            {
                "label": f"{created_label} · {run['dataset']} · {run['model_type']} · {run['run_id']}",
                "value": run["run_id"],
            }
        )
    selected = current_value if any(option["value"] == current_value for option in options) else options[0]["value"]
    return options, selected, html.Span(f"Nájdených behov: {len(runs)}", className="status-success")


@callback(
    Output("result-content", "children"),
    Input("result-run-select", "value"),
    prevent_initial_call=True,
)
def render_selected_run(run_id):
    if not run_id:
        raise PreventUpdate
    try:
        result = load_run(run_id)
        config = result.get("config", {})
        scalars = result.get("scalars", {})
        loss_series = [
            ("Train loss", scalars.get("train/loss", [])),
            ("Validation loss", scalars.get("val/loss", [])),
        ]
        accuracy_series = [
            ("Train accuracy", scalars.get("train/accuracy", [])),
            ("Validation accuracy", scalars.get("val/accuracy", [])),
        ]
        history_available = any(points for _, points in loss_series) or any(points for _, points in accuracy_series)
        if not history_available:
            return placeholder("V TensorBoard event súbore sa nenašli priebehové hodnoty. Skúste iný beh.")

        test_accuracy = float(result.get("test_accuracy", 0.0))
        test_loss = float(result.get("test_loss", 0.0))
        runtime = float(result.get("runtime_seconds", 0.0))
        if config.get("model_type") == "ANN":
            model_description = "ANN · [" + ", ".join(str(x) for x in config.get("hidden_sizes", [])) + "]"
        else:
            model_description = "CNN · " + str(config.get("architecture", ""))

        config_items = [
            ("Dataset", config.get("dataset", "—")),
            ("Model", model_description),
            ("Epochy", str(config.get("epochs", "—"))),
            ("Batch size", str(config.get("batch_size", "—"))),
            ("Learning rate", str(config.get("learning_rate", "—"))),
            ("Train / validation", f"{int(float(config.get('train_ratio', 0.8)) * 100)} / {100 - int(float(config.get('train_ratio', 0.8)) * 100)} %"),
            ("Optimalizátor", config.get("optimizer", "Adam")),
            ("Zariadenie", config.get("device", "—")),
        ]
        count_items = [
            html.Div([html.Small("TEST ACCURACY"), html.Strong(f"{test_accuracy * 100:.2f} %")], className="metric-card metric-card--accent"),
            html.Div([html.Small("TEST LOSS"), html.Strong(f"{test_loss:.4f}")], className="metric-card"),
            html.Div([html.Small("ČAS TRÉNINGU"), html.Strong(_runtime(runtime))], className="metric-card"),
        ]
        config_rows = [
            html.Div([html.Span(label), html.Strong(value)], className="config-row", key=label)
            for label, value in config_items
        ]
        return html.Div(
            className="results-layout",
            children=[
                html.Div(count_items, className="metrics-grid"),
                html.Div(
                    className="charts-grid",
                    children=[
                        html.Div(dcc.Graph(figure=_figure("Chyba počas tréningu", "Loss", loss_series), config={"displayModeBar": False}, responsive=True), className="chart-card"),
                        html.Div(dcc.Graph(figure=_figure("Presnosť počas tréningu", "Accuracy", accuracy_series), config={"displayModeBar": False}, responsive=True), className="chart-card"),
                    ],
                ),
                html.Section(
                    className="step-card run-details",
                    children=[
                        html.Div([html.H2("Konfigurácia behu"), html.P(f"ID behu: {result.get('run_id', run_id)}")], className="details-heading"),
                        html.Div(config_rows, className="config-grid"),
                        html.P("Priebeh grafov bol načítaný z TensorBoard event súboru tohto behu.", className="field-note"),
                    ],
                ),
            ],
        )
    except Exception as exc:
        return html.Div(
            [html.H3("Výsledky sa nepodarilo načítať"), html.P(str(exc)), html.P("Skontrolujte, či je tréning dokončený a jeho TensorBoard event súbor existuje.")],
            className="error-card",
        )

