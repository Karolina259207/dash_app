from __future__ import annotations

import dash
from dash import Input, Output, State, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from backend.data import cache_bundle, clear_bundle, load_dataset
from backend.models import CNN_ARCHITECTURES
from backend.training import run_training


dash.register_page(__name__, path="/", name="Tréning", order=0)

DATASET_OPTIONS = [
    {"label": "Heart Disease · tabuľkové dáta", "value": "heart"},
    {"label": "MNIST · číslice 28 × 28", "value": "MNIST"},
    {"label": "CIFAR-10 · farebné obrázky", "value": "CIFAR-10"},
    {"label": "CIFAR-100 · 100 tried", "value": "CIFAR-100"},
    {"label": "Vlastné CSV · URL", "value": "custom_url"},
    {"label": "Vlastné CSV · lokálny súbor", "value": "custom_file"},
]

CNN_OPTIONS = [{"label": label, "value": value} for value, label in CNN_ARCHITECTURES.items()]


def empty_note(text: str) -> html.Div:
    return html.Div(text, className="empty-note")


def field(label: str, control, note: str | None = None) -> html.Div:
    children = [html.Label(label, className="field-label"), control]
    if note:
        children.append(html.Div(note, className="field-note"))
    return html.Div(children, className="field")


layout = html.Div(
    className="page page-training",
    children=[
        html.Div(
            className="page-heading",
            children=[
                html.Div("WORKSPACE / MODEL TRAINING", className="eyebrow"),
                html.H1("Pripravte dáta. Natrénujte model."),
                html.P(
                    "Postupujte po krokoch. Ďalšia časť sa zobrazí po potvrdení predchádzajúcej, "
                    "pričom už vyplnené nastavenia zostanú dostupné."
                ),
            ],
        ),
        dcc.Store(id="dataset-state", data=None, storage_type="session"),
        dcc.Store(id="model-config-state", data=None, storage_type="session"),
        html.Section(
            className="step-card",
            children=[
                html.Div(
                    className="step-heading",
                    children=[
                        html.Div("01", className="step-number"),
                        html.Div(
                            [html.H2("Načítanie dát"), html.P("Vyberte pripravený dataset alebo vlastný CSV zdroj.")]
                        ),
                    ],
                ),
                html.Div(
                    className="fields-grid fields-grid--two",
                    children=[
                        field(
                            "Zdroj datasetu",
                            dcc.Dropdown(
                                id="dataset-source",
                                options=DATASET_OPTIONS,
                                value="MNIST",
                                clearable=False,
                                className="input-control",
                            ),
                        ),
                        html.Div(
                            id="custom-url-row",
                            style={"display": "none"},
                            children=[
                                field(
                                    "URL adresa CSV",
                                    dcc.Input(
                                        id="custom-url",
                                        type="url",
                                        value="",
                                        placeholder="https://example.com/data.csv",
                                        className="input-control",
                                    ),
                                )
                            ],
                        ),
                        html.Div(
                            id="custom-file-row",
                            style={"display": "none"},
                            children=[
                                field(
                                    "Lokálny CSV súbor",
                                    dcc.Upload(
                                        id="custom-file-upload",
                                        children=html.Div(["Presuňte CSV sem alebo ", html.Span("vyberte súbor", className="upload-link")]),
                                        className="upload-area",
                                        multiple=False,
                                    ),
                                ),
                                html.Div(id="uploaded-file-name", className="field-note"),
                            ],
                        ),
                        html.Div(
                            id="target-column-row",
                            style={"display": "none"},
                            children=[
                                field(
                                    "Názov cieľového stĺpca",
                                    dcc.Input(
                                        id="target-column",
                                        type="text",
                                        value="target",
                                        placeholder="target",
                                        className="input-control",
                                    ),
                                    "Stĺpec, ktorý sa má predikovať; ostatné stĺpce sa použijú ako príznaky.",
                                )
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="actions-row",
                    children=[
                        html.Button("Načítať dáta", id="load-data-btn", n_clicks=0, className="button button-primary"),
                        html.Button("Vyčistiť výber", id="reset-data-btn", n_clicks=0, className="button button-quiet"),
                        html.Div(
                            "Načítanie môže pri prvom použití obrazových datasetov stiahnuť dáta.",
                            className="field-note action-note",
                        ),
                    ],
                ),
                html.Div("Načítavam dataset…", id="dataset-working-message", className="working-message", style={"display": "none"}),
                html.Div(id="dataset-status", className="status-line"),
                html.Div(
                    id="dataset-summary",
                    children=empty_note("Zatiaľ nie sú načítané dáta. Vyberte dataset a stlačte Načítať dáta."),
                ),
            ],
        ),
        html.Section(
            id="model-step",
            className="step-card",
            style={"display": "none"},
            children=[
                html.Div(
                    className="step-heading",
                    children=[
                        html.Div("02", className="step-number"),
                        html.Div([html.H2("Úloha a architektúra"), html.P("Vyberte typ siete a nastavte jej architektúru.")]),
                    ],
                ),
                html.Div(
                    className="fields-grid fields-grid--two",
                    children=[
                        field(
                            "Typ neurónovej siete",
                            dcc.Dropdown(
                                id="model-select",
                                options=[{"label": "ANN · plne prepojená", "value": "ANN"}],
                                value="ANN",
                                clearable=False,
                                className="input-control",
                            ),
                        ),
                        html.Div(id="task-summary", className="readout-card", children="Úloha: klasifikácia"),
                    ],
                ),
                html.Div(
                    id="ann-branch",
                    className="branch-panel",
                    children=[
                        html.Div(className="branch-title", children=[html.Span("ANN"), "Plne prepojená sieť"]),
                        html.Div(
                            className="fields-grid fields-grid--two",
                            children=[
                                field(
                                    "Počet skrytých vrstiev",
                                    dcc.Input(id="hidden-layer-count", type="number", min=1, max=8, step=1, value=2, className="input-control"),
                                ),
                                field(
                                    "Neuróny v skrytých vrstvách",
                                    dcc.Input(id="hidden-sizes", type="text", value="128, 64", placeholder="128, 64, 32", className="input-control"),
                                    "Zadajte celé čísla oddelené čiarkami. Počet musí sedieť s počtom vrstiev.",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="cnn-branch",
                    className="branch-panel",
                    style={"display": "none"},
                    children=[
                        html.Div(className="branch-title", children=[html.Span("CNN"), "Konvolučná sieť"]),
                        field(
                            "CNN architektúra",
                            dcc.Dropdown(id="cnn-architecture", options=CNN_OPTIONS, value="resnet18", clearable=False, className="input-control"),
                            "Model sa vytvorí s weights=None; vstupný kanál aj posledná vrstva sa upravia podľa datasetu.",
                        ),
                    ],
                ),
                html.Div(id="architecture-status", className="status-line"),
                html.Button("Potvrdiť architektúru a pokračovať", id="configure-model-btn", n_clicks=0, className="button button-primary"),
            ],
        ),
        html.Section(
            id="training-step",
            className="step-card",
            style={"display": "none"},
            children=[
                html.Div(
                    className="step-heading",
                    children=[
                        html.Div("03", className="step-number"),
                        html.Div([html.H2("Nastavenie tréningu"), html.P("Skontrolujte parametre a spustite učenie.")]),
                    ],
                ),
                html.Div(id="confirmed-model-summary", className="readout-strip"),
                html.Div(
                    className="fields-grid fields-grid--four",
                    children=[
                        field("Epochy", dcc.Input(id="epochs", type="number", min=1, max=100, step=1, value=3, className="input-control")),
                        field("Batch size", dcc.Input(id="batch-size", type="number", min=1, max=512, step=1, value=32, className="input-control")),
                        field("Learning rate", dcc.Input(id="learning-rate", type="number", min=0.000001, max=1, step="any", value=0.001, className="input-control")),
                        html.Div(
                            className="optimizer-readout",
                            children=[html.Div("OPTIMALIZÁTOR", className="field-label"), html.Strong("Adam"), html.Small("Pevne nastavený podľa notebooku")],
                        ),
                    ],
                ),
                html.Div(
                    className="split-control",
                    children=[
                        html.Div([html.Label("Rozdelenie train / validation", className="field-label"), html.P("Pomer sa aplikuje iba na pôvodnú tréningovú časť; test zostáva oddelený.", className="field-note")]),
                        dcc.Slider(
                            id="train-ratio",
                            min=50,
                            max=95,
                            step=5,
                            value=80,
                            marks={50: "50 / 50", 65: "65 / 35", 80: "80 / 20", 95: "95 / 5"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                ),
                html.Div(
                    className="actions-row training-actions",
                    children=[
                        html.Button("Spustiť tréning", id="run-training-btn", n_clicks=0, className="button button-primary button-large"),
                        dcc.Link("Pozrieť výsledky →", href="/results", className="inline-link"),
                    ],
                ),
                html.Div("Tréning prebieha. Aplikácia počíta metriky pre každú epochu…", id="training-working-message", className="working-message", style={"display": "none"}),
                dcc.Loading(html.Div(id="training-status", className="status-line"), type="circle", color="#5d72f4"),
                html.Div(id="training-result-summary"),
            ],
        ),
    ],
)


@callback(
    Output("custom-url-row", "style"),
    Output("custom-file-row", "style"),
    Output("target-column-row", "style"),
    Output("uploaded-file-name", "children"),
    Input("dataset-source", "value"),
    Input("custom-file-upload", "filename"),
)
def show_custom_source_fields(source, filename):
    url_style = {"display": "block"} if source == "custom_url" else {"display": "none"}
    file_style = {"display": "block"} if source == "custom_file" else {"display": "none"}
    target_style = {"display": "block"} if source in {"custom_url", "custom_file"} else {"display": "none"}
    return url_style, file_style, target_style, (f"Vybraný súbor: {filename}" if filename else "")


@callback(
    Output("dataset-state", "data"),
    Output("dataset-status", "children"),
    Output("dataset-summary", "children"),
    Input("load-data-btn", "n_clicks"),
    Input("reset-data-btn", "n_clicks"),
    State("session-store", "data"),
    State("dataset-source", "value"),
    State("custom-url", "value"),
    State("target-column", "value"),
    State("custom-file-upload", "contents"),
    prevent_initial_call=True,
    running=[
        (Output("load-data-btn", "disabled"), True, False),
        (Output("reset-data-btn", "disabled"), True, False),
        (Output("dataset-working-message", "style"), {"display": "block"}, {"display": "none"}),
    ],
)
def handle_dataset_action(_load_clicks, _reset_clicks, session_data, source, custom_url, target_col, upload_contents):
    if not session_data or not session_data.get("session_id"):
        raise PreventUpdate
    session_id = session_data["session_id"]
    if ctx.triggered_id == "reset-data-btn":
        clear_bundle(session_id)
        return None, html.Span("Výber bol vyčistený. Načítajte dataset, aby ste mohli pokračovať.", className="status-neutral"), empty_note("Zatiaľ nie sú načítané dáta.")

    try:
        bundle = load_dataset(
            source,
            custom_url=custom_url,
            upload_contents=upload_contents,
            target_col=(target_col or "target").strip(),
        )
        cache_bundle(session_id, bundle)
        summary_children = [
            html.Div([html.Small("DATASET"), html.Strong(bundle.name)], className="summary-item"),
            html.Div([html.Small("VZORKY"), html.Strong(f"{bundle.train_count:,}".replace(",", " "))], className="summary-item"),
            html.Div([html.Small("TRIEDY"), html.Strong(str(bundle.num_classes))], className="summary-item"),
            html.Div([html.Small("TVAR VSTUPU"), html.Strong(str(bundle.input_shape))], className="summary-item"),
        ]
        if bundle.kind == "tabular":
            previews = ", ".join(bundle.feature_names[:8])
            if len(bundle.feature_names) > 8:
                previews += "…"
            summary_children.append(html.Div([html.Span("Vstupné príznaky: ", className="summary-label"), previews], className="feature-preview"))
        summary_children.append(
            html.Div(
                "Testovacia množina sa oddeľuje pri príprave tréningu a nepoužíva sa na ladenie modelu.",
                className="field-note summary-footnote",
            )
        )
        summary = html.Div(className="dataset-summary-grid", children=summary_children)
        message = html.Span("Dataset je pripravený. Teraz vyberte kompatibilný model.", className="status-success")
        state = {
            "dataset_name": bundle.name,
            "dataset_kind": bundle.kind,
            "num_classes": bundle.num_classes,
            "input_shape": list(bundle.input_shape),
            "num_features": len(bundle.feature_names),
            "samples": bundle.train_count,
            "test_samples": bundle.test_count,
        }
        return state, message, summary
    except Exception as exc:
        # Pri chybe sa zachová posledný platný dataset a používateľ môže opraviť vstup.
        return no_update, html.Span(str(exc), className="status-error"), no_update


@callback(
    Output("model-step", "style"),
    Input("dataset-state", "data"),
)
def reveal_model_step(dataset_state):
    return {"display": "block"} if dataset_state else {"display": "none"}


@callback(
    Output("model-select", "options"),
    Output("model-select", "value"),
    Output("task-summary", "children"),
    Input("dataset-state", "data"),
    State("model-select", "value"),
)
def choose_compatible_models(dataset_state, current_model):
    if not dataset_state:
        raise PreventUpdate
    options = [{"label": "ANN · plne prepojená", "value": "ANN"}]
    if dataset_state.get("dataset_kind") == "image":
        options.append({"label": "CNN · konvolučná", "value": "CNN"})
    model = current_model if any(item["value"] == current_model for item in options) else "ANN"
    task_text = (
        f"Úloha: viac-triedna klasifikácia · {dataset_state.get('num_classes', '?')} tried"
    )
    return options, model, task_text


@callback(
    Output("ann-branch", "style"),
    Output("cnn-branch", "style"),
    Input("model-select", "value"),
    Input("dataset-state", "data"),
)
def show_model_branch(model_type, dataset_state):
    if not dataset_state:
        return {"display": "none"}, {"display": "none"}
    show_ann = model_type == "ANN"
    show_cnn = model_type == "CNN" and dataset_state.get("dataset_kind") == "image"
    return (
        {"display": "block"} if show_ann else {"display": "none"},
        {"display": "block"} if show_cnn else {"display": "none"},
    )


@callback(
    Output("model-config-state", "data"),
    Output("architecture-status", "children"),
    Input("configure-model-btn", "n_clicks"),
    State("session-store", "data"),
    State("dataset-state", "data"),
    State("model-select", "value"),
    State("hidden-layer-count", "value"),
    State("hidden-sizes", "value"),
    State("cnn-architecture", "value"),
    prevent_initial_call=True,
)
def confirm_architecture(_clicks, session_data, dataset_state, model_type, layer_count, hidden_text, architecture):
    if not session_data or not dataset_state:
        raise PreventUpdate
    if model_type not in {"ANN", "CNN"}:
        return None, html.Span("Vyberte typ modelu.", className="status-error")
    config = {
        "session_id": session_data.get("session_id"),
        "dataset_name": dataset_state.get("dataset_name"),
        "model_type": model_type,
    }
    if model_type == "ANN":
        try:
            layer_count = int(layer_count)
            hidden_sizes = [int(piece.strip()) for piece in str(hidden_text).split(",") if piece.strip()]
        except (TypeError, ValueError):
            return None, html.Span("Počty neurónov musia byť celé čísla oddelené čiarkami.", className="status-error")
        if layer_count < 1 or layer_count > 8:
            return None, html.Span("Počet skrytých vrstiev musí byť v rozsahu 1 až 8.", className="status-error")
        if len(hidden_sizes) != layer_count:
            return None, html.Span(f"Zadali ste {len(hidden_sizes)} hodnôt, ale počet vrstiev je {layer_count}.", className="status-error")
        if any(size < 1 or size > 4096 for size in hidden_sizes):
            return None, html.Span("Každá skrytá vrstva musí mať 1 až 4096 neurónov.", className="status-error")
        config.update({"hidden_sizes": hidden_sizes, "layer_count": layer_count})
        arch_label = "[" + ", ".join(str(x) for x in hidden_sizes) + "]"
    else:
        if dataset_state.get("dataset_kind") != "image":
            return None, html.Span("CNN je dostupná iba pre obrazové datasety.", className="status-error")
        if architecture not in CNN_ARCHITECTURES:
            return None, html.Span("Vyberte podporovanú CNN architektúru.", className="status-error")
        config.update({"architecture": architecture})
        arch_label = CNN_ARCHITECTURES[architecture]
    return config, html.Span(f"Architektúra potvrdená: {model_type} · {arch_label}.", className="status-success")


@callback(
    Output("training-step", "style"),
    Output("confirmed-model-summary", "children"),
    Input("model-config-state", "data"),
    Input("dataset-state", "data"),
    Input("model-select", "value"),
    State("session-store", "data"),
)
def reveal_training_step(config_state, dataset_state, model_type, session_data):
    if not config_state or not dataset_state or not session_data:
        return {"display": "none"}, ""
    valid = (
        config_state.get("dataset_name") == dataset_state.get("dataset_name")
        and config_state.get("model_type") == model_type
        and config_state.get("session_id") == session_data.get("session_id")
    )
    if not valid:
        return {"display": "none"}, ""
    if model_type == "ANN":
        architecture = "Skryté vrstvy: [" + ", ".join(str(x) for x in config_state.get("hidden_sizes", [])) + "]"
    else:
        architecture = "Architektúra: " + CNN_ARCHITECTURES.get(config_state.get("architecture", ""), config_state.get("architecture", "CNN"))
    text = f"{dataset_state['dataset_name']}  ·  {model_type}  ·  {architecture}"
    return {"display": "block"}, text


@callback(
    Output("training-status", "children"),
    Output("training-result-summary", "children"),
    Input("run-training-btn", "n_clicks"),
    State("session-store", "data"),
    State("dataset-state", "data"),
    State("model-config-state", "data"),
    State("model-select", "value"),
    State("hidden-layer-count", "value"),
    State("hidden-sizes", "value"),
    State("cnn-architecture", "value"),
    State("epochs", "value"),
    State("batch-size", "value"),
    State("learning-rate", "value"),
    State("train-ratio", "value"),
    prevent_initial_call=True,
    running=[
        (Output("run-training-btn", "disabled"), True, False),
        (Output("training-working-message", "style"), {"display": "block"}, {"display": "none"}),
    ],
)

def start_training(
    _clicks,
    session_data,
    dataset_state,
    model_config,
    model_type,
    layer_count,
    hidden_text,
    architecture,
    epochs,
    batch_size,
    learning_rate,
    train_percent,
):
    import traceback

    try:
        # 1. Kontrola načítaných dát a konfigurácie
        if not session_data or not dataset_state or not model_config:
            return (
                html.Span(
                    "Najprv načítajte MNIST a potvrďte architektúru CNN.",
                    className="status-error",
                ),
                no_update,
            )

        session_id = session_data.get("session_id")

        if (
            model_config.get("session_id") != session_id
            or model_config.get("dataset_name")
            != dataset_state.get("dataset_name")
            or model_config.get("model_type") != model_type
        ):
            return (
                html.Span(
                    "Nastavenie modelu sa zmenilo. "
                    "Znova potvrďte architektúru.",
                    className="status-error",
                ),
                no_update,
            )

        # 2. Kontrola architektúry
        if model_type == "CNN":
            if architecture != model_config.get("architecture"):
                return (
                    html.Span(
                        "Potvrďte zvolenú CNN architektúru ešte raz.",
                        className="status-error",
                    ),
                    no_update,
                )

        elif model_type == "ANN":
            current_sizes = [
                int(piece.strip())
                for piece in str(hidden_text).split(",")
                if piece.strip()
            ]

            if (
                int(layer_count) != model_config.get("layer_count")
                or current_sizes != model_config.get("hidden_sizes")
            ):
                return (
                    html.Span(
                        "Potvrďte nastavenie ANN ešte raz.",
                        className="status-error",
                    ),
                    no_update,
                )

        # 3. Kontrola parametrov tréningu
        if epochs is None or batch_size is None or learning_rate is None:
            raise ValueError(
                "Vyplňte epochy, batch size a learning rate."
            )

        epochs = int(epochs)
        batch_size = int(batch_size)
        learning_rate = float(learning_rate)

        if not 1 <= epochs <= 100:
            raise ValueError("Počet epoch musí byť od 1 do 100.")

        if not 1 <= batch_size <= 512:
            raise ValueError("Batch size musí byť od 1 do 512.")

        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate musí byť väčší ako 0 a najviac 1.")

        if train_percent is None:
            raise ValueError("Vyberte pomer train/validation.")

        config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_ratio": float(train_percent) / 100.0,
            "model_type": model_type,
            "hidden_sizes": model_config.get("hidden_sizes", [128, 64]),
            "architecture": model_config.get(
                "architecture", "resnet18"
            ),
        }

        # 4. Spustenie skutočného tréningu v backende
        result = run_training(session_id, config)

        # 5. Spracovanie výsledkov
        duration = float(result["runtime_seconds"])
        minutes, seconds = divmod(int(duration), 60)

        status = html.Span(
            (
                "Tréning dokončený. "
                f"Test loss: {result['test_loss']:.4f} · "
                f"Test accuracy: "
                f"{result['test_accuracy'] * 100:.2f} % · "
                f"Čas: {minutes:02d}:{seconds:02d}."
            ),
            className="status-success",
        )

        summary = html.Div(
            className="completed-run-card",
            children=[
                html.Div(
                    "POSLEDNÝ DOKONČENÝ BEH",
                    className="eyebrow",
                ),
                html.H3(
                    f"{result['config']['dataset']} · "
                    f"{result['config']['model_type']}"
                ),
                html.P(f"Beh: {result['run_id']}"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Small("TEST ACCURACY"),
                                html.Strong(
                                    f"{result['test_accuracy'] * 100:.2f} %"
                                ),
                            ],
                            className="summary-item",
                        ),
                        html.Div(
                            [
                                html.Small("TEST LOSS"),
                                html.Strong(
                                    f"{result['test_loss']:.4f}"
                                ),
                            ],
                            className="summary-item",
                        ),
                        html.Div(
                            [
                                html.Small("TRVANIE"),
                                html.Strong(
                                    f"{minutes:02d}:{seconds:02d}"
                                ),
                            ],
                            className="summary-item",
                        ),
                    ],
                    className="dataset-summary-grid",
                ),
                dcc.Link(
                    "Otvoriť grafy a detailné výsledky →",
                    href="/results",
                    className="inline-link",
                ),
            ],
        )

        return status, summary

    except Exception as exc:
        # Vypíše kompletnú chybu a jej pôvod do terminálu.
        traceback.print_exc()

        # Zobrazí chybu aj v rozhraní aplikácie.
        return (
            html.Span(
                f"Tréning zlyhal ({type(exc).__name__}): {exc}",
                className="status-error",
            ),
            no_update,
        )
