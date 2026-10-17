from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from backend.training import RUNS_DIR


def list_runs() -> list[dict[str, Any]]:
    """Vráti dokončené tréningy, ktoré majú metadata aj TensorBoard event súbor."""
    runs: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return runs
    for metadata_path in RUNS_DIR.glob("*/metadata.json"):
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            event_files = list(metadata_path.parent.glob("events.out.tfevents.*"))
            if data.get("status") != "completed" or not event_files:
                continue
            config = data.get("config", {})
            runs.append(
                {
                    "run_id": data["run_id"],
                    "dataset": config.get("dataset", "Dataset"),
                    "model_type": config.get("model_type", "Model"),
                    "created_at": config.get("created_at", data["run_id"]),
                    "metadata": data,
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
    runs.sort(key=lambda item: item["created_at"], reverse=True)
    return runs


def load_run(run_id: str) -> dict[str, Any]:
    """Načíta bezpečne vybraný beh a vytiahne skutočné scaláre z TensorBoard eventov."""
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Neplatný identifikátor tréningového behu.")
    run_dir = (RUNS_DIR / run_id).resolve()
    if run_dir.parent != RUNS_DIR.resolve():
        raise ValueError("Neplatná cesta k tréningovému behu.")
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Výsledky vybraného behu sa nenašli.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    event_files = list(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError("Pre vybraný beh sa nenašli TensorBoard výsledky.")

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    scalars: dict[str, list[dict[str, float | int]]] = {}
    for tag in tags:
        scalars[tag] = [
            {"step": int(event.step), "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
    metadata["scalars"] = scalars
    return metadata
