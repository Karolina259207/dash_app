from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from backend.data import PROJECT_ROOT, get_bundle, make_loaders
from backend.models import CNN_ARCHITECTURES, build_model

RUNS_DIR = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        count = int(yb.size(0))
        total_loss += float(loss.item()) * count
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_items += count
    if not total_items:
        raise ValueError("Tréningová množina je prázdna.")
    return total_loss / total_items, total_correct / total_items


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        count = int(yb.size(0))
        total_loss += float(loss.item()) * count
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_items += count
    if not total_items:
        raise ValueError("Validačná alebo testovacia množina je prázdna.")
    return total_loss / total_items, total_correct / total_items


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def run_training(session_id: str, config: dict[str, Any]) -> dict[str, Any]:
    """Spustí jeden beh a zapisuje loss/accuracy aj hyperparametre do TensorBoardu."""
    bundle = get_bundle(session_id)
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    train_ratio = float(config["train_ratio"])
    model_type = str(config["model_type"])
    hidden_sizes = [int(x) for x in config.get("hidden_sizes", [128, 64])]
    architecture = str(config.get("architecture", "resnet18"))

    if not 1 <= epochs <= 100:
        raise ValueError("Počet epoch musí byť od 1 do 100.")
    if not 1 <= batch_size <= 512:
        raise ValueError("Batch size musí byť od 1 do 512.")
    if not 0 < learning_rate <= 1:
        raise ValueError("Learning rate musí byť väčší ako 0 a najviac 1.")
    if model_type == "CNN" and bundle.kind != "image":
        raise ValueError("CNN je dostupná len pre obrazové datasety.")
    if model_type == "CNN" and architecture not in CNN_ARCHITECTURES:
        raise ValueError("Vyberte jednu z podporovaných CNN architektúr.")
    if model_type == "ANN" and (not hidden_sizes or any(n < 1 or n > 4096 for n in hidden_sizes)):
        raise ValueError("ANN musí mať kladné počty neurónov v skrytých vrstvách.")

    set_seed(42)
    train_loader, val_loader, test_loader, split_counts = make_loaders(
        bundle, train_ratio=train_ratio, batch_size=batch_size
    )
    device = get_device()
    model = build_model(
        model_type,
        bundle.input_shape,
        bundle.num_classes,
        hidden_sizes=hidden_sizes,
        architecture=architecture,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)  # Adam je pevne daný podľa zadania.

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_saved = {
        "dataset": bundle.name,
        "dataset_kind": bundle.kind,
        "model_type": model_type,
        "hidden_sizes": hidden_sizes if model_type == "ANN" else [],
        "architecture": architecture if model_type == "CNN" else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_ratio": train_ratio,
        "optimizer": "Adam",
        "device": str(device),
        "input_shape": list(bundle.input_shape),
        "num_classes": bundle.num_classes,
        "class_names": bundle.class_names,
        "split_counts": split_counts,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "metadata.json", {"run_id": run_id, "status": "running", "config": config_saved})

    writer = SummaryWriter(log_dir=str(run_dir))
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    try:
        writer.add_text("run/config", json.dumps(config_saved, ensure_ascii=False, indent=2), 0)
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = _train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/accuracy", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/accuracy", val_acc, epoch)
            writer.flush()

        test_loss, test_acc = _evaluate(model, test_loader, criterion, device)
        runtime_seconds = time.perf_counter() - start
        writer.add_scalar("test/loss", test_loss, epochs + 1)
        writer.add_scalar("test/accuracy", test_acc, epochs + 1)
        writer.add_scalar("runtime/seconds", runtime_seconds, epochs)
        writer.add_custom_scalars(
            {
                "Porovnanie": {
                    "Chyba": ["Multiline", ["train/loss", "val/loss"]],
                    "Presnosť": ["Multiline", ["train/accuracy", "val/accuracy"]],
                }
            }
        )
        writer.add_hparams(
            {
                "dataset": bundle.name,
                "model_type": model_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "train_ratio": train_ratio,
                "optimizer": "Adam",
                "device": str(device),
            },
            {
                "test_accuracy": float(test_acc),
                "test_loss": float(test_loss),
                "runtime_seconds": float(runtime_seconds),
            },
        )
        writer.flush()
        result = {
            "run_id": run_id,
            "status": "completed",
            "config": config_saved,
            "history": history,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "runtime_seconds": float(runtime_seconds),
        }
        writer.close()
        _write_json(run_dir / "metadata.json", result)
        return result
    except Exception as exc:
        writer.close()
        failed = {
            "run_id": run_id,
            "status": "failed",
            "config": config_saved,
            "error": str(exc),
        }
        _write_json(run_dir / "metadata.json", failed)
        raise
