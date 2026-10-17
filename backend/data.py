from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

HEART_DISEASE_URL = (
    "https://raw.githubusercontent.com/JayantGoel001/Heart-Disease-Classification/"
    "master/heart.csv"
)
HEART_TARGET_COLUMN = "target"
IMAGE_DATASETS = {"MNIST", "CIFAR-10", "CIFAR-100"}


@dataclass
class DataBundle:
    name: str
    kind: str  # "tabular" alebo "image"
    input_shape: tuple[int, ...]
    num_classes: int
    class_names: list[str]
    train_count: int
    test_count: int
    feature_names: list[str]
    X: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None


# Cache ostáva iba v pamäti servera; do dcc.Store sa ukladá len malý identifikátor.
DATA_CACHE: dict[str, DataBundle] = {}


def cache_bundle(session_id: str, bundle: DataBundle) -> None:
    DATA_CACHE[session_id] = bundle


def get_bundle(session_id: str) -> DataBundle:
    bundle = DATA_CACHE.get(session_id)
    if bundle is None:
        raise ValueError(
            "Dáta sa v aktuálnej serverovej relácii nenašli. Načítajte dataset znovu."
        )
    return bundle


def clear_bundle(session_id: str) -> None:
    DATA_CACHE.pop(session_id, None)


def _decode_upload(contents: str) -> bytes:
    if not contents or "," not in contents:
        raise ValueError("Súbor CSV sa nepodarilo prečítať. Nahrajte platný CSV súbor.")
    _mime, encoded = contents.split(",", 1)
    try:
        return base64.b64decode(encoded)
    except Exception as exc:
        raise ValueError("Obsah nahraného súboru nie je platný Base64/CSV.") from exc


def _load_csv_frame(
    *,
    source: str,
    custom_url: str | None = None,
    upload_contents: str | None = None,
) -> pd.DataFrame:
    try:
        if source == "heart":
            frame = pd.read_csv(HEART_DISEASE_URL)
        elif source == "custom_url":
            if not custom_url or not custom_url.strip():
                raise ValueError("Zadajte URL adresu CSV súboru.")
            frame = pd.read_csv(custom_url.strip())
        elif source == "custom_file":
            if not upload_contents:
                raise ValueError("Najprv vyberte CSV súbor na nahratie.")
            frame = pd.read_csv(io.BytesIO(_decode_upload(upload_contents)))
        else:
            raise ValueError("Neznámy zdroj tabuľkových dát.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Dataset sa nepodarilo načítať. Skontrolujte URL/súbor, internetové pripojenie "
            "a názvy stĺpcov, potom to skúste znova."
        ) from exc
    if frame.empty:
        raise ValueError("CSV súbor neobsahuje žiadne riadky.")
    return frame


def _make_tabular_bundle(
    frame: pd.DataFrame,
    *,
    name: str,
    target_col: str,
) -> DataBundle:
    if target_col not in frame.columns:
        columns = ", ".join(str(c) for c in frame.columns[:15])
        raise ValueError(
            f"V CSV chýba cieľový stĺpec '{target_col}'. Dostupné stĺpce: {columns}"
        )

    frame = frame.dropna(subset=[target_col]).copy()
    if len(frame) < 20:
        raise ValueError("Dataset musí obsahovať aspoň 20 riadkov s cieľovou hodnotou.")

    y_raw = frame[target_col].astype(str).str.strip()
    y_codes, unique_labels = pd.factorize(y_raw, sort=True)
    if len(unique_labels) < 2:
        raise ValueError("Cieľový stĺpec musí obsahovať aspoň dve triedy.")

    features = frame.drop(columns=[target_col]).copy()
    if features.shape[1] == 0:
        raise ValueError("CSV nemá žiadne vstupné príznaky okrem cieľového stĺpca.")

    # Zakóduje aj prípadné textové kategórie a doplní chýbajúce číselné hodnoty.
    try:
        features = pd.get_dummies(features, dummy_na=True, dtype=float)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.astype(np.float64)
        medians = features.median(numeric_only=True).fillna(0.0)
        features = features.fillna(medians).fillna(0.0)
    except Exception as exc:
        raise ValueError("Vstupné stĺpce CSV sa nepodarilo previesť na číselné príznaky.") from exc

    X = features.to_numpy(dtype=np.float32)
    y = np.asarray(y_codes, dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("Po spracovaní CSV zostali v príznakoch nečíselné hodnoty.")

    return DataBundle(
        name=name,
        kind="tabular",
        input_shape=(int(X.shape[1]),),
        num_classes=len(unique_labels),
        class_names=[str(value) for value in unique_labels],
        train_count=len(X),
        test_count=max(1, int(round(len(X) * 0.15))),
        feature_names=[str(column) for column in features.columns],
        X=X,
        y=y,
    )


def load_dataset(
    source: str,
    *,
    custom_url: str | None = None,
    upload_contents: str | None = None,
    target_col: str = HEART_TARGET_COLUMN,
) -> DataBundle:
    """Načíta zvolený dataset. Táto funkcia sa volá až po kliknutí používateľa."""
    if source == "heart":
        frame = _load_csv_frame(source="heart")
        return _make_tabular_bundle(
            frame, name="Heart Disease", target_col=HEART_TARGET_COLUMN
        )
    if source == "custom_url":
        frame = _load_csv_frame(source=source, custom_url=custom_url)
        return _make_tabular_bundle(frame, name="Vlastný CSV (URL)", target_col=target_col)
    if source == "custom_file":
        frame = _load_csv_frame(source=source, upload_contents=upload_contents)
        return _make_tabular_bundle(frame, name="Vlastný CSV súbor", target_col=target_col)

    image_specs = {
        "MNIST": ("mnist", 1, 28, 28, 10),
        "CIFAR-10": ("cifar10", 3, 32, 32, 10),
        "CIFAR-100": ("cifar100", 3, 32, 32, 100),
    }
    if source not in image_specs:
        raise ValueError("Vyberte jeden z ponúknutých datasetov.")

    _folder, channels, height, width, num_classes = image_specs[source]
    if source == "MNIST":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
        class_names = [str(i) for i in range(10)]
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if source == "CIFAR-10":
            train_dataset = datasets.CIFAR10(
                DATA_ROOT, train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                DATA_ROOT, train=False, download=True, transform=transform
            )
            class_names = list(train_dataset.classes)
        else:
            train_dataset = datasets.CIFAR100(
                DATA_ROOT, train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR100(
                DATA_ROOT, train=False, download=True, transform=transform
            )
            class_names = list(train_dataset.classes)

    return DataBundle(
        name=source,
        kind="image",
        input_shape=(channels, height, width),
        num_classes=num_classes,
        class_names=class_names,
        train_count=len(train_dataset),
        test_count=len(test_dataset),
        feature_names=[],
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )


def make_loaders(
    bundle: DataBundle,
    *,
    train_ratio: float = 0.8,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    """Vytvorí train/validation/test loadery; test ostáva mimo nastavovania modelu."""
    if not 0.50 <= train_ratio <= 0.95:
        raise ValueError("Pomer train musí byť v rozsahu 50 % až 95 % train/validation dát.")
    if batch_size < 1:
        raise ValueError("Batch size musí byť kladné celé číslo.")

    generator = torch.Generator().manual_seed(2025)
    if bundle.kind == "tabular":
        assert bundle.X is not None and bundle.y is not None
        all_indices = np.arange(len(bundle.y))
        try:
            train_val_idx, test_idx = train_test_split(
                all_indices,
                test_size=0.15,
                random_state=2025,
                stratify=bundle.y,
            )
        except ValueError:
            train_val_idx, test_idx = train_test_split(
                all_indices, test_size=0.15, random_state=2025
            )
        try:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                train_size=train_ratio,
                random_state=2026,
                stratify=bundle.y[train_val_idx],
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                train_val_idx, train_size=train_ratio, random_state=2026
            )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(bundle.X[train_idx]).astype(np.float32)
        X_val = scaler.transform(bundle.X[val_idx]).astype(np.float32)
        X_test = scaler.transform(bundle.X[test_idx]).astype(np.float32)
        y_train, y_val, y_test = bundle.y[train_idx], bundle.y[val_idx], bundle.y[test_idx]
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
        test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    else:
        assert bundle.train_dataset is not None and bundle.test_dataset is not None
        n_total = len(bundle.train_dataset)
        n_train = max(1, min(n_total - 1, int(round(n_total * train_ratio))))
        n_val = n_total - n_train
        train_ds, val_ds = random_split(
            bundle.train_dataset,
            [n_train, n_val],
            generator=generator,
        )
        test_ds = bundle.test_dataset

    workers = 0  # bezpečné aj pri spúšťaní aplikácie vo Windows
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(batch_size, 64), shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(batch_size, 64), shuffle=False, num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    counts = {"train": len(train_ds), "validation": len(val_ds), "test": len(test_ds)}
    return train_loader, val_loader, test_loader, counts
