"""
Task 3: Deep Learning Model Training (HybridURLNet).
Trains the PyTorch residual tabular network on GPU/CPU with class-weighted cross-entropy.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from url_pipeline.dl_model import HybridURLNet

STAGED_PATH = ROOT / "reports" / "features_staged.parquet"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def train_model(
    input_path: str | Path | None = None,
    epochs: int = 5,
    batch_size: int = 1024,
) -> dict:
    t_start = time.time()
    in_path = Path(input_path) if input_path else STAGED_PATH
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Airflow Task: train_model] Loading staged features from {in_path}...")
    df = pd.read_parquet(in_path)

    y_raw = df["target_label"].astype(str)
    X = df.drop(columns=["target_label"]).fillna(0.0)
    feature_names = list(X.columns)
    del df
    gc.collect()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    label_mapping = {int(idx): str(c) for idx, c in enumerate(le.classes_)}
    num_classes = len(label_mapping)
    print(f"Target classes ({num_classes}): {label_mapping}")

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=y
    )
    del X, y, y_raw
    gc.collect()

    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    del X_train, X_test
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Compute balanced weights to handle class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    model = HybridURLNet(input_dim=len(feature_names), hidden_dim=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training HybridURLNet for {epochs} epochs...")
    best_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, total_seen = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_b)
            total_seen += len(y_b)

        scheduler.step()

        # Validation
        model.eval()
        t_preds, t_trues = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                logits = model(X_b.to(device))
                t_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                t_trues.extend(y_b.numpy())

        acc = accuracy_score(t_trues, t_preds)
        f1 = f1_score(t_trues, t_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch}/{epochs} | Loss: {running_loss/max(total_seen, 1):.4f} | Val Acc: {acc*100:.2f}% | Val Macro-F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)

    # Save artifacts
    model_save_path = MODEL_DIR / "url_dl_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "hidden_dim": 128,
        "num_classes": num_classes,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
    }, str(model_save_path))

    joblib.dump(scaler, MODEL_DIR / "url_dl_scaler.joblib")
    joblib.dump(le, MODEL_DIR / "url_dl_label_encoder.joblib")

    rep = classification_report(
        t_trues, t_preds,
        target_names=[label_mapping[i] for i in range(num_classes)],
        output_dict=True,
        zero_division=0,
    )
    with open(MODEL_DIR / "url_dl_metrics.json", "w") as f:
        json.dump(rep, f, indent=2)
    with open(REPORT_DIR / "url_dl_metrics.json", "w") as f:
        json.dump(rep, f, indent=2)

    elapsed = time.time() - t_start
    summary = {
        "status": "TRAINED",
        "model_path": str(model_save_path),
        "accuracy": round(acc, 4),
        "macro_f1": round(best_f1, 4),
        "device": str(device),
        "elapsed_seconds": round(elapsed, 2),
    }
    print(f"[Airflow Task: train_model] SUCCESS: {summary}")
    return summary


if __name__ == "__main__":
    train_model(epochs=3)
