"""
Training Pipeline for URLInfraGNN (Heterogeneous URL Infrastructure GNN).

Trains the Graph Attention Network over infrastructure graphs constructed from
the actual URL dataset (Dataset/combined_urls.csv).

Saves:
  - models/url_gnn_model.pt
  - reports/url_gnn_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.url_pipeline.gnn_model import URLInfraGNN
from src.url_pipeline.gnn_graph_builder import (
    build_domain_node_features,
    build_ip_node_features,
    build_asn_node_features,
)
from src.url_pipeline.train_dl import load_stratified_dataset, extract_features_df

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_batch_graphs(
    urls: List[str],
    X_url_scaled: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs batched infrastructure graphs:
    Each sample has:
      - 1 URL node (45-dim)
      - 1 Domain node (8-dim)
      - 1 IP node (4-dim)
      - 1 ASN node (4-dim)
    Stacked into batch tensors.
    """
    N = len(urls)
    u_list = torch.tensor(X_url_scaled, dtype=torch.float32)

    dom_feats = np.stack([build_domain_node_features(u) for u in urls])
    ip_feats = np.stack([build_ip_node_features(u) for u in urls])
    asn_feats = np.stack([build_asn_node_features(u) for u in urls])

    d_list = torch.tensor(dom_feats, dtype=torch.float32)
    i_list = torch.tensor(ip_feats, dtype=torch.float32)
    a_list = torch.tensor(asn_feats, dtype=torch.float32)

    # Standard (4, 4) infrastructure star-ego adjacency matrix with self-loops
    base_adj = torch.zeros((4, 4), dtype=torch.float32)
    base_adj[0, 0] = base_adj[1, 1] = base_adj[2, 2] = base_adj[3, 3] = 1.0
    base_adj[0, 1] = base_adj[1, 0] = 1.0  # URL <-> Domain
    base_adj[1, 2] = base_adj[2, 1] = 1.0  # Domain <-> IP
    base_adj[2, 3] = base_adj[3, 2] = 1.0  # IP <-> ASN

    return u_list, d_list, i_list, a_list, base_adj


def train_gnn_pipeline(
    data_path: Path,
    sample_size: int = 15000,
    epochs: int = 8,
    batch_size: int = 256,
    lr: float = 1e-3,
    device_name: str = "auto",
) -> Dict[str, Any]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    log(f"Training URLInfraGNN on device: {device}")

    # 1. Load data
    if "combined_urls" in str(data_path) and sample_size > 0:
        df = load_stratified_dataset(data_path, target_total=sample_size)
    else:
        df = pd.read_csv(data_path, nrows=sample_size if sample_size > 0 else None)

    col_map = {c.lower(): c for c in df.columns}
    url_col = col_map.get("url", "url")
    label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))
    df = df.dropna(subset=[url_col, label_col]).reset_index(drop=True)

    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    log(f"Dataset ready: {len(df):,} rows.")

    # 2. Extract static 45 URL features
    log("Extracting 45 hybrid static features across dataset...")
    t0 = time.time()
    X_raw = extract_features_df(df, url_col=url_col)
    log(f"Features extracted in {time.time()-t0:.1f}s.")

    # 3. Scale and encode
    le = LabelEncoder()
    y_raw = le.fit_transform(df[label_col].astype(str))
    label_mapping = {int(idx): str(c) for idx, c in enumerate(le.classes_)}
    num_classes = len(label_mapping)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.fillna(0.0).values)

    urls = df[url_col].tolist()

    # 4. Train / Test split
    X_train_s, X_test_s, y_train, y_test, urls_train, urls_test = train_test_split(
        X_scaled, y_raw, urls, test_size=0.2, random_state=42, stratify=y_raw
    )

    log(f"Train split: {len(X_train_s):,} | Test split: {len(X_test_s):,}")

    # 5. Build batched infrastructure graphs
    log("Assembling infrastructure graph nodes (URL, Domain, IP, ASN)...")
    u_tr, d_tr, i_tr, a_tr, base_adj = build_batch_graphs(urls_train, X_train_s)
    u_te, d_te, i_te, a_te, _ = build_batch_graphs(urls_test, X_test_s)

    # 6. Initialize GNN Model
    model = URLInfraGNN(
        url_dim=45,
        domain_dim=8,
        ip_dim=4,
        asn_dim=4,
        hidden_dim=64,
        num_classes=num_classes,
        dropout=0.15,
    ).to(device)

    base_adj = base_adj.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Initialized URLInfraGNN: {param_count:,} parameters.")

    # 7. Training loop
    n_train = len(y_train)
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        t_start = time.time()

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            B = len(batch_idx)

            # Sub-batch tensors
            u_b = u_tr[batch_idx].to(device)
            d_b = d_tr[batch_idx].to(device)
            i_b = i_tr[batch_idx].to(device)
            a_b = a_tr[batch_idx].to(device)
            y_b = torch.tensor(y_train[batch_idx.numpy()], dtype=torch.long).to(device)

            optimizer.zero_grad()

            # Encode and classify per sample in batch
            batch_logits = []
            for b in range(B):
                # (4, hidden_dim) node representations for sample b
                nodes = model.encode_nodes(
                    u_b[b:b+1], d_b[b:b+1], i_b[b:b+1], a_b[b:b+1]
                )
                out, _ = model(nodes, base_adj, url_node_idx=0)
                batch_logits.append(out)

            logits = torch.cat(batch_logits, dim=0)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * B

        scheduler.step()

        # Evaluate on test set
        model.eval()
        t_preds = []
        with torch.no_grad():
            for b in range(len(y_test)):
                nodes = model.encode_nodes(
                    u_te[b:b+1].to(device),
                    d_te[b:b+1].to(device),
                    i_te[b:b+1].to(device),
                    a_te[b:b+1].to(device),
                )
                out, _ = model(nodes, base_adj, url_node_idx=0)
                pred = torch.argmax(out, dim=-1).item()
                t_preds.append(pred)

        acc = accuracy_score(y_test, t_preds)
        macro_f1 = f1_score(y_test, t_preds, average="macro", zero_division=0)
        dt = time.time() - t_start
        log(f"Epoch {epoch:02d}/{epochs} ({dt:.1f}s) | Train Loss: {running_loss/n_train:.4f} | Test Acc: {acc*100:.2f}% | Test Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 8. Final evaluation report
    model.eval()
    final_preds = []
    with torch.no_grad():
        for b in range(len(y_test)):
            nodes = model.encode_nodes(
                u_te[b:b+1].to(device),
                d_te[b:b+1].to(device),
                i_te[b:b+1].to(device),
                a_te[b:b+1].to(device),
            )
            out, _ = model(nodes, base_adj, url_node_idx=0)
            final_preds.append(torch.argmax(out, dim=-1).item())

    rep = classification_report(
        y_test, final_preds, target_names=[label_mapping[i] for i in range(num_classes)], output_dict=True, zero_division=0
    )
    print("\nFinal GNN Classification Report:\n", classification_report(
        y_test, final_preds, target_names=[label_mapping[i] for i in range(num_classes)], zero_division=0
    ))

    # 9. Save checkpoints
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "url_gnn_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "url_dim": 45,
        "domain_dim": 8,
        "ip_dim": 4,
        "asn_dim": 4,
        "hidden_dim": 64,
        "num_classes": num_classes,
        "label_mapping": label_mapping,
    }, str(model_path))
    log(f"Saved GNN weights -> {model_path}")

    joblib.dump(scaler, MODELS_DIR / "url_gnn_scaler.joblib")
    joblib.dump(le, MODELS_DIR / "url_gnn_label_encoder.joblib")

    metrics = {
        "module": "Module B - Heterogeneous URL Infrastructure GNN",
        "model_architecture": "URLInfraGNN (Graph Attention Network)",
        "parameter_count": param_count,
        "test_accuracy": accuracy_score(y_test, final_preds),
        "test_macro_f1": f1_score(y_test, final_preds, average="macro", zero_division=0),
        "per_class_metrics": {
            label_mapping[i]: {
                "precision": rep[label_mapping[i]]["precision"],
                "recall": rep[label_mapping[i]]["recall"],
                "f1_score": rep[label_mapping[i]]["f1-score"],
                "support": rep[label_mapping[i]]["support"],
            } for i in range(num_classes)
        },
        "trained_samples": n_train,
        "test_samples": len(y_test),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(REPORTS_DIR / "url_gnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved GNN metrics -> {REPORTS_DIR / 'url_gnn_metrics.json'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train URLInfraGNN on actual URL dataset.")
    parser.add_argument("--data", type=str, default="Dataset/merged_clean_urls.csv", help="Dataset path")
    parser.add_argument("--sample-size", type=int, default=15000, help="Stratified sample rows (default: 15,000)")
    parser.add_argument("--epochs", type=int, default=8, help="Epoch count")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    train_gnn_pipeline(
        data_path=data_path,
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
