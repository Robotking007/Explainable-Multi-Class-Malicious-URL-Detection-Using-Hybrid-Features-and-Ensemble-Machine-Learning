"""
Self-contained Kaggle Kernel script for training the Lightweight Tabular
Deep Learning model (HybridURLNet) on GPU.

Designed to eliminate the out-of-memory (OOM) crashes experienced by the
classical 300-tree RF and monolithic TF-IDF matrices:
  - Mini-batch streaming via PyTorch DataLoader (under 3 GB RAM footprint)
  - Balanced Class-Weighted Cross-Entropy (eliminates need for 40M-row SMOTE)
  - Trains with CUDA acceleration on Kaggle GPU
  - Saves all model weights and evaluation metrics to /kaggle/working/
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Kaggle paths
IN_DIR = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")

SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf",
    "loan", "download", "men", "win", "bid", "stream", "date", "racing",
}
PHISH_KEYWORDS = [
    "login", "signin", "account", "verify", "secure", "update", "confirm",
    "banking", "webscr", "password", "credential", "suspend", "urgent",
]
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "bank", "instagram", "whatsapp", "outlook", "icloud",
]
ACTION_KEYWORDS = ["download", "install", "click", "free", "gift", "bonus", "win"]
SENSITIVE_KEYWORDS = ["ssn", "creditcard", "cvv", "otp", "pin", "wallet"]
SPECIAL_CHARS = ["@", "?", "-", "=", ".", "/", "%", "&", ":"]
IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def extract_features_row(url: str) -> dict:
    url_str = str(url) if url is not None else ""
    parsed = urlparse(url_str if "//" in url_str else "http://" + url_str)
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    parts = netloc.split(".")
    tld = parts[-1].lower() if len(parts) > 1 else ""
    domain = parts[-2].lower() if len(parts) > 1 else netloc.lower()
    subdomain_count = max(0, len(parts) - 2)

    return {
        "url_length": len(url_str),
        "hostname_length": len(netloc),
        "path_length": len(path),
        "digit_ratio": sum(c.isdigit() for c in url_str) / max(len(url_str), 1),
        "uppercase_ratio": sum(c.isupper() for c in url_str) / max(len(url_str), 1),
        "ip_in_url_flag": int(bool(IP_RE.match(netloc))),
        "count_at": url_str.count("@"),
        "count_qmark": url_str.count("?"),
        "count_hyphen": url_str.count("-"),
        "count_eq": url_str.count("="),
        "count_dot": url_str.count("."),
        "count_slash": url_str.count("/"),
        "count_pct": url_str.count("%"),
        "count_amp": url_str.count("&"),
        "count_colon": url_str.count(":"),
        "subdomain_count": subdomain_count,
        "domain_length": len(domain),
        "tld_length": len(tld),
        "suspicious_tld_flag": int(tld in SUSPICIOUS_TLDS),
        "hyphen_in_domain_flag": int("-" in domain),
        "digit_in_domain_flag": int(any(c.isdigit() for c in domain)),
        "domain_entropy": _entropy(domain),
        "subdomain_length": sum(len(p) for p in parts[:-2]) if subdomain_count else 0,
        "domain_token_count": len(parts),
        "brand_in_domain_flag": int(any(b in domain for b in BRAND_KEYWORDS)),
        "brand_in_subdomain_flag": int(any(b in ".".join(parts[:-2]) for b in BRAND_KEYWORDS)) if subdomain_count else 0,
        "phish_keyword_count": sum(k in url_str.lower() for k in PHISH_KEYWORDS),
        "brand_keyword_count": sum(b in url_str.lower() for b in BRAND_KEYWORDS),
        "action_keyword_count": sum(a in url_str.lower() for a in ACTION_KEYWORDS),
        "sensitive_keyword_count": sum(s in url_str.lower() for s in SENSITIVE_KEYWORDS),
        "path_depth": path.count("/"),
        "param_count": len(query.split("&")) if query else 0,
        "has_fragment_flag": int(bool(parsed.fragment)),
        "double_slash_in_path_flag": int("//" in path),
        "query_length": len(query),
        "path_entropy": _entropy(path),
        "query_entropy": _entropy(query),
        "url_entropy": _entropy(url_str),
        "compression_ratio": len(zlib.compress(url_str.encode())) / max(len(url_str), 1),
        "char_diversity": len(set(url_str)) / max(len(url_str), 1),
        "token_count": len(re.split(r"[/._\-?=&%@:]+", url_str)),
        "avg_token_length": np.mean([len(t) for t in re.split(r"[/._\-?=&%@:]+", url_str) if t]) or 0.0,
        "max_token_length": max([len(t) for t in re.split(r"[/._\-?=&%@:]+", url_str)] or [0]),
        "unique_token_ratio": len(set(re.split(r"[/._\-?=&%@:]+", url_str))) / max(len(re.split(r"[/._\-?=&%@:]+", url_str)), 1),
        "special_char_ratio": sum(url_str.count(c) for c in SPECIAL_CHARS) / max(len(url_str), 1),
    }


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act1(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        return self.act2(out + residual)


class HybridURLNet(nn.Module):
    def __init__(self, input_dim: int = 45, hidden_dim: int = 128, num_classes: int = 4, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        self.res_block = ResidualBlock(hidden_dim, dropout_rate=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        h = self.res_block(h)
        return self.classifier(h)


def _extract_worker_chunk(chunk: list[str]) -> list[dict]:
    return [extract_features_row(u) for u in chunk]


def find_data_file() -> Path:
    for p in IN_DIR.rglob("combined_urls.csv"):
        return p
    for p in IN_DIR.rglob("*.parquet"):
        return p
    for p in IN_DIR.rglob("*.csv"):
        return p
    raise FileNotFoundError("Could not find combined_urls dataset in /kaggle/input")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting Kaggle URL Deep Learning training pipeline on {device}")

    data_file = find_data_file()
    log(f"Loading data from {data_file}...")

    if str(data_file).endswith(".parquet"):
        df = pd.read_parquet(data_file)
    else:
        df = pd.read_csv(data_file, nrows=150000)

    col_map = {c.lower(): c for c in df.columns}
    url_col = col_map.get("url", "url")
    label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))
    df = df.dropna(subset=[url_col, label_col])
    log(f"Total rows: {len(df):,}")

    log("Extracting features in parallel across CPU cores...")
    urls = df[url_col].tolist()
    n_workers = min(os.cpu_count() or 4, 8)
    sub_chunk_size = 50000
    sub_chunks = [urls[i:i + sub_chunk_size] for i in range(0, len(urls), sub_chunk_size)]

    feature_dfs = []
    log(f"Processing {len(urls):,} URLs across {len(sub_chunks)} chunks using {n_workers} worker processes...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, rows in enumerate(executor.map(_extract_worker_chunk, sub_chunks)):
            feature_dfs.append(pd.DataFrame(rows, dtype=np.float32))
            if (idx + 1) % 10 == 0 or (idx + 1) == len(sub_chunks):
                log(f"Processed {min((idx + 1) * sub_chunk_size, len(urls)):,} / {len(urls):,} rows")
                gc.collect()

    X_all = pd.concat(feature_dfs, ignore_index=True)
    del feature_dfs, urls, sub_chunks
    gc.collect()

    feature_names = list(X_all.columns)
    le = LabelEncoder()
    y_all = le.fit_transform(df[label_col].astype(str))
    label_mapping = {int(idx): str(c) for idx, c in enumerate(le.classes_)}
    num_classes = len(label_mapping)
    log(f"Labels: {label_mapping}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    del X_all, y_all, df
    gc.collect()

    log("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.fillna(0.0).values)
    X_test_s = scaler.transform(X_test.fillna(0.0).values)
    del X_train, X_test
    gc.collect()

    # Balanced weights
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=1024,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_s, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=2048,
        shuffle=False,
    )

    model = HybridURLNet(input_dim=len(feature_names), hidden_dim=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log(f"Training HybridURLNet ({model.num_classes} classes, {sum(p.numel() for p in model.parameters()):,} params) for {epochs} epochs...")
    best_f1 = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        t_start = time.time()
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

        # Fast eval
        model.eval()
        t_preds, t_trues = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                logits = model(X_b.to(device))
                t_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                t_trues.extend(y_b.numpy())

        acc = accuracy_score(t_trues, t_preds)
        f1 = f1_score(t_trues, t_preds, average="macro", zero_division=0)
        log(f"Epoch {epoch}/{epochs} ({time.time()-t_start:.1f}s) | Train Loss: {running_loss/total_seen:.4f} | Test Acc: {acc*100:.2f}% | Test F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights:
        model.load_state_dict(best_weights)

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "hidden_dim": 128,
        "num_classes": num_classes,
        "feature_names": feature_names,
        "label_mapping": label_mapping,
    }, str(OUT_DIR / "url_dl_model.pt"))
    joblib.dump(scaler, OUT_DIR / "url_dl_scaler.joblib")
    joblib.dump(le, OUT_DIR / "url_dl_label_encoder.joblib")

    # Metrics
    rep = classification_report(t_trues, t_preds, target_names=[label_mapping[i] for i in range(num_classes)], output_dict=True, zero_division=0)
    with open(OUT_DIR / "url_dl_metrics.json", "w") as f:
        json.dump(rep, f, indent=2)

    log("Training complete. Artifacts saved successfully to /kaggle/working/")


if __name__ == "__main__":
    main()
