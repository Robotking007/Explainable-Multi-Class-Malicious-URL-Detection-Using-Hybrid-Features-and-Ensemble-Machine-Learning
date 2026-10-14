"""
Self-contained Kaggle Kernel script for training the Lightweight Tabular
Deep Learning model (HybridURLNet) on GPU.

- Mini-batch streaming via PyTorch DataLoader
- Balanced Class-Weighted Cross-Entropy
- 45 hybrid features (lexical, host, keywords, structural, statistical)
- Stratified sampling across all 27M rows: preserves 100% minority classes (malware, defacement) + representative benign & phishing
- In-memory ThreadPoolExecutor for fast feature extraction without IPC overhead
- Trains with CUDA acceleration on Kaggle GPU
- Automatic model registry publish to Hugging Face Hub (siddudavant/cyberxai-models)
- Error handling: writes CRASH_TRACEBACK.txt on error
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse
import zlib

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

# Kaggle paths & HF config
IN_DIR = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")
HF_TOKEN = ""  # Populated dynamically by kernel push script or env
HF_REPO_ID = "siddudavant/cyberxai-models"

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
TOKEN_SPLIT_RE = re.compile(r"[/._\-?=&%@:]+")

DEFAULT_ZERO_ROW = {
    "url_length": 0.0,
    "hostname_length": 0.0,
    "path_length": 0.0,
    "digit_ratio": 0.0,
    "uppercase_ratio": 0.0,
    "ip_in_url_flag": 0.0,
    "count_at": 0.0,
    "count_qmark": 0.0,
    "count_hyphen": 0.0,
    "count_eq": 0.0,
    "count_dot": 0.0,
    "count_slash": 0.0,
    "count_pct": 0.0,
    "count_amp": 0.0,
    "count_colon": 0.0,
    "subdomain_count": 0.0,
    "domain_length": 0.0,
    "tld_length": 0.0,
    "suspicious_tld_flag": 0.0,
    "hyphen_in_domain_flag": 0.0,
    "digit_in_domain_flag": 0.0,
    "domain_entropy": 0.0,
    "subdomain_length": 0.0,
    "domain_token_count": 0.0,
    "brand_in_domain_flag": 0.0,
    "brand_in_subdomain_flag": 0.0,
    "phish_keyword_count": 0.0,
    "brand_keyword_count": 0.0,
    "action_keyword_count": 0.0,
    "sensitive_keyword_count": 0.0,
    "path_depth": 0.0,
    "param_count": 0.0,
    "has_fragment_flag": 0.0,
    "double_slash_in_path_flag": 0.0,
    "query_length": 0.0,
    "path_entropy": 0.0,
    "query_entropy": 0.0,
    "url_entropy": 0.0,
    "compression_ratio": 0.0,
    "char_diversity": 0.0,
    "token_count": 0.0,
    "avg_token_length": 0.0,
    "max_token_length": 0.0,
    "unique_token_ratio": 0.0,
    "special_char_ratio": 0.0,
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def extract_features_row(url: str) -> dict:
    try:
        url_str = str(url) if url is not None else ""
        url_lower = url_str.lower()
        parsed = urlparse(url_str if "//" in url_str else "http://" + url_str)
        netloc = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""

        parts = netloc.split(".")
        tld = parts[-1].lower() if len(parts) > 1 else ""
        domain = parts[-2].lower() if len(parts) > 1 else netloc.lower()
        subdomain_count = max(0, len(parts) - 2)

        toks = [len(t) for t in TOKEN_SPLIT_RE.split(url_str) if t]
        all_toks = TOKEN_SPLIT_RE.split(url_str)
        u_len = max(len(url_str), 1)

        return {
            "url_length": float(len(url_str)),
            "hostname_length": float(len(netloc)),
            "path_length": float(len(path)),
            "digit_ratio": float(sum(c.isdigit() for c in url_str) / u_len),
            "uppercase_ratio": float(sum(c.isupper() for c in url_str) / u_len),
            "ip_in_url_flag": float(bool(IP_RE.match(netloc))),
            "count_at": float(url_str.count("@")),
            "count_qmark": float(url_str.count("?")),
            "count_hyphen": float(url_str.count("-")),
            "count_eq": float(url_str.count("=")),
            "count_dot": float(url_str.count(".")),
            "count_slash": float(url_str.count("/")),
            "count_pct": float(url_str.count("%")),
            "count_amp": float(url_str.count("&")),
            "count_colon": float(url_str.count(":")),
            "subdomain_count": float(subdomain_count),
            "domain_length": float(len(domain)),
            "tld_length": float(len(tld)),
            "suspicious_tld_flag": float(tld in SUSPICIOUS_TLDS),
            "hyphen_in_domain_flag": float("-" in domain),
            "digit_in_domain_flag": float(any(c.isdigit() for c in domain)),
            "domain_entropy": float(_entropy(domain)),
            "subdomain_length": float(sum(len(p) for p in parts[:-2]) if subdomain_count else 0),
            "domain_token_count": float(len(parts)),
            "brand_in_domain_flag": float(any(b in domain for b in BRAND_KEYWORDS)),
            "brand_in_subdomain_flag": float(any(b in ".".join(parts[:-2]) for b in BRAND_KEYWORDS) if subdomain_count else 0),
            "phish_keyword_count": float(sum(k in url_lower for k in PHISH_KEYWORDS)),
            "brand_keyword_count": float(sum(b in url_lower for b in BRAND_KEYWORDS)),
            "action_keyword_count": float(sum(a in url_lower for a in ACTION_KEYWORDS)),
            "sensitive_keyword_count": float(sum(s in url_lower for s in SENSITIVE_KEYWORDS)),
            "path_depth": float(path.count("/")),
            "param_count": float(len(query.split("&")) if query else 0),
            "has_fragment_flag": float(bool(parsed.fragment)),
            "double_slash_in_path_flag": float("//" in path),
            "query_length": float(len(query)),
            "path_entropy": float(_entropy(path)),
            "query_entropy": float(_entropy(query)),
            "url_entropy": float(_entropy(url_str)),
            "compression_ratio": float(len(zlib.compress(url_str.encode("utf-8", "replace"))) / u_len),
            "char_diversity": float(len(set(url_str)) / u_len),
            "token_count": float(len(all_toks)),
            "avg_token_length": float(sum(toks) / len(toks)) if toks else 0.0,
            "max_token_length": float(max(toks)) if toks else 0.0,
            "unique_token_ratio": float(len(set(all_toks)) / max(len(all_toks), 1)),
            "special_char_ratio": float(sum(url_str.count(c) for c in SPECIAL_CHARS) / u_len),
        }
    except Exception:
        return DEFAULT_ZERO_ROW.copy()


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


def find_data_file() -> Path:
    log("Inspecting /kaggle/input directory...")
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_file():
            log(f"  Found file: {p} ({p.stat().st_size:,} bytes)")

    for p in IN_DIR.rglob("combined_urls.csv"):
        log(f"Selected dataset file: {p}")
        return p
    for p in IN_DIR.rglob("*.parquet"):
        log(f"Selected parquet file: {p}")
        return p
    for p in IN_DIR.rglob("*.csv"):
        log(f"Selected csv file: {p}")
        return p
    raise FileNotFoundError("Could not find combined_urls.csv anywhere in /kaggle/input")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting Kaggle URL Deep Learning training pipeline on {device}")

    data_file = find_data_file()
    log(f"Loading data from {data_file}...")

    if str(data_file).endswith(".parquet"):
        df = pd.read_parquet(data_file)
        col_map = {c.lower(): c for c in df.columns}
        url_col = col_map.get("url", "url")
        label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))
        df = df.dropna(subset=[url_col, label_col])
    else:
        log("Streaming dataset from CSV in chunks with stratified balance across all 27M rows...")
        chunks = []
        # Retain 100% of minority attack classes (malware & defacement),
        # and sample 100k benign + 100k phishing uniformly across the full dataset.
        target_per_maj = 100_000
        total_rows_approx = 27_022_952
        maj_sample_rate = target_per_maj / max(total_rows_approx / 2, 1)

        chunksize = 1_000_000
        for i, chunk in enumerate(pd.read_csv(data_file, chunksize=chunksize)):
            col_map = {c.lower(): c for c in chunk.columns}
            u_col = col_map.get("url", "url")
            l_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))
            chunk = chunk.dropna(subset=[u_col, l_col])
            chunk[u_col] = chunk[u_col].astype(str).str.strip()
            chunk[l_col] = chunk[l_col].astype(str).str.lower().str.strip()

            chunk = chunk[chunk[u_col].str.len() > 3]
            chunk = chunk[chunk[l_col].isin(["benign", "phishing", "malware", "defacement"])]

            # Separate minority vs majority classes
            is_minority = chunk[l_col].isin(["defacement", "malware"])
            minority_df = chunk[is_minority]
            majority_df = chunk[~is_minority].sample(frac=maj_sample_rate, random_state=42 + i) if not chunk[~is_minority].empty else chunk[~is_minority]

            sampled_chunk = pd.concat([minority_df, majority_df])
            chunks.append(sampled_chunk[[u_col, l_col]])
            if (i + 1) % 5 == 0 or (i + 1) >= 27:
                log(f"  Chunk {i+1}: running sampled total = {sum(len(c) for c in chunks):,} rows")

        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        col_map = {c.lower(): c for c in df.columns}
        url_col = col_map.get("url", "url")
        label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))

    log(f"Total balanced dataset rows: {len(df):,}")
    log(f"Class distribution:\n{df[label_col].value_counts()}")

    log("Extracting 45 hybrid features using in-memory ThreadPoolExecutor...")
    urls = df[url_col].tolist()
    labels = df[label_col].tolist()
    del df
    gc.collect()

    batch_chunk_size = 25000
    all_feature_rows = []
    t_feat_start = time.time()

    with ThreadPoolExecutor(max_workers=4) as executor:
        for b_start in range(0, len(urls), batch_chunk_size):
            b_urls = urls[b_start:b_start + batch_chunk_size]
            batch_rows = list(executor.map(extract_features_row, b_urls))
            all_feature_rows.extend(batch_rows)
            log(f"  Extracted {min(b_start + len(b_urls), len(urls)):,} / {len(urls):,} URLs ({time.time() - t_feat_start:.1f}s)")

    log("Constructing feature DataFrame...")
    X_all = pd.DataFrame(all_feature_rows, dtype=np.float32).fillna(0.0)
    del all_feature_rows, urls
    gc.collect()

    feature_names = list(X_all.columns)
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    label_mapping = {int(idx): str(c) for idx, c in enumerate(le.classes_)}
    num_classes = len(label_mapping)
    log(f"Target classes ({num_classes}): {label_mapping}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all.values, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    del X_all, y_all, labels
    gc.collect()

    log("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
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
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log(f"Training HybridURLNet ({model.num_classes} classes, {sum(p.numel() for p in model.parameters()):,} params) for {epochs} epochs on {device}...")
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
        log(f"Epoch {epoch}/{epochs} ({time.time()-t_start:.1f}s) | Train Loss: {running_loss/max(total_seen, 1):.4f} | Test Acc: {acc*100:.2f}% | Test F1: {f1:.4f}")

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

    # Direct push to Hugging Face Hub from Kaggle
    hf_tok = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if hf_tok:
        log(f"Uploading trained model artifacts directly to Hugging Face Hub ({HF_REPO_ID})...")
        try:
            try:
                import huggingface_hub
            except ImportError:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)

            from huggingface_hub import HfApi
            api = HfApi(token=hf_tok)
            for artifact_name in [
                "url_dl_model.pt",
                "url_dl_scaler.joblib",
                "url_dl_label_encoder.joblib",
                "url_dl_metrics.json",
            ]:
                fpath = OUT_DIR / artifact_name
                if fpath.exists():
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=artifact_name,
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        token=hf_tok,
                    )
                    log(f"Uploaded {artifact_name} to Hugging Face: {HF_REPO_ID}")
            log("All artifacts successfully published to Hugging Face Hub!")
        except Exception as e:
            log(f"Hugging Face Hub direct upload warning: {e}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        main()
    except Exception as exc:
        import traceback
        with open(OUT_DIR / "CRASH_TRACEBACK.txt", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise
