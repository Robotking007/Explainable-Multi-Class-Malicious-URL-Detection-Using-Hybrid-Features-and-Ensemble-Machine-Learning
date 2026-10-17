"""
Training pipeline for the Lightweight Tabular Deep Learning model (HybridURLNet).

Features:
  - Uses the 45 engineered hybrid features (lexical, host, content, structural, statistical)
  - PyTorch mini-batch training (OOM-safe, streaming via DataLoader)
  - Balanced Class-Weighted Cross-Entropy (eliminates need for 40M-row SMOTE explosion)
  - Cosine Annealing learning rate schedule + AdamW
  - Dynamic INT8 quantization option for green edge inference
  - Outputs metrics JSON and saves checkpoint compatible with the serving API
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.dl_model import HybridURLNet, save_checkpoint
from url_pipeline.features import extract_features_row

MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


from concurrent.futures import ThreadPoolExecutor
import os


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _extract_chunk(urls: list[str]) -> list[dict]:
    return [extract_features_row(u) for u in urls]


def extract_features_df(df: pd.DataFrame, url_col: str = "url", max_workers: Optional[int] = None) -> pd.DataFrame:
    """Extracts all 53 hybrid features (lexical, host, structural, keyword, statistical, and Abuse.ch) in parallel."""
    # Pre-warm singleton threat index on main thread to avoid thread contention on first miss
    try:
        try:
            from url_pipeline.abuse_features import get_abuse_index
        except ImportError:
            from src.url_pipeline.abuse_features import get_abuse_index
        get_abuse_index()
    except Exception:
        pass

    urls = [str(u) for u in df[url_col]]
    n_workers = max_workers or min(os.cpu_count() or 4, 8)
    chunk_size = max(1, len(urls) // (n_workers * 4))
    url_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]


    dfs = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for chunk_result in executor.map(_extract_chunk, url_chunks):
            dfs.append(pd.DataFrame(chunk_result, dtype=np.float32))

    return pd.concat(dfs, ignore_index=True)



def load_stratified_dataset(data_path: Path, target_total: int = 100000) -> pd.DataFrame:
    """Streams combined_urls.csv / malicious_phish.csv and injects fresh Abuse.ch + PhishTank samples to collect a balanced sample."""
    log(f"Streaming from actual dataset {data_path} to assemble ~{target_total:,} balanced rows...")
    target_per_class = target_total // 4
    collected = {c: [] for c in ["benign", "phishing", "malware", "defacement"]}

    dataset_dir = data_path.parent

    # 1. Inject fresh Abuse.ch ThreatFox & URLhaus malware URLs
    try:
        try:
            from url_pipeline.abuse_features import get_abuse_index
        except ImportError:
            from src.url_pipeline.abuse_features import get_abuse_index
        idx = get_abuse_index()
        malware_urls = list(idx.tf_urls | idx.uh_urls)[:target_per_class // 3]
        if malware_urls:
            collected["malware"].append(pd.DataFrame({"url": malware_urls, "type": "malware"}))
            log(f"Injected {len(malware_urls):,} real-time Abuse.ch (ThreatFox/URLhaus) malware samples.")
    except Exception as e:
        log(f"Warning injecting Abuse.ch data: {e}")

    # 2. Inject fresh PhishTank verified phishing URLs
    pt_path = dataset_dir / "verified_online.csv"
    if pt_path.exists():
        try:
            pt_df = pd.read_csv(pt_path, usecols=["url"], nrows=target_per_class // 3)
            pt_df["type"] = "phishing"
            collected["phishing"].append(pt_df)
            log(f"Injected {len(pt_df):,} real-time PhishTank verified phishing samples.")
        except Exception as e:
            log(f"Warning injecting PhishTank data: {e}")

    # 2b. Inject top authoritative root & popular benign domains so the model learns legitimate clean apex structures
    top_benign = [
        "google.com", "www.google.com", "youtube.com", "www.youtube.com", "facebook.com",
        "wikipedia.org", "en.wikipedia.org", "amazon.com", "www.amazon.com", "yahoo.com",
        "reddit.com", "www.reddit.com", "netflix.com", "github.com", "linkedin.com",
        "microsoft.com", "apple.com", "instagram.com", "twitter.com", "x.com",
        "cloudflare.com", "bing.com", "ebay.com", "twitch.tv", "adobe.com",
        "spotify.com", "cnn.com", "bbc.co.uk", "nytimes.com", "theguardian.com",
        "stackoverflow.com", "medium.com", "quora.com", "imdb.com", "pinterest.com",
        "tumblr.com", "craigslist.org", "aliexpress.com", "walmart.com", "espn.com",
        "chase.com", "bankofamerica.com", "wellsfargo.com", "paypal.com", "stripe.com",
        "dropbox.com", "salesforce.com", "wordpress.org", "harvard.edu", "mit.edu",
        "stanford.edu", "berkeley.edu", "ox.ac.uk", "cam.ac.uk", "nih.gov", "cdc.gov",
        "nasa.gov", "who.int", "un.org", "weather.com", "forbes.com", "bloomberg.com",
        "reuters.com", "wsj.com", "usatoday.com", "archive.org", "vimeo.com", "brown.edu"
    ]
    benign_expanded = []
    for d in top_benign:
        benign_expanded.append(d)
        benign_expanded.append(f"https://{d}")
        benign_expanded.append(f"http://{d}/")
        if not d.startswith("www.") and "." in d and not d.startswith("en."):
            benign_expanded.append(f"www.{d}")
    benign_unique = list(set(benign_expanded))
    collected["benign"].append(pd.DataFrame({"url": benign_unique, "type": "benign"}))
    log(f"Injected {len(benign_unique)} authoritative benign root & portal URLs.")

    # 3. Read in chunks of 100k from base corpus
    chunksize = 100000
    for chunk in pd.read_csv(data_path, chunksize=chunksize, usecols=["url", "type"]):
        chunk = chunk.dropna(subset=["url", "type"])
        chunk["type"] = chunk["type"].astype(str).str.lower().str.strip()
        # Filter out non-URL garbage strings
        chunk = chunk[chunk["url"].astype(str).str.contains(r"\.", na=False)]
        chunk = chunk[~chunk["url"].astype(str).str.startswith(("#", "$", "%"))]

        for cls_name in list(collected.keys()):
            current_count = sum(len(x) for x in collected[cls_name])
            if current_count < target_per_class:
                sub = chunk[chunk["type"] == cls_name]
                needed = target_per_class - current_count
                if len(sub) > 0:
                    collected[cls_name].append(sub.iloc[:needed])

        counts = {c: sum(len(x) for x in chunks) for c, chunks in collected.items()}
        if all(counts[c] >= target_per_class for c in collected):
            break

    dfs = []
    for cls_name, chunk_list in collected.items():
        if chunk_list:
            dfs.append(pd.concat(chunk_list, ignore_index=True))

    full_df = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    log(f"Assembled stratified sample: {len(full_df):,} total rows with Abuse.ch & PhishTank feeds.")
    return full_df


def train_dl_pipeline(
    data_path: Path,
    sample_size: int = 50000,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device_name: str = "auto",
    random_state: int = 42,
    save_prefix: str = "url_dl",
) -> Dict[str, Any]:
    """Runs the Deep Learning training pipeline."""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    log(f"Using compute device: {device}")

    # 1. Load dataset
    log(f"Loading data from {data_path} (target size={sample_size})...")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    if ("combined_urls" in str(data_path) or "malicious_phish" in str(data_path) or "merged_clean" in str(data_path)) and sample_size > 0:
        df = load_stratified_dataset(data_path, target_total=sample_size)

    elif str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        if sample_size > 0:
            df = pd.read_csv(data_path, nrows=sample_size * 2)
        else:
            df = pd.read_csv(data_path)

    # Standardize column names
    col_map = {c.lower(): c for c in df.columns}
    url_col = col_map.get("url", "url")
    label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))

    df = df.dropna(subset=[url_col, label_col])
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    log(f"Dataset ready: {len(df):,} rows.")
    log(f"Class distribution:\n{df[label_col].value_counts()}")

    # 2. Extract features
    log("Extracting hybrid features (45 lexical, host, keyword, structural, statistical)...")
    t0 = time.time()
    X_raw = extract_features_df(df, url_col=url_col)
    feature_names = list(X_raw.columns)
    log(f"Extracted {len(feature_names)} features for {len(X_raw):,} rows in {time.time()-t0:.1f}s.")

    # 3. Encode labels
    le = LabelEncoder()
    y_raw = le.fit_transform(df[label_col].astype(str))
    label_mapping = {int(idx): str(cls_name) for idx, cls_name in enumerate(le.classes_)}
    num_classes = len(label_mapping)
    log(f"Target classes ({num_classes}): {label_mapping}")

    # 4. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=random_state, stratify=y_raw
    )
    log(f"Train split: {len(X_train):,} | Test split: {len(X_test):,}")

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0.0).values)
    X_test_scaled = scaler.transform(X_test.fillna(0.0).values)

    # 6. Balanced class weights (Energy-efficient alternative to 40M-row SMOTE)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
    log(f"Computed balanced class weights: {dict(zip(classes, [round(w, 3) for w in weights]))}")

    # 7. PyTorch DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 8. Instantiate Model
    model = HybridURLNet(
        input_dim=len(feature_names),
        hidden_dim=128,
        num_classes=num_classes,
        dropout_rate=0.2,
    ).to(device)
    log(f"Instantiated HybridURLNet: {model.count_parameters():,} trainable parameters.")

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 9. Training Loop
    log(f"Starting training for {epochs} epochs...")
    best_f1 = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Evaluate on test set
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.to(device)
                logits = model(X_b)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                test_preds.extend(preds)
                test_trues.extend(y_b.numpy())

        test_acc = accuracy_score(test_trues, test_preds)
        test_macro_f1 = f1_score(test_trues, test_preds, average="macro", zero_division=0)

        log(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Acc: {test_acc*100:.2f}% | Test Macro F1: {test_macro_f1:.4f}"
        )

        if test_macro_f1 > best_f1:
            best_f1 = test_macro_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # 10. Final Detailed Evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            logits = model(X_b)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(y_b.numpy())

    final_acc = accuracy_score(all_trues, all_preds)
    final_f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)
    precision, recall, fscore, support = precision_recall_fscore_support(
        all_trues, all_preds, zero_division=0
    )

    per_class_metrics = {}
    for idx, cls_name in label_mapping.items():
        per_class_metrics[cls_name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(fscore[idx]),
            "support": int(support[idx]),
        }

    report_str = classification_report(
        all_trues, all_preds, target_names=[label_mapping[i] for i in range(num_classes)], zero_division=0
    )
    log(f"Final Classification Report:\n{report_str}")

    metrics_payload = {
        "module": "Module B - URL Tabular Deep Learning",
        "model_architecture": "HybridURLNet (PyTorch Tabular Residual)",
        "parameter_count": model.count_parameters(),
        "input_features": len(feature_names),
        "test_accuracy": float(final_acc),
        "test_macro_f1": float(final_f1),
        "per_class_metrics": per_class_metrics,
        "classes": list(label_mapping.values()),
        "trained_samples": len(X_train),
        "test_samples": len(X_test),
        "quantization_supported": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 11. Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = MODEL_DIR / f"{save_prefix}_model.pt"
    scaler_path = MODEL_DIR / f"{save_prefix}_scaler.joblib"
    le_path = MODEL_DIR / f"{save_prefix}_label_encoder.joblib"
    metrics_path = REPORTS_DIR / f"{save_prefix}_metrics.json"

    # Save scaler and label encoder
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)

    # Save model checkpoint
    scaler_state = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
    }
    save_checkpoint(
        model=model,
        scaler_state=scaler_state,
        feature_names=feature_names,
        label_mapping=label_mapping,
        save_path=pt_path,
        extra_metrics=metrics_payload,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    log(f"Saved model checkpoint -> {pt_path}")
    log(f"Saved scaler -> {scaler_path}")
    log(f"Saved label encoder -> {le_path}")
    log(f"Saved evaluation metrics -> {metrics_path}")

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(description="Train Tabular Deep Learning model for URL detection.")
    default_data = (
        "Dataset/merged_clean_urls.csv"
        if (ROOT / "Dataset/merged_clean_urls.csv").exists()
        else ("Dataset/malicious_phish.csv" if (ROOT / "Dataset/malicious_phish.csv").exists() else "Dataset/combined_urls.csv")
    )
    parser.add_argument("--data", type=str, default=default_data, help="Path to URL dataset")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of rows to sample for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cpu/cuda/auto)")
    parser.add_argument("--prefix", type=str, default="url_dl", help="Prefix for saved model artifacts")

    args = parser.parse_args()
    data_path = ROOT / args.data
    train_dl_pipeline(
        data_path=data_path,
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
        save_prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
