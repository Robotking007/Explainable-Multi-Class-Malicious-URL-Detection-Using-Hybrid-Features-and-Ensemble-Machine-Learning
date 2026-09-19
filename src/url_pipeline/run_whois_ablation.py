"""
WHOIS ablation study, reproducing [P1]'s own methodology at [P1]'s own
scale (Khedekar & Pawar, Frontiers in Computer Science, 2026, Section 4.6
"Impact of WHOIS features on detection performance").

[P1] trained on exactly 5,000 URLs specifically because that's the scale
live WHOIS lookups are practical at; this script reproduces that choice
rather than attempting WHOIS across this project's full 1M+-row corpus
(see whois_features.py's docstring for why that's infeasible).

Pipeline: sample 5,000 URLs (stratified across the 4 classes) from
Dataset/combined_urls.csv -> extract the existing 45 engineered features +
5 WHOIS features per URL (live lookups, cached per domain) -> train the
same RF+XGBoost+LR+NB Voting Classifier twice, once with WHOIS features
included and once without -> report both results side by side, exactly
matching [P1] Table 5's "With WHOIS" / "Without WHOIS" comparison.

Usage:
    python src/url_pipeline/run_whois_ablation.py --sample-size 5000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import build_static_features  # noqa: E402
from url_pipeline.whois_features import get_whois_features_for_url, WHOIS_DEFAULTS  # noqa: E402

DATA_PATH = ROOT / "Dataset" / "combined_urls.csv"
REPORT_DIR = ROOT / "reports"
WHOIS_COLS = list(WHOIS_DEFAULTS.keys())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_xgb_device() -> str:
    try:
        import xgboost as xgb_mod
        d = xgb_mod.DMatrix(np.zeros((4, 2)), label=[0, 1, 0, 1])
        xgb_mod.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


def build_whois_block(urls: pd.Series, max_workers: int = 20) -> pd.DataFrame:
    """WHOIS lookups are I/O-bound (waiting on a network socket per query),
    so they parallelize well even under Python's GIL -- run concurrently via
    a thread pool rather than sequentially. Measured directly: sequential
    lookups projected ~2.6 hours for 3,999 URLs (partly because of the
    now-fixed unbounded timeout, partly because each lookup blocks the next
    for no reason); concurrent lookups with 20 workers turn that into
    (wall time) / ~20, i.e. minutes rather than hours."""
    import concurrent.futures

    n = len(urls)
    results = [None] * n
    t0 = time.time()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(get_whois_features_for_url, u): i for i, u in enumerate(urls)}
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = dict(WHOIS_DEFAULTS)
            completed += 1
            if completed % 100 == 0 or completed == n:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (n - completed) / rate if rate > 0 else 0
                log(f"  WHOIS lookups: {completed}/{n} ({elapsed:.0f}s elapsed, "
                    f"~{remaining:.0f}s remaining, {max_workers} concurrent workers)")

    return pd.DataFrame(results, columns=WHOIS_COLS)


def train_and_eval(X_train, y_train, X_test, y_test, xgb_device, label):
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # k_neighbors must be < smallest class size; at 5,000-URL scale this is
    # never binding (SMOTE's default k=5 is fine), but keeps this script
    # robust for smaller ad-hoc runs too.
    min_class_size = y_train.value_counts().min()
    k_neighbors = max(1, min(5, min_class_size - 1))
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_bal, y_bal = smote.fit_resample(X_train_s, y_train)

    le = LabelEncoder()
    y_bal_enc = le.fit_transform(y_bal)
    y_test_enc = le.transform(y_test)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=6)
    xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42,
                         n_jobs=6, eval_metric="mlogloss", tree_method="hist", device=xgb_device)
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=4)
    nb = GaussianNB()
    voting = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lr", lr), ("nb", nb)],
                               voting="soft", n_jobs=None)
    voting.fit(X_bal, y_bal_enc)

    proba = voting.predict_proba(X_test_s)
    y_pred = proba.argmax(axis=1)
    acc = accuracy_score(y_test_enc, y_pred)
    prec = precision_score(y_test_enc, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test_enc, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test_enc, proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    log(f"[{label}] Accuracy={acc:.4f} Macro-P={prec:.4f} Macro-R={rec:.4f} "
        f"Macro-F1={f1:.4f} Macro-ROC-AUC={roc_auc:.4f}")
    return {"accuracy": acc, "macro_precision": prec, "macro_recall": rec,
            "macro_f1": f1, "macro_roc_auc": roc_auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=5000,
                     help="Matches [P1]'s own dataset size (5,000 URLs) -- the scale live WHOIS is practical at.")
    args = ap.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)

    log(f"Loading and stratified-sampling {args.sample_size} URLs from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["url", "type"])
    df = df.groupby("type", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max(1, int(args.sample_size * len(g) / len(df)))), random_state=42),
        include_groups=True,
    )
    log(f"Sampled {len(df)} rows, class distribution:\n{df['type'].value_counts()}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["type"])

    log("Extracting the existing 45 engineered features...")
    X_train_static = build_static_features(train_df["url"]).reset_index(drop=True)
    X_test_static = build_static_features(test_df["url"]).reset_index(drop=True)

    log(f"Running live WHOIS lookups for {len(train_df)} train URLs (this takes a while -- "
        f"~2-4s per unique domain, cached per domain)...")
    X_train_whois = build_whois_block(train_df["url"].reset_index(drop=True))
    log(f"Running live WHOIS lookups for {len(test_df)} test URLs...")
    X_test_whois = build_whois_block(test_df["url"].reset_index(drop=True))

    log(f"WHOIS availability in train sample: {X_train_whois['whois_available'].mean():.1%}")

    y_train = train_df["type"].reset_index(drop=True)
    y_test = test_df["type"].reset_index(drop=True)

    xgb_device = detect_xgb_device()
    log(f"XGBoost device: {xgb_device}")

    log("=== Training WITHOUT WHOIS ===")
    without_metrics = train_and_eval(X_train_static, y_train, X_test_static, y_test, xgb_device, "without_whois")

    log("=== Training WITH WHOIS ===")
    X_train_with = pd.concat([X_train_static, X_train_whois], axis=1)
    X_test_with = pd.concat([X_test_static, X_test_whois], axis=1)
    with_metrics = train_and_eval(X_train_with, y_train, X_test_with, y_test, xgb_device, "with_whois")

    log("=== [P1]-style comparison ===")
    log(f"Without WHOIS: {without_metrics}")
    log(f"With WHOIS:    {with_metrics}")
    delta_f1 = with_metrics["macro_f1"] - without_metrics["macro_f1"]
    log(f"Delta macro-F1 (with - without): {delta_f1:+.4f} "
        f"([P1] reported +0.20% F1 on their 5,000-URL set)")

    with open(REPORT_DIR / "url_whois_ablation_metrics.json", "w") as f:
        json.dump({
            "sample_size": len(df),
            "without_whois": without_metrics,
            "with_whois": with_metrics,
            "delta_macro_f1": delta_f1,
            "whois_availability_rate": float(X_train_whois["whois_available"].mean()),
        }, f, indent=2, default=str)
    log(f"Saved to {REPORT_DIR / 'url_whois_ablation_metrics.json'}")


if __name__ == "__main__":
    main()
