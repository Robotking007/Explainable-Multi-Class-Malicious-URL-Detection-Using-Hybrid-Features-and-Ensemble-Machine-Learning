"""
Train a lightweight SHAP-explanation surrogate for the URL Voting Classifier.

Why this exists: the production Random Forest inside the Voting Classifier
(models/url_voting_classifier.joblib) has 300 unconstrained-depth trees --
depth ~60-72, ~53,000-56,000 leaves each (measured directly; see
reports/train_full_run.log context). Exact TreeSHAP over a forest that size
takes 90+ seconds and multiple GB per single-row explanation, which made the
live API endpoint look hung. That forest is genuinely fine for *prediction*
(that's what it was tuned for); it is simply the wrong shape for real-time
per-request SHAP.

This script trains a small, shallow Random Forest (bounded depth, few
estimators) on the SAME feature space (same saved scaler, same saved n-gram
vectorizer, same selected_cols) and the same 4-class labels, on a subsample
of malicious_phish.csv. It is used ONLY to compute fast SHAP attributions at
serving time -- src/api/main.py's /api/url/predict endpoint still reports
the predicted class and confidence from the real production Voting
Classifier; only the "top contributing factors" come from this surrogate.
That distinction is stated explicitly in the API response
(`explanation_model` field) and in the frontend, so nobody mistakes the
surrogate's structure for the production model's.

Usage:
    python src/url_pipeline/train_shap_surrogate.py --sample-size 40000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import build_static_features, normalize_url  # noqa: E402

DATA_PATH = ROOT / "Dataset" / "malicious_phish.csv"
MODEL_DIR = ROOT / "models"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=40000)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    log("Loading saved production artifacts (scaler, vectorizer, selected_cols)...")
    scaler = joblib.load(MODEL_DIR / "url_scaler.joblib")
    vectorizer = joblib.load(MODEL_DIR / "url_ngram_vectorizer.joblib")
    le = joblib.load(MODEL_DIR / "url_label_encoder.joblib")
    import json
    with open(MODEL_DIR / "url_selected_features.json") as f:
        selected_cols = json.load(f)
    ngram_cols = [c for c in selected_cols if c.startswith("ngram_")]
    static_cols = [c for c in selected_cols if not c.startswith("ngram_")]

    log(f"Loading and subsampling {args.sample_size} rows from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH).dropna(subset=["url", "type"])
    df = df.groupby("type", group_keys=False).apply(
        lambda g: g.sample(min(len(g), int(args.sample_size * len(g) / len(df))), random_state=args.random_state),
        include_groups=True,
    )
    log(f"Subsample class distribution:\n{df['type'].value_counts()}")

    log("Extracting features (static + reusing production n-gram vectorizer)...")
    t0 = time.time()
    static_feats = build_static_features(df["url"])
    norm_urls = df["url"].astype(str).apply(normalize_url)
    ngram_vec = vectorizer.transform(norm_urls).toarray()
    ngram_feats = pd.DataFrame(ngram_vec, columns=[f"ngram_{i}" for i in range(ngram_vec.shape[1])], index=static_feats.index)
    X_full = pd.concat([static_feats.reset_index(drop=True), ngram_feats.reset_index(drop=True)], axis=1)
    X = X_full[selected_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=selected_cols)
    y = le.transform(df["type"].reset_index(drop=True))
    log(f"Feature extraction done in {time.time() - t0:.1f}s. Shape: {X_scaled.shape}")

    log(f"Training shallow surrogate RF (n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
    t0 = time.time()
    surrogate = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth,
        random_state=args.random_state, n_jobs=1,  # n_jobs=1: see explain.py note on serving overhead
    )
    surrogate.fit(X_scaled, y)
    log(f"Surrogate trained in {time.time() - t0:.1f}s")

    depths = [surrogate.estimators_[i].get_depth() for i in range(min(10, args.n_estimators))]
    leaves = [surrogate.estimators_[i].get_n_leaves() for i in range(min(10, args.n_estimators))]
    log(f"Surrogate tree depths (first 10): {depths}")
    log(f"Surrogate tree leaf counts (first 10): {leaves}")

    acc = surrogate.score(X_scaled, y)
    log(f"Surrogate train-set accuracy (sanity check, not a generalization metric): {acc:.4f}")

    joblib.dump(surrogate, MODEL_DIR / "url_shap_surrogate_rf.joblib")
    log(f"Saved to {MODEL_DIR / 'url_shap_surrogate_rf.joblib'}")


if __name__ == "__main__":
    main()
