"""
Module D (URL branch) - SHAP explanation for the trained Voting Classifier.

Loads the artifacts produced by train.py and, for one or more input URLs,
prints the predicted class, confidence, and the top contributing features
per the "top contributing factors" explanation contract described in
Section 5.4. SHAP is computed against the Random-Forest member of the
voting ensemble (the tree-based member gives exact TreeExplainer support;
the ensemble's own soft-voted probability is still what is reported as the
prediction/confidence).

Usage:
    python src/url_pipeline/explain.py "http://paypal-secure-login.tk/verify/account"
"""
from __future__ import annotations

import sys
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import extract_features_row, normalize_url  # noqa: E402
from model_registry import ensure_model  # noqa: E402

MODEL_DIR = ROOT / "models"


@lru_cache(maxsize=1)
def load_artifacts():
    """Cached so a long-lived process (e.g. the API server) loads these once,
    not once per request -- joblib.load on the voting classifier is not
    instant."""
    voting = joblib.load(ensure_model("url_voting_classifier.joblib"))
    scaler = joblib.load(ensure_model("url_scaler.joblib"))
    le = joblib.load(ensure_model("url_label_encoder.joblib"))
    vectorizer = joblib.load(ensure_model("url_ngram_vectorizer.joblib"))
    with open(ensure_model("url_selected_features.json")) as f:
        selected_cols = json.load(f)

    # These estimators were trained with n_jobs=-1, which is right for
    # fitting 300 trees on ~1.4M rows once, but wrong for serving: every
    # single-row prediction would otherwise spawn a fresh joblib/loky
    # multiprocess worker pool, which is pure overhead for one row and, on
    # this machine's tight memory margin (MemoryError was hit twice during
    # training with the same setting), can hang the whole server under
    # concurrent requests. Force single-threaded inference here.
    for name, est in getattr(voting, "named_estimators_", {}).items():
        if hasattr(est, "n_jobs"):
            est.n_jobs = 1

    # By explicit request: use the actual production RF for SHAP, not the
    # fast shallow surrogate (train_shap_surrogate.py). Known consequence,
    # confirmed directly earlier: the production RF has 300
    # unconstrained-depth trees (depth ~60-72, ~53,000-56,000 leaves each),
    # and exact TreeSHAP over a forest that shape takes 90+ seconds and
    # multiple GB per single request. That's a real cost, accepted here on
    # purpose rather than worked around, so predictions and their
    # explanations both come from the one real model, with no substitute in
    # the loop.
    explainer_model = dict(voting.named_estimators_)["rf"]
    explanation_model_note = "production_rf (full model, unconstrained depth -- SHAP computation can take 60-90s+ per request)"
    explainer = shap.TreeExplainer(explainer_model)

    return voting, scaler, le, vectorizer, selected_cols, explainer, explanation_model_note


def featurize_url(url: str, vectorizer, selected_cols) -> pd.DataFrame:
    static_feats = extract_features_row(url)
    ngram_vec = vectorizer.transform([normalize_url(url)]).toarray()[0]
    ngram_feats = {f"ngram_{i}": v for i, v in enumerate(ngram_vec)}
    all_feats = {**static_feats, **ngram_feats}
    row = {c: all_feats.get(c, 0.0) for c in selected_cols}
    return pd.DataFrame([row])[selected_cols]


def predict_url(url: str, top_n: int = 8) -> dict:
    """Structured, JSON-serializable version of the explanation, for reuse by
    the API server (src/api/main.py) as well as the CLI below."""
    voting, scaler, le, vectorizer, selected_cols, explainer, explanation_model_note = load_artifacts()
    X_raw = featurize_url(url, vectorizer, selected_cols)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=selected_cols)

    proba = voting.predict_proba(X_scaled)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = str(le.inverse_transform([pred_idx])[0])
    confidence = float(proba[pred_idx])
    probabilities = {str(c): float(p) for c, p in zip(le.classes_, proba)}

    shap_values = explainer.shap_values(X_scaled, check_additivity=False)

    n_feat = len(selected_cols)
    if isinstance(shap_values, list):
        class_shap = shap_values[pred_idx][0]
    else:
        arr = np.asarray(shap_values)
        class_axis = next((ax for ax, sz in enumerate(arr.shape) if sz == len(le.classes_) and sz != n_feat), None)
        if class_axis is not None:
            class_shap = np.take(arr[0], pred_idx, axis=class_axis - 1)
        else:
            class_shap = arr[0]

    base_value = float(np.mean(explainer.expected_value[pred_idx] if hasattr(explainer.expected_value, "__len__") else explainer.expected_value))
    contrib = pd.Series(class_shap, index=selected_cols).sort_values(key=np.abs, ascending=False).head(top_n)
    top_features = [
        {
            "feature": feat,
            "shap_value": float(val),
            "raw_value": float(X_raw.iloc[0][feat]),
            "direction": "toward" if val > 0 else "away_from",
        }
        for feat, val in contrib.items()
    ]

    return {
        "url": url,
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "base_value": base_value,
        "net_push": float(sum(f["shap_value"] for f in top_features)),
        "top_features": top_features,
        "explanation_model": explanation_model_note,
    }


def explain_url(url: str, top_n: int = 8):
    """CLI-friendly wrapper: prints predict_url()'s result and also returns
    (pred_class, confidence, contrib) for backward compatibility."""
    result = predict_url(url, top_n=top_n)
    print(f"\nURL: {result['url']}")
    print(f"Predicted class: {result['predicted_class']}  (confidence: {result['confidence']:.4f})")
    print(f"Class probabilities: { {k: round(v, 4) for k, v in result['probabilities'].items()} }")
    print(f"\nTop {top_n} contributing factors (SHAP, RF component):")
    for f in result["top_features"]:
        direction = "pushes toward" if f["direction"] == "toward" else "pushes away from"
        print(f"  {f['feature']:30s} {f['shap_value']:+.4f}  ({direction} '{result['predicted_class']}')  [value={f['raw_value']:.4g}]")
    contrib = pd.Series(
        {f["feature"]: f["shap_value"] for f in result["top_features"]}
    )
    return result["predicted_class"], result["confidence"], contrib


if __name__ == "__main__":
    urls = sys.argv[1:] or [
        "http://paypal-secure-login-verify.tk/account/confirm.php",
        "https://www.google.com",
        "http://192.168.1.1/admin/login.php?redirect=http://evil.com",
    ]
    for u in urls:
        explain_url(u)
