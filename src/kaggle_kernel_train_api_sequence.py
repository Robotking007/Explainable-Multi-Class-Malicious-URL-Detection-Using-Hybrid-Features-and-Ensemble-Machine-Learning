"""
Self-contained Module A'' (API call sequence malware family baseline)
training script for a Kaggle kernel. Standalone copy of
src/malware_pipeline/train_api_sequence_baseline.py, inlined since a
Kaggle kernel can't import this repo's package layout.

Dataset expected at /kaggle/input/cyberxai-malapi2019/ (labels.txt,
mal-api-2019/1000_calls.txt). Outputs saved to /kaggle/working/.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap"], check=False)

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

OUT_DIR = Path("/kaggle/working")
MAX_NGRAM_FEATURES = 500


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


def compute_fnr_fpr(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fnr, fpr = {}, {}
    total = cm.sum()
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        fnr[cls] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fnr, fpr


def main():
    log("Listing /kaggle/input/ ...")
    for p in Path("/kaggle/input").rglob("*"):
        log(f"  {p}")

    labels_path = next(Path("/kaggle/input").rglob("labels.txt"))
    calls_path = next(Path("/kaggle/input").rglob("1000_calls.txt"))

    log(f"Loading {calls_path} and {labels_path} ...")
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]
    with open(calls_path) as f:
        sequences = [line.strip().replace(",", " ") for line in f if line.strip()]
    assert len(labels) == len(sequences), f"{len(labels)} labels vs {len(sequences)} sequences"
    log(f"Loaded {len(sequences)} samples. Family distribution:\n{pd.Series(labels).value_counts()}")

    le = LabelEncoder()
    y = le.fit_transform(labels)
    classes = list(le.classes_)

    seq_train, seq_test, y_train, y_test = train_test_split(
        sequences, y, test_size=0.2, random_state=42, stratify=y
    )

    log(f"Building word-level TF-IDF over API-call-ID tokens (1-3 grams, max_features={MAX_NGRAM_FEATURES})...")
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=MAX_NGRAM_FEATURES)
    X_train = vectorizer.fit_transform(seq_train)
    X_test = vectorizer.transform(seq_test)
    feature_names = vectorizer.get_feature_names_out()
    log(f"Feature matrix: {X_train.shape[1]} features")

    xgb_device = detect_xgb_device()
    log(f"Training RF + XGBoost[{xgb_device}] ...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42,
        n_jobs=-1, eval_metric="mlogloss", tree_method="hist", device=xgb_device,
    )
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    proba = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2.0
    y_pred = proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc = float("nan")

    y_test_labels = le.inverse_transform(y_test)
    y_pred_labels = le.inverse_transform(y_pred)
    fnr, fpr = compute_fnr_fpr(y_test_labels, y_pred_labels, classes)
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)

    log(f"Accuracy={acc:.4f} Macro-P={prec:.4f} Macro-R={rec:.4f} Macro-F1={f1:.4f} Macro-ROC-AUC={roc_auc:.4f}")

    log("Computing SHAP values on a 300-row test sample ...")
    sample_idx = np.random.RandomState(42).choice(X_test.shape[0], min(300, X_test.shape[0]), replace=False)
    sample = X_test[sample_idx].toarray()
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample, check_additivity=False)
    n_feat = len(feature_names)
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        arr = np.abs(shap_values)
        feat_axis = next(ax for ax, sz in enumerate(arr.shape) if sz == n_feat)
        other_axes = tuple(ax for ax in range(arr.ndim) if ax != feat_axis)
        mean_abs = arr.mean(axis=other_axes)
    top_features = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(15)
    log(f"Top-15 SHAP features (API-call n-grams):\n{top_features}")

    joblib.dump(rf, OUT_DIR / "malware_api_sequence_rf.joblib")
    joblib.dump(xgb, OUT_DIR / "malware_api_sequence_xgb.joblib")
    joblib.dump(le, OUT_DIR / "malware_api_sequence_label_encoder.joblib")
    joblib.dump(vectorizer, OUT_DIR / "malware_api_sequence_vectorizer.joblib")

    metrics_out = {
        "classes": classes, "n_train": X_train.shape[0], "n_test": X_test.shape[0],
        "n_features": X_train.shape[1], "accuracy": acc, "macro_precision": prec,
        "macro_recall": rec, "macro_f1": f1, "macro_roc_auc": roc_auc,
        "per_class_fnr": fnr, "per_class_fpr": fpr,
        "top_shap_features": top_features.to_dict(), "classification_report": report,
    }
    with open(OUT_DIR / "malware_module_a_api_sequence_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    log("Saved model artifacts and metrics.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "CRASH_TRACEBACK.txt", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise
