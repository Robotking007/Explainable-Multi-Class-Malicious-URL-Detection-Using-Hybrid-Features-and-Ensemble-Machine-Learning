"""
Module B - training entry point.

Pipeline (per Section 5.2 / Image 1 / [P1],[P2],[P3]):
  URL -> {lexical, host, content, structural, statistical} features
      -> char n-gram TF-IDF block
      -> Random-Forest-importance feature selection (Top-K)
      -> SMOTE class balancing (train split only)
      -> soft-voting ensemble (RandomForest + XGBoost + LogisticRegression + NaiveBayes)
      -> per-class Accuracy/Precision/Recall/F1/ROC-AUC/FNR/FPR (Section 9)

Dataset: Dataset/malicious_phish.csv (651,199 URLs; benign/phishing/malware/defacement),
which matches the 4-class taxonomy used in [P2] exactly.

Usage:
    python src/url_pipeline/train.py --sample-size 0 --top-k 30 --ngram-features 60
    (--sample-size 0 means "use the full dataset"; pass an int to subsample for a quick run)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import build_static_features, build_ngram_features  # noqa: E402

DATA_PATH = ROOT / "Dataset" / "malicious_phish.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def detect_xgb_device() -> str:
    """Probe whether XGBoost's CUDA backend actually works on this machine
    (an NVIDIA GPU being present isn't sufficient -- the build must have CUDA
    support compiled in). Falls back to CPU on any failure. Only XGBoost
    benefits from GPU here: scikit-learn's RandomForest/LogisticRegression/
    GaussianNB have no GPU backend without RAPIDS cuML, which isn't installed."""
    try:
        import xgboost as xgb_mod
        d = xgb_mod.DMatrix(np.zeros((4, 2)), label=[0, 1, 0, 1])
        xgb_mod.train({"tree_method": "hist", "device": "cuda"}, d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


def compute_fnr_fpr(y_true, y_pred, classes):
    """Per-class False Negative Rate / False Positive Rate from a multi-class confusion matrix."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--top-k", type=int, default=30, help="Top-K engineered features to keep (RF importance)")
    ap.add_argument("--ngram-features", type=int, default=100,
                     help="char n-gram TF-IDF vocab size (150 OOM-killed a full 20M-row Kaggle run -- "
                          "see build_ngram_features()'s docstring in features.py)")
    ap.add_argument("--smote-target", type=int, default=1_200_000,
                     help="Cap for minority-class SMOTE oversampling (not full parity with the majority class -- "
                          "see kaggle_kernel_train_url.py's SMOTE_TARGET_PER_MINORITY_CLASS for why)")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--data-path", type=str, default=str(DATA_PATH),
                     help="CSV with 'url' and 'type' columns (default: Dataset/malicious_phish.csv)")
    args = ap.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    data_path = Path(args.data_path)
    log(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path).dropna(subset=["url", "type"])
    if args.sample_size and args.sample_size < len(df):
        df = df.groupby("type", group_keys=False).apply(
            lambda g: g.sample(
                min(len(g), int(args.sample_size * len(g) / len(df))), random_state=args.random_state
            )
        )
        log(f"Subsampled to {len(df)} rows (stratified)")
    log(f"Class distribution:\n{df['type'].value_counts()}")

    log("Splitting train/test (stratified, 80/20)")
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=df["type"]
    )

    log(f"Extracting static features on {len(train_df)} train + {len(test_df)} test URLs ...")
    t0 = time.time()
    X_train_static = build_static_features(train_df["url"])
    X_test_static = build_static_features(test_df["url"])
    log(f"Static feature extraction done in {time.time() - t0:.1f}s. Shape: {X_train_static.shape}")

    log(f"Building char n-gram TF-IDF block (max_features={args.ngram_features}, fit on train only)")
    X_train_ngram, X_test_ngram, vectorizer = build_ngram_features(
        train_df["url"], test_df["url"], max_features=args.ngram_features
    )

    X_train_full = pd.concat([X_train_static.reset_index(drop=True), X_train_ngram.reset_index(drop=True)], axis=1)
    X_test_full = pd.concat([X_test_static.reset_index(drop=True), X_test_ngram.reset_index(drop=True)], axis=1)
    y_train = train_df["type"].reset_index(drop=True)
    y_test = test_df["type"].reset_index(drop=True)

    log(f"Combined feature matrix: {X_train_full.shape[1]} features "
        f"({X_train_static.shape[1]} engineered + {X_train_ngram.shape[1]} n-gram)")

    # ---- Feature selection: Random Forest importance -> Top-K engineered features ----
    # (kept separate from the n-gram block, which is already dimensionality-limited)
    log("Ranking engineered features by Random Forest importance ...")
    # n_jobs=1: with the dataset now 2.4M+ rows (up from ~830k), n_jobs=6
    # here would fork 6 worker processes each building an unconstrained-depth
    # tree concurrently -- the exact pattern that crashed
    # train_static_baseline.py's RF fit twice at a smaller row count (587k)
    # than even this pre-SMOTE selector sees. Same fix applied here
    # preemptively rather than waiting to hit it again.
    # Also subsample + cap depth: unconstrained-depth RF fit on the FULL
    # 16.7M+-row train split OOM-killed the Kaggle kernel twice (see
    # kaggle_kernel_train_url.py's matching comment) -- a shallow forest
    # on a 2M-row subsample gives an equally reliable importance ranking.
    rng = np.random.RandomState(args.random_state)
    sample_size = min(2_000_000, len(X_train_static))
    sample_idx = rng.choice(len(X_train_static), size=sample_size, replace=False)
    rf_selector = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=args.random_state, n_jobs=1
    )
    rf_selector.fit(X_train_static.iloc[sample_idx], y_train.iloc[sample_idx])
    importances = pd.Series(rf_selector.feature_importances_, index=X_train_static.columns)
    top_k_cols = importances.sort_values(ascending=False).head(args.top_k).index.tolist()
    log(f"Top-{args.top_k} engineered features selected:\n{importances.sort_values(ascending=False).head(args.top_k)}")

    selected_cols = top_k_cols + list(X_train_ngram.columns)
    X_train_sel = X_train_full[selected_cols]
    X_test_sel = X_test_full[selected_cols]

    # Free the large intermediate matrices (static+ngram, pre-selection) once
    # the selected-column subset is carved out -- this machine runs with as
    # little as ~4GB free once normal desktop apps (VS Code, Defender, etc.)
    # are accounted for, and holding onto ~97-column matrices we no longer
    # need alongside the ~90-column ones we do was pure waste that
    # contributed to two separate MemoryError/ArrayMemoryError crashes on
    # this same pipeline at this scale.
    del X_train_full, X_test_full, X_train_static, X_test_static, X_train_ngram, X_test_ngram, rf_selector
    import gc
    gc.collect()

    # ---- Scale (helps LR/NB); float32 halves memory vs. the float64
    # default for every large matrix downstream (SMOTE input/output, model
    # training input) -- meaningful at ~90 columns x ~850k+ rows. ----
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_cols, dtype=np.float32)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_cols, dtype=np.float32)
    del X_train_sel, X_test_sel
    gc.collect()

    # ---- SMOTE on train split only ----
    # Capped, targeted: only classes below args.smote_target get
    # oversampled, and only up to that cap -- NOT full parity with the
    # majority class. At large row counts, full parity means tens of
    # millions of mostly-synthetic rows, which is both impractical and
    # actively hurts generalization (SMOTE's linear interpolation over
    # sparse n-gram features manufactures unrealistic points at extreme
    # oversampling ratios). Remaining imbalance is handled via
    # sample_weight at fit time instead (see below).
    log(f"Class counts before SMOTE:\n{y_train.value_counts()}")
    counts = y_train.value_counts()
    sampling_strategy = {cls: args.smote_target for cls, n in counts.items() if n < args.smote_target}
    log(f"SMOTE sampling_strategy (capped, targeted): {sampling_strategy}")
    smote = SMOTE(random_state=args.random_state, sampling_strategy=sampling_strategy or "auto")
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    X_train_bal_np = X_train_bal.astype(np.float32).to_numpy()
    del X_train_bal, X_train_scaled
    gc.collect()
    log(f"Class counts after capped SMOTE:\n{pd.Series(y_train_bal).value_counts()}")

    # Move the SMOTE-balanced matrix to a disk-backed memmap on D: (same
    # technique that fixed train_static_baseline.py's pool-building crash).
    # This run previously got all the way through RF + XGBoost fitting
    # (5.5+ hours) only to crash inside LogisticRegression.fit(): sklearn's
    # check_array upcasts to float64 for LR internally (3.13GB for this
    # matrix), and by that point the already-fitted, unconstrained-depth RF
    # (300 trees over 4.67M rows) left too little RAM headroom for that
    # one more allocation -- the same "big object already resident + one
    # more allocation" pattern as every other memory crash this project
    # hit. Keeping X_train_bal itself off the RAM budget (paged from disk
    # instead) frees exactly the headroom LR's internal copy needs, without
    # touching RF/XGBoost's data, depth, or row count at all.
    tmp_dir = REPORT_DIR / "_tmp_arrays"
    tmp_dir.mkdir(exist_ok=True)
    mm_path = tmp_dir / "X_train_bal_url.f32"
    X_train_bal = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=X_train_bal_np.shape)
    X_train_bal[:] = X_train_bal_np
    X_train_bal.flush()
    del X_train_bal_np
    gc.collect()

    # ---- Encode labels (needed for XGBoost + multi-class ROC-AUC) ----
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_bal)
    y_test_enc = le.transform(y_test)
    classes = list(le.classes_)

    # ---- Voting Classifier: RF + XGBoost + LR + NB ([P2]) ----
    # Remaining imbalance after the capped SMOTE (benign/phishing still
    # outnumber the capped defacement/malware classes) is handled via
    # sample_weight rather than more synthetic oversampling --
    # VotingClassifier.fit(sample_weight=...) forwards the same weights to
    # every member that supports them (RF, XGBoost, LR, GaussianNB all do),
    # so this one array balances the loss for all four voters without
    # needing per-estimator class_weight.
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight("balanced", y_train_enc)
    log(f"Computed balanced sample_weight over {len(np.unique(y_train_enc))} classes.")

    xgb_device = detect_xgb_device()
    log(f"Training Voting Classifier (RF + XGBoost[{xgb_device}] + LR + NB) ...")
    # n_jobs=1: same reasoning as rf_selector above, and now fitting on the
    # SMOTE-balanced pool (all 4 classes resampled up to the majority
    # class's count -- several million rows), where the crash risk is even
    # higher than the pre-SMOTE selector's.
    # max_depth/max_samples: unconstrained-depth RF fit on ~18.8M SMOTE-
    # balanced rows OOM-killed the Kaggle kernel (see kaggle_kernel_train_
    # url.py's matching comment for the crash history and reasoning).
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=30, max_samples=0.3,
        random_state=args.random_state, n_jobs=1,
    )
    xgb = XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.08, random_state=args.random_state,
        n_jobs=6, eval_metric="mlogloss", tree_method="hist", device=xgb_device,
    )
    lr = LogisticRegression(max_iter=1000, random_state=args.random_state, n_jobs=4)
    nb = GaussianNB()

    # n_jobs=None (sequential) here on purpose: VotingClassifier's own n_jobs
    # forks one OS process per estimator via joblib, and each fork gets its own
    # pickled copy of the ~1GB training matrix -- on a 16GB-RAM machine that
    # multiplies memory use ~4-5x and reliably OOMs (MemoryError during
    # result-pickling back from a worker). Fitting the 4 estimators
    # sequentially avoids that; each estimator still parallelizes internally
    # via its own n_jobs=-1 (RF, LR) or built-in multithreading (XGBoost).
    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", lr), ("nb", nb)], voting="soft", n_jobs=None
    )
    t0 = time.time()
    voting.fit(X_train_bal, y_train_enc, sample_weight=sample_weight)
    log(f"Voting Classifier trained in {time.time() - t0:.1f}s")

    # ---- Evaluate ----
    log("Evaluating on held-out test split ...")
    y_pred_enc = voting.predict(X_test_scaled)
    y_proba = voting.predict_proba(X_test_scaled)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        roc_auc_macro = roc_auc_score(y_test_enc, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        roc_auc_macro = float("nan")

    fnr, fpr = compute_fnr_fpr(y_test, y_pred, classes)

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log(f"Accuracy: {acc:.4f} | Macro-Precision: {prec_macro:.4f} | "
        f"Macro-Recall: {rec_macro:.4f} | Macro-F1: {f1_macro:.4f} | Macro-ROC-AUC: {roc_auc_macro:.4f}")
    log(f"Per-class classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    log(f"Per-class FNR: {fnr}")
    log(f"Per-class FPR: {fpr}")

    # ---- Persist artifacts ----
    joblib.dump(voting, MODEL_DIR / "url_voting_classifier.joblib")
    joblib.dump(scaler, MODEL_DIR / "url_scaler.joblib")
    joblib.dump(le, MODEL_DIR / "url_label_encoder.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "url_ngram_vectorizer.joblib")
    with open(MODEL_DIR / "url_selected_features.json", "w") as f:
        json.dump(selected_cols, f, indent=2)

    metrics_out = {
        "n_train": len(X_train_bal),
        "n_test": len(X_test_scaled),
        "n_features": len(selected_cols),
        "top_k_engineered_features": top_k_cols,
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "macro_roc_auc": roc_auc_macro,
        "per_class_fnr": fnr,
        "per_class_fpr": fpr,
        "classification_report": report_dict,
    }
    with open(REPORT_DIR / "url_module_b_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    log(f"Saved model artifacts to {MODEL_DIR} and metrics to {REPORT_DIR}/url_module_b_metrics.json")


if __name__ == "__main__":
    main()
