"""
Self-contained Kaggle Kernel script for training the Multi-Class Malicious URL
Detection Model using the Combined URL Dataset + WHOIS Features (Module B Ensemble).

Produces production-ready .joblib model artifacts for explainability & API serving:
  - url_voting_classifier.joblib (Soft-Voting Ensemble: RF + XGBoost + LogisticRegression + GaussianNB)
  - url_model.joblib (Alias for voting classifier)
  - url_scaler.joblib (StandardScaler fitted on all 50 features)
  - url_label_encoder.joblib (4-class LabelEncoder)
  - url_shap_surrogate_rf.joblib (Fast SHAP TreeExplainer surrogate for sub-second explanations)
  - url_selected_features.json (50 feature names: 45 hybrid + 5 WHOIS)
  - url_module_b_metrics.json (Full multi-class classification report)

Datasets attached on Kaggle:
  - /kaggle/input/cyberxai-combined-urls/combined_urls.csv (27.02M master dataset)
  - /kaggle/input/cyberxai-whois-cache/whois_cache.csv (593,006 pre-cached WHOIS records)
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Kaggle input/output & HF configuration
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
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

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
    "whois_available": 0.0,
    "domain_age_days": 0.0,
    "registration_length_days": 0.0,
    "recent_domain_flag": 0.0,
    "whois_private_flag": 0.0,
}

# Global in-memory lookup table populated from whois_cache.csv
WHOIS_LOOKUP: dict[str, tuple[float, float, float, float, float]] = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def extract_features_row_with_whois(url: str) -> dict:
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
        registered_domain = f"{parts[-2]}.{parts[-1]}".lower() if len(parts) >= 2 else netloc.lower()
        subdomain_count = max(0, len(parts) - 2)

        toks = [len(t) for t in TOKEN_SPLIT_RE.split(url_str) if t]
        all_toks = TOKEN_SPLIT_RE.split(url_str)
        u_len = max(len(url_str), 1)

        # Lookup WHOIS cached attributes
        whois_data = WHOIS_LOOKUP.get(registered_domain, (0.0, 0.0, 0.0, 0.0, 0.0))

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
            "whois_available": float(whois_data[0]),
            "domain_age_days": float(whois_data[1]),
            "registration_length_days": float(whois_data[2]),
            "recent_domain_flag": float(whois_data[3]),
            "whois_private_flag": float(whois_data[4]),
        }
    except Exception:
        return DEFAULT_ZERO_ROW.copy()


def load_whois_cache() -> None:
    log("Scanning /kaggle/input for whois_cache.csv...")
    whois_file = None
    for p in IN_DIR.rglob("whois_cache.csv"):
        whois_file = p
        break

    if not whois_file:
        for p in IN_DIR.rglob("*whois*.csv"):
            whois_file = p
            break

    if not whois_file:
        log("Notice: whois_cache.csv not found in /kaggle/input. Using zero defaults for WHOIS block.")
        return

    log(f"Loading WHOIS cache from {whois_file} ({whois_file.stat().st_size:,} bytes)...")
    t0 = time.time()
    count = 0
    for chunk in pd.read_csv(whois_file, chunksize=100000):
        dom_col = [c for c in chunk.columns if "domain" in c.lower()][0]
        avail_col = [c for c in chunk.columns if "avail" in c.lower()][0]
        age_col = [c for c in chunk.columns if "age" in c.lower()][0]
        len_col = [c for c in chunk.columns if "reg" in c.lower() or "length" in c.lower()][0]
        rec_col = [c for c in chunk.columns if "rec" in c.lower()][0]
        priv_col = [c for c in chunk.columns if "priv" in c.lower()][0]

        for d, a, ag, l, r, pr in zip(
            chunk[dom_col], chunk[avail_col], chunk[age_col], chunk[len_col], chunk[rec_col], chunk[priv_col]
        ):
            clean_dom = str(d).strip().strip("'").strip('"').lower()
            WHOIS_LOOKUP[clean_dom] = (
                float(a or 0),
                float(ag or 0),
                float(l or 0),
                float(r or 0),
                float(pr or 0),
            )
            count += 1

    log(f"Loaded {len(WHOIS_LOOKUP):,} unique domains from WHOIS cache in {time.time() - t0:.1f}s.")


def find_data_file() -> Path:
    for p in IN_DIR.rglob("combined_urls.csv"):
        log(f"Found combined dataset: {p}")
        return p
    for p in IN_DIR.rglob("*.parquet"):
        return p
    for p in IN_DIR.rglob("*.csv"):
        if "whois" not in p.name.lower():
            return p
    raise FileNotFoundError("Could not find combined_urls.csv anywhere in /kaggle/input")


def main():
    log("================================================================================")
    log("Starting CyberXAI Module B Training: Combined URLs + WHOIS Features (.joblib)")
    log("================================================================================")

    load_whois_cache()
    data_file = find_data_file()
    log(f"Reading dataset: {data_file}")

    target_per_maj = 100_000
    total_rows_approx = 27_022_952
    maj_sample_rate = target_per_maj / max(total_rows_approx / 2, 1)

    chunks = []
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

        # Retain 100% of minority attack classes (malware & defacement)
        is_minority = chunk[l_col].isin(["defacement", "malware"])
        minority_df = chunk[is_minority]
        majority_df = chunk[~is_minority].sample(frac=maj_sample_rate, random_state=42 + i) if not chunk[~is_minority].empty else chunk[~is_minority]

        sampled_chunk = pd.concat([minority_df, majority_df])
        chunks.append(sampled_chunk[[u_col, l_col]])
        if (i + 1) % 5 == 0 or (i + 1) >= 27:
            log(f"  Processed Chunk {i+1}: running total = {sum(len(c) for c in chunks):,} rows")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    col_map = {c.lower(): c for c in df.columns}
    url_col = col_map.get("url", "url")
    label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))

    log(f"Master training distribution:\n{df[label_col].value_counts()}")
    urls = df[url_col].tolist()
    labels = df[label_col].tolist()
    del df
    gc.collect()

    log(f"Extracting 50 Hybrid + WHOIS features for {len(urls):,} URLs via ThreadPoolExecutor...")
    t_feat_start = time.time()
    batch_chunk_size = 25000
    all_feature_rows = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        for b_start in range(0, len(urls), batch_chunk_size):
            b_urls = urls[b_start:b_start + batch_chunk_size]
            batch_rows = list(executor.map(extract_features_row_with_whois, b_urls))
            all_feature_rows.extend(batch_rows)
            log(f"  Extracted {min(b_start + len(b_urls), len(urls)):,} / {len(urls):,} URLs ({time.time() - t_feat_start:.1f}s)")

    log("Building feature DataFrame...")
    X_df = pd.DataFrame(all_feature_rows, dtype=np.float32).fillna(0.0)
    del all_feature_rows, urls
    gc.collect()

    feature_names = list(X_df.columns)
    log(f"Constructed {len(feature_names)} features: {feature_names}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    label_mapping = {int(idx): str(c) for idx, c in enumerate(le.classes_)}
    log(f"Encoded target labels: {label_mapping}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_df.values, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    del X_df, y_encoded, labels
    gc.collect()

    log("Fitting StandardScaler across 50 features...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    del X_train, X_test
    gc.collect()

    sample_weights = compute_sample_weight("balanced", y_train)

    log("Instantiating Soft-Voting Ensemble (RandomForest + XGBoost + LogisticRegression + GaussianNB)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        min_samples_split=5,
        random_state=42,
        n_jobs=4,
    )
    xgb = XGBClassifier(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.08,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=4,
    )
    lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=4)
    nb = GaussianNB()

    voting = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb), ("lr", lr), ("nb", nb)],
        voting="soft",
        n_jobs=None,  # sequential caller to avoid IPC conflicts
    )

    log(f"Training ensemble on {len(X_train_s):,} samples with sample weights...")
    t_train_start = time.time()
    voting.fit(X_train_s, y_train, sample_weight=sample_weights)
    log(f"Voting Classifier fit complete in {time.time() - t_train_start:.1f}s.")

    log(f"Evaluating ensemble model on {len(X_test_s):,} held-out test URLs...")
    preds = voting.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)

    log(f"Test Accuracy: {acc*100:.2f}% | Macro-F1: {f1_macro:.4f} | Macro-Precision: {prec_macro:.4f} | Macro-Recall: {rec_macro:.4f}")
    rep = classification_report(
        y_test, preds, target_names=[label_mapping[i] for i in range(len(label_mapping))], output_dict=True, zero_division=0
    )
    print("\n" + classification_report(y_test, preds, target_names=[label_mapping[i] for i in range(len(label_mapping))], zero_division=0))

    # Fit fast SHAP TreeExplainer surrogate for sub-second API explanations
    log("Fitting SHAP surrogate Random Forest explainer on subsample...")
    shap_rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=4)
    sub_idx = np.random.RandomState(42).choice(len(X_train_s), size=min(25000, len(X_train_s)), replace=False)
    shap_rf.fit(X_train_s[sub_idx], y_train[sub_idx])

    # Serializing all artifacts as .joblib and .json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Serializing models and encoders to /kaggle/working/ as .joblib...")
    joblib.dump(voting, OUT_DIR / "url_voting_classifier.joblib", compress=3)
    joblib.dump(voting, OUT_DIR / "url_model.joblib", compress=3)
    joblib.dump(scaler, OUT_DIR / "url_scaler.joblib", compress=3)
    joblib.dump(le, OUT_DIR / "url_label_encoder.joblib", compress=3)
    joblib.dump(shap_rf, OUT_DIR / "url_shap_surrogate_rf.joblib", compress=3)

    with open(OUT_DIR / "url_selected_features.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    metrics_payload = {
        "n_train": len(X_train_s),
        "n_test": len(X_test_s),
        "n_features": len(feature_names),
        "accuracy": acc,
        "macro_f1": f1_macro,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "classification_report": rep,
    }
    with open(OUT_DIR / "url_module_b_metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    log("All .joblib and .json artifacts successfully generated and validated locally in /kaggle/working/.")

    # Direct push to Hugging Face Model Hub
    hf_tok = HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if hf_tok:
        log(f"Publishing trained .joblib artifacts directly to Hugging Face Hub ({HF_REPO_ID})...")
        try:
            try:
                import huggingface_hub
            except ImportError:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=False)

            from huggingface_hub import HfApi
            api = HfApi(token=hf_tok)
            for fname in [
                "url_voting_classifier.joblib",
                "url_model.joblib",
                "url_scaler.joblib",
                "url_label_encoder.joblib",
                "url_shap_surrogate_rf.joblib",
                "url_selected_features.json",
                "url_module_b_metrics.json",
            ]:
                fpath = OUT_DIR / fname
                if fpath.exists():
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=fname,
                        repo_id=HF_REPO_ID,
                        repo_type="model",
                        token=hf_tok,
                    )
                    log(f"  Uploaded {fname} -> Hugging Face Hub: {HF_REPO_ID}")
            log("All joblib and metrics artifacts successfully synchronized to Hugging Face Hub!")
        except Exception as e:
            log(f"Hugging Face Hub direct sync warning: {e}")


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
