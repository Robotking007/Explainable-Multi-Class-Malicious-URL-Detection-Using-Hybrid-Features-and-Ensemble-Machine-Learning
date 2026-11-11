"""
Upload all trained PyTorch, GNN, Ensemble, and Malware models to Hugging Face Hub
(`siddudavant/cyberxai-models`) and enforce private repository visibility.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "siddudavant/cyberxai-models"

FILES_TO_SYNC = [
    # Deep Learning HybridURLNet (Kaggle Cloud Trained)
    "url_dl_model.pt",
    "url_dl_scaler.joblib",
    "url_dl_label_encoder.joblib",
    "url_dl_metrics.json",
    # Graph Neural Network URLInfraGNN (Kaggle Cloud Trained)
    "url_gnn_model.pt",
    "url_gnn_scaler.joblib",
    "url_gnn_label_encoder.joblib",
    "url_gnn_metrics.json",
    # Classical Ensemble & Explainers
    "url_voting_classifier.joblib",
    "url_model.joblib",
    "url_scaler.joblib",
    "url_label_encoder.joblib",
    "url_ngram_vectorizer.joblib",
    "url_selected_features.json",
    "url_shap_surrogate_rf.joblib",
    "url_module_b_metrics.json",
    # Malware Analysis Modules (Dynamic Memory, Static PE, API Sequence)
    "malware_dynamic_rf_multiclass_family.joblib",
    "malware_dynamic_xgb_multiclass_family.joblib",
    "malware_dynamic_label_encoder_multiclass_family.joblib",
    "malware_dynamic_rf_binary_benign_vs_malware.joblib",
    "malware_dynamic_xgb_binary_benign_vs_malware.joblib",
    "malware_dynamic_label_encoder_binary_benign_vs_malware.joblib",
    "malware_static_rf.joblib",
    "malware_static_xgb.joblib",
    "malware_api_sequence_rf.joblib",
    "malware_api_sequence_xgb.joblib",
    "malware_api_sequence_vectorizer.joblib",
    "malware_api_sequence_label_encoder.joblib",
    "malware_module_a_api_sequence_metrics.json",
    "malware_module_a_dynamic_baseline_metrics.json",
    "malware_module_a_static_baseline_metrics.json",
]

HF_MODEL_CARD = """---
language:
- en
license: mit
tags:
- cybersecurity
- malicious-url-detection
- malware-detection
- graph-neural-network
- deep-learning
- xai
- explainable-ai
- shap
- pytorch
datasets:
- combined-urls
- bodmas
- ember-2018
- obfuscated-malmem2022
- malapi-2019
metrics:
- accuracy
- f1
- precision
- recall
---

# CyberXAI Model Registry (`siddudavant/cyberxai-models`)

Private model artifact repository for **CyberXAI: Explainable Multi-Class Malicious URL & Malware Detection Platform**.

## Model Benchmarks Summary

| Model / Artifact | Architecture | Input Dim | Key Evaluation Metrics |
| :--- | :--- | :--- | :--- |
| `url_gnn_model.pt` | `URLInfraGNN` (2-Layer Pure PyTorch GAT) | 45 URL + 8 Domain + 4 IP + 4 ASN | **89.77% Acc**, **0.8947 Macro F1**, **0.9550 Malware F1** (`n=25,145` test) |
| `url_dl_model.pt` | `HybridURLNet` (Tabular Residual MLP + SiLU) | 45 Hybrid URL Features | **89.46% Acc**, **0.8916 Macro F1**, **0.9552 Malware F1** (`n=33,526` test) |
| `url_voting_classifier.joblib` | Soft-Voting Ensemble (RF + XGB + LR + GNB) | 50 (45 Hybrid + 5 WHOIS) | **81.85% Acc**, **0.8163 Macro F1**, **0.8893 Malware F1** (`n=76,627` test) |
| `malware_dynamic_*_binary_*.joblib` | RF + XGBoost (CIC-MalMem-2022 Binary) | 55 Volatility Memory Stats | **100.00% Acc**, **1.0000 Macro F1**, **1.0000 ROC-AUC** (`n=11,612` test) |
| `malware_dynamic_*_multiclass_*.joblib` | RF + XGBoost (CIC-MalMem-2022 4-Class) | 55 Volatility Memory Stats | **87.81% Acc**, **0.8153 Macro F1**, **0.9729 Macro ROC-AUC** (`n=11,612` test) |
| `malware_static_*.joblib` | RF + XGBoost (BODMAS + EMBER-2018) | 2,381 Static PE Features | **98.11% Internal Acc** (0.9982 AUC) / **96.64% External EMBER Acc** (0.9942 AUC) |
| `malware_api_sequence_*.joblib` | RF + XGBoost (MalAPI2019 8-Family) | 500 Word 1-3 Gram TF-IDF | **67.30% Acc**, **0.6885 Macro F1**, **0.9175 Macro ROC-AUC** (`n=1,422` test) |
"""


def _load_hf_token() -> str:
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token.strip()
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("HF_TOKEN not found in environment or .env")


def main() -> None:
    hf_token = _load_hf_token()
    api = HfApi(token=hf_token)
    who = api.whoami()
    print(f"Authenticated as Hugging Face user: {who['name']}")

    # Ensure repository exists and is private
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
    api.update_repo_settings(repo_id=REPO_ID, repo_type="model", private=True)
    info = api.repo_info(repo_id=REPO_ID, repo_type="model")
    print(f"Repository {REPO_ID} visibility: private={info.private}")

    models_dir = ROOT / "models"
    reports_dir = ROOT / "reports"

    # Update Hugging Face Model Card README.md
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(HF_MODEL_CARD)
        tmp_path = Path(tmp.name)
    try:
        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            token=hf_token,
        )
        print("[OK] Updated Hugging Face Model Card (README.md)")
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"Beginning synchronization of {len(FILES_TO_SYNC)} artifacts to {REPO_ID}...")
    success_count = 0
    for idx, filename in enumerate(FILES_TO_SYNC, start=1):
        local_path = models_dir / filename
        if not local_path.exists():
            local_path = reports_dir / filename
        if not local_path.exists():
            print(f"[{idx}/{len(FILES_TO_SYNC)}] Skipped (local file not found): {filename}")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"[{idx}/{len(FILES_TO_SYNC)}] Uploading {filename} ({size_mb:.2f} MB)...", flush=True)
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model",
                token=hf_token,
            )
            elapsed = time.time() - t0
            print(f"    [OK] {filename} successfully uploaded ({elapsed:.1f}s)", flush=True)
            success_count += 1
        except Exception as e:
            print(f"    [ERROR] Failed to upload {filename}: {e}", flush=True)

    print(
        f"\nCompleted: {success_count}/{len(FILES_TO_SYNC)} models and artifacts synced "
        f"to private repo https://huggingface.co/{REPO_ID}"
    )


if __name__ == "__main__":
    main()
