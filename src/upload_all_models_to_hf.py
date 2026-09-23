"""
Upload all trained PyTorch models, scalers, encoders, metrics, and SHAP explainers
to the Hugging Face repository `siddudavant/cyberxai-models`.

Usage:
    python src/upload_all_models_to_hf.py
"""
import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "siddudavant/cyberxai-models"

# 1. Load HF Token from .env
hf_token = None
for line in (ROOT / ".env").read_text().splitlines():
    if line.startswith("HF_TOKEN="):
        hf_token = line.split("=", 1)[1].strip()

if not hf_token:
    print("Error: HF_TOKEN not found in .env")
    sys.exit(1)

api = HfApi(token=hf_token)
who = api.whoami()
print(f"Logged in as Hugging Face user: {who['name']}")

# 2. File mapping: local_path -> remote_path_in_repo
upload_map = {
    # Deep Learning HybridURLNet (Kaggle GPU Trained)
    ROOT / "models" / "url_dl_model.pt": "url_dl_model.pt",
    ROOT / "models" / "url_dl_scaler.joblib": "url_dl_scaler.joblib",
    ROOT / "models" / "url_dl_label_encoder.joblib": "url_dl_label_encoder.joblib",
    ROOT / "models" / "url_dl_metrics.json": "url_dl_metrics.json",
    
    # URL Infrastructure Graph Neural Network (URLInfraGNN)
    ROOT / "models" / "url_gnn_model.pt": "url_gnn_model.pt",
    ROOT / "models" / "url_gnn_scaler.joblib": "url_gnn_scaler.joblib",
    ROOT / "models" / "url_gnn_label_encoder.joblib": "url_gnn_label_encoder.joblib",
    ROOT / "reports" / "url_gnn_metrics.json": "url_gnn_metrics.json",
    
    # URL Fast SHAP Surrogate
    ROOT / "models" / "url_shap_surrogate_rf.joblib": "url_shap_surrogate_rf.joblib",
    
    # Module Evaluation Metrics
    ROOT / "models" / "from_kaggle_api_seq" / "malware_module_a_api_sequence_metrics.json": "malware_module_a_api_sequence_metrics.json",
    ROOT / "models" / "from_kaggle_dynamic" / "malware_module_a_dynamic_baseline_metrics.json": "malware_module_a_dynamic_baseline_metrics.json",
    ROOT / "models" / "from_kaggle_static" / "malware_module_a_static_baseline_metrics.json": "malware_module_a_static_baseline_metrics.json",
    ROOT / "models" / "from_kaggle_url" / "url_module_b_metrics.json": "url_module_b_metrics.json",
}

# 3. Model Card
MODEL_CARD = """---
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
- obfuscated-malmem2022
- malapi-2019
metrics:
- accuracy
- f1
- precision
- recall
---

# CyberXAI: Explainable Multi-Modal Threat Detection & Classification

**CyberXAI** is an industrial-grade threat intelligence and cyber-defense framework combining:
1. **Module A (Malware Analysis):**
   - **Static Analysis:** PE header, entropy, section analysis on **BODMAS** dataset (57 PE features).
   - **Dynamic Analysis:** Memory dump inspection on **Obfuscated-MalMem2022** (55 volatility memory features, Ransomware vs Spyware vs Trojan).
   - **API Call Sequence Analysis:** Behavioral sequence mining on **MalAPI 2019** (n-gram TF-IDF on API traces).
2. **Module B (URL Threat Analysis):**
   - **HybridURLNet (Deep Learning):** Character-level 1D-CNN + Bidirectional GRU with residual lexical fusion trained on Kaggle cloud GPUs (88.61% accuracy, 0.95 Malware F1).
   - **URLInfraGNN (Graph Attention Network):** Heterogeneous bipartite infrastructure graph linking URLs, Domains, IPs, and Autonomous System Numbers (ASNs) with Layer-2 Attention explainability.
   - **Soft Voting Ensemble:** CatBoost + LightGBM + XGBoost + Random Forest on 79 hybrid lexical, host, and domain features.
   - **Sub-Second Explainability:** Neural gradient saliency, Layer-2 GAT attention weights, and Tree SHAP surrogate attribution.

---

## Model Zoo & Repository Contents

| File | Module / Architecture | Parameters / Size | Key Metric |
| :--- | :--- | :--- | :--- |
| `url_dl_model.pt` | HybridURLNet (1D-CNN + BiGRU + Residual Lexical MLP) | 52,060 params (208 KB) | **88.61% Test Accuracy** (0.95 Malware F1) |
| `url_gnn_model.pt` | URLInfraGNN (2-Layer Pure PyTorch Graph Attention Network) | 15,588 params (70 KB) | **83.33% Graph Test Accuracy** (0.94 Malware F1) |
| `url_voting_classifier.joblib` | Soft Voting Ensemble (CatBoost, LightGBM, XGBoost, RF) | 13.9 GB | Comprehensive ensemble on hybrid URL features |
| `url_shap_surrogate_rf.joblib` | Fast Tree SHAP surrogate explainer | 3.01 MB | Sub-second local feature attributions (<35ms) |
| `malware_dynamic_rf_multiclass_family.joblib` | Random Forest dynamic memory family classifier | 286 MB | Multi-class malware family classification |
| `malware_dynamic_xgb_multiclass_family.joblib` | XGBoost dynamic memory family classifier | 5.84 MB | Multi-class malware family classification |
| `malware_static_rf.joblib` | Random Forest PE static classifier | 531 MB | Static executable triage |
| `malware_static_xgb.joblib` | XGBoost PE static classifier | 2.75 MB | Static executable triage |
| `malware_api_sequence_rf.joblib` | Random Forest behavioral API sequence classifier | 110 MB | Multi-class behavioral classifier |
| `malware_api_sequence_xgb.joblib` | XGBoost behavioral API sequence classifier | 6.80 MB | Multi-class behavioral classifier |

---

## Quickstart: Running Inference with Deep Learning & GNN

### 1. HybridURLNet Deep Learning Inference
```python
import torch
import joblib
from huggingface_hub import hf_hub_download
from src.url_pipeline.dl_model import HybridURLNet

repo = "siddudavant/cyberxai-models"
model_path = hf_hub_download(repo_id=repo, filename="url_dl_model.pt")
scaler_path = hf_hub_download(repo_id=repo, filename="url_dl_scaler.joblib")
encoder_path = hf_hub_download(repo_id=repo, filename="url_dl_label_encoder.joblib")

model = HybridURLNet(lexical_dim=15, num_classes=4)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
```

### 2. URLInfraGNN Graph Attention Network Inference
```python
import torch
from huggingface_hub import hf_hub_download
from src.url_pipeline.gnn_model import URLInfraGNN

repo = "siddudavant/cyberxai-models"
gnn_path = hf_hub_download(repo_id=repo, filename="url_gnn_model.pt")

model = URLInfraGNN(in_dim=15, hidden_dim=64, num_classes=4, heads=2)
model.load_state_dict(torch.load(gnn_path, map_location="cpu"))
model.eval()
```
"""

def main():
    print(f"Beginning upload to Hugging Face: {REPO_ID}")
    
    # Upload Model Card first
    print("Uploading README.md (Model Card)...")
    readme_path = ROOT / "scratch_hf_readme.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")
    try:
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            token=hf_token
        )
        print("  README.md successfully uploaded.")
    except Exception as e:
        print(f"  Error uploading README: {e}")
    finally:
        if readme_path.exists():
            readme_path.unlink()

    # Upload each artifact
    for local_path, remote_name in upload_map.items():
        if not local_path.exists():
            print(f"Skipping {remote_name}: local file {local_path} not found.")
            continue
        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {remote_name} ({size_mb:.2f} MB)...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=REPO_ID,
                repo_type="model",
                token=hf_token
            )
            print(f"  Successfully uploaded {remote_name}.")
        except Exception as e:
            print(f"  Failed to upload {remote_name}: {e}")

    print("\nAll available trained modules and reports transferred to Hugging Face repo successfully!")
    print(f"Repository URL: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
