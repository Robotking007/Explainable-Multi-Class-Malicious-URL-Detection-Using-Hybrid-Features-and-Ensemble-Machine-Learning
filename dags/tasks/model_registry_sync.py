"""
Task 5: Model Registry & Hugging Face Hub Deployment.
Synchronizes validated model artifacts to Hugging Face Hub repository siddudavant/cyberxai-models.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DEFAULT_REPO_ID = "siddudavant/cyberxai-models"

ARTIFACT_FILENAMES = [
    "url_dl_model.pt",
    "url_dl_scaler.joblib",
    "url_dl_label_encoder.joblib",
    "url_dl_metrics.json",
]


def _get_hf_token() -> str:
    # 1. Environment variable
    env_tok = os.environ.get("HF_TOKEN")
    if env_tok:
        return env_tok

    # 2. Local .env file
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()

    raise ValueError("HF_TOKEN not found in environment or .env file.")


def sync_to_huggingface(repo_id: str = DEFAULT_REPO_ID) -> dict:
    print(f"[Airflow Task: sync_to_huggingface] Publishing artifacts to {repo_id}...")
    token = _get_hf_token()
    api = HfApi(token=token)

    who = api.whoami()
    print(f"Authenticated as Hugging Face user: {who['name']}")

    uploaded_files = []
    for fn in ARTIFACT_FILENAMES:
        local_path = MODEL_DIR / fn
        if not local_path.exists():
            print(f"Warning: {local_path} not found locally, skipping.")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"Uploading {fn} ({size_mb:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=fn,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        uploaded_files.append(fn)

    summary = {
        "status": "DEPLOYED",
        "repo_id": repo_id,
        "repo_url": f"https://huggingface.co/{repo_id}",
        "uploaded_files": uploaded_files,
        "total_files": len(uploaded_files),
    }
    print(f"[Airflow Task: sync_to_huggingface] SUCCESS: {summary}")
    return summary


if __name__ == "__main__":
    sync_to_huggingface()
