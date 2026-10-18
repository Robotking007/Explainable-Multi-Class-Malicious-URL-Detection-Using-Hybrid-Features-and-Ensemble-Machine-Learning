"""
Upload any local model files (.pkl, .joblib, .pt, .json) that are missing from Hugging Face Hub.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPO_ID = "siddudavant/cyberxai-models"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_hf_token() -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        sys.exit(".env not found")
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("HF_TOKEN not found in .env")


def main():
    token = load_hf_token()
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    hf_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type="model"))

    local_candidates = [
        p for p in (
            list(MODEL_DIR.glob("*.joblib"))
            + list(MODEL_DIR.glob("*.pkl"))
            + list(MODEL_DIR.glob("*.pt"))
            + list(MODEL_DIR.glob("*.json"))
        ) if not p.name.endswith(".stale_5gb")
    ]

    missing = [p for p in local_candidates if p.name not in hf_files]
    log(f"Found {len(missing)} files missing on Hugging Face ({REPO_ID}). Uploading...")

    for i, path in enumerate(missing, 1):
        sz_mb = path.stat().st_size / 1e6
        log(f"[{i}/{len(missing)}] Uploading {path.name} ({sz_mb:.2f} MB) ...")
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=REPO_ID,
                repo_type="model",
                token=token,
            )
            log(f"  -> Uploaded in {time.time() - t0:.1f}s")
        except Exception as e:
            log(f"  -> Warning uploading {path.name}: {e}")

    log("All missing models have been synchronized with Hugging Face Hub.")


if __name__ == "__main__":
    main()
