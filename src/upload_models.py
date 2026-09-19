"""
Push trained model artifacts (models/*.joblib, models/*.json) to the
private Hugging Face Hub repo used to store them, instead of committing
~5GB+ of binaries into git (models/ is gitignored -- see .gitignore).

Reads the write token from .env's HF_TOKEN (never hardcoded, never
committed -- same handling as the abuse.ch key in this file). Uploads only
the files that exist locally, so this is safe to re-run at any point (e.g.
after the malware baselines finish, again later after the URL model
finishes) without needing every model trained yet.

Usage:
    python src/upload_models.py
    python src/upload_models.py --files malware_static_rf.joblib malware_static_xgb.joblib
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPO_ID_SUFFIX = "cyberxai-models"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_token() -> str:
    env_path = ROOT / ".env"
    if not env_path.exists():
        sys.exit(".env not found -- add a line HF_TOKEN=hf_... to it first")
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("HF_TOKEN not found in .env")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                     help="Specific filenames under models/ to upload (default: every .joblib/.json there)")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, whoami

    token = _load_token()
    who = whoami(token=token)
    repo_id = f"{who['name']}/{REPO_ID_SUFFIX}"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
    log(f"Target repo: https://huggingface.co/{repo_id}")

    if args.files:
        paths = [MODEL_DIR / f for f in args.files]
    else:
        paths = sorted(MODEL_DIR.glob("*.joblib")) + sorted(MODEL_DIR.glob("*.json"))

    if not paths:
        log(f"No model files found in {MODEL_DIR} -- nothing to upload.")
        return

    for path in paths:
        if not path.exists():
            log(f"SKIP (not found): {path.name}")
            continue
        size_mb = path.stat().st_size / (1024 * 1024)
        log(f"Uploading {path.name} ({size_mb:.1f} MB) ...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        log(f"  done in {time.time() - t0:.1f}s")

    log("All uploads complete.")


if __name__ == "__main__":
    main()
