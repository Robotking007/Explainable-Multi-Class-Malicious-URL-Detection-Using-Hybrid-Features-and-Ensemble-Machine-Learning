"""
Upload (or update) Dataset/whois_cache.csv as a private Kaggle dataset, so
the WHOIS-enrichment kernel can resume from existing progress rather than
starting over on every session.

Usage:
    python src/kaggle_upload_whois_cache.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
HANDLE = "siddudavant/cyberxai-whois-cache"
STAGE_DIR = Path("D:/tmp_scratch/kaggle_datasets/cyberxai-whois-cache")


def main():
    token = TOKEN_PATH.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    import kagglehub

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "Dataset" / "whois_cache.csv"
    dst = STAGE_DIR / "whois_cache.csv"
    shutil.copy2(src, dst)

    n_lines = sum(1 for _ in open(src, encoding="utf-8")) - 1
    kagglehub.dataset_upload(HANDLE, str(STAGE_DIR), version_notes=f"{n_lines} domains cached")
    print(f"Uploaded ({n_lines} domains) to https://www.kaggle.com/datasets/{HANDLE}")


if __name__ == "__main__":
    main()
