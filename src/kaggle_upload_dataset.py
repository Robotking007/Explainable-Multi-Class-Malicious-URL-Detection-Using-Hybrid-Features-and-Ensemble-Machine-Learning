"""
One-off helper: upload Dataset/combined_urls.csv to a private Kaggle
Dataset, so a Kaggle Notebook can train Module B on Kaggle's compute
instead of locally (this machine's 16GB RAM has repeatedly bottlenecked
the SMOTE-balanced 4.67M-row fit).

Reads the token from ~/.kaggle/access_token (never inlined in a command
line -- avoids the secret ever appearing in shell history/process args).
kagglehub.dataset_upload() creates datasets as private by default
(hardcoded in its own source), matching the explicit requirement here.

Usage:
    python src/kaggle_upload_dataset.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
DATASET_HANDLE = "siddudavant/cyberxai-combined-urls"
STAGE_DIR = Path("D:/tmp_scratch/kaggle_dataset")


def main():
    token = TOKEN_PATH.read_text().strip()

    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    import kagglehub

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    src = ROOT / "Dataset" / "combined_urls.csv"
    dst = STAGE_DIR / "combined_urls.csv"
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        import shutil
        shutil.copy2(src, dst)

    kagglehub.dataset_upload(
        DATASET_HANDLE, str(STAGE_DIR),
        version_notes="2.5M-row merged URL dataset for CyberXAI Module B",
    )
    print(f"Uploaded to https://www.kaggle.com/datasets/{DATASET_HANDLE}")


if __name__ == "__main__":
    main()
