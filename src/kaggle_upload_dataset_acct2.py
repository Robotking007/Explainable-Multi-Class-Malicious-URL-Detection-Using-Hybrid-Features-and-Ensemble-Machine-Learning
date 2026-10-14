"""
Same as kaggle_upload_dataset.py, but uploads under the SECOND Kaggle
account (siddhantudavant) -- set up because the primary account
(siddudavant) has a limited weekly compute quota that repeatedly cut
off the URL training kernel. Keeping a second account's dataset/kernel
in parallel means URL Module B training isn't blocked on the primary
account's quota resetting.

Reads the token from ~/.kaggle/access_token_2 (never inlined in a
command line).

Usage:
    python src/kaggle_upload_dataset_acct2.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token_2"
DATASET_HANDLE = "siddhantudavant/cyberxai-combined-urls"
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
        version_notes="Consolidated, cleaned, deduplicated, sorted 27.02M-row master URL dataset for CyberXAI Module B (account-2 mirror)",
    )
    print(f"Uploaded to https://www.kaggle.com/datasets/{DATASET_HANDLE}")


if __name__ == "__main__":
    main()
