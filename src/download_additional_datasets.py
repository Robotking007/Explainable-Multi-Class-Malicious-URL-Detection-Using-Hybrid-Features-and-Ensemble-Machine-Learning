"""
Download additional URL datasets from Kaggle and open threat intelligence feeds.
"""
from __future__ import annotations

import io
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_kaggle_additional():
    token_path = Path.home() / ".kaggle" / "access_token"
    if not token_path.exists():
        log("No Kaggle token found, skipping Kaggle downloads.")
        return

    from kagglehub.config import set_kaggle_api_token
    import kagglehub

    token = token_path.read_text().strip()
    set_kaggle_api_token(token)

    candidates = [
        "muhammadfaizanhassan/phiusiil-phishing-url-website",
        "samahsadiq/benign-and-malicious-urls",
        "antonyj453/urldataset",
        "harisudhan411/phishing-and-legitimate-urls",
    ]

    for handle in candidates:
        safe_prefix = handle.replace("/", "_")
        log(f"Downloading Kaggle dataset: {handle} ...")
        try:
            download_dir = Path(kagglehub.dataset_download(handle))
            log(f"Downloaded to {download_dir}")
            for csv_file in download_dir.rglob("*.csv"):
                dest = DATASET_DIR / f"{safe_prefix}_{csv_file.name}"
                shutil.copy2(csv_file, dest)
                sz_mb = dest.stat().st_size / 1e6
                log(f"  -> Saved {dest.name} ({sz_mb:.2f} MB)")
        except Exception as e:
            log(f"  Warning downloading {handle}: {e}")


def refresh_openphish():
    log("Refreshing live zero-day OpenPhish feed...")
    url = "https://openphish.com/feed.txt"
    dest = DATASET_DIR / "openphish_feed.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CyberXAI-Threat-Intel/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            dest.write_bytes(content)
            lines = [l for l in content.decode("utf-8", errors="ignore").splitlines() if l.strip()]
            log(f"Saved {len(lines):,} active phishing zero-day URLs to {dest.name}")
    except Exception as e:
        log(f"Warning fetching OpenPhish: {e}")


if __name__ == "__main__":
    download_kaggle_additional()
    refresh_openphish()
