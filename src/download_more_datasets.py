"""
Download additional URL datasets from Kaggle and open threat feeds:
  1. Kaggle datasets:
     - siddharthkumar25/malicious-and-benign-urls (~450k URLs)
     - samahs/malicious-urls-dataset
     - antoniolanza1979/phishing-urls-dataset
  2. OpenPhish Live Zero-Day Phishing Feed (https://openphish.com/feed.txt)
  3. Tranco Top Benign Domains List (https://tranco-list.eu/top-1m.csv.zip)
"""
from __future__ import annotations

import io
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_kaggle_datasets() -> None:
    token_path = Path.home() / ".kaggle" / "access_token"
    if not token_path.exists():
        log("No Kaggle access token found in ~/.kaggle/access_token. Skipping Kaggle downloads.")
        return

    token = token_path.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    import kagglehub

    candidates = [
        "siddharthkumar25/malicious-and-benign-urls",
        "samahs/malicious-urls-dataset",
        "antoniolanza1979/phishing-urls-dataset",
        "saurabhshahane/cyber-security-malicious-url-dataset"
    ]

    for handle in candidates:
        safe_name = handle.replace("/", "_")
        log(f"Fetching Kaggle dataset: {handle} ...")
        try:
            download_dir = Path(kagglehub.dataset_download(handle))
            copied = 0
            for csv_file in download_dir.glob("*.csv"):
                dest = DATASET_DIR / f"{safe_name}_{csv_file.name}"
                shutil.copy2(csv_file, dest)
                size_mb = dest.stat().st_size / 1e6
                log(f"  -> Saved {dest.name} ({size_mb:.1f} MB)")
                copied += 1
            if copied == 0:
                log(f"  No CSV files found in {download_dir}")
        except Exception as e:
            log(f"  Warning downloading {handle}: {e}")


def download_openphish_feed() -> None:
    log("Fetching OpenPhish live phishing feed...")
    url = "https://openphish.com/feed.txt"
    dest = DATASET_DIR / "openphish_feed.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CyberXAI-Threat-Intel/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            dest.write_bytes(content)
            lines = content.decode("utf-8", errors="ignore").splitlines()
            log(f"Saved {len(lines):,} active phishing URLs to {dest.name}")
    except Exception as e:
        log(f"Warning fetching OpenPhish feed: {e}")


def download_tranco_top_domains() -> None:
    dest_csv = DATASET_DIR / "tranco_top_1m.csv"
    if dest_csv.exists() and dest_csv.stat().st_size > 1000000:
        log(f"Tranco top 1M domains already exists: {dest_csv.name} ({dest_csv.stat().st_size / 1e6:.1f} MB)")
        return

    log("Downloading Tranco Top 1M benign domains list (research gold standard)...")
    url = "https://tranco-list.eu/top-1m.csv.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CyberXAI-Research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if csv_names:
                    target_name = csv_names[0]
                    dest_csv.write_bytes(zf.read(target_name))
                    log(f"Saved Tranco top 1M to {dest_csv.name} ({dest_csv.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        log(f"Warning downloading Tranco top 1M: {e}")


def main():
    download_kaggle_datasets()
    download_openphish_feed()
    download_tranco_top_domains()
    log("Dataset download complete.")


if __name__ == "__main__":
    main()
