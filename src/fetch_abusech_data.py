"""
Pull additional threat-intel data from abuse.ch, now that using the stored
Auth-Key has been explicitly authorized (it was previously kept
stored-but-unused by instruction):

  1. ThreatFox full CSV export (IOCs from the past 6 months, includes
     malware-delivery and botnet C2 URLs/domains) -- genuinely new data,
     not overlapping with the URLhaus dump already merged.
  2. Refreshed URLhaus bulk CSV (same public dump Dataset/file.csv.txt came
     from, re-pulled to pick up anything added since that download).

Auth-Key is read from .env, sent only as the Auth-Key header (URLhaus) or
embedded in abuse.ch's own documented export URL path (ThreatFox) -- never
logged or printed.

Usage:
    python src/fetch_abusech_data.py
"""
from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_auth_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("ABUSE_CH_AUTH_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ABUSE_CH_AUTH_KEY not found in .env")


def fetch_threatfox(auth_key: str) -> Path:
    url = f"https://threatfox-api.abuse.ch/v2/files/exports/{auth_key}/full.csv.zip"
    log("Downloading ThreatFox full CSV export...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    names = zf.namelist()
    log(f"ThreatFox zip contains: {names}")
    csv_name = next(n for n in names if n.endswith(".csv"))
    out_path = DATASET_DIR / "threatfox_full.csv"
    with zf.open(csv_name) as src, open(out_path, "wb") as dst:
        dst.write(src.read())
    log(f"Saved ThreatFox export to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def refresh_urlhaus(auth_key: str) -> Path:
    log("Refreshing URLhaus bulk CSV dump...")
    resp = requests.get(
        "https://urlhaus.abuse.ch/downloads/csv/",
        headers={"Auth-Key": auth_key},
        timeout=120,
    )
    resp.raise_for_status()
    out_path = DATASET_DIR / "file.csv.txt"
    old_size = out_path.stat().st_size if out_path.exists() else 0

    # This endpoint returns a ZIP (confirmed via its PK magic-byte header),
    # not a plain CSV -- writing resp.content directly (an earlier version
    # of this function did) corrupts the file with raw zip bytes under a
    # .txt name. Extract the CSV inside instead.
    if resp.content[:2] == b"PK":
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv") or n.endswith(".txt"))
        data = zf.read(csv_name)
    else:
        data = resp.content

    out_path.write_bytes(data)
    log(f"Refreshed URLhaus dump: {old_size / 1e6:.1f} MB -> {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


def main():
    auth_key = load_auth_key()
    fetch_threatfox(auth_key)
    refresh_urlhaus(auth_key)
    log("Done.")


if __name__ == "__main__":
    main()
