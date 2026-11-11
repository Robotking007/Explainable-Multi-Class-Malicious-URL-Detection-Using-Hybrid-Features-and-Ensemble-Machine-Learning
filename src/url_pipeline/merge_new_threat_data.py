"""
Merge newly fetched threat intelligence feeds (ThreatFox, PhishTank, URLhaus)
into the clean dataset, strictly enforcing:
  1. Zero duplicate URLs (non-repeated records across sources).
  2. Zero null, NaN, or whitespace entries.
  3. Canonical 4-class taxonomy: benign, phishing, defacement, malware.
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
VALID_CLASSES = {"benign", "phishing", "defacement", "malware"}
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return None
    s = str(raw_url).strip().strip("\"' \t\r\n")
    if not s or s.lower() in ("nan", "none", "null", "undefined", "n/a", "-", "--"):
        return None
    s = SCHEME_RE.sub("", s).strip("\"' \t\r\n")
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        return None
    if len(s) < 3 or not any(c.isalnum() for c in s):
        return None
    return s


def parse_threatfox() -> list[tuple[str, str]]:
    tf_path = DATASET_DIR / "threatfox_full.csv"
    if not tf_path.exists():
        return []
    records = []
    log(f"Parsing fresh ThreatFox IOCs from {tf_path.name}...")
    with open(tf_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#") or len(row) < 5:
                continue
            ioc_type = row[3].strip().strip('"').lower()
            ioc_val = row[2].strip().strip('"')
            if ioc_type in ("url", "domain", "ip:port"):
                c_u = clean_url(ioc_val)
                if c_u:
                    records.append((c_u, "malware"))
    log(f"Extracted {len(records):,} clean malware IOCs from ThreatFox.")
    return records


def parse_phishtank() -> list[tuple[str, str]]:
    pt_path = DATASET_DIR / "verified_online.csv"
    if not pt_path.exists():
        return []
    records = []
    log(f"Parsing verified PhishTank feed from {pt_path.name}...")
    with open(pt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        url_idx = 1
        if header:
            for idx, col in enumerate(header):
                if col.strip().lower() == "url":
                    url_idx = idx
                    break
        for row in reader:
            if len(row) > url_idx:
                c_u = clean_url(row[url_idx])
                if c_u:
                    records.append((c_u, "phishing"))
    log(f"Extracted {len(records):,} clean phishing URLs from PhishTank.")
    return records


def parse_urlhaus() -> list[tuple[str, str]]:
    uh_path = DATASET_DIR / "file.csv.txt"
    if not uh_path.exists():
        return []
    records = []
    log(f"Parsing URLhaus dump from {uh_path.name}...")
    with open(uh_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip().strip('"') for p in line.split('","')]
            if len(parts) >= 3:
                c_u = clean_url(parts[2])
                if c_u:
                    records.append((c_u, "malware"))
            elif "," in line:
                tokens = line.split(",")
                for tok in tokens:
                    if "://" in tok or "." in tok:
                        c_u = clean_url(tok)
                        if c_u:
                            records.append((c_u, "malware"))
    log(f"Extracted {len(records):,} clean malware URLs from URLhaus.")
    return records


def main():
    new_candidates = []
    new_candidates.extend(parse_threatfox())
    new_candidates.extend(parse_phishtank())
    new_candidates.extend(parse_urlhaus())

    log(f"Total raw candidates collected from fresh feeds: {len(new_candidates):,}")

    # Deduplicate candidate list itself
    candidate_dict: dict[str, str] = {}
    for u, t in new_candidates:
        if u not in candidate_dict and t in VALID_CLASSES:
            candidate_dict[u] = t

    log(f"Unique new candidates after internal deduplication: {len(candidate_dict):,}")

    master_path = DATASET_DIR / "merged_clean_urls.csv"
    if master_path.exists():
        log(f"Checking candidates against master dataset {master_path.name} to remove duplicates...")
        with open(master_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    u_clean = row[0].strip()
                    if u_clean in candidate_dict:
                        del candidate_dict[u_clean]

    log(f"Genuinely new, non-repeated threat records: {len(candidate_dict):,}")

    out_update = DATASET_DIR / "fresh_threat_intel_update.csv"
    with open(out_update, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "type"])
        for u, t in candidate_dict.items():
            if u and t:
                writer.writerow([u, t])

    log(f"Saved {len(candidate_dict):,} clean, non-null, deduplicated threat intelligence URLs to {out_update}")


if __name__ == "__main__":
    main()
