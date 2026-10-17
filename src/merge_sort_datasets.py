"""
Module to merge, clean, deduplicate, and sort all URL datasets:
  1. Kaggle sid321axn (Dataset/malicious_phish.csv) - 651k rows (multi-class)
  2. Kaggle taruntiwarihp (Dataset/phishing_site_urls.csv) - 549k rows (binary good/bad)
  3. Abuse.ch ThreatFox (Dataset/threatfox_full.csv) - 112k verified IOCs (malware)
  4. Abuse.ch URLhaus (Dataset/file.csv.txt) - 73k verified malware URLs (malware)
  5. PhishTank (Dataset/verified_online.csv) - verified zero-day phishing feed (phishing)
  6. Authoritative benign root & portal domains - top global clean web infrastructure

Cleaning rules:
  - Validates URLs have a domain/host with a dot ('.')
  - Strips wrapping quotes and whitespace
  - Drops corrupt/binary noise strings (control chars, non-ASCII garbage)
  - Resolves collisions with security priority: malware > phishing > defacement > benign
  - Deterministically sorts by class ('type') and URL
  - Outputs to Dataset/merged_clean_urls.csv
"""
from __future__ import annotations

import csv
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
OUTPUT_CSV = DATASET_DIR / "merged_clean_urls.csv"

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_INVALID_START_RE = re.compile(r"^[#$%\s\x00-\x1f]")

LABEL_PRIORITY = {
    "malware": 4,
    "phishing": 3,
    "defacement": 2,
    "benign": 1,
}

AUTHORITATIVE_BENIGN_DOMAINS = [
    "google.com", "www.google.com", "youtube.com", "www.youtube.com", "facebook.com",
    "wikipedia.org", "en.wikipedia.org", "amazon.com", "www.amazon.com", "yahoo.com",
    "reddit.com", "www.reddit.com", "netflix.com", "github.com", "linkedin.com",
    "microsoft.com", "apple.com", "instagram.com", "twitter.com", "x.com",
    "cloudflare.com", "bing.com", "ebay.com", "twitch.tv", "adobe.com",
    "spotify.com", "cnn.com", "bbc.co.uk", "nytimes.com", "theguardian.com",
    "stackoverflow.com", "medium.com", "quora.com", "imdb.com", "pinterest.com",
    "tumblr.com", "craigslist.org", "aliexpress.com", "walmart.com", "espn.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "paypal.com", "stripe.com",
    "dropbox.com", "salesforce.com", "wordpress.org", "harvard.edu", "mit.edu",
    "stanford.edu", "berkeley.edu", "ox.ac.uk", "cam.ac.uk", "nih.gov", "cdc.gov",
    "nasa.gov", "who.int", "un.org", "weather.com", "forbes.com", "bloomberg.com",
    "reuters.com", "wsj.com", "usatoday.com", "archive.org", "vimeo.com", "brown.edu"
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_valid_url_string(s: str) -> bool:
    """Checks if a string is a valid printable URL candidates."""
    if not s or len(s) < 3 or len(s) > 2048:
        return False
    if _INVALID_START_RE.match(s):
        return False
    # Must have at least one dot for domain/host/TLD/IP
    if "." not in s:
        return False
    # Reject binary non-ASCII control characters
    for c in s:
        code = ord(c)
        if code < 32 or code > 126:
            return False
    return True


def normalize_for_dedup(raw: str) -> str:
    """Normalizes URL key for deduplication."""
    s = raw.strip().strip("'\"")
    s = _SCHEME_RE.sub("", s)
    return s.lower().rstrip("/")


def merge_and_sort_all_datasets() -> Path:
    log("Starting dataset merge and sort process...")
    t0 = time.time()

    # In-memory store: norm_key -> (original_url, label)
    dataset_records: Dict[str, Tuple[str, str]] = {}
    stats = Counter()

    def add_record(raw_url: str, label: str, source_name: str) -> None:
        stats[f"raw_{source_name}"] += 1
        clean_url = str(raw_url).strip().strip("'\"")
        if not is_valid_url_string(clean_url):
            stats["invalid_discarded"] += 1
            return

        norm_label = label.strip().lower()
        if norm_label in ("good", "legitimate", "safe", "ham"):
            norm_label = "benign"
        elif norm_label in ("bad", "spam"):
            norm_label = "phishing"

        if norm_label not in LABEL_PRIORITY:
            stats["unknown_label_discarded"] += 1
            return

        key = normalize_for_dedup(clean_url)
        if not key:
            stats["invalid_discarded"] += 1
            return

        if key in dataset_records:
            stats["duplicate_seen"] += 1
            existing_url, existing_label = dataset_records[key]
            # Keep higher priority security label
            if LABEL_PRIORITY[norm_label] > LABEL_PRIORITY[existing_label]:
                dataset_records[key] = (clean_url, norm_label)
                stats["collision_promoted"] += 1
        else:
            dataset_records[key] = (clean_url, norm_label)
            stats["unique_added"] += 1

    # 1. Kaggle malicious_phish.csv (651k rows multi-class)
    p_mal_phish = DATASET_DIR / "malicious_phish.csv"
    if p_mal_phish.exists():
        log(f"Loading {p_mal_phish.name} ...")
        with open(p_mal_phish, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    add_record(row[0], row[1], "kaggle_malicious_phish")

    # 2. Kaggle phishing_site_urls.csv (549k rows binary)
    p_phish_sites = DATASET_DIR / "phishing_site_urls.csv"
    if p_phish_sites.exists():
        log(f"Loading {p_phish_sites.name} ...")
        with open(p_phish_sites, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    add_record(row[0], row[1], "kaggle_phishing_site_urls")

    # 3. Abuse.ch ThreatFox (112k IOCs)
    p_threatfox = DATASET_DIR / "threatfox_full.csv"
    if p_threatfox.exists():
        log(f"Loading {p_threatfox.name} ...")
        with open(p_threatfox, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0] or row[0].startswith("#") or len(row) < 3:
                    continue
                ioc_val = row[2].strip()
                add_record(ioc_val, "malware", "abusech_threatfox")

    # 4. Abuse.ch URLhaus (73k URLs)
    p_urlhaus = DATASET_DIR / "file.csv.txt"
    if p_urlhaus.exists():
        log(f"Loading {p_urlhaus.name} ...")
        with open(p_urlhaus, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or not row[0] or row[0].startswith("#") or len(row) < 3:
                    continue
                url_val = row[2].strip()
                add_record(url_val, "malware", "abusech_urlhaus")

    # 5. PhishTank verified_online.csv
    p_phishtank = DATASET_DIR / "verified_online.csv"
    if p_phishtank.exists():
        log(f"Loading {p_phishtank.name} ...")
        with open(p_phishtank, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and len(row) >= 2:
                    add_record(row[1], "phishing", "phishtank_online")

    # 6. Authoritative benign root & portal domains
    log("Injecting authoritative benign domains and variations...")
    for d in AUTHORITATIVE_BENIGN_DOMAINS:
        add_record(d, "benign", "authoritative_benign")
        add_record(f"https://{d}", "benign", "authoritative_benign")
        add_record(f"http://{d}/", "benign", "authoritative_benign")
        if not d.startswith("www.") and "." in d and not d.startswith("en."):
            add_record(f"www.{d}", "benign", "authoritative_benign")

    log(f"Merged total {len(dataset_records):,} unique URLs across all sources.")
    log(f"Discarded invalid rows: {stats['invalid_discarded']:,} | Collisions upgraded: {stats['collision_promoted']:,}")

    # Sorting
    log("Sorting merged dataset by class ('type') and URL...")
    records_list = list(dataset_records.values())
    # Sort deterministically: primary key: type, secondary key: lowercase URL
    records_list.sort(key=lambda item: (item[1], item[0].lower()))

    # Write out to CSV
    log(f"Writing sorted clean dataset to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["url", "type"])
        for orig_url, label in records_list:
            writer.writerow([orig_url, label])

    file_size_mb = OUTPUT_CSV.stat().st_size / 1e6
    log(f"Successfully saved {len(records_list):,} clean rows to {OUTPUT_CSV.name} ({file_size_mb:.1f} MB) in {time.time()-t0:.1f}s.")

    # Class breakdown
    class_counts = Counter(item[1] for item in records_list)
    log("Class breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(records_list)) * 100
        log(f"  - {cls_name:<12}: {count:,} ({pct:.1f}%)")

    return OUTPUT_CSV


if __name__ == "__main__":
    merge_and_sort_all_datasets()
