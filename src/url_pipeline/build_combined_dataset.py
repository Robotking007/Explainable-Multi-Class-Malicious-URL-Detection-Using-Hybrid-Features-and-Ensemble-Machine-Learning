"""
Merge every usable URL dataset in Dataset/ into a single, deduplicated,
and sorted training corpus for Module B.

Sources are merged in an explicit priority order (highest first) -- for the small
fraction of URLs that appear in more than one source with a DIFFERENT label,
the highest-priority source's label wins:

  1.  Dataset/verified_online.csv            - PhishTank live feed (phishing-only, human-verified)
  2.  Dataset/threatfox_full.csv             - abuse.ch ThreatFox authenticated IOCs (malware-only)
  3.  Dataset/file.csv.txt                    - abuse.ch URLhaus malware dump (if present)
  4.  Dataset/MUD_malicious_urls_2026_V2.csv - 2026 multi-class corpus (if present)
  5.  Dataset/malicious_phish.csv             - Core 4-class benchmark (651k rows)
  6.  Dataset/robotking_dataset/combined_urls.csv - Curated 4-class balanced corpus
  7.  Dataset/Malicious URL v3.csv           - 4-class corpus
  8.  Dataset/Malicious URL Detection.csv    - 4-class corpus
  9.  Dataset/merged_urls_dataset.csv         - multi-class (benign, phishing, malware; adult dropped)
  10. Dataset/PhishVN.../dataset_url.csv      - PhishVN v3.1.0 (Vietnamese research dataset)
  11. Dataset/Web page phishing detection/dataset_B_05_2020.csv - Web page phishing dataset
  12. Dataset/MyDataSET.xlsx                  - Curated phishing & legitimate URLs
  13. Dataset/train-00000-of-00001.parquet    - AlienVault OTX IoCs (domain/url)
  14. Dataset/zhengxian_12M_urls.csv          - 12M-row Kaggle dataset (legitimate/phishing)
  15. Dataset/somanshumahajan_6M_urls.csv     - 6.1M-row Kaggle dataset (0=benign, 1=phishing)
  16. Dataset/kmack_phishing_urls.csv         - HF verified phishing/benign dataset
  17. Dataset/phreshphish_urls.csv            - HF fresh 2025-2026 phishing URL corpus
  18. Dataset/naren1704_urls.csv              - HF multi-source compiled phishing dataset
  19. Dataset/ASA_Phishing_URL_Dataset.csv   - legitimate / phishing
  20. Dataset/URL dataset.csv                 - legitimate / phishing
  21. Dataset/StealthPhisher2025.csv          - legitimate / phishing
  22. Dataset/Phishing URLs.csv               - phishing-only
  23. Dataset/Dataset.csv                     - binary 0=benign / 1=phishing
  24. Dataset/DomainGuard-50K.csv             - binary 0=benign / 1=phishing
  25. Dataset/dataset2.csv                    - binary 0=benign / 1=phishing
  26. Dataset/url_features_extracted1.csv     - binary inverted (1=benign, 0=phishing)
  27. Dataset/PhiUSIIL_Phishing_URL_Dataset.csv - binary inverted (1=benign, 0=phishing)
  28. Dataset/balanced_urls.csv               - ambiguous binary (benign kept only)
  29. Dataset/Dataset I.csv                   - ambiguous binary (benign kept only)
  30. Dataset/Dataset II.csv                  - ambiguous binary (benign kept only)
  31. Dataset/satyammishra_1.5M_benign_domains.csv - business domains (all benign)
  32. Dataset/combined_urls.csv               - historical combined URLs (catch-all baseline)

Processing Pipeline:
  - URLs are normalized and cleaned (schemes stripped, wrapping quotes removed,
    whitespace/control chars stripped, empty/corrupt URLs discarded).
  - Labels are canonicalized to the 4 valid classes: {"benign", "phishing", "defacement", "malware"}.
  - Processing runs via an indexed on-disk SQLite engine with PRIMARY KEY(url), guaranteeing:
      1. Zero-memory-explosion streaming (safe on 16GB RAM machines).
      2. Priority-based conflict resolution (first seen wins).
      3. Perfect deduplication.
      4. Exact alphabetical sorting (A to Z) on URL.
  - Output is streamed to a temporary CSV and atomically replaces Dataset/combined_urls.csv.

Usage:
    python src/url_pipeline/build_combined_dataset.py
    python src/url_pipeline/build_combined_dataset.py --sort-by url
    python src/url_pipeline/build_combined_dataset.py --skip-existing-combined
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, Union

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"
VALID_CLASSES = {"benign", "phishing", "defacement", "malware"}

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_url(raw_url: Any) -> Optional[str]:
    """Clean and normalize URL for deduplication and consistent model ingestion.
    Strips schemes (http://, https://, etc.) and surrounding quotes, whitespace,
    and discards empty/corrupt/null/binary values.
    """
    if raw_url is None:
        return None
    s = str(raw_url).strip().strip("\"' \t\r\n")
    if not s or s.lower() in ("nan", "none", "null", "undefined", "n/a", "-", "--", "http://", "https://", "ftp://"):
        return None
    # Strip scheme
    s = _SCHEME_RE.sub("", s).strip("\"' \t\r\n")
    # Discard non-printable characters or control characters (ord < 32 or ord == 127)
    if any(ord(c) < 32 or ord(c) == 127 for c in s):
        return None
    # Minimum valid length and must contain at least one alphanumeric char
    if len(s) < 3 or not any(c.isalnum() for c in s):
        return None
    return s


def extract_clean_urls(raw_url: Any) -> list[str]:
    """Clean raw URL input. If multiple URLs are concatenated with comma, splits them.
    Filters out null values, whitespace, control characters, quotes, and invalid strings.
    """
    if raw_url is None:
        return []
    s = str(raw_url).strip().strip("\"' \t\r\n")
    if not s or s.lower() in ("nan", "none", "null", "undefined", "n/a", "-", "--", "http://", "https://", "ftp://"):
        return []

    # Check if multiple URLs are packed into one cell separated by comma
    if "," in s and any(k in s for k in ("://", ".com", ".org", ".net", ".io", ".xyz")):
        parts = s.split(",")
    else:
        parts = [s]

    cleaned = []
    for p in parts:
        c_u = clean_url(p)
        if c_u:
            cleaned.append(c_u)
    return cleaned


def map_label(raw_val: Any, mapping: Union[Dict, str, Callable, None]) -> Optional[str]:
    """Map raw dataset label to canonical 4-class taxonomy:
    benign, phishing, defacement, malware.
    Strictly returns None for null/empty/invalid/unmapped values.
    """
    if raw_val is None:
        return None
    s_raw = str(raw_val).strip()
    if not s_raw or s_raw.lower() in ("nan", "none", "null", "undefined", "n/a", ""):
        return None

    if isinstance(mapping, str):
        return mapping if mapping in VALID_CLASSES else None
    if callable(mapping):
        res = mapping(raw_val)
        return res if res in VALID_CLASSES else None
    if isinstance(mapping, dict):
        mapped = mapping.get(raw_val)
        if mapped is None and isinstance(raw_val, str):
            mapped = mapping.get(s_raw.lower())
            if mapped is None:
                try:
                    mapped = mapping.get(int(raw_val))
                except (ValueError, TypeError):
                    pass
        elif mapped is None and isinstance(raw_val, (int, float)):
            try:
                mapped = mapping.get(str(int(raw_val)) if isinstance(raw_val, float) and raw_val.is_integer() else str(raw_val))
            except Exception:
                mapped = mapping.get(str(raw_val))
        return mapped if mapped in VALID_CLASSES else None

    # Fallback standard normalization if mapping is None
    s = s_raw.lower()
    if s in ("benign", "legitimate", "good", "ham", "0", "0.0"):
        return "benign"
    elif s in ("phishing", "phish", "1", "1.0"):
        return "phishing"
    elif s in ("defacement", "deface"):
        return "defacement"
    elif s in ("malware", "exploit", "botnet_cc", "payload_delivery"):
        return "malware"
    return None


def stream_tuples_from_csv(
    path: Path,
    url_col: Union[str, int],
    label_col: Union[str, int, None],
    mapping: Any,
    chunksize: int = 100000,
    header: Any = "infer",
    names: Any = None,
    comment: Optional[str] = None,
    skipinitialspace: bool = False,
    condition_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Generator[list[Tuple[str, str]], None, None]:
    """Stream batches of (cleaned_url, canonical_type) tuples from a CSV file."""
    usecols = None
    if names is None and header != "None" and header is not None:
        if label_col is not None and url_col != label_col:
            usecols = [url_col, label_col]
        elif url_col is not None:
            usecols = [url_col]

    read_kwargs = {
        "chunksize": chunksize,
        "low_memory": False,
        "comment": comment,
        "skipinitialspace": skipinitialspace,
    }
    if header is not None:
        read_kwargs["header"] = header
    if names is not None:
        read_kwargs["names"] = names
    if usecols is not None:
        read_kwargs["usecols"] = usecols

    for chunk in pd.read_csv(path, **read_kwargs):
        if condition_fn is not None:
            chunk = condition_fn(chunk)
            if chunk.empty:
                continue

        u_series = chunk[url_col] if url_col in chunk.columns else chunk.iloc[:, url_col]
        if label_col is not None:
            l_series = chunk[label_col] if label_col in chunk.columns else chunk.iloc[:, label_col]
        else:
            l_series = [mapping] * len(chunk)

        batch = []
        for raw_u, raw_l in zip(u_series, l_series):
            c_type = map_label(raw_l, mapping) if label_col is not None else mapping
            if c_type not in VALID_CLASSES:
                continue
            for c_url in extract_clean_urls(raw_u):
                batch.append((c_url, c_type))

        if batch:
            yield batch


def stream_tuples_from_text(
    path: Path,
    mapping: str,
    chunksize: int = 100000,
) -> Generator[list[Tuple[str, str]], None, None]:
    """Stream batches of (cleaned_url, canonical_type) from a raw line-separated text file."""
    if mapping not in VALID_CLASSES:
        return
    batch = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for c_url in extract_clean_urls(line):
                batch.append((c_url, mapping))
                if len(batch) >= chunksize:
                    yield batch
                    batch = []
    if batch:
        yield batch


def get_dataset_sources(skip_existing_combined: bool = False) -> list[Dict[str, Any]]:
    """Define all dataset sources in explicit priority order."""
    sources: list[Dict[str, Any]] = [
        # 1. PhishTank live feed (human verified phishing)
        {
            "name": "PhishTank Live Feed",
            "path": DATASET_DIR / "verified_online.csv",
            "url_col": "url",
            "label_col": None,
            "mapping": "phishing",
        },
        # 2. ThreatFox IOCs (abuse.ch authenticated malware)
        {
            "name": "ThreatFox Malware IOCs",
            "path": DATASET_DIR / "threatfox_full.csv",
            "header": None,
            "comment": "#",
            "skipinitialspace": True,
            "names": [
                "first_seen_utc", "ioc_id", "ioc_value", "ioc_type", "threat_type",
                "fk_malware", "malware_alias", "malware_printable", "last_seen_utc",
                "confidence_level", "is_compromised", "reference", "tags", "anonymous", "reporter"
            ],
            "url_col": "ioc_value",
            "label_col": None,
            "mapping": "malware",
            "condition_fn": lambda df: df[df["ioc_type"].isin(["url", "domain"])],
        },
        # 3. URLhaus dump (if present)
        {
            "name": "URLhaus Malware Dump",
            "path": DATASET_DIR / "file.csv.txt",
            "comment": "#",
            "header": None,
            "names": ["id", "dateadded", "url", "url_status", "last_online", "threat", "tags", "urlhaus_link", "reporter"],
            "url_col": "url",
            "label_col": None,
            "mapping": "malware",
        },
        # 4. MUD 2026 V2 (if present)
        {
            "name": "MUD Malicious URLs 2026 V2",
            "path": DATASET_DIR / "MUD_malicious_urls_2026_V2.csv",
            "url_col": "url",
            "label_col": "type",
            "mapping": None,
        },
        # 5. Core 4-class benchmark (no header in CSV)
        {
            "name": "Malicious Phish 4-Class Baseline",
            "path": DATASET_DIR / "malicious_phish.csv",
            "header": None,
            "names": ["url", "type"],
            "url_col": "url",
            "label_col": "type",
            "mapping": {
                "benign": "benign",
                "phishing": "phishing",
                "defacement": "defacement",
                "malware": "malware",
            },
        },
        # 6. robotking_dataset (user's curated 4-class corpus)
        {
            "name": "RobotKing Curated 4-Class URLs",
            "path": DATASET_DIR / "robotking_dataset" / "combined_urls.csv",
            "url_col": "url",
            "label_col": "type",
            "mapping": None,
        },
        # 7. Malicious URL v3
        {
            "name": "Malicious URL v3",
            "path": DATASET_DIR / "Malicious URL v3.csv",
            "url_col": "url",
            "label_col": "type",
            "mapping": {
                "benign": "benign",
                "phishing": "phishing",
                "defacement": "defacement",
                "malware": "malware",
            },
        },
        # 8. Malicious URL Detection
        {
            "name": "Malicious URL Detection",
            "path": DATASET_DIR / "Malicious URL Detection.csv",
            "url_col": "url",
            "label_col": "type",
            "mapping": {
                "benign": "benign",
                "legitimate": "benign",
                "phishing": "phishing",
                "defacement": "defacement",
                "malware": "malware",
                "Exploit": "malware",
                "exploit": "malware",
            },
        },
        # 9. Merged URLs Dataset (dropping out-of-scope adult rows)
        {
            "name": "Merged URLs Dataset (Non-Adult)",
            "path": DATASET_DIR / "merged_urls_dataset.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {"benign": "benign", "phish": "phishing", "malware": "malware"},
        },
        # 10. PhishVN research dataset
        {
            "name": "PhishVN Vietnamese Research Dataset",
            "path": DATASET_DIR / "PhishVN A Time-Stamped Vietnamese URL Phishing Dat" / "data" / "dataset_url.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {"benign": "benign", "phishing": "phishing"},
        },
        # 11. Web page phishing detection dataset B
        {
            "name": "Web Page Phishing Detection (Dataset B)",
            "path": DATASET_DIR / "Web page phishing detection" / "dataset_B_05_2020.csv",
            "url_col": "url",
            "label_col": "status",
            "mapping": {"legitimate": "benign", "phishing": "phishing"},
        },
        # 12. MyDataSET (Excel format, inverted: 1=benign, 0=phishing)
        {
            "name": "MyDataSET Curated Phishing/Benign",
            "path": DATASET_DIR / "MyDataSET.xlsx",
            "custom_loader": "excel",
            "url_col": "URL",
            "label_col": "ClassLabel",
            "mapping": {1: "benign", 0: "phishing", "1": "benign", "0": "phishing"},
        },
        # 13. AlienVault OTX indicators (Parquet format)
        {
            "name": "AlienVault OTX IoCs",
            "path": DATASET_DIR / "train-00000-of-00001.parquet",
            "custom_loader": "parquet",
            "url_col": "indicator",
            "label_col": "label",
            "condition_fn": lambda df: df[df["indicator_type"].isin(["url", "domain", "hostname"])],
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 14. Zhengxian 12M URLs
        {
            "name": "Zhengxian 12M URLs",
            "path": DATASET_DIR / "zhengxian_12M_urls.csv",
            "url_col": "input",
            "label_col": "label",
            "mapping": {"legitimate": "benign", "phishing": "phishing"},
        },
        # 15. Somanshumahajan 6M URLs
        {
            "name": "Somanshumahajan 6.1M URLs",
            "path": DATASET_DIR / "somanshumahajan_6M_urls.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 16. KMack Phishing URLs
        {
            "name": "KMack Phishing URLs",
            "path": DATASET_DIR / "kmack_phishing_urls.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 17. PhreshPhish URLs (Fresh 2025-2026)
        {
            "name": "PhreshPhish 2025-2026 URLs",
            "path": DATASET_DIR / "phreshphish_urls.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {"benign": "benign", "phish": "phishing"},
        },
        # 18. Naren1704 URLs
        {
            "name": "Naren1704 Compiled Phishing URLs",
            "path": DATASET_DIR / "naren1704_urls.csv",
            "url_col": "text",
            "label_col": "label",
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 19. ASA Phishing URL Dataset
        {
            "name": "ASA Phishing URL Dataset",
            "path": DATASET_DIR / "ASA_Phishing_URL_Dataset.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {"legitimate": "benign", "phishing": "phishing"},
        },
        # 20. URL dataset.csv
        {
            "name": "URL dataset.csv",
            "path": DATASET_DIR / "URL dataset.csv",
            "url_col": "url",
            "label_col": "type",
            "mapping": {"legitimate": "benign", "phishing": "phishing"},
        },
        # 21. StealthPhisher2025
        {
            "name": "StealthPhisher2025",
            "path": DATASET_DIR / "StealthPhisher2025.csv",
            "url_col": "URL",
            "label_col": "Label",
            "mapping": {"Legitimate": "benign", "Phishing": "phishing", "legitimate": "benign", "phishing": "phishing"},
        },
        # 22. Phishing URLs.csv (Phishing only)
        {
            "name": "Phishing URLs.csv",
            "path": DATASET_DIR / "Phishing URLs.csv",
            "url_col": "url",
            "label_col": "Type",
            "mapping": {"Phishing": "phishing", "phishing": "phishing"},
        },
        # 23. Dataset.csv
        {
            "name": "Dataset.csv",
            "path": DATASET_DIR / "Dataset.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 24. DomainGuard-50K
        {
            "name": "DomainGuard-50K",
            "path": DATASET_DIR / "DomainGuard-50K.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 25. dataset2.csv
        {
            "name": "dataset2.csv",
            "path": DATASET_DIR / "dataset2.csv",
            "url_col": "URL",
            "label_col": "Label",
            "mapping": {0: "benign", 1: "phishing", "0": "benign", "1": "phishing"},
        },
        # 26. url_features_extracted1 (INVERTED: 1=benign, 0=phishing)
        {
            "name": "url_features_extracted1 (Inverted)",
            "path": DATASET_DIR / "url_features_extracted1.csv",
            "url_col": "URL",
            "label_col": "ClassLabel",
            "mapping": {1.0: "benign", 0.0: "phishing", 1: "benign", 0: "phishing", "1.0": "benign", "0.0": "phishing"},
        },
        # 27. PhiUSIIL Phishing URL Dataset (INVERTED: 1=benign, 0=phishing)
        {
            "name": "PhiUSIIL Phishing URL Dataset (Inverted)",
            "path": DATASET_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
            "url_col": "URL",
            "label_col": "label",
            "mapping": {1: "benign", 0: "phishing", 1.0: "benign", 0.0: "phishing", "1": "benign", "0": "phishing"},
        },
        # 28. balanced_urls (ambiguous binary, keep benign only)
        {
            "name": "Balanced URLs (Benign Only)",
            "path": DATASET_DIR / "balanced_urls.csv",
            "url_col": "url",
            "label_col": "label",
            "mapping": {"benign": "benign"},
        },
        # 29. Dataset I (ambiguous binary, keep benign only)
        {
            "name": "Dataset I (Benign Only)",
            "path": DATASET_DIR / "Dataset I.csv",
            "url_col": "Websites",
            "label_col": "Lable",
            "mapping": {"Benign": "benign", "benign": "benign"},
        },
        # 30. Dataset II (ambiguous binary, keep benign only)
        {
            "name": "Dataset II (Benign Only)",
            "path": DATASET_DIR / "Dataset II.csv",
            "url_col": "URL",
            "label_col": "type",
            "mapping": {"Benign": "benign", "benign": "benign"},
        },
        # 31. satyammishra 1.5M benign domains
        {
            "name": "Satyammishra 1.5M Benign Domains",
            "path": DATASET_DIR / "satyammishra_1.5M_benign_domains.csv",
            "url_col": "website",
            "label_col": None,
            "mapping": "benign",
        },
        # 32. Phishing Website Detection Dataset (Train/Test text files)
        {
            "name": "Phishing Website Detection (Benign Train)",
            "path": DATASET_DIR / "Phishing Website Detection Dataset" / "Data" / "Train" / "benign_Train.txt",
            "custom_loader": "text",
            "mapping": "benign",
        },
        {
            "name": "Phishing Website Detection (Benign Test)",
            "path": DATASET_DIR / "Phishing Website Detection Dataset" / "Data" / "Test" / "benign_Test.txt",
            "custom_loader": "text",
            "mapping": "benign",
        },
        {
            "name": "Phishing Website Detection (Phishing Train)",
            "path": DATASET_DIR / "Phishing Website Detection Dataset" / "Data" / "Train" / "malign_Train.txt",
            "custom_loader": "text",
            "mapping": "phishing",
        },
        {
            "name": "Phishing Website Detection (Phishing Test)",
            "path": DATASET_DIR / "Phishing Website Detection Dataset" / "Data" / "Test" / "malign_Test.txt",
            "custom_loader": "text",
            "mapping": "phishing",
        },
    ]

    # 33. Historical combined_urls.csv (catch-all baseline to retain any legacy rows)
    if not skip_existing_combined:
        combined_path = DATASET_DIR / "combined_urls.csv"
        if combined_path.exists():
            sources.append({
                "name": "Existing Combined URLs (Baseline Catch-All)",
                "path": combined_path,
                "url_col": "url",
                "label_col": "type",
                "mapping": None,
            })

    return sources


def load_source_batches(src: Dict[str, Any]) -> Generator[list[Tuple[str, str]], None, None]:
    """Load batches of (url, type) from a source descriptor."""
    path = src["path"]
    loader_type = src.get("custom_loader", "csv")

    if loader_type == "text":
        yield from stream_tuples_from_text(
            path=path,
            mapping=src["mapping"],
        )

    elif loader_type == "excel":
        df = pd.read_excel(path)
        u_col = src["url_col"]
        l_col = src["label_col"]
        mapping = src["mapping"]
        batch = []
        for raw_u, raw_l in zip(df[u_col], df[l_col]):
            c_l = map_label(raw_l, mapping)
            if c_l not in VALID_CLASSES:
                continue
            for c_u in extract_clean_urls(raw_u):
                batch.append((c_u, c_l))
        if batch:
            yield batch

    elif loader_type == "parquet":
        df = pd.read_parquet(path)
        cond = src.get("condition_fn")
        if cond is not None:
            df = cond(df)
        u_col = src["url_col"]
        l_col = src["label_col"]
        mapping = src["mapping"]
        batch = []
        for raw_u, raw_l in zip(df[u_col], df[l_col]):
            c_l = map_label(raw_l, mapping)
            if c_l not in VALID_CLASSES:
                continue
            for c_u in extract_clean_urls(raw_u):
                batch.append((c_u, c_l))
        if batch:
            yield batch

    else:
        yield from stream_tuples_from_csv(
            path=path,
            url_col=src["url_col"],
            label_col=src["label_col"],
            mapping=src["mapping"],
            header=src.get("header", "infer"),
            names=src.get("names"),
            comment=src.get("comment"),
            skipinitialspace=src.get("skipinitialspace", False),
            condition_fn=src.get("condition_fn"),
        )


def build_combined_dataset(
    output_path: Path = DATASET_DIR / "combined_urls.csv",
    sort_by: str = "url",
    temp_db_path: Path = DATASET_DIR / ".merge_temp.db",
    skip_existing_combined: bool = False,
) -> None:
    """Merge, preprocess, deduplicate, and sort all URL datasets in Dataset/ into a single file."""
    start_time = time.time()
    log("=" * 80)
    log("STARTING CYBERXAI URL DATASET MERGE, PREPROCESSING & SORTING PIPELINE")
    log("=" * 80)
    log(f"Output Target : {output_path}")
    log(f"Sorting Mode  : Alphabetical ({sort_by})")
    log(f"Temp DB Path  : {temp_db_path}")

    if temp_db_path.exists():
        try:
            temp_db_path.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(temp_db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF;")
    cur.execute("PRAGMA journal_mode = OFF;")
    cur.execute("PRAGMA cache_size = -200000;")  # ~200MB memory cache
    cur.execute("PRAGMA temp_store = MEMORY;")

    # url is PRIMARY KEY: guarantees instant deduplication and priority order
    cur.execute("CREATE TABLE urls (url TEXT PRIMARY KEY, type TEXT) WITHOUT ROWID;")
    conn.commit()

    sources = get_dataset_sources(skip_existing_combined=skip_existing_combined)
    total_raw_rows = 0
    sources_loaded = 0

    log(f"\nProcessing {len(sources)} candidate sources in priority order...")

    for idx, src in enumerate(sources, 1):
        name = src["name"]
        path = src["path"]

        if not path.exists():
            log(f"  [{idx:02d}/{len(sources):02d}] MISSING: {name} ({path.name})")
            continue

        log(f"  [{idx:02d}/{len(sources):02d}] Loading: {name} ({path.name})...")
        t_src = time.time()
        src_raw = 0
        src_inserted_before = cur.execute("SELECT COUNT(*) FROM urls;").fetchone()[0]

        try:
            for batch in load_source_batches(src):
                src_raw += len(batch)
                cur.executemany("INSERT OR IGNORE INTO urls VALUES (?, ?);", batch)
            conn.commit()

            src_inserted_after = cur.execute("SELECT COUNT(*) FROM urls;").fetchone()[0]
            new_unique = src_inserted_after - src_inserted_before
            dt_src = time.time() - t_src
            total_raw_rows += src_raw
            sources_loaded += 1
            log(f"         Loaded {src_raw:,} rows | +{new_unique:,} new unique URLs ({dt_src:.1f}s)")
        except Exception as e:
            log(f"         FAILED loading {name}: {e}")

    total_unique = cur.execute("SELECT COUNT(*) FROM urls;").fetchone()[0]
    log("\n" + "=" * 80)
    log(f"Deduplication complete: {total_unique:,} unique URLs from {total_raw_rows:,} raw source rows.")
    log(f"Sources successfully ingested: {sources_loaded}")

    # Query class distribution
    class_counts = dict(cur.execute("SELECT type, COUNT(*) FROM urls GROUP BY type;").fetchall())
    log("\nClass distribution in merged corpus:")
    for cls in sorted(VALID_CLASSES):
        count = class_counts.get(cls, 0)
        pct = (count / total_unique) * 100 if total_unique > 0 else 0
        log(f"  {cls:12s} : {count:>12,} ({pct:6.2f}%)")

    # Determine order query
    if sort_by == "url":
        order_query = "SELECT url, type FROM urls ORDER BY url ASC;"
    elif sort_by == "type_url":
        order_query = "SELECT url, type FROM urls ORDER BY type ASC, url ASC;"
    else:
        order_query = "SELECT url, type FROM urls;"

    log(f"\nSorting and streaming {total_unique:,} clean rows to output...")
    tmp_out = output_path.with_suffix(".tmp.csv")
    if tmp_out.exists():
        tmp_out.unlink()

    t_export = time.time()
    cur.execute(order_query)

    with open(tmp_out, "w", encoding="utf-8", newline="") as f_out:
        f_out.write("url,type\n")
        chunk_lines = []
        exported_count = 0
        while True:
            rows = cur.fetchmany(100000)
            if not rows:
                break
            for u, t in rows:
                # Escape any embedded commas or quotes in CSV
                if "," in u or '"' in u or "\n" in u:
                    escaped_u = '"' + u.replace('"', '""') + '"'
                else:
                    escaped_u = u
                chunk_lines.append(f"{escaped_u},{t}\n")
            f_out.writelines(chunk_lines)
            exported_count += len(chunk_lines)
            chunk_lines.clear()

    conn.close()
    try:
        temp_db_path.unlink()
    except Exception:
        pass

    # Atomically replace target
    if output_path.exists():
        backup_path = output_path.with_suffix(".csv.bak")
        try:
            output_path.replace(backup_path)
        except Exception as e:
            log(f"Notice during backup: {e}")

    try:
        tmp_out.replace(output_path)
    except Exception:
        import shutil
        shutil.move(str(tmp_out), str(output_path))

    # Remove backup if successful
    backup_path = output_path.with_suffix(".csv.bak")
    if backup_path.exists():
        try:
            backup_path.unlink()
        except Exception:
            pass

    total_time = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    log("\n" + "=" * 80)
    log(f"SUCCESS: Saved final dataset to {output_path}")
    log(f"Total Rows : {exported_count:,}")
    log(f"File Size  : {file_size_mb:.2f} MB")
    log(f"Total Time : {total_time:.1f} s ({total_time / 60:.2f} min)")
    log("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Merge, preprocess, deduplicate, and sort all URL datasets into combined_urls.csv"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATASET_DIR / "combined_urls.csv"),
        help="Target output CSV path",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="url",
        choices=["url", "type_url", "none"],
        help="Sorting criteria: 'url' (alphabetical A-Z) or 'type_url' (by class then URL)",
    )
    parser.add_argument(
        "--skip-existing-combined",
        action="store_true",
        help="Skip merging rows from the pre-existing combined_urls.csv",
    )
    args = parser.parse_args()

    build_combined_dataset(
        output_path=Path(args.output),
        sort_by=args.sort_by,
        skip_existing_combined=args.skip_existing_combined,
    )


if __name__ == "__main__":
    main()
