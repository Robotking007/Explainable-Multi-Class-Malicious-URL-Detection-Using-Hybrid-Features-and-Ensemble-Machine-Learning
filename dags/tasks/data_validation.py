"""
Task 1: Data Ingestion & Quality Validation.
Validates dataset presence, schema, null constraints, and class distribution.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Dataset" / "combined_urls.csv"
EXPECTED_CLASSES = {"benign", "phishing", "defacement", "malware"}


def validate_dataset(data_path: str | Path | None = None) -> dict:
    target_path = Path(data_path) if data_path else DATA_PATH
    print(f"[Airflow Task: validate_dataset] Checking dataset at {target_path}...")

    if not target_path.exists():
        raise FileNotFoundError(f"Dataset not found at {target_path}. Run build_combined_dataset.py first.")

    file_size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Read schema & first 100k rows for fast pre-flight check
    sample_df = pd.read_csv(target_path, nrows=100000)
    cols = [c.lower() for c in sample_df.columns]

    if "url" not in cols:
        raise ValueError(f"Missing required 'url' column in {target_path}. Found: {sample_df.columns.tolist()}")

    type_col = "type" if "type" in cols else ("label" if "label" in cols else None)
    if not type_col:
        raise ValueError(f"Missing target label/type column in {target_path}. Found: {sample_df.columns.tolist()}")

    # Check for nulls in sample
    null_urls = sample_df["url"].isnull().sum()
    null_types = sample_df[type_col].isnull().sum()
    if null_urls > 0 or null_types > 0:
        raise ValueError(f"Dataset contains null values: {null_urls} null URLs, {null_types} null labels.")

    # Check class coverage
    present_classes = set(sample_df[type_col].str.lower().unique())
    print(f"Sample classes detected: {present_classes}")

    missing_classes = EXPECTED_CLASSES - present_classes
    if missing_classes:
        print(f"Warning: Classes {missing_classes} not seen in first 100k rows (will be verified in full stream).")

    summary = {
        "status": "VALIDATED",
        "dataset_path": str(target_path),
        "file_size_mb": round(file_size_mb, 2),
        "columns": list(sample_df.columns),
        "sample_classes": list(present_classes),
    }
    print(f"[Airflow Task: validate_dataset] SUCCESS: {summary}")
    return summary


if __name__ == "__main__":
    validate_dataset()
