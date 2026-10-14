"""
Task 2: Hybrid Feature Extraction & Stratified Sampling.
Extracts all 45 lexical, host, keyword, structural, and statistical features.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from url_pipeline.features import extract_features_row

DATA_PATH = ROOT / "Dataset" / "combined_urls.csv"
OUTPUT_PATH = ROOT / "reports" / "features_staged.parquet"


def _worker_chunk(chunk: list[str]) -> list[dict]:
    return [extract_features_row(u) for u in chunk]


def extract_features(
    sample_size: int = 100000,
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict:
    t_start = time.time()
    in_path = Path(data_path) if data_path else DATA_PATH
    out_path = Path(output_path) if output_path else OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Airflow Task: extract_features] Loading dataset from {in_path}...")

    # Stratified streaming: capture 100% of defacement and malware, sample benign/phishing
    chunks = []
    chunksize = 500000
    target_majority_each = sample_size // 2

    # Fast approximate sampling rate
    maj_sample_rate = min(1.0, max(0.001, target_majority_each / 13_000_000))

    print(f"Sampling minority classes 100% + ~{maj_sample_rate * 100:.2f}% of majority classes...")

    for i, chunk in enumerate(pd.read_csv(in_path, chunksize=chunksize)):
        col_map = {c.lower(): c for c in chunk.columns}
        u_col = col_map.get("url", "url")
        l_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))
        chunk = chunk.dropna(subset=[u_col, l_col])

        is_minority = chunk[l_col].str.lower().isin(["defacement", "malware"])
        minority_df = chunk[is_minority]
        majority_df = chunk[~is_minority].sample(frac=maj_sample_rate, random_state=42 + i) if not chunk[~is_minority].empty else chunk[~is_minority]

        sampled = pd.concat([minority_df, majority_df])
        chunks.append(sampled[[u_col, l_col]])

        if sum(len(c) for c in chunks) >= sample_size * 2:
            break

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    col_map = {c.lower(): c for c in df.columns}
    u_col = col_map.get("url", "url")
    l_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))

    print(f"Total sampled rows: {len(df):,}")
    print(f"Class distribution:\n{df[l_col].value_counts()}")

    urls = df[u_col].tolist()
    labels = df[l_col].tolist()
    del df
    gc.collect()

    print("Extracting 45 hybrid features across CPU cores...")
    n_workers = min(os.cpu_count() or 4, 8)
    sub_chunk_size = 25000
    sub_chunks = [urls[i:i + sub_chunk_size] for i in range(0, len(urls), sub_chunk_size)]

    feature_dfs = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for idx, rows in enumerate(executor.map(_worker_chunk, sub_chunks)):
            feature_dfs.append(pd.DataFrame(rows, dtype=np.float32))

    X_all = pd.concat(feature_dfs, ignore_index=True)
    del feature_dfs, urls, sub_chunks
    gc.collect()

    X_all["target_label"] = labels
    X_all.to_parquet(out_path, index=False)
    elapsed = time.time() - t_start

    summary = {
        "status": "COMPLETED",
        "output_path": str(out_path),
        "total_rows": len(X_all),
        "num_features": X_all.shape[1] - 1,
        "elapsed_seconds": round(elapsed, 2),
    }
    print(f"[Airflow Task: extract_features] SUCCESS: {summary}")
    return summary


if __name__ == "__main__":
    extract_features(sample_size=10000)
