"""
Bulk, checkpointed WHOIS enrichment for the full production URL corpus
(Dataset/combined_urls.csv), by explicit request -- full coverage attempted
despite the known risk (registrar rate-limiting/blocking at this volume,
~618k unique registered domains), not the originally-recommended bounded
sample.

Checkpointed and resumable: results append to Dataset/whois_cache.csv as
they complete (not held only in memory), so a laptop sleep/shutdown or
session restart loses at most the last incomplete batch, not hours of
progress. Re-running this script automatically skips domains already in
the cache.

Usage:
    python src/url_pipeline/whois_bulk_enrich.py
    python src/url_pipeline/whois_bulk_enrich.py --workers 30
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import sys
import time
from pathlib import Path

import pandas as pd
import tldextract

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.whois_features import get_whois_features, WHOIS_DEFAULTS  # noqa: E402

DATA_PATH = ROOT / "Dataset" / "combined_urls.csv"
CACHE_PATH = ROOT / "Dataset" / "whois_cache.csv"
CACHE_COLS = ["domain"] + list(WHOIS_DEFAULTS.keys())
CHECKPOINT_EVERY = 500


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_done_domains() -> set:
    if not CACHE_PATH.exists():
        return set()
    done = set()
    with open(CACHE_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                done.add(row[0])
    return done


def append_batch(rows: list[dict]) -> None:
    is_new = not CACHE_PATH.exists()
    with open(CACHE_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_COLS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()

    log(f"Loading {DATA_PATH} and extracting unique registered domains...")
    df = pd.read_csv(DATA_PATH, usecols=["url"])
    domains_series = df["url"].astype(str).apply(lambda u: tldextract.extract(u).registered_domain)
    all_domains = sorted(set(d for d in domains_series if d))
    log(f"Total unique registered domains: {len(all_domains)}")

    done = load_done_domains()
    remaining = [d for d in all_domains if d not in done]
    log(f"Already cached: {len(done)}. Remaining to fetch: {len(remaining)}.")

    if not remaining:
        log("Nothing left to do -- cache is already complete.")
        return

    t0 = time.time()
    completed_this_run = 0
    batch = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_domain = {pool.submit(get_whois_features, d): d for d in remaining}
        for future in concurrent.futures.as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                feats = future.result()
            except Exception:
                feats = dict(WHOIS_DEFAULTS)
            row = {"domain": domain, **feats}
            batch.append(row)
            completed_this_run += 1

            if len(batch) >= CHECKPOINT_EVERY:
                append_batch(batch)
                batch = []
                elapsed = time.time() - t0
                rate = completed_this_run / elapsed if elapsed > 0 else 0
                remaining_count = len(remaining) - completed_this_run
                eta_hours = (remaining_count / rate / 3600) if rate > 0 else float("inf")
                log(f"Checkpoint: {completed_this_run}/{len(remaining)} done this run "
                    f"({len(done) + completed_this_run}/{len(all_domains)} total) -- "
                    f"{rate:.2f} domains/s, ETA {eta_hours:.1f}h")

    if batch:
        append_batch(batch)

    log(f"Run complete. {completed_this_run} domains processed in {time.time() - t0:.0f}s. "
        f"Total cached: {len(done) + completed_this_run}/{len(all_domains)}.")


if __name__ == "__main__":
    main()
