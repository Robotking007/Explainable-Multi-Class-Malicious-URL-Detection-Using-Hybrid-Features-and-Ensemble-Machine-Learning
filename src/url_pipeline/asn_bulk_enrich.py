"""
Bulk, checkpointed ASN/IP-prefix enrichment for the full production URL
corpus (Dataset/combined_urls.csv).

Two-phase pipeline, matching asn_features.py:
  1. DNS-resolve every unique registered domain to an IP address (threaded,
     since DNS is fast and -- unlike per-registrar WHOIS -- is not prone to
     bulk-automation blocking at reasonable concurrency).
  2. Batch the resolved IPs through Team Cymru's free bulk-whois service
     (whois.cymru.com:43) in chunks, one TCP round-trip per chunk instead
     of one per domain -- this is the step WHOIS enrichment couldn't do
     (WHOIS has no bulk protocol; every registrar is queried individually
     and rate-limits independently, which is why the WHOIS chain needed
     multiple Kaggle sessions to cover ~618k domains).

Checkpointed and resumable: results append to Dataset/asn_cache.csv as
each batch completes, so a laptop sleep/shutdown or session restart loses
at most the last incomplete batch, not hours of progress. Re-running this
script automatically skips domains already in the cache.

Usage:
    python src/url_pipeline/asn_bulk_enrich.py
    python src/url_pipeline/asn_bulk_enrich.py --dns-workers 100 --cymru-batch-size 1000
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
from url_pipeline.asn_features import ASN_DEFAULTS, cymru_bulk_lookup, resolve_ip  # noqa: E402

DATA_PATH = ROOT / "Dataset" / "combined_urls.csv"
CACHE_PATH = ROOT / "Dataset" / "asn_cache.csv"
CACHE_COLS = ["domain", "resolved_ip"] + list(ASN_DEFAULTS.keys())
CHECKPOINT_EVERY_DOMAINS = 2000


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
    ap.add_argument("--dns-workers", type=int, default=100)
    ap.add_argument("--cymru-batch-size", type=int, default=1000)
    ap.add_argument("--dns-timeout", type=float, default=3.0)
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
    domain_batch: list[str] = []
    ip_of_domain: dict[str, str] = {}

    def flush_batch():
        nonlocal domain_batch, ip_of_domain, completed_this_run
        if not domain_batch:
            return
        unique_ips = sorted(set(ip_of_domain.values()))
        asn_by_ip: dict[str, dict] = {}
        # Cymru's docs recommend keeping a single bulk query under a few
        # thousand IPs; chunk defensively even within one flush.
        for i in range(0, len(unique_ips), args.cymru_batch_size):
            chunk = unique_ips[i:i + args.cymru_batch_size]
            asn_by_ip.update(cymru_bulk_lookup(chunk))

        rows = []
        for d in domain_batch:
            ip = ip_of_domain.get(d)
            feats = asn_by_ip.get(ip) if ip else None
            if feats is None:
                feats = dict(ASN_DEFAULTS)
            rows.append({"domain": d, "resolved_ip": ip or "", **feats})
        append_batch(rows)
        completed_this_run += len(domain_batch)
        elapsed = time.time() - t0
        rate = completed_this_run / elapsed if elapsed > 0 else 0
        remaining_count = len(remaining) - completed_this_run
        eta_min = (remaining_count / rate / 60) if rate > 0 else float("inf")
        log(f"Checkpoint: {completed_this_run}/{len(remaining)} done this run "
            f"({len(done) + completed_this_run}/{len(all_domains)} total) -- "
            f"{rate:.1f} domains/s, ETA {eta_min:.1f}min")
        domain_batch = []
        ip_of_domain = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.dns_workers) as pool:
        future_to_domain = {pool.submit(resolve_ip, d, args.dns_timeout): d for d in remaining}
        for future in concurrent.futures.as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                ip = future.result()
            except Exception:
                ip = None
            domain_batch.append(domain)
            if ip:
                ip_of_domain[domain] = ip

            if len(domain_batch) >= CHECKPOINT_EVERY_DOMAINS:
                flush_batch()

    flush_batch()

    log(f"Run complete. {completed_this_run} domains processed in {time.time() - t0:.0f}s. "
        f"Total cached: {len(done) + completed_this_run}/{len(all_domains)}.")


if __name__ == "__main__":
    main()
