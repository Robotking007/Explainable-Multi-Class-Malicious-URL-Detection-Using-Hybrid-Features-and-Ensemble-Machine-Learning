"""
Self-contained WHOIS bulk enrichment script for a Kaggle kernel (CPU-only
-- no GPU needed, pure network I/O). Standalone copy of
src/url_pipeline/whois_bulk_enrich.py, inlined since a Kaggle kernel can't
import this repo's package layout.

Reads the domain list from the cyberxai-combined-urls dataset and any
already-completed lookups from the cyberxai-whois-cache dataset (if
attached -- won't exist on the very first run). Writes incremental
checkpoints to /kaggle/working/whois_cache.csv, same as the local script,
so a Kaggle session timeout (kernels get capped at several hours) loses at
most the last incomplete batch, not the whole run. The orchestrating
kaggle_whois_chain.py script re-uploads this output as the next run's input
dataset and re-pushes automatically until every domain is covered.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-whois", "tldextract"], check=False)

import concurrent.futures
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import tldextract

OUT_DIR = Path("/kaggle/working")
CACHE_OUT_PATH = OUT_DIR / "whois_cache.csv"
CHECKPOINT_EVERY = 500
WORKERS = 30
# Leaves headroom before Kaggle's session cap kills the kernel mid-write.
MAX_RUNTIME_SECONDS = 8.5 * 3600

WHOIS_DEFAULTS = {
    "whois_available": 0, "domain_age_days": 0, "registration_length_days": 0,
    "recent_domain_flag": 0, "whois_private_flag": 0,
}
CACHE_COLS = ["domain"] + list(WHOIS_DEFAULTS.keys())
PRIVACY_MARKERS = ["redacted", "privacy", "withheld", "protect", "whoisguard", "private"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _first_date(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        dates = [v for v in value if isinstance(v, datetime)]
        return min(dates) if dates else None
    if isinstance(value, datetime):
        return value
    return None


def _ensure_aware(dt):
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def get_whois_features(registered_domain: str, timeout: int = 4) -> dict:
    if not registered_domain or "." not in registered_domain:
        return dict(WHOIS_DEFAULTS)
    try:
        import whois as whois_lib
        w = whois_lib.whois(registered_domain, timeout=timeout)
    except Exception:
        return dict(WHOIS_DEFAULTS)

    creation = _first_date(getattr(w, "creation_date", None))
    expiration = _first_date(getattr(w, "expiration_date", None))
    if creation is None:
        return dict(WHOIS_DEFAULTS)

    now = datetime.now(timezone.utc)
    creation_aware = _ensure_aware(creation)
    age_days = max((now - creation_aware).days, 0)

    reg_length_days = 0
    if expiration is not None:
        expiration_aware = _ensure_aware(expiration)
        reg_length_days = max((expiration_aware - creation_aware).days, 0)

    raw_text = str(dict(w)).lower()
    is_private = int(any(marker in raw_text for marker in PRIVACY_MARKERS))

    return {
        "whois_available": 1, "domain_age_days": age_days,
        "registration_length_days": reg_length_days,
        "recent_domain_flag": int(age_days < 90), "whois_private_flag": is_private,
    }


def find_input(filename: str):
    matches = list(Path("/kaggle/input").rglob(filename))
    return matches[0] if matches else None


def main():
    log("Listing /kaggle/input/ ...")
    for p in Path("/kaggle/input").rglob("*"):
        log(f"  {p}")

    urls_path = find_input("combined_urls.csv")
    if urls_path is None:
        raise FileNotFoundError("combined_urls.csv not found under /kaggle/input")

    log(f"Loading {urls_path} and extracting unique registered domains...")
    df = pd.read_csv(urls_path, usecols=["url"])
    domains_series = df["url"].astype(str).apply(lambda u: tldextract.extract(u).registered_domain)
    all_domains = sorted(set(d for d in domains_series if d))
    log(f"Total unique registered domains: {len(all_domains)}")

    done = set()
    prior_cache_path = find_input("whois_cache.csv")
    if prior_cache_path is not None:
        log(f"Loading prior progress from {prior_cache_path} ...")
        with open(prior_cache_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    done.add(row[0])
        log(f"Already cached from prior runs: {len(done)}")
        # Seed this run's output with everything already done, so the
        # output dataset is always the FULL cumulative cache, not just
        # this session's delta -- simplifies the orchestrator's merge step.
        import shutil
        shutil.copy2(prior_cache_path, CACHE_OUT_PATH)
    else:
        log("No prior cache found -- starting fresh.")

    remaining = [d for d in all_domains if d not in done]
    log(f"Remaining to fetch: {len(remaining)}")
    if not remaining:
        log("Nothing left to do.")
        return

    def append_batch(rows):
        is_new = not CACHE_OUT_PATH.exists()
        with open(CACHE_OUT_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_COLS)
            if is_new:
                writer.writeheader()
            writer.writerows(rows)

    t0 = time.time()
    completed = 0
    batch = []
    stopped_early = False

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_domain = {pool.submit(get_whois_features, d): d for d in remaining}
        for future in concurrent.futures.as_completed(future_to_domain):
            if time.time() - t0 > MAX_RUNTIME_SECONDS:
                log("Approaching Kaggle session time limit -- stopping early, saving progress.")
                stopped_early = True
                for f in future_to_domain:
                    f.cancel()
                break
            domain = future_to_domain[future]
            try:
                feats = future.result()
            except Exception:
                feats = dict(WHOIS_DEFAULTS)
            batch.append({"domain": domain, **feats})
            completed += 1
            if len(batch) >= CHECKPOINT_EVERY:
                append_batch(batch)
                batch = []
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                log(f"Checkpoint: {completed}/{len(remaining)} this run "
                    f"({len(done) + completed}/{len(all_domains)} total) -- {rate:.2f}/s")

    if batch:
        append_batch(batch)

    total_done = len(done) + completed
    log(f"Run finished ({'time limit reached' if stopped_early else 'all remaining processed'}). "
        f"{completed} domains this session. Total cached: {total_done}/{len(all_domains)}.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / "CRASH_TRACEBACK.txt", "w") as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        raise
