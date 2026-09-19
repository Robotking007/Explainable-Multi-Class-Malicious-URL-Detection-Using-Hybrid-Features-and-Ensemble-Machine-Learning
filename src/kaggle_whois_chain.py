"""
Orchestrates the WHOIS-enrichment kernel across as many Kaggle sessions as
it takes to cover every domain: watch the current run -> download its
output -> merge into the local, cumulative Dataset/whois_cache.csv ->
re-upload as the next session's starting point -> push a new kernel
version -> repeat, until the log reports full coverage or no more
progress is being made.

Usage:
    python src/kaggle_whois_chain.py
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
USER_NAME = "siddudavant"
KERNEL_SLUG = "cyberxai-whois-enrichment"
LOCAL_CACHE = ROOT / "Dataset" / "whois_cache.csv"
STAGE_OUT = ROOT / "models" / "from_kaggle_whois"
POLL_SECONDS = 120
MAX_SESSIONS = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def count_domains(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def merge_cache(downloaded: Path):
    """Merge a session's output cache into the local cumulative cache,
    keeping the first (existing) entry for any domain appearing in both --
    the downloaded file already contains everything the local file had
    (kaggle_kernel_whois_enrich.py seeds its own output from the prior
    cache dataset), so this is really just "adopt the newer, larger file"
    but written defensively in case that assumption ever breaks."""
    import csv
    existing = {}
    if LOCAL_CACHE.exists():
        with open(LOCAL_CACHE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["domain"]] = row
    with open(downloaded, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] not in existing:
                existing[row["domain"]] = row

    fieldnames = ["domain", "whois_available", "domain_age_days",
                  "registration_length_days", "recent_domain_flag", "whois_private_flag"]
    with open(LOCAL_CACHE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())
    return len(existing)


def total_domain_count() -> int:
    import pandas as pd
    import tldextract
    df = pd.read_csv(ROOT / "Dataset" / "combined_urls.csv", usecols=["url"])
    domains = df["url"].astype(str).apply(lambda u: tldextract.extract(u).registered_domain)
    return len(set(d for d in domains if d))


def main():
    token = TOKEN_PATH.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelSessionStatusRequest, ApiListKernelSessionOutputRequest,
        ApiDownloadKernelOutputRequest,
    )
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    total = total_domain_count()
    log(f"Target: {total} total unique domains.")

    for session_num in range(1, MAX_SESSIONS + 1):
        before = count_domains(LOCAL_CACHE)
        log(f"=== Session {session_num}: {before}/{total} done so far ===")
        if before >= total:
            log("Already fully covered -- done.")
            return

        client = build_kaggle_client()
        status_req = ApiGetKernelSessionStatusRequest()
        status_req.user_name = USER_NAME
        status_req.kernel_slug = KERNEL_SLUG
        terminal = {KernelWorkerStatus.COMPLETE, KernelWorkerStatus.ERROR, KernelWorkerStatus.CANCEL_ACKNOWLEDGED}

        while True:
            with client:
                status = client.kernels.kernels_api_client.get_kernel_session_status(status_req)
            log(f"  Kernel status: {status.status}")
            if status.status in terminal:
                break
            time.sleep(POLL_SECONDS)

        STAGE_OUT.mkdir(parents=True, exist_ok=True)
        list_req = ApiListKernelSessionOutputRequest()
        list_req.user_name = USER_NAME
        list_req.kernel_slug = KERNEL_SLUG
        with client:
            listing = client.kernels.kernels_api_client.list_kernel_session_output(list_req)

        downloaded_cache = None
        for f in (listing.files or []):
            file_name = getattr(f, "file_name", None) or str(f)
            if file_name != "whois_cache.csv":
                continue
            dl_req = ApiDownloadKernelOutputRequest()
            dl_req.owner_slug = USER_NAME
            dl_req.kernel_slug = KERNEL_SLUG
            dl_req.file_path = file_name
            with client:
                redirect = client.kernels.kernels_api_client.download_kernel_output(dl_req)
            dest = STAGE_OUT / file_name
            urllib.request.urlretrieve(redirect.url, dest)
            downloaded_cache = dest
            log(f"  Downloaded session output: {dest}")

        if downloaded_cache is None:
            log("  No whois_cache.csv in this session's output -- stopping chain (check for a crash).")
            return

        after = merge_cache(downloaded_cache)
        log(f"  Merged: {after}/{total} domains cached now.")

        if after >= total:
            log("Full coverage reached.")
            return
        if after <= before:
            log("  No progress made this session -- stopping chain to avoid looping forever.")
            return

        log("  Re-uploading merged cache and pushing next session...")
        import subprocess
        subprocess.run(["python", str(ROOT / "src" / "kaggle_upload_whois_cache.py")], check=True)
        subprocess.run(["python", str(ROOT / "src" / "kaggle_push_whois_kernel.py")], check=True)
        log("  Next session pushed.")

    log(f"Reached MAX_SESSIONS={MAX_SESSIONS} -- stopping chain. Re-run this script to continue further.")


if __name__ == "__main__":
    main()
