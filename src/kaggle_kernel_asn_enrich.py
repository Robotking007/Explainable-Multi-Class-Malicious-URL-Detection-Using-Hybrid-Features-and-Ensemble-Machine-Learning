"""
Self-contained ASN/IP-prefix bulk enrichment script for a Kaggle kernel
(CPU-only -- no GPU needed, pure network I/O). Standalone copy of
src/url_pipeline/asn_bulk_enrich.py, inlined since a Kaggle kernel can't
import this repo's package layout.

Two-phase per checkpoint batch: threaded DNS resolution (domain -> IP),
then one Team Cymru bulk-whois TCP round-trip per batch of unique IPs
(whois.cymru.com:43) to get ASN/BGP-prefix/registry/country -- no API key,
no per-registrar rate-limit fragmentation (unlike WHOIS), so this is
expected to finish in well under one Kaggle session, unlike the WHOIS
enrichment which needed a multi-session chain.

Reads the domain list from the cyberxai-combined-urls dataset and any
already-completed lookups from a prior run's cache dataset (if attached).
Writes incremental checkpoints to /kaggle/working/asn_cache.csv.
"""
import concurrent.futures
import csv
import socket
import time
from pathlib import Path

import pandas as pd
import tldextract

OUT_DIR = Path("/kaggle/working")
CACHE_OUT_PATH = OUT_DIR / "asn_cache.csv"
CHECKPOINT_EVERY_DOMAINS = 2000
DNS_WORKERS = 100
DNS_TIMEOUT = 3.0
CYMRU_BATCH_SIZE = 1000
CYMRU_HOST = "whois.cymru.com"
CYMRU_PORT = 43
# Leaves headroom before Kaggle's session cap kills the kernel mid-write.
MAX_RUNTIME_SECONDS = 8.5 * 3600

ASN_DEFAULTS = {
    "asn_available": 0, "asn": 0, "bgp_prefix_length": 0,
    "registry": "", "asn_country_code": "",
}
CACHE_COLS = ["domain", "resolved_ip"] + list(ASN_DEFAULTS.keys())


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_ip(domain: str, timeout: float = DNS_TIMEOUT):
    if not domain:
        return None
    try:
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            return socket.gethostbyname(domain)
        finally:
            socket.setdefaulttimeout(old_timeout)
    except Exception:
        return None


def cymru_bulk_lookup(ips: list, timeout: float = 10.0) -> dict:
    if not ips:
        return {}
    query = "begin\nverbose\n" + "\n".join(ips) + "\nend\n"
    try:
        with socket.create_connection((CYMRU_HOST, CYMRU_PORT), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(query.encode("ascii"))
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks).decode("utf-8", errors="ignore")
    except Exception:
        return {}

    results = {}
    for line in raw.splitlines():
        if "|" not in line or line.strip().lower().startswith("as"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        as_number, ip, bgp_prefix, cc, registry = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            asn_int = int(as_number)
        except ValueError:
            continue
        prefix_len = 0
        if "/" in bgp_prefix:
            try:
                prefix_len = int(bgp_prefix.rsplit("/", 1)[1])
            except ValueError:
                prefix_len = 0
        results[ip] = {
            "asn_available": 1, "asn": asn_int, "bgp_prefix_length": prefix_len,
            "registry": registry.lower(), "asn_country_code": cc.upper(),
        }
    return results


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
    prior_cache_path = find_input("asn_cache.csv")
    if prior_cache_path is not None:
        log(f"Loading prior progress from {prior_cache_path} ...")
        with open(prior_cache_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    done.add(row[0])
        log(f"Already cached from prior runs: {len(done)}")
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
    domain_batch = []
    ip_of_domain = {}
    stopped_early = False

    def flush_batch():
        nonlocal domain_batch, ip_of_domain, completed
        if not domain_batch:
            return
        unique_ips = sorted(set(ip_of_domain.values()))
        asn_by_ip = {}
        for i in range(0, len(unique_ips), CYMRU_BATCH_SIZE):
            chunk = unique_ips[i:i + CYMRU_BATCH_SIZE]
            asn_by_ip.update(cymru_bulk_lookup(chunk))

        rows = []
        for d in domain_batch:
            ip = ip_of_domain.get(d)
            feats = asn_by_ip.get(ip) if ip else None
            if feats is None:
                feats = dict(ASN_DEFAULTS)
            rows.append({"domain": d, "resolved_ip": ip or "", **feats})
        append_batch(rows)
        completed += len(domain_batch)
        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        log(f"Checkpoint: {completed}/{len(remaining)} this run "
            f"({len(done) + completed}/{len(all_domains)} total) -- {rate:.1f}/s")
        domain_batch = []
        ip_of_domain = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=DNS_WORKERS) as pool:
        future_to_domain = {pool.submit(resolve_ip, d): d for d in remaining}
        for future in concurrent.futures.as_completed(future_to_domain):
            if time.time() - t0 > MAX_RUNTIME_SECONDS:
                log("Approaching Kaggle session time limit -- stopping early, saving progress.")
                stopped_early = True
                for f in future_to_domain:
                    f.cancel()
                break
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
