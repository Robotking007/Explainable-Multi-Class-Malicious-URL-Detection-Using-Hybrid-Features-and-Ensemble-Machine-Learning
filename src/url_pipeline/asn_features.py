"""
Module B - ASN / IP-prefix host feature extraction.

Ma, Saul, Savage & Voelker ("Beyond Blacklists: Learning to Detect
Malicious Web Sites from Suspicious URLs" / "Identifying Suspicious URLs:
An Application of Large-Scale Online Learning") found IP-address-based
host features -- the resolving IP's Autonomous System Number (ASN), BGP
routing prefix, and registry/allocation info -- to be strong signals:
malicious infrastructure clusters on specific hosting ASNs and IP ranges
in a way lexical features alone can't see.

This project's WHOIS enrichment (whois_features.py) already covers
domain-registration signals; ASN/IP-prefix covers the complementary
*hosting infrastructure* signal the same paper's feature set calls out
separately from WHOIS.

Two steps, both live network operations, no API key required:
  1. DNS resolution: registered domain -> IP address (socket.gethostbyname).
  2. IP -> ASN mapping via Team Cymru's free bulk-whois service
     (whois.cymru.com:43, "begin/verbose/<ips>/end" protocol) -- no
     registration/API key, and unlike per-registrar WHOIS this is a single
     service with no per-registrar rate-limit fragmentation, so it's
     practical to batch many IPs in one TCP round-trip.

Used the same way WHOIS is: NOT part of the main production training
pipeline by default (a single live DNS+ASN lookup costs ~0.1-2s, still too
slow for 20M+ rows inline), but available for:
  - single-URL live predictions (src/api/main.py)
  - bulk, checkpointed full-coverage enrichment (asn_bulk_enrich.py),
    producing Dataset/asn_cache.csv, to be joined into the training
    feature matrix by registered domain -- same pattern as whois_cache.csv.

Features returned:
  asn_available     - 1 if the domain resolved and an ASN was found
  asn               - the numeric ASN (0 if unavailable), e.g. 15169 for Google
  bgp_prefix_length - the /n prefix length of the announced BGP prefix (0 if unavailable)
  registry          - which RIR allocated the block: arin/ripencc/apnic/lacnic/afrinic ("" if unavailable)
  asn_country_code  - the ASN's registered country code ("" if unavailable)
"""
from __future__ import annotations

import socket
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import normalize_url  # noqa: E402

ASN_DEFAULTS = {
    "asn_available": 0,
    "asn": 0,
    "bgp_prefix_length": 0,
    "registry": "",
    "asn_country_code": "",
}

CYMRU_HOST = "whois.cymru.com"
CYMRU_PORT = 43


def resolve_ip(domain: str, timeout: float = 3.0) -> str | None:
    """DNS A-record resolution for a registered domain. Returns None on any
    failure (NXDOMAIN, timeout, no A record) -- a domain that doesn't
    resolve is itself a signal (matches WHOIS's whois_available=0 pattern),
    not an error to propagate."""
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


def cymru_bulk_lookup(ips: list[str], timeout: float = 10.0) -> dict[str, dict]:
    """Team Cymru bulk-whois IP-to-ASN lookup for up to a few thousand IPs
    in one TCP round-trip. Returns {ip: {asn, bgp_prefix_length, registry,
    asn_country_code}}; IPs Cymru has no mapping for (or the whole batch on
    a connection failure) are simply absent from the result, left to the
    caller to fill with ASN_DEFAULTS."""
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

    results: dict[str, dict] = {}
    for line in raw.splitlines():
        if "|" not in line or line.strip().lower().startswith("as"):
            continue  # skip the header row ("AS | IP | BGP Prefix | CC | Registry | ...")
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        as_number, ip, bgp_prefix, cc, registry = parts[0], parts[1], parts[2], parts[3], parts[4]
        if ip not in ips and ip not in results:
            # Cymru echoes back the queried IP verbatim in normal operation;
            # this branch only guards against an unexpected/malformed reply.
            pass
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
            "asn_available": 1,
            "asn": asn_int,
            "bgp_prefix_length": prefix_len,
            "registry": registry.lower(),
            "asn_country_code": cc.upper(),
        }
    return results


@lru_cache(maxsize=4096)
def get_asn_features(registered_domain: str, timeout: float = 3.0) -> dict:
    """Live DNS + ASN lookup for one registered domain. Cached per-process
    by domain. Returns ASN_DEFAULTS on any resolution/lookup failure."""
    ip = resolve_ip(registered_domain, timeout=timeout)
    if ip is None:
        return dict(ASN_DEFAULTS)
    result = cymru_bulk_lookup([ip], timeout=timeout + 5.0)
    feats = result.get(ip)
    if feats is None:
        return dict(ASN_DEFAULTS)
    return feats


def get_asn_features_for_url(url: str, timeout: float = 3.0) -> dict:
    """Convenience wrapper: extract the registered domain from a raw URL
    and look up its ASN/IP-prefix features."""
    import tldextract
    norm = normalize_url(url)
    ext = tldextract.extract(norm)
    registered_domain = ext.registered_domain or (ext.domain + "." + ext.suffix if ext.suffix else "")
    return get_asn_features(registered_domain, timeout=timeout)
