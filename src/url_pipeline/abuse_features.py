"""
Module B - Abuse.ch Threat Intelligence feature extraction.

Extracts empirical threat-intel signals from Abuse.ch ThreatFox and URLhaus:
  - abuse_threatfox_match: 1 if URL, domain, or host is in ThreatFox authenticated IOC database
  - abuse_urlhaus_match: 1 if URL or domain is in URLhaus malware database
  - abuse_confidence_score: normalized confidence score [0.0, 1.0] from ThreatFox / URLhaus
  - abuse_payload_delivery_flag: 1 if classified as payload_delivery
  - abuse_c2_flag: 1 if classified as botnet C2 / command & control
  - abuse_online_flag: 1 if URLhaus reports the payload is actively online
  - abuse_tld_risk_score: empirical abuse frequency for the apex TLD across abuse.ch dumps
  - abuse_ioc_match_any: 1 if matched in any abuse.ch threat intelligence stream
"""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Set, Tuple
from urllib.parse import urlparse

import tldextract

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "Dataset"


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


SHARED_HOSTING_DOMAINS = {
    "google.com", "github.com", "microsoft.com", "apple.com", "amazon.com",
    "discord.com", "dropbox.com", "cloudflare.com", "workers.dev", "amazonaws.com",
    "live.com", "sharepoint.com", "firebaseio.com", "blob.core.windows.net",
    "baidu.com", "yahoo.com", "wikipedia.org", "youtube.com", "facebook.com",
    "instagram.com", "twitter.com", "x.com", "linkedin.com", "reddit.com",
    "medium.com", "wordpress.com", "blogspot.com", "telegram.org", "t.me",
    "vk.com", "whatsapp.com", "zoom.us", "bit.ly", "tinyurl.com"
}

HIGH_RISK_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "loan", "buzz", "club",
    "work", "fit", "icu", "cam", "vip", "monster", "rest", "bar", "surf",
    "casa", "sbs", "cfd", "bid", "racing", "win", "stream", "download"
}

LOW_RISK_TLDS = {"com", "org", "net", "edu", "gov", "mil", "io", "ca", "au", "de", "jp", "uk"}


def _norm_url(raw: str) -> str:
    s = str(raw).strip()
    s = _SCHEME_RE.sub("", s)
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
        s = _SCHEME_RE.sub("", s)
    return s.lower().rstrip("/")



class AbuseThreatIntelIndex:
    """Singleton index holding parsed Abuse.ch ThreatFox & URLhaus indicators."""

    _instance = None

    def __init__(self):
        self.tf_urls: Set[str] = set()
        self.tf_domains: Set[str] = set()
        self.tf_ips: Set[str] = set()
        self.tf_confidence: Dict[str, float] = {}
        self.tf_threat_types: Dict[str, str] = {}

        self.uh_urls: Set[str] = set()
        self.uh_domains: Set[str] = set()
        self.uh_online_urls: Set[str] = set()

        self.tld_counts: Counter = Counter()
        self.total_iocs = 0

        self._load_threatfox()
        self._load_urlhaus()

    def _load_threatfox(self):
        tf_file = DATASET_DIR / "threatfox_full.csv"
        if not tf_file.exists():
            return

        try:
            with open(tf_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, skipinitialspace=True)
                for row in reader:
                    if not row or not row[0] or row[0].startswith("#") or len(row) < 10:
                        continue
                    # ThreatFox columns:
                    # first_seen, ioc_id, ioc_value, ioc_type, threat_type, fk_malware, alias, printable, last_seen, confidence_level, ...
                    ioc_val = _norm_url(row[2])
                    ioc_type = row[3].strip().lower()
                    threat_type = row[4].strip().lower()
                    try:
                        conf = float(row[9].strip()) / 100.0
                    except Exception:
                        conf = 0.75

                    ext = tldextract.extract(ioc_val)
                    reg_domain = ext.registered_domain.lower() if ext.registered_domain else ""

                    if ioc_type == "url":
                        self.tf_urls.add(ioc_val)
                    elif ioc_type == "domain":
                        if reg_domain not in SHARED_HOSTING_DOMAINS:
                            self.tf_domains.add(ioc_val)
                    elif "ip" in ioc_type:
                        # strip port if present
                        ip_clean = ioc_val.split(":")[0]
                        self.tf_ips.add(ip_clean)

                    self.tf_confidence[ioc_val] = conf
                    self.tf_threat_types[ioc_val] = threat_type

                    # TLD risk tracking
                    if ext.suffix:
                        self.tld_counts[ext.suffix.lower()] += 1
                    self.total_iocs += 1
        except Exception as e:
            print(f"[abuse_features] Warning loading ThreatFox: {e}")

    def _load_urlhaus(self):
        uh_file = DATASET_DIR / "file.csv.txt"
        if not uh_file.exists():
            return

        try:
            with open(uh_file, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f, skipinitialspace=True)
                for row in reader:
                    if not row or not row[0] or row[0].startswith("#") or len(row) < 5:
                        continue
                    # URLhaus columns: id, dateadded, url, url_status, last_online, threat, tags, urlhaus_link, reporter
                    raw_url = row[2]
                    url_norm = _norm_url(raw_url)
                    status = row[3].strip().lower()

                    self.uh_urls.add(url_norm)
                    if status == "online":
                        self.uh_online_urls.add(url_norm)

                    ext = tldextract.extract(url_norm)
                    reg_domain = ext.registered_domain.lower() if ext.registered_domain else ""
                    if reg_domain and reg_domain not in SHARED_HOSTING_DOMAINS:
                        self.uh_domains.add(reg_domain)
                    if ext.suffix:
                        self.tld_counts[ext.suffix.lower()] += 1
                    self.total_iocs += 1
        except Exception as e:
            print(f"[abuse_features] Warning loading URLhaus: {e}")


@lru_cache(maxsize=1)
def get_abuse_index() -> AbuseThreatIntelIndex:
    return AbuseThreatIntelIndex()


def extract_abuse_features_row(raw_url: str) -> Dict[str, float]:
    """
    Extracts Abuse.ch ThreatFox and URLhaus intelligence features for one URL.
    Returns standard numeric features suitable for tree ensembles and neural models.
    """
    idx = get_abuse_index()
    url = _norm_url(raw_url)

    # Parse components
    ext = tldextract.extract(url)
    domain = ext.registered_domain.lower() if ext.registered_domain else ""
    tld = ext.suffix.lower() if ext.suffix else ""
    host = (ext.subdomain + "." + domain).strip(".") if ext.subdomain else domain

    # IP check
    host_no_port = host.split(":")[0]

    # Shared hosting check
    is_shared = (domain in SHARED_HOSTING_DOMAINS) or (host in SHARED_HOSTING_DOMAINS)

    # ThreatFox match checks (URLs and direct IPs always match; domains only match if not shared host)
    tf_url_hit = url in idx.tf_urls
    tf_domain_hit = (not is_shared) and (domain in idx.tf_domains or host in idx.tf_domains)
    tf_ip_hit = host_no_port in idx.tf_ips
    tf_hit = int(tf_url_hit or tf_domain_hit or tf_ip_hit)

    # URLhaus match checks (exact URLs always match; apex domains only match if not shared host)
    uh_url_hit = url in idx.uh_urls
    uh_domain_hit = (not is_shared) and (domain in idx.uh_domains or host in idx.uh_domains)
    uh_hit = int(uh_url_hit or uh_domain_hit)
    uh_online = int(url in idx.uh_online_urls)

    # Confidence score
    conf = 0.0
    if tf_hit:
        conf = max(
            idx.tf_confidence.get(url, 0.0),
            idx.tf_confidence.get(domain, 0.0) if not is_shared else 0.0,
            idx.tf_confidence.get(host, 0.0) if not is_shared else 0.0,
            idx.tf_confidence.get(host_no_port, 0.0),
            0.85,
        )
    elif uh_hit:
        conf = 0.90 if uh_online else 0.70

    # Threat types
    threat_type = (
        idx.tf_threat_types.get(url)
        or (idx.tf_threat_types.get(domain) if not is_shared else "")
        or (idx.tf_threat_types.get(host) if not is_shared else "")
        or ""
    )
    is_payload = int("payload" in threat_type or uh_hit)
    is_c2 = int("c2" in threat_type or "botnet" in threat_type)

    # TLD abuse risk score
    if tld in HIGH_RISK_TLDS:
        tld_risk = 0.85
    elif tld in {"gov", "edu", "mil"}:
        tld_risk = 0.0
    elif tld in LOW_RISK_TLDS:
        tld_risk = 0.05
    else:
        tld_count = idx.tld_counts.get(tld, 0)
        tld_risk = float(min(tld_count / 1000.0, 0.65))

    any_hit = int(tf_hit or uh_hit)

    return {
        "abuse_threatfox_match": float(tf_hit),
        "abuse_urlhaus_match": float(uh_hit),
        "abuse_confidence_score": float(conf),
        "abuse_payload_delivery_flag": float(is_payload),
        "abuse_c2_flag": float(is_c2),
        "abuse_online_flag": float(uh_online),
        "abuse_tld_risk_score": float(tld_risk),
        "abuse_ioc_match_any": float(any_hit),
    }
