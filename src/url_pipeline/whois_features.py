"""
Module B - WHOIS feature extraction ([P1]'s WHOIS feature block).

[P1] (Khedekar & Pawar, Frontiers in Computer Science, 2026) found that
adding 5 WHOIS-derived features (domain age, registration length, recency,
privacy, availability) to a lexical+host+content+structural feature set
improved F1 by 0.2% on their 5,000-URL dataset, and called domain
registration details "highly significant" for distinguishing legitimate
from malicious sites.

This project's main training corpus (Dataset/combined_urls.csv) is
1,052,060 URLs -- WHOIS is a live network query per domain (this library
measures ~2-4s/query against a real registrar/registry server), so running
it for every row is not feasible: even deduplicated to unique registered
domains, bulk WHOIS querying at that volume would take many hours and
risks being rate-limited or blocked by registrars, whose WHOIS services are
not meant for bulk automated use.

What IS feasible, and provided here:
  1. `get_whois_features(domain)` -- real, live WHOIS lookup for ONE domain,
     cached in-process so repeated lookups of the same registered domain
     (e.g. two URLs on the same site) cost one network round-trip, not two.
     Used by the API for single-URL live predictions (src/api/main.py),
     where a 2-4s WHOIS lookup for one user-submitted URL is affordable.
  2. `run_whois_ablation.py` (sibling script) -- reproduces [P1]'s own
     ablation methodology (train with vs. without WHOIS features) at
     [P1]'s own scale: a 5,000-URL stratified sample, matching the size
     [P1] used specifically because it's the scale WHOIS is practical at.

WHOIS is NOT part of the main 1M+-row production training pipeline
(train.py) for the reason above; this is a deliberate, documented scope
decision, not an oversight.

Features returned (matching [P1] Table 2's WHOIS block):
  whois_available          - 1 if a WHOIS record was found at all
  domain_age_days          - days since creation_date (0 if unavailable)
  registration_length_days - days between creation_date and expiration_date
  recent_domain_flag       - 1 if registered within the last 90 days
  whois_private_flag       - 1 if registrant identity looks redacted/private
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from url_pipeline.features import normalize_url  # noqa: E402

WHOIS_DEFAULTS = {
    "whois_available": 0,
    "domain_age_days": 0,
    "registration_length_days": 0,
    "recent_domain_flag": 0,
    "whois_private_flag": 0,
}

PRIVACY_MARKERS = ["redacted", "privacy", "withheld", "protect", "whoisguard", "private"]


def _first_date(value):
    """python-whois sometimes returns a single datetime, sometimes a list of
    them (one per WHOIS server queried) -- normalize to the earliest one."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        dates = [v for v in value if isinstance(v, datetime)]
        return min(dates) if dates else None
    if isinstance(value, datetime):
        return value
    return None


def _ensure_aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@lru_cache(maxsize=4096)
def get_whois_features(registered_domain: str, timeout: int = 4) -> dict:
    """Live WHOIS lookup for one registered domain (e.g. 'paypal.com', not a
    full URL or subdomain). Cached per-process by domain, since many URLs
    share a domain. Returns WHOIS_DEFAULTS on any failure/timeout -- a
    missing WHOIS record is itself a signal (matches [P1]'s
    whois_available=0 case), not an error to propagate.

    NOTE: the `timeout` kwarg was previously accepted but never actually
    passed to the underlying whois.whois() call, which defaults to a 10s
    socket timeout and can chain through multiple referral WHOIS servers per
    lookup -- observed directly to make a 5,000-URL ablation run project at
    ~2.6 hours for the training split alone. Now forwarded, and lowered to a
    4s default, so a genuinely unresponsive server fails fast instead of
    eating 10+ seconds per (of thousands of) failed lookups."""
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
        "whois_available": 1,
        "domain_age_days": age_days,
        "registration_length_days": reg_length_days,
        "recent_domain_flag": int(age_days < 90),
        "whois_private_flag": is_private,
    }


def get_whois_features_for_url(url: str, timeout: int = 5) -> dict:
    """Convenience wrapper: extract the registered domain from a raw URL and
    look up its WHOIS features."""
    import tldextract
    norm = normalize_url(url)
    ext = tldextract.extract(norm)
    registered_domain = ext.registered_domain or (ext.domain + "." + ext.suffix if ext.suffix else "")
    return get_whois_features(registered_domain, timeout=timeout)
