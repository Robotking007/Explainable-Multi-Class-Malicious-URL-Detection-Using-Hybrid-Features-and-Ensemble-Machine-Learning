"""
Module B - URL static feature extraction.

Implements the lexical, host/structural, and statistical feature blocks
described in the project synopsis (Section 5.2 / [P1],[P2],[P3]):

  - Lexical:    length ratios, digit/uppercase ratios, HTTPS flag,
                IP-in-URL flag, special-character counts.
  - Host-based: subdomain count, domain length, TLD checks, hyphen/digit
                flags, domain entropy. (Live WHOIS lookups are NOT run
                inline here -- see whois_enrich.py for an optional,
                rate-limited enrichment step for a small sample, since a
                per-URL WHOIS query is infeasible at 650k-row scale and
                requires live network access.)
  - Content:    phishing/brand/action/sensitive keyword counts computed
                directly on the URL string (no live page fetch, since
                this dataset only supplies the URL).
  - Structural: path depth, parameter count, fragment/double-slash flags,
                path/query entropy and length.
  - Statistical: overall URL entropy, compression ratio, character
                diversity.

Char-level TF-IDF/n-gram features (the "TF-IDF/N-gram" block in Image 1)
are produced separately by build_ngram_features() so they can be fit only
on the training split (avoiding leakage) and concatenated with the
engineered numeric features below.
"""
from __future__ import annotations

import math
import re
import zlib
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract

SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf",
    "loan", "download", "men", "win", "bid", "stream", "date", "racing",
}

PHISH_KEYWORDS = [
    "login", "signin", "account", "verify", "secure", "update", "confirm",
    "banking", "webscr", "password", "credential", "suspend", "urgent",
]
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "bank", "instagram", "whatsapp", "outlook", "icloud",
]
ACTION_KEYWORDS = ["download", "install", "click", "free", "gift", "bonus", "win"]
SENSITIVE_KEYWORDS = ["ssn", "creditcard", "cvv", "otp", "pin", "wallet"]

IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)
SPECIAL_CHARS = ["@", "?", "-", "=", ".", "/", "%", "&", ":"]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = pd.Series(list(s)).value_counts()
    probs = counts / len(s)
    return float(-(probs * np.log2(probs)).sum())


_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


def normalize_url(raw_url: str) -> str:
    """Strip any scheme the caller supplied so every URL is featurized on equal
    footing. This dataset (Dataset/malicious_phish.csv) was assembled such that
    ~92% of benign rows carry no http(s):// prefix at all while ~96-100% of
    malware/defacement rows do -- an artifact of how each class was collected,
    not a real phishing signal. Left unstripped, a model trained on this data
    learns "has a scheme prefix" as a proxy for "malicious" and will
    misclassify an ordinary, fully-qualified URL like https://www.google.com.
    Stripping the scheme here, uniformly, at both train and inference time,
    removes that leak.

    Also strips a literal wrapping single-quote (e.g. "'example.com/x'"),
    a CSV-quoting artifact found in MUD_malicious_urls_2026_V2.csv (and
    propagated into several sources merged from it) that survives pandas'
    default parsing since single quotes aren't its quote character. Found
    via 8,356 affected rows (0.32% of combined_urls.csv) making WHOIS
    lookups fail on corrupted domains like "'aepc1.com" during the bulk
    WHOIS enrichment run -- a small fraction of rows, but a real, fixable
    data bug, not just registrar rate-limiting.

    merged_urls_dataset.csv has its own variant of this: the quote comes
    AFTER the scheme ("http://'example.com/x'"), so the quote-strip must
    run after scheme-strip, not before -- checking for it first (as an
    earlier version of this function did) silently missed every row from
    that source, since the string doesn't start with "'" until the scheme
    is already gone."""
    s = str(raw_url).strip()
    s = _SCHEME_RE.sub("", s)
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1]
        s = _SCHEME_RE.sub("", s)  # handles the hypothetical "'http://domain'" case too
    return s


def _safe_urlparse(url: str):
    u = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", u):
        u = "http://" + u
    try:
        return urlparse(u)
    except Exception:
        return urlparse("http://invalid")


def extract_features_row(raw_url: str) -> dict:
    url = normalize_url(raw_url)
    parsed = _safe_urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    ext = tldextract.extract(url)
    domain = ext.domain or ""
    subdomain = ext.subdomain or ""
    suffix = ext.suffix or ""
    registered_domain = ext.registered_domain or hostname

    lower_url = url.lower()

    feats = {}

    # ---- Lexical ----
    feats["url_length"] = len(url)
    feats["hostname_length"] = len(hostname)
    feats["path_length"] = len(path)
    digits = sum(c.isdigit() for c in url)
    uppers = sum(c.isupper() for c in url)
    feats["digit_ratio"] = digits / max(len(url), 1)
    feats["uppercase_ratio"] = uppers / max(len(url), 1)
    # NOTE: no https_flag feature -- normalize_url() strips any scheme the
    # caller supplied (see its docstring for why), so scheme is not
    # observable here and would otherwise be a constant, uninformative column.
    feats["ip_in_url_flag"] = int(bool(IP_RE.match(hostname)))
    for ch in SPECIAL_CHARS:
        feats[f"count_{ch if ch.isalnum() else {'@':'at','?':'qmark','-':'hyphen','=':'eq','.':'dot','/':'slash','%':'pct','&':'amp',':':'colon'}[ch]}"] = url.count(ch)

    # ---- Host-based (no live WHOIS; structural/lexical host signals only) ----
    feats["subdomain_count"] = subdomain.count(".") + (1 if subdomain else 0)
    feats["domain_length"] = len(domain)
    feats["tld_length"] = len(suffix)
    feats["suspicious_tld_flag"] = int(suffix.split(".")[-1] in SUSPICIOUS_TLDS if suffix else False)
    feats["hyphen_in_domain_flag"] = int("-" in domain)
    feats["digit_in_domain_flag"] = int(any(c.isdigit() for c in domain))
    feats["domain_entropy"] = _shannon_entropy(registered_domain)
    feats["brand_in_subdomain_flag"] = int(
        any(b in subdomain.lower() for b in BRAND_KEYWORDS) and domain.lower() not in BRAND_KEYWORDS
    )
    # Added per [P1]/[P3]'s host-feature set (subdomain_length, domain_token_count)
    # and to widen the brand check from subdomain-only to the full domain
    # string (P1's "brand_domain" checks the whole domain, not just the
    # subdomain -- e.g. "paypal-verify-account.com" itself, not only a
    # subdomain like "paypal.evil.com").
    feats["subdomain_length"] = len(subdomain)
    feats["domain_token_count"] = len([t for t in re.split(r"[-.]", domain) if t])
    feats["brand_in_domain_flag"] = int(any(b in domain.lower() for b in BRAND_KEYWORDS))

    # ---- Content keyword counts (computed on URL string; no page fetch) ----
    feats["phish_keyword_count"] = sum(k in lower_url for k in PHISH_KEYWORDS)
    feats["brand_keyword_count"] = sum(k in lower_url for k in BRAND_KEYWORDS)
    feats["action_keyword_count"] = sum(k in lower_url for k in ACTION_KEYWORDS)
    feats["sensitive_keyword_count"] = sum(k in lower_url for k in SENSITIVE_KEYWORDS)

    # ---- Structural ----
    feats["path_depth"] = path.count("/")
    feats["param_count"] = query.count("&") + (1 if query else 0)
    feats["has_fragment_flag"] = int(bool(fragment))
    feats["double_slash_in_path_flag"] = int("//" in path)
    feats["query_length"] = len(query)
    feats["path_entropy"] = _shannon_entropy(path)
    feats["query_entropy"] = _shannon_entropy(query)

    # ---- Statistical ----
    feats["url_entropy"] = _shannon_entropy(url)
    compressed_len = len(zlib.compress(url.encode("utf-8", errors="ignore")))
    feats["compression_ratio"] = compressed_len / max(len(url), 1)
    feats["char_diversity"] = len(set(url)) / max(len(url), 1)
    # Added per [P1]/[P3]'s statistical-feature set: token/special-char
    # distribution measures, computed by splitting the URL on common
    # separators (the same characters counted individually above).
    tokens = [t for t in re.split(r"[/\-_.?=&%@:]+", url) if t]
    token_lengths = [len(t) for t in tokens]
    feats["token_count"] = len(tokens)
    feats["avg_token_length"] = float(np.mean(token_lengths)) if token_lengths else 0.0
    feats["max_token_length"] = max(token_lengths) if token_lengths else 0
    feats["unique_token_ratio"] = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    special_chars = sum(1 for c in url if not c.isalnum())
    feats["special_char_ratio"] = special_chars / max(len(url), 1)

    return feats


def build_static_features(urls: pd.Series) -> pd.DataFrame:
    """Vectorized-ish wrapper: apply extract_features_row over a Series of URLs."""
    records = urls.astype(str).apply(extract_features_row)
    df = pd.DataFrame(list(records))
    # float64->float32: at ~20M-row scale this DataFrame briefly coexists
    # with the (also large) n-gram block during pd.concat in train.py --
    # halving it here reduces that peak. See build_ngram_features' own
    # float32 cast below for the OOM history that motivated both.
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df


def build_ngram_features(train_urls: pd.Series, test_urls: pd.Series, max_features: int = 100):
    """Char-level TF-IDF n-gram block ([P1]/[P3] 'TF-IDF/N-gram' feature type).
    Fit ONLY on the training split to avoid leakage; transform both splits.

    OOM history: at ~20M rows, TfidfVectorizer's default float64 sparse
    output, densified via .toarray(), needs rows*max_features*8 bytes --
    at max_features=150 that's ~19GB for a 16M-row train split alone,
    which OOM-killed a full production training run on Kaggle after 4.3h
    of otherwise-successful static feature extraction (no traceback --
    the Linux OOM killer just SIGKILLs the process). Casting the sparse
    matrix to float32 BEFORE densifying (not after) means the dense array
    is float32 from the moment it's allocated, roughly halving this
    block's peak memory."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_norm = train_urls.astype(str).apply(normalize_url)
    test_norm = test_urls.astype(str).apply(normalize_url)

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3), max_features=max_features)
    X_train = vectorizer.fit_transform(train_norm).astype(np.float32)
    X_test = vectorizer.transform(test_norm).astype(np.float32)
    cols = [f"ngram_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train.toarray(), columns=cols)
    test_df = pd.DataFrame(X_test.toarray(), columns=cols)
    return train_df, test_df, vectorizer
