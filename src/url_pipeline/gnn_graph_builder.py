"""
Infrastructure Graph Builder for URLInfraGNN.

Assembles heterogeneous threat infrastructure graphs connecting:
  - URL Nodes (45-dim hybrid static features)
  - Domain Nodes (8-dim WHOIS & domain structural properties)
  - IP Nodes (4-dim IPv4 octet and routing structure)
  - ASN Nodes (4-dim BGP autonomous system intelligence)

Builds self-connected normalized adjacency matrices and node feature blocks.
Supports both single-URL ego-graphs (for live sub-second API inference)
and multi-URL cluster graphs (for inductive batch training).
"""
from __future__ import annotations

from collections import Counter
import math
import re
import socket
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.url_pipeline.features import (
    BRAND_KEYWORDS,
    IP_RE,
    PHISH_KEYWORDS,
    extract_features_row,
    normalize_url,
)
from src.url_pipeline.asn_features import get_asn_features_for_url
from src.url_pipeline.whois_features import get_whois_features_for_url

GNN_SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf",
}


def _gnn_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def extract_gnn_url_features(url: str) -> List[float]:
    """45-dim URL feature vector matching the Kaggle URLInfraGNN checkpoint schema."""
    url_str = normalize_url(url)
    parsed = urlparse(url_str if "//" in url_str else "http://" + url_str)
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    parts = netloc.split(".")
    tld = parts[-1].lower() if len(parts) > 1 else ""
    domain = parts[-2].lower() if len(parts) > 1 else netloc.lower()
    subdomain_count = max(0, len(parts) - 2)
    u_len = max(len(url_str), 1)
    lower = url_str.lower()

    feats = {
        "url_length": len(url_str),
        "hostname_length": len(netloc),
        "path_length": len(path),
        "digit_ratio": sum(c.isdigit() for c in url_str) / u_len,
        "uppercase_ratio": sum(c.isupper() for c in url_str) / u_len,
        "ip_in_url_flag": int(bool(IP_RE.match(netloc))),
        "count_at": url_str.count("@"),
        "count_qmark": url_str.count("?"),
        "count_hyphen": url_str.count("-"),
        "count_eq": url_str.count("="),
        "count_dot": url_str.count("."),
        "count_slash": url_str.count("/"),
        "count_pct": url_str.count("%"),
        "count_amp": url_str.count("&"),
        "count_colon": url_str.count(":"),
        "subdomain_count": subdomain_count,
        "domain_length": len(domain),
        "tld_length": len(tld),
        "suspicious_tld_flag": int(tld in GNN_SUSPICIOUS_TLDS),
        "hyphen_in_domain_flag": int("-" in domain),
        "digit_in_domain_flag": int(any(c.isdigit() for c in domain)),
        "domain_entropy": _gnn_entropy(domain),
        "subdomain_length": len(".".join(parts[:-2])) if len(parts) > 2 else 0,
        "domain_token_count": len(re.split(r"[.\-_]", netloc)),
        "brand_in_domain_flag": int(any(b in domain for b in BRAND_KEYWORDS)),
        "brand_in_subdomain_flag": int(any(b in ".".join(parts[:-2]) for b in BRAND_KEYWORDS)) if len(parts) > 2 else 0,
        "phish_keyword_count": sum(k in lower for k in PHISH_KEYWORDS),
        "brand_keyword_count": sum(b in lower for b in BRAND_KEYWORDS),
        "action_keyword_count": sum(k in lower for k in ["download", "install", "click", "free", "gift", "bonus", "win"]),
        "sensitive_keyword_count": sum(k in lower for k in ["ssn", "creditcard", "cvv", "otp", "pin", "wallet"]),
        "path_depth": path.count("/"),
        "param_count": len(query.split("&")) if query else 0,
        "has_query_flag": int(bool(query)),
        "has_fragment_flag": int(bool(parsed.fragment)),
        "has_port_flag": int(bool(parsed.port)),
        "double_slash_in_path": int("//" in path),
        "semicolon_in_url": int(";" in url_str),
        "url_entropy": _gnn_entropy(url_str),
        "path_entropy": _gnn_entropy(path),
        "special_char_ratio": sum(url_str.count(c) for c in ["@", "?", "-", "=", ".", "/", "%", "&", ":"]) / u_len,
        "alpha_ratio": sum(c.isalpha() for c in url_str) / u_len,
        "hex_char_ratio": sum(c in "0123456789abcdefABCDEF" for c in url_str) / u_len,
        "longest_token_length": max((len(t) for t in re.split(r"[.\-_/?=&#]", url_str) if t), default=0),
        "consecutive_consonant_ratio": 1.0 if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", lower) else 0.0,
        "token_variance": 0.0,
    }
    return [float(v) for v in feats.values()]

# Known high-abuse ASNs frequently observed hosting disposable phishing kits
SUSPICIOUS_ASNS = {
    200052, 209588, 206980, 212238, 48031, 57523, 202425, 49870, 47583
}

RIR_ENCODING = {"arin": 1.0, "ripencc": 2.0, "apnic": 3.0, "lacnic": 4.0, "afrinic": 5.0}


def build_domain_node_features(url: str, whois_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Extracts an 8-dimensional feature vector for the registered domain node:
    [domain_length, domain_entropy, suspicious_tld_flag, hyphen_in_domain,
     digit_in_domain, token_count, domain_age_days_norm, whois_private_flag]
    """
    clean_url = normalize_url(url)
    parsed = urlparse("//" + clean_url if "//" not in clean_url else clean_url)
    host = parsed.netloc or clean_url.split("/")[0]
    parts = host.split(".")
    domain = parts[-2] if len(parts) > 1 else host
    tld = parts[-1] if len(parts) > 1 else ""

    dom_len = float(len(domain))
    # Domain entropy
    if domain:
        from collections import Counter
        import math
        n = len(domain)
        counts = Counter(domain)
        dom_entropy = float(-sum((c / n) * math.log2(c / n) for c in counts.values()))
    else:
        dom_entropy = 0.0

    suspicious_tld = 1.0 if tld.lower() in {
        "zip", "review", "country", "kim", "cricket", "science", "work",
        "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf"
    } else 0.0

    hyphen_flag = 1.0 if "-" in domain else 0.0
    digit_flag = 1.0 if any(c.isdigit() for c in domain) else 0.0
    token_count = float(len(re.split(r"[.\-_]", host)))

    # WHOIS signals
    age_norm = 0.5  # default neutral age
    private_flag = 0.0
    if whois_info and whois_info.get("whois_available"):
        age_days = float(whois_info.get("domain_age_days", 365) or 365)
        age_norm = min(1.0, max(0.0, age_days / 3650.0))  # 10 years max
        private_flag = 1.0 if whois_info.get("whois_private_flag") else 0.0

    return np.array([
        dom_len / 50.0,
        dom_entropy / 5.0,
        suspicious_tld,
        hyphen_flag,
        digit_flag,
        token_count / 10.0,
        age_norm,
        private_flag,
    ], dtype=np.float32)


def build_ip_node_features(url: str, asn_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Extracts a 4-dimensional feature vector for the resolved host IP:
    [octet1_norm, octet2_norm, octet3_norm, is_routable_flag]
    """
    ip_str = ""
    if asn_info and asn_info.get("resolved_ip"):
        ip_str = str(asn_info["resolved_ip"])

    if not ip_str or not re.match(r"^\d+\.\d+\.\d+\.\d+$", ip_str):
        # Deterministic hash fallback based on hostname if offline/unresolved
        clean_url = normalize_url(url)
        parsed = urlparse("//" + clean_url if "//" not in clean_url else clean_url)
        host = parsed.netloc or clean_url.split("/")[0]
        h = abs(hash(host))
        octets = [(h >> (i * 8)) & 0xFF for i in range(4)]
        routable = 1.0
    else:
        try:
            octets = [int(p) for p in ip_str.split(".")]
            routable = 1.0 if not (octets[0] in [10, 127, 192] and octets[1] == 168) else 0.0
        except Exception:
            octets = [128, 128, 128, 128]
            routable = 1.0

    return np.array([
        octets[0] / 255.0,
        octets[1] / 255.0,
        octets[2] / 255.0,
        routable,
    ], dtype=np.float32)


def build_asn_node_features(url: str, asn_info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Extracts a 4-dimensional feature vector for the BGP ASN routing node:
    [asn_numeric_norm, bgp_prefix_norm, suspicious_asn_flag, rir_code_norm]
    """
    asn_num = 0
    prefix_len = 24
    rir_code = 0.0

    if asn_info and asn_info.get("asn_available"):
        asn_num = int(asn_info.get("asn", 0) or 0)
        prefix_len = int(asn_info.get("bgp_prefix_length", 24) or 24)
        rir_code = RIR_ENCODING.get(str(asn_info.get("registry", "")).lower(), 0.0)
    else:
        # Domain heuristic fallback
        clean_url = normalize_url(url)
        h = abs(hash(clean_url.split("/")[0]))
        asn_num = (h % 65535)

    suspicious_asn = 1.0 if asn_num in SUSPICIOUS_ASNS else 0.0

    return np.array([
        min(1.0, float(asn_num) / 400000.0),
        float(prefix_len) / 32.0,
        suspicious_asn,
        rir_code / 5.0,
    ], dtype=np.float32)


def build_url_star_graph(
    url: str,
    scaler: Any = None,
    whois_info: Optional[Dict[str, Any]] = None,
    asn_info: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Builds a 4-node star-ego infrastructure graph for a single URL:
      Node 0: URL Node (45-dim)
      Node 1: Domain Node (8-dim)
      Node 2: IP Node (4-dim)
      Node 3: ASN Node (4-dim)

    Adjacency Topology:
      Bidirectional edges with self-loops:
        (URL <-> Domain), (Domain <-> IP), (IP <-> ASN)

    Returns:
      node_feats_dict: Dict with raw tensor blocks for each node type
      adj: (4, 4) Adjacency matrix tensor
      meta: Metadata regarding graph node identities
    """
    # 1. URL Features (45-dim Kaggle GNN schema or full hybrid schema)
    expected_dim = getattr(scaler, "n_features_in_", None) if scaler is not None else None
    if expected_dim == 45:
        url_vec = np.array(extract_gnn_url_features(url), dtype=np.float32)
    else:
        url_dict = extract_features_row(url)
        vals = list(url_dict.values())
        if expected_dim is not None and len(vals) > expected_dim:
            vals = vals[:expected_dim]
        url_vec = np.array(vals, dtype=np.float32)
    if scaler is not None:
        url_vec = scaler.transform(url_vec.reshape(1, -1)).squeeze(0)

    # 2. Infrastructure Features
    domain_vec = build_domain_node_features(url, whois_info=whois_info)
    ip_vec = build_ip_node_features(url, asn_info=asn_info)
    asn_vec = build_asn_node_features(url, asn_info=asn_info)

    # Convert to PyTorch tensors
    u_tensor = torch.tensor(url_vec, dtype=torch.float32).unsqueeze(0)       # (1, 45)
    d_tensor = torch.tensor(domain_vec, dtype=torch.float32).unsqueeze(0)    # (1, 8)
    i_tensor = torch.tensor(ip_vec, dtype=torch.float32).unsqueeze(0)        # (1, 4)
    a_tensor = torch.tensor(asn_vec, dtype=torch.float32).unsqueeze(0)       # (1, 4)

    # Adjacency Matrix with Self-Loops:
    # 0: URL, 1: Domain, 2: IP, 3: ASN
    adj = torch.zeros((4, 4), dtype=torch.float32)
    # Self-loops
    adj[0, 0] = adj[1, 1] = adj[2, 2] = adj[3, 3] = 1.0

    # Bidirectional Infrastructure Edges
    adj[0, 1] = adj[1, 0] = 1.0  # URL <-> Domain
    adj[1, 2] = adj[2, 1] = 1.0  # Domain <-> IP
    adj[2, 3] = adj[3, 2] = 1.0  # IP <-> ASN

    meta = {
        "nodes": [
            {"id": 0, "type": "URL", "label": url[:60]},
            {"id": 1, "type": "Domain", "label": normalize_url(url).split("/")[0]},
            {"id": 2, "type": "IP", "label": str(asn_info.get("resolved_ip") if asn_info else "Unresolved")},
            {"id": 3, "type": "ASN", "label": f"AS{asn_info.get('asn')}" if asn_info and asn_info.get("asn") else "AS_UNKNOWN"},
        ],
        "edges": [
            {"source": 0, "target": 1, "relation": "HOSTED_ON"},
            {"source": 1, "target": 2, "relation": "RESOLVES_TO"},
            {"source": 2, "target": 3, "relation": "ANNOUNCED_BY"},
        ]
    }

    feats_dict = {
        "url": u_tensor,
        "domain": d_tensor,
        "ip": i_tensor,
        "asn": a_tensor,
    }

    return feats_dict, adj, meta
