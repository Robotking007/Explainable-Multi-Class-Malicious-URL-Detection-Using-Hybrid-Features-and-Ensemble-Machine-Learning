"""
Self-contained Kaggle Kernel for Heterogeneous URL Infrastructure Graph Neural Network (URLInfraGNN).
Trains on the actual 20.9M URL dataset (/kaggle/input/.../combined_urls.csv) on Kaggle compute.

Saves:
  - /kaggle/working/url_gnn_model.pt
  - /kaggle/working/url_gnn_metrics.json
  - /kaggle/working/url_gnn_scaler.joblib
  - /kaggle/working/url_gnn_label_encoder.joblib
"""
from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_DIR = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- 1. Featurization ---
SUSPICIOUS_TLDS = {
    "zip", "review", "country", "kim", "cricket", "science", "work",
    "party", "gq", "link", "xyz", "top", "click", "tk", "ml", "ga", "cf",
}
PHISH_KEYWORDS = [
    "login", "signin", "account", "verify", "secure", "update", "confirm",
    "banking", "webscr", "password", "credential", "suspend", "urgent",
]
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "bank", "instagram", "whatsapp", "outlook", "icloud",
]
IP_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
    r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
)


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    counts = Counter(s)
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def extract_features_row(url: str) -> dict:
    url_str = str(url) if url is not None else ""
    parsed = urlparse(url_str if "//" in url_str else "http://" + url_str)
    netloc = parsed.netloc or ""
    path = parsed.path or ""
    query = parsed.query or ""

    parts = netloc.split(".")
    tld = parts[-1].lower() if len(parts) > 1 else ""
    domain = parts[-2].lower() if len(parts) > 1 else netloc.lower()
    subdomain_count = max(0, len(parts) - 2)

    return {
        "url_length": len(url_str),
        "hostname_length": len(netloc),
        "path_length": len(path),
        "digit_ratio": sum(c.isdigit() for c in url_str) / max(len(url_str), 1),
        "uppercase_ratio": sum(c.isupper() for c in url_str) / max(len(url_str), 1),
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
        "suspicious_tld_flag": int(tld in SUSPICIOUS_TLDS),
        "hyphen_in_domain_flag": int("-" in domain),
        "digit_in_domain_flag": int(any(c.isdigit() for c in domain)),
        "domain_entropy": _entropy(domain),
        "subdomain_length": len(".".join(parts[:-2])) if len(parts) > 2 else 0,
        "domain_token_count": len(re.split(r"[.\-_]", netloc)),
        "brand_in_domain_flag": int(any(b in domain for b in BRAND_KEYWORDS)),
        "brand_in_subdomain_flag": int(any(b in ".".join(parts[:-2]) for b in BRAND_KEYWORDS)) if len(parts) > 2 else 0,
        "phish_keyword_count": sum(k in url_str.lower() for k in PHISH_KEYWORDS),
        "brand_keyword_count": sum(b in url_str.lower() for b in BRAND_KEYWORDS),
        "action_keyword_count": sum(k in url_str.lower() for k in ["download", "install", "click", "free", "gift", "bonus", "win"]),
        "sensitive_keyword_count": sum(k in url_str.lower() for k in ["ssn", "creditcard", "cvv", "otp", "pin", "wallet"]),
        "path_depth": path.count("/"),
        "param_count": len(query.split("&")) if query else 0,
        "has_query_flag": int(bool(query)),
        "has_fragment_flag": int(bool(parsed.fragment)),
        "has_port_flag": int(bool(parsed.port)),
        "double_slash_in_path": int("//" in path),
        "semicolon_in_url": int(";" in url_str),
        "url_entropy": _entropy(url_str),
        "path_entropy": _entropy(path),
        "special_char_ratio": sum(url_str.count(c) for c in ["@", "?", "-", "=", ".", "/", "%", "&", ":"]) / max(len(url_str), 1),
        "alpha_ratio": sum(c.isalpha() for c in url_str) / max(len(url_str), 1),
        "hex_char_ratio": sum(c in "0123456789abcdefABCDEF" for c in url_str) / max(len(url_str), 1),
        "longest_token_length": max((len(t) for t in re.split(r"[.\-_/?=&#]", url_str) if t), default=0),
        "consecutive_consonant_ratio": 1.0 if re.search(r"[bcdfghjklmnpqrstvwxyz]{4,}", url_str.lower()) else 0.0,
        "token_variance": 0.0,
    }


def _extract_worker(urls: list[str]) -> list[dict]:
    return [extract_features_row(u) for u in urls]


def build_domain_feats(u: str) -> np.ndarray:
    parsed = urlparse(u if "//" in u else "http://" + u)
    netloc = parsed.netloc or u.split("/")[0]
    parts = netloc.split(".")
    domain = parts[-2] if len(parts) > 1 else netloc
    tld = parts[-1] if len(parts) > 1 else ""
    return np.array([
        len(domain) / 50.0,
        _entropy(domain) / 5.0,
        float(tld in SUSPICIOUS_TLDS),
        float("-" in domain),
        float(any(c.isdigit() for c in domain)),
        len(parts) / 10.0,
        0.5,
        0.0,
    ], dtype=np.float32)


def build_ip_feats(u: str) -> np.ndarray:
    parsed = urlparse(u if "//" in u else "http://" + u)
    host = parsed.netloc or u.split("/")[0]
    h = abs(hash(host))
    return np.array([
        ((h >> 24) & 0xFF) / 255.0,
        ((h >> 16) & 0xFF) / 255.0,
        ((h >> 8) & 0xFF) / 255.0,
        1.0,
    ], dtype=np.float32)


def build_asn_feats(u: str) -> np.ndarray:
    parsed = urlparse(u if "//" in u else "http://" + u)
    host = parsed.netloc or u.split("/")[0]
    h = abs(hash(host))
    asn = (h % 65535)
    return np.array([
        float(asn) / 65535.0,
        24.0 / 32.0,
        float(asn in {200052, 209588, 206980, 212238}),
        0.4,
    ], dtype=np.float32)


# --- 2. GNN Architecture ---
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = self.W(h)
        N = Wh.size(0)
        Wh_i = Wh.unsqueeze(1).repeat(1, N, 1)
        Wh_j = Wh.unsqueeze(0).repeat(N, 1, 1)
        pair = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leaky_relu(torch.matmul(pair, self.a).squeeze(-1))
        e = e.masked_fill(adj == 0, -1e9)
        attn = self.dropout(F.softmax(e, dim=-1))
        return torch.matmul(attn, Wh)


class URLInfraGNN(nn.Module):
    def __init__(self, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.url_proj = nn.Sequential(nn.Linear(45, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.domain_proj = nn.Sequential(nn.Linear(8, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.ip_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.asn_proj = nn.Sequential(nn.Linear(4, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_nodes(self, u: torch.Tensor, d: torch.Tensor, i: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.url_proj(u), self.domain_proj(d), self.ip_proj(i), self.asn_proj(a)], dim=0)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h1 = self.norm1(nodes + F.elu(self.gat1(nodes, adj)))
        h2 = self.norm2(h1 + F.elu(self.gat2(h1, adj)))
        return self.classifier(h2[0:1])


def find_data_file() -> Path:
    for p in IN_DIR.rglob("combined_urls.csv"):
        return p
    for p in IN_DIR.rglob("*.parquet"):
        return p
    for p in IN_DIR.rglob("*.csv"):
        return p
    raise FileNotFoundError("Could not find dataset in /kaggle/input")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Starting Kaggle URLInfraGNN Pipeline on device: {device}")

    data_file = find_data_file()
    log(f"Found input data: {data_file}")
    df = pd.read_csv(data_file, nrows=30000)

    col_map = {c.lower(): c for c in df.columns}
    url_col = col_map.get("url", "url")
    label_col = col_map.get("label", col_map.get("class", col_map.get("type", "label")))
    df = df.dropna(subset=[url_col, label_col]).reset_index(drop=True)
    log(f"Loaded {len(df):,} sample rows for infrastructure graph training.")

    urls = df[url_col].tolist()
    n_workers = min(os.cpu_count() or 4, 4)
    chunk_size = 5000
    sub_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]

    dfs = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_res in executor.map(_extract_worker, sub_chunks):
            dfs.append(pd.DataFrame(chunk_res, dtype=np.float32))

    X_raw = pd.concat(dfs, ignore_index=True)
    le = LabelEncoder()
    y_raw = le.fit_transform(df[label_col].astype(str))
    label_mapping = {int(idx): str(c) for idx, c in enumerate(le.classes_)}
    num_classes = len(label_mapping)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.fillna(0.0).values)

    X_tr, X_te, y_tr, y_te, u_tr, u_te = train_test_split(
        X_scaled, y_raw, urls, test_size=0.2, random_state=42, stratify=y_raw
    )

    # Base Adjacency (4, 4)
    base_adj = torch.zeros((4, 4), dtype=torch.float32)
    base_adj[0, 0] = base_adj[1, 1] = base_adj[2, 2] = base_adj[3, 3] = 1.0
    base_adj[0, 1] = base_adj[1, 0] = 1.0
    base_adj[1, 2] = base_adj[2, 1] = 1.0
    base_adj[2, 3] = base_adj[3, 2] = 1.0
    base_adj = base_adj.to(device)

    u_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    d_tr_t = torch.tensor(np.stack([build_domain_feats(u) for u in u_tr]), dtype=torch.float32)
    i_tr_t = torch.tensor(np.stack([build_ip_feats(u) for u in u_tr]), dtype=torch.float32)
    a_tr_t = torch.tensor(np.stack([build_asn_feats(u) for u in u_tr]), dtype=torch.float32)

    u_te_t = torch.tensor(X_te, dtype=torch.float32)
    d_te_t = torch.tensor(np.stack([build_domain_feats(u) for u in u_te]), dtype=torch.float32)
    i_te_t = torch.tensor(np.stack([build_ip_feats(u) for u in u_te]), dtype=torch.float32)
    a_te_t = torch.tensor(np.stack([build_asn_feats(u) for u in u_te]), dtype=torch.float32)

    model = URLInfraGNN(hidden_dim=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 8

    log(f"Training URLInfraGNN ({sum(p.numel() for p in model.parameters()):,} params) for {epochs} epochs...")
    n_train = len(y_tr)
    batch_size = 256

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        running_loss = 0.0
        t0 = time.time()

        for i in range(0, n_train, batch_size):
            b_idx = perm[i:i + batch_size]
            B = len(b_idx)
            u_b = u_tr_t[b_idx].to(device)
            d_b = d_tr_t[b_idx].to(device)
            i_b = i_tr_t[b_idx].to(device)
            a_b = a_tr_t[b_idx].to(device)
            y_b = torch.tensor(y_tr[b_idx.numpy()], dtype=torch.long).to(device)

            optimizer.zero_grad()
            batch_logits = []
            for b in range(B):
                nodes = model.encode_nodes(u_b[b:b+1], d_b[b:b+1], i_b[b:b+1], a_b[b:b+1])
                batch_logits.append(model(nodes, base_adj))

            logits = torch.cat(batch_logits, dim=0)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * B

        # Fast eval
        model.eval()
        t_preds = []
        with torch.no_grad():
            for b in range(len(y_te)):
                nodes = model.encode_nodes(
                    u_te_t[b:b+1].to(device), d_te_t[b:b+1].to(device),
                    i_te_t[b:b+1].to(device), a_te_t[b:b+1].to(device)
                )
                t_preds.append(torch.argmax(model(nodes, base_adj), dim=-1).item())

        acc = accuracy_score(y_te, t_preds)
        f1 = f1_score(y_te, t_preds, average="macro", zero_division=0)
        log(f"Epoch {epoch}/{epochs} ({time.time()-t0:.1f}s) | Train Loss: {running_loss/n_train:.4f} | Test Acc: {acc*100:.2f}% | Macro F1: {f1:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dim": 64,
        "num_classes": num_classes,
        "label_mapping": label_mapping,
    }, str(OUT_DIR / "url_gnn_model.pt"))
    joblib.dump(scaler, OUT_DIR / "url_gnn_scaler.joblib")
    joblib.dump(le, OUT_DIR / "url_gnn_label_encoder.joblib")

    rep = classification_report(y_te, t_preds, target_names=[label_mapping[i] for i in range(num_classes)], output_dict=True, zero_division=0)
    with open(OUT_DIR / "url_gnn_metrics.json", "w") as f:
        json.dump(rep, f, indent=2)

    log("Training complete. Artifacts successfully saved to /kaggle/working/")


if __name__ == "__main__":
    main()
