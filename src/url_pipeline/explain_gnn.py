"""
GNN Inference and Graph Explainability Engine for URLInfraGNN.

Performs sub-second inference over heterogeneous threat infrastructure graphs,
extracting exact Graph Attention Network (GAT) attention coefficients (alpha_ij)
to show which infrastructure elements (Domain, IP, ASN) drive the threat decision.
"""
from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from src.url_pipeline.gnn_model import URLInfraGNN
from src.url_pipeline.gnn_graph_builder import build_url_star_graph
from src.url_pipeline.whois_features import get_whois_features_for_url
from src.url_pipeline.asn_features import get_asn_features_for_url

MODELS_DIR = ROOT / "models"


@lru_cache(maxsize=1)
def load_gnn_artifacts():
    """Loads the trained GNN model weights, scaler, and label encoder."""
    model_path = MODELS_DIR / "url_gnn_model.pt"
    scaler_path = MODELS_DIR / "url_gnn_scaler.joblib"
    le_path = MODELS_DIR / "url_gnn_label_encoder.joblib"

    if not model_path.exists():
        return None, None, None

    checkpoint = torch.load(str(model_path), map_location="cpu")
    model = URLInfraGNN(
        url_dim=checkpoint.get("url_dim", 45),
        domain_dim=checkpoint.get("domain_dim", 8),
        ip_dim=checkpoint.get("ip_dim", 4),
        asn_dim=checkpoint.get("asn_dim", 4),
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_classes=checkpoint.get("num_classes", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    le = joblib.load(le_path) if le_path.exists() else None

    return model, scaler, le


def predict_url_gnn(
    url: str,
    include_network_intel: bool = True,
) -> Dict[str, Any]:
    """
    Classifies a URL using the Heterogeneous URL Infrastructure GNN.
    Returns:
      - Predicted threat class and multi-class probability distribution
      - Graph topology (nodes, edges)
      - Normalized Graph Attention Network (GAT) weights showing relative
        attribution between URL string, Domain, IP, and ASN nodes.
    """
    t0 = time.time()
    model, scaler, le = load_gnn_artifacts()

    if model is None:
        raise FileNotFoundError(
            f"Trained GNN model not found at {MODELS_DIR / 'url_gnn_model.pt'}. "
            "Please run 'python src/url_pipeline/train_gnn.py' first."
        )

    # 1. Fetch WHOIS & ASN host intelligence
    whois_info = None
    asn_info = None
    if include_network_intel:
        try:
            whois_info = get_whois_features_for_url(url)
        except Exception:
            whois_info = {"whois_available": False}
        try:
            asn_info = get_asn_features_for_url(url)
        except Exception:
            asn_info = {"asn_available": False}

    # 2. Build 4-node star-ego infrastructure graph
    feats_dict, adj, meta = build_url_star_graph(
        url=url,
        scaler=scaler,
        whois_info=whois_info,
        asn_info=asn_info,
    )

    # 3. Encode nodes and run GAT forward pass
    with torch.no_grad():
        nodes = model.encode_nodes(
            feats_dict["url"],
            feats_dict["domain"],
            feats_dict["ip"],
            feats_dict["asn"],
        )
        logits, attentions = model(nodes, adj, url_node_idx=0, return_attention=True)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    # 4. Map probabilities to class names
    classes = ["benign", "defacement", "malware", "phishing"]
    if le is not None and hasattr(le, "classes_"):
        classes = list(le.classes_)

    pred_idx = int(np.argmax(probs))
    predicted_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    prob_dict = {cls_name: float(round(float(p), 4)) for cls_name, p in zip(classes, probs)}

    # 5. Extract Graph Attention Edge Weights (Layer 2 Attention Map)
    # Layer 2 attention: (4, 4) tensor showing how much node i attends to node j
    l2_attn = attentions[1].squeeze(0).numpy() if attentions else np.eye(4)

    # Extract attention weights for the primary URL node (Node 0)
    url_to_self = float(l2_attn[0, 0])
    url_to_domain = float(l2_attn[0, 1])

    # Domain to IP (Node 1 -> Node 2)
    domain_to_ip = float(l2_attn[1, 2])
    # IP to ASN (Node 2 -> Node 3)
    ip_to_asn = float(l2_attn[2, 3])

    graph_attributions = [
        {
            "node": "URL Lexical & Structural Features",
            "node_type": "URL",
            "attention_weight": round(url_to_self, 4),
            "description": "Intrinsic URL length, digit ratio, phishing keywords, and character distributions",
        },
        {
            "node": "Domain Registration & WHOIS",
            "node_type": "Domain",
            "attention_weight": round(url_to_domain, 4),
            "description": f"Domain entropy, TLD abuse flags, creation age ({whois_info.get('domain_age_days', 'N/A') if whois_info else 'N/A'} days)",
        },
        {
            "node": "Host IP Resolution",
            "node_type": "IP",
            "attention_weight": round(domain_to_ip, 4),
            "description": f"Resolved host IPv4 address ({asn_info.get('resolved_ip', 'Unresolved') if asn_info else 'Unresolved'})",
        },
        {
            "node": "BGP Autonomous System (ASN)",
            "node_type": "ASN",
            "attention_weight": round(ip_to_asn, 4),
            "description": f"Network routing provider (AS{asn_info.get('asn', 'Unknown') if asn_info else 'Unknown'})",
        },
    ]

    total_time = round(time.time() - t0, 3)

    return {
        "url": url,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": prob_dict,
        "model_architecture": "URLInfraGNN (Graph Attention Network)",
        "graph_topology": meta,
        "graph_attributions": graph_attributions,
        "whois_info": whois_info,
        "asn_info": asn_info,
        "latency_seconds": total_time,
    }


if __name__ == "__main__":
    test_url = sys.argv[1] if len(sys.argv) > 1 else "http://paypal-security-check.cf/login.php"
    print(f"Analyzing URL with Infrastructure GNN: {test_url}")
    res = predict_url_gnn(test_url, include_network_intel=False)
    print(json.dumps(res, indent=2))
