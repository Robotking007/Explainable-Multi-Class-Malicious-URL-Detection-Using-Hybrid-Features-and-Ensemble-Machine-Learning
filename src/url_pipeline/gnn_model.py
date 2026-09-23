"""
URLInfraGNN: Pure PyTorch Graph Attention Network (GAT) for Heterogeneous
URL Infrastructure Threat Classification.

Designed without heavy C++ / CUDA binary dependencies (such as torch-geometric
or DGL) so it runs natively and identically across any local CPU or cloud GPU.

Models infrastructure topology:
  Nodes:
    - URL Node: 45-dimensional hybrid static features
    - Domain Node: 8-dimensional WHOIS / domain structural features
    - IP Node: 4-dimensional IP octet & routing features
    - ASN Node: 4-dimensional BGP prefix & autonomous system features

  Edges:
    - URL <-> Domain (hosted on)
    - Domain <-> IP (resolves to)
    - IP <-> ASN (announced by)
    - Cross-URL edges if co-hosted on identical infrastructure

Provides native graph explainability through extracted GAT attention weights (alpha_ij).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Self-Attention Graph Layer (Veličković et al., GAT).
    Computes message passing with learnable attention coefficients:
      alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )
      h_i' = sigma( sum_j alpha_ij W h_j )
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h: Node feature matrix of shape (N, in_features)
            adj: Adjacency matrix of shape (N, N) where adj[i, j] > 0 indicates an edge.
            return_attention: If True, returns normalized attention weights matrix.

        Returns:
            h_prime: Updated node representations of shape (N, out_features)
            attention: (N, N) attention weights if return_attention=True, else None
        """
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)

        # Self-attention mechanism: compute a^T [Wh_i || Wh_j] for all pairs
        # Broadcast Wh to (N, N, out_features)
        Wh_i = Wh.unsqueeze(1).repeat(1, N, 1)  # (N, N, out_features)
        Wh_j = Wh.unsqueeze(0).repeat(N, 1, 1)  # (N, N, out_features)
        pair_h = torch.cat([Wh_i, Wh_j], dim=-1)  # (N, N, 2 * out_features)

        e = self.leaky_relu(torch.matmul(pair_h, self.a).squeeze(-1))  # (N, N)

        # Mask non-connected edges with large negative value
        mask = (adj == 0)
        e = e.masked_fill(mask, -1e9)

        # Normalize attention coefficients over neighbor nodes
        attention = F.softmax(e, dim=-1)  # (N, N)
        attention_dropped = self.dropout(attention)

        # Message aggregation
        h_prime = torch.matmul(attention_dropped, Wh)  # (N, out_features)

        if return_attention:
            return h_prime, attention
        return h_prime, None


class URLInfraGNN(nn.Module):
    """
    Heterogeneous URL Infrastructure Graph Neural Network.
    Projects heterogeneous node types into a common latent dimension,
    performs multi-layer GAT message passing, and outputs threat class logits.
    """
    def __init__(
        self,
        url_dim: int = 45,
        domain_dim: int = 8,
        ip_dim: int = 4,
        asn_dim: int = 4,
        hidden_dim: int = 64,
        num_classes: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Heterogeneous node feature encoders
        self.url_proj = nn.Sequential(
            nn.Linear(url_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.domain_proj = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.ip_proj = nn.Sequential(
            nn.Linear(ip_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.asn_proj = nn.Sequential(
            nn.Linear(asn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 2-Layer Graph Attention Network
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Classifier head operating on the URL node's graph-enriched representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def encode_nodes(
        self,
        url_feats: torch.Tensor,
        domain_feats: torch.Tensor,
        ip_feats: torch.Tensor,
        asn_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Projects heterogeneous features to common latent space and stacks them."""
        u_emb = self.url_proj(url_feats)       # (N_url, hidden)
        d_emb = self.domain_proj(domain_feats) # (N_dom, hidden)
        i_emb = self.ip_proj(ip_feats)         # (N_ip, hidden)
        a_emb = self.asn_proj(asn_feats)       # (N_asn, hidden)
        return torch.cat([u_emb, d_emb, i_emb, a_emb], dim=0)

    def forward(
        self,
        node_feats: torch.Tensor,
        adj: torch.Tensor,
        url_node_idx: int = 0,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through GNN.
        Args:
            node_feats: Combined node feature matrix (N, hidden_dim)
            adj: Adjacency matrix (N, N)
            url_node_idx: Index of the primary URL node to classify (default: 0)
            return_attention: Whether to return attention maps for explainability
        """
        # Layer 1 + Residual + LayerNorm
        h1, attn1 = self.gat1(node_feats, adj, return_attention=return_attention)
        h1 = self.norm1(node_feats + F.elu(h1))

        # Layer 2 + Residual + LayerNorm
        h2, attn2 = self.gat2(h1, adj, return_attention=return_attention)
        h2 = self.norm2(h1 + F.elu(h2))

        # Classify primary URL node
        url_rep = h2[url_node_idx:url_node_idx + 1]
        logits = self.classifier(url_rep)

        attentions = [attn1, attn2] if return_attention else None
        return logits, attentions

    def quantize(self) -> nn.Module:
        """Applies PyTorch dynamic INT8 quantization for edge and low-latency deployments."""
        return torch.quantization.quantize_dynamic(
            self, {nn.Linear}, dtype=torch.qint8
        )
