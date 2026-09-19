"""
Push src/kaggle_kernel_whois_enrich.py to Kaggle as a private, CPU-only
script kernel (no GPU needed for network I/O -- also sidesteps the
2-concurrent-GPU-session cap entirely). Attached to both the URL dataset
(for the domain list) and the WHOIS cache dataset (for resuming).

Usage:
    python src/kaggle_push_whois_kernel.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
KERNEL_SLUG = "siddudavant/cyberxai-whois-enrichment"
KERNEL_ID = 135020299  # set after first successful push
DATASET_SOURCES = ["siddudavant/cyberxai-combined-urls", "siddudavant/cyberxai-whois-cache"]
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_whois_enrich.py"


def main():
    token = TOKEN_PATH.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    code = SCRIPT_PATH.read_text(encoding="utf-8")

    req = ApiSaveKernelRequest()
    if KERNEL_ID:
        req.id = KERNEL_ID
    req.slug = KERNEL_SLUG
    req.new_title = "CyberXAI WHOIS Enrichment"
    req.text = code
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = False
    req.enable_internet = True
    req.dataset_data_sources = DATASET_SOURCES
    req.competition_data_sources = []
    req.kernel_data_sources = []
    req.model_data_sources = []
    req.category_ids = []

    client = build_kaggle_client()
    with client:
        result = client.kernels.kernels_api_client.save_kernel(req)
    return result.to_dict()


if __name__ == "__main__":
    print("Push result:", main())
