"""
Push src/kaggle_kernel_asn_enrich.py to Kaggle as a private, CPU-only
script kernel (no GPU needed for network I/O -- also sidesteps the
2-concurrent-GPU-session cap entirely). Attached to the URL dataset (for
the domain list); once a first run produces an ASN cache, re-run
kaggle_upload_asn_cache.py and add "siddudavant/cyberxai-asn-cache" to
DATASET_SOURCES below for resumable re-pushes (this enrichment is
expected to finish in a single session, unlike WHOIS, so a resume path is
a fallback, not the expected path).

Usage:
    python src/kaggle_push_asn_kernel.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
KERNEL_SLUG = "siddudavant/cyberxai-asn-enrichment"
KERNEL_ID = 135077441
DATASET_SOURCES = ["siddudavant/cyberxai-combined-urls"]
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_asn_enrich.py"


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
    req.new_title = "CyberXAI ASN Enrichment"
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
