"""
Push src/kaggle_kernel_train_static_malware.py to Kaggle as a private
script kernel with GPU enabled, attached to the cyberxai-static-malware
dataset.

Usage:
    python src/kaggle_push_static_kernel.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
KERNEL_SLUG = "siddudavant/cyberxai-static-malware-training"
KERNEL_ID = 135013090  # set after first push, to update rather than hit a title conflict
DATASET_SOURCE = "siddudavant/cyberxai-static-malware"
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_train_static_malware.py"


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
    req.new_title = "CyberXAI Static Malware Training"
    req.text = code
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = False
    req.enable_internet = True
    req.dataset_data_sources = [DATASET_SOURCE]
    req.competition_data_sources = []
    req.kernel_data_sources = []
    req.model_data_sources = []
    req.category_ids = []

    client = build_kaggle_client()
    with client:
        result = client.kernels.kernels_api_client.save_kernel(req)
    print("Push result:", result.to_dict())


if __name__ == "__main__":
    main()
