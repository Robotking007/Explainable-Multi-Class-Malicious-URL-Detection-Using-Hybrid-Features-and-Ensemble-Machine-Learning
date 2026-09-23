"""
Upload the MalAPI2019 dataset and push the API-call-sequence training
kernel to Kaggle (CPU-only -- tiny dataset, no GPU benefit, and avoids
competing for the 2-GPU-session slot the URL kernel is using).

Usage:
    python src/kaggle_push_api_sequence_kernel.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
DATASET_HANDLE = "siddudavant/cyberxai-malapi2019"
KERNEL_SLUG = "siddudavant/cyberxai-api-sequence-training"
KERNEL_ID = 135071594  # set after first successful push
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_train_api_sequence.py"
DATA_STAGE_DIR = ROOT / "Dataset" / "malapi2019"


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
    req.new_title = "CyberXAI API Sequence Training"
    req.text = code
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = False
    req.enable_internet = True
    req.dataset_data_sources = [DATASET_HANDLE]
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
