"""
Push src/kaggle_kernel_train_url_dl.py to Kaggle as a private GPU-enabled
script kernel to run Deep Learning URL Module B training on cloud compute.

Supports multi-token rotation (access_token, access_token_2, access_token_3).

Usage:
    python src/kaggle_push_kernel_dl.py [token_file] [username]
    (defaults: access_token_3, siddudavant)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = sys.argv[1] if len(sys.argv) > 1 else "access_token_2"
USER_NAME = sys.argv[2] if len(sys.argv) > 2 else "siddhantudavant"
TOKEN_PATH = Path.home() / ".kaggle" / TOKEN_FILE

KERNEL_SLUG = f"{USER_NAME}/cyberxai-url-module-b-deep-learning-training"
DATASET_SOURCE = f"{USER_NAME}/cyberxai-combined-urls"
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_train_url_dl.py"


def main():
    if not TOKEN_PATH.exists():
        # Fallback to access_token
        active_token_path = Path.home() / ".kaggle" / "access_token"
    else:
        active_token_path = TOKEN_PATH

    print(f"Reading Kaggle token from {active_token_path}...")
    token = active_token_path.read_text().strip()

    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)

    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    code = SCRIPT_PATH.read_text(encoding="utf-8")

    req = ApiSaveKernelRequest()
    req.slug = KERNEL_SLUG
    req.new_title = "CyberXAI URL Module B Deep Learning Training"
    req.text = code
    req.language = "python"
    req.kernel_type = "script"
    req.is_private = True
    req.enable_gpu = True
    req.enable_internet = True
    req.dataset_data_sources = [DATASET_SOURCE]
    req.competition_data_sources = []
    req.kernel_data_sources = []
    req.model_data_sources = []
    req.category_ids = []

    client = build_kaggle_client()
    with client:
        result = client.kernels.kernels_api_client.save_kernel(req)

    print("Push result:", result.to_dict() if hasattr(result, "to_dict") else result)
    print(f"Kernel URL: https://www.kaggle.com/code/{KERNEL_SLUG.split('/', 1)[1]}"
          f" (or https://www.kaggle.com/{KERNEL_SLUG})")


if __name__ == "__main__":
    main()
