"""
Same as kaggle_push_kernel.py, but pushes under the SECOND Kaggle
account (siddhantudavant) -- see kaggle_upload_dataset_acct2.py's
docstring for why. Same script content (kaggle_kernel_train_url.py),
attached to this account's own copy of the dataset.

Reads the token from ~/.kaggle/access_token_2 (never inlined in a
command line).

Usage:
    python src/kaggle_push_kernel_acct2.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token_2"
KERNEL_SLUG = "siddhantudavant/cyberxai-url-training"
KERNEL_ID = 135204067
DATASET_SOURCE = "siddhantudavant/cyberxai-combined-urls"
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_train_url_dl.py"


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
    req.new_title = "CyberXAI URL Module B Deep Learning Training (acct2)"
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
