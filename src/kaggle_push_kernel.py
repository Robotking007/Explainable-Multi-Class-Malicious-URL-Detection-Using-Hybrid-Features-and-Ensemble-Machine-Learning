"""
One-off helper: push src/kaggle_kernel_train_url.py to Kaggle as a private
script kernel with GPU enabled, attached to the cyberxai-combined-urls
dataset, then start running it -- so Module B training runs on Kaggle's
compute (survives this laptop shutting down) instead of locally.

Uses kagglesdk directly (the internal library both kagglehub and the
classic `kaggle` CLI build on) since the installable `kaggle` PyPI package
(1.7.4.5) predates its own documented token-auth support (>=1.8.0, not yet
on PyPI) and kagglehub itself doesn't expose a kernel-push helper. This
replicates the exact field mapping the classic client's kernels_push()
uses internally.

Reads the token from ~/.kaggle/access_token (never inlined in a command).

Usage:
    python src/kaggle_push_kernel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_FILE = sys.argv[1] if len(sys.argv) > 1 else "access_token"
USER_NAME = sys.argv[2] if len(sys.argv) > 2 else "siddudavant"
TOKEN_PATH = Path.home() / ".kaggle" / TOKEN_FILE

KERNEL_SLUG = f"{USER_NAME}/cyberxai-url-module-b-training"
KERNEL_ID = 135009902 if USER_NAME == "siddudavant" else None
DATASET_SOURCE = "siddudavant/cyberxai-combined-urls"
SCRIPT_PATH = ROOT / "src" / "kaggle_kernel_train_url.py"


def main():
    if not TOKEN_PATH.exists():
        active_token_path = Path.home() / ".kaggle" / "access_token"
    else:
        active_token_path = TOKEN_PATH

    print(f"Reading Kaggle token from {active_token_path}...")
    token = active_token_path.read_text().strip()

    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)

    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    # Load HF_TOKEN from .env to inject into the kernel
    hf_token = ""
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1].strip()

    code = SCRIPT_PATH.read_text(encoding="utf-8")
    if hf_token:
        code = code.replace('HF_TOKEN = ""', f'HF_TOKEN = "{hf_token}"')
        print(f"Injected HF_TOKEN (length: {len(hf_token)}) into kernel code for direct Hugging Face upload.")

    req = ApiSaveKernelRequest()
    if KERNEL_ID:
        req.id = KERNEL_ID
    req.slug = KERNEL_SLUG
    req.new_title = "CyberXAI URL Module B Training"
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
