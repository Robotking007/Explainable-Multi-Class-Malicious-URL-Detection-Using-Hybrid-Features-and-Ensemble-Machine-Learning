"""
Wait until a GPU session slot frees up (Kaggle caps batch GPU sessions at
2 per account), then push the dynamic malware training kernel. Polls both
currently-running kernels' status and retries the push on the
"Maximum batch GPU session count" error until it succeeds.

Usage:
    python src/kaggle_push_dynamic_when_free.py
"""
from __future__ import annotations

import time
from pathlib import Path

TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
WATCHED_KERNELS = ["cyberxai-url-module-b-training", "cyberxai-static-malware-training"]
POLL_SECONDS = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    token = TOKEN_PATH.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest

    ROOT = Path(__file__).resolve().parents[1]
    code = (ROOT / "src" / "kaggle_kernel_train_dynamic_malware.py").read_text(encoding="utf-8")

    def try_push():
        req = ApiSaveKernelRequest()
        req.slug = "siddudavant/cyberxai-dynamic-malware-training"
        req.new_title = "CyberXAI Dynamic Malware Training"
        req.text = code
        req.language = "python"
        req.kernel_type = "script"
        req.is_private = True
        req.enable_gpu = True
        req.enable_internet = True
        req.dataset_data_sources = ["siddudavant/cyberxai-dynamic-malware"]
        req.competition_data_sources = []
        req.kernel_data_sources = []
        req.model_data_sources = []
        req.category_ids = []
        client2 = build_kaggle_client()
        with client2:
            return client2.kernels.kernels_api_client.save_kernel(req)

    client = build_kaggle_client()
    while True:
        with client:
            statuses = {}
            for slug in WATCHED_KERNELS:
                req = ApiGetKernelSessionStatusRequest()
                req.user_name = "siddudavant"
                req.kernel_slug = slug
                statuses[slug] = client.kernels.kernels_api_client.get_kernel_session_status(req).status
        log(f"Statuses: {statuses}")

        running_count = sum(1 for s in statuses.values() if s == KernelWorkerStatus.RUNNING)
        if running_count < 2:
            log(f"Only {running_count} GPU session(s) running -- attempting dynamic kernel push...")
            result = try_push()
            result_dict = result.to_dict()
            log(f"Push result: {result_dict}")
            if "error" not in result_dict:
                log("Dynamic kernel pushed successfully.")
                return
            log("Push rejected (likely a race with another session grabbing the slot first) -- retrying after poll interval.")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
