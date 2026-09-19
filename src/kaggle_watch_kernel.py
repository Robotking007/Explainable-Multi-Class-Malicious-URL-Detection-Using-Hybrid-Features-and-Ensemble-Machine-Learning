"""
Poll the Kaggle kernel's status until it finishes (COMPLETE/ERROR/CANCELLED),
then download its output files (trained models) to models/from_kaggle/.

Usage:
    python src/kaggle_watch_kernel.py [kernel-slug] [output-subdir]
    (defaults: cyberxai-url-module-b-training, from_kaggle)
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
USER_NAME = "siddudavant"
KERNEL_SLUG = sys.argv[1] if len(sys.argv) > 1 else "cyberxai-url-module-b-training"
OUT_SUBDIR = sys.argv[2] if len(sys.argv) > 2 else "from_kaggle"
OUT_DIR = ROOT / "models" / OUT_SUBDIR
POLL_SECONDS = 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    token = TOKEN_PATH.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(token)
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelSessionStatusRequest,
        ApiListKernelSessionOutputRequest,
        ApiDownloadKernelOutputRequest,
    )
    from kagglesdk.kernels.types.kernels_enums import KernelWorkerStatus

    client = build_kaggle_client()
    status_req = ApiGetKernelSessionStatusRequest()
    status_req.user_name = USER_NAME
    status_req.kernel_slug = KERNEL_SLUG

    # status.status is a KernelWorkerStatus ENUM, not a string -- comparing
    # it against string literals (an earlier bug in this script) silently
    # never matches, so the loop polled forever without ever detecting
    # ERROR/COMPLETE and downloading/reporting anything.
    terminal_states = {KernelWorkerStatus.COMPLETE, KernelWorkerStatus.ERROR, KernelWorkerStatus.CANCEL_ACKNOWLEDGED}
    state = None
    while True:
        with client:
            status = client.kernels.kernels_api_client.get_kernel_session_status(status_req)
        state = status.status
        log(f"Kernel status: {state}")
        if state in terminal_states:
            break
        time.sleep(POLL_SECONDS)

    log(f"Kernel finished with status: {state}")
    if state != KernelWorkerStatus.COMPLETE:
        log("Kernel did not complete successfully -- attempting to download "
            "whatever output exists anyway (e.g. CRASH_TRACEBACK.txt) before giving up.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    list_req = ApiListKernelSessionOutputRequest()
    list_req.user_name = USER_NAME
    list_req.kernel_slug = KERNEL_SLUG
    with client:
        listing = client.kernels.kernels_api_client.list_kernel_session_output(list_req)

    files = listing.files or []
    log(f"{len(files)} output file(s) found.")
    for f in files:
        file_name = getattr(f, "file_name", None) or getattr(f, "name", None) or str(f)
        log(f"Downloading {file_name} ...")
        dl_req = ApiDownloadKernelOutputRequest()
        dl_req.owner_slug = USER_NAME
        dl_req.kernel_slug = KERNEL_SLUG
        dl_req.file_path = file_name
        with client:
            redirect = client.kernels.kernels_api_client.download_kernel_output(dl_req)
        urllib.request.urlretrieve(redirect.url, OUT_DIR / file_name)
        log(f"  saved to {OUT_DIR / file_name}")

    log(f"All output downloaded to {OUT_DIR}")


if __name__ == "__main__":
    main()
