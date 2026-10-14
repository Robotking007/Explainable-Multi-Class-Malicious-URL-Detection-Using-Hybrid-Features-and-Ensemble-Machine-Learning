"""
Poll the Kaggle kernel's status until it finishes (COMPLETE/ERROR/CANCELLED),
then download its output files (trained models) to models/from_kaggle/.

Usage:
    python src/kaggle_watch_kernel.py [kernel-slug] [output-subdir] [user-name] [token-file-name]
    (defaults: cyberxai-url-module-b-training, from_kaggle, siddudavant, access_token)

    The last two args let this watch a kernel pushed under the second
    Kaggle account (siddhantudavant, used to work around the primary
    account's weekly compute-quota cap) -- e.g.:
    python src/kaggle_watch_kernel.py cyberxai-url-module-b-training from_kaggle_acct2 siddhantudavant access_token_2
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = sys.argv[1] if len(sys.argv) > 1 else "cyberxai-url-module-b-training"
OUT_SUBDIR = sys.argv[2] if len(sys.argv) > 2 else "from_kaggle"
USER_NAME = sys.argv[3] if len(sys.argv) > 3 else "siddudavant"
TOKEN_FILE_NAME = sys.argv[4] if len(sys.argv) > 4 else "access_token"
TOKEN_PATH = Path.home() / ".kaggle" / TOKEN_FILE_NAME
OUT_DIR = ROOT / "models" / OUT_SUBDIR
POLL_SECONDS = 20


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
        # A transient local network/DNS blip (this laptop's own connection,
        # not Kaggle) previously killed this whole script -- the actual
        # Kaggle kernel kept running fine, but the local watcher died and
        # had to be noticed and restarted by hand. Retrying a few times
        # before giving up means a brief local outage no longer requires
        # manual intervention.
        for attempt in range(5):
            try:
                with client:
                    status = client.kernels.kernels_api_client.get_kernel_session_status(status_req)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                log(f"Status check failed (attempt {attempt + 1}/5), retrying in 30s: {e}")
                time.sleep(30)
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
        # _tmp_arrays/* are the multi-GB memmap scratch files the training
        # scripts write to /kaggle/working/ as a memory-crash workaround
        # (see kaggle_kernel_train_static_malware.py) -- not real outputs,
        # and downloading them would waste a lot of time/bandwidth for
        # nothing usable.
        if file_name.startswith("_tmp_arrays/") or file_name.startswith("_tmp_arrays\\"):
            log(f"Skipping scratch file {file_name}")
            continue
        log(f"Downloading {file_name} ...")
        dl_req = ApiDownloadKernelOutputRequest()
        dl_req.owner_slug = USER_NAME
        dl_req.kernel_slug = KERNEL_SLUG
        dl_req.file_path = file_name
        with client:
            redirect = client.kernels.kernels_api_client.download_kernel_output(dl_req)
        dest = OUT_DIR / file_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(redirect.url, dest)
        log(f"  saved to {dest}")

    log(f"All output downloaded to {OUT_DIR}")


if __name__ == "__main__":
    main()
