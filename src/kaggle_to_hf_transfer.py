"""
One-off: move url_voting_classifier.joblib from the completed Kaggle
kernel straight to HF Hub -- download once to a temp file (retried,
chunked), upload, then delete the temp file. No permanent local copy.

Usage:
    python src/kaggle_to_hf_transfer.py
"""
from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
USER_NAME = "siddudavant"
KERNEL_SLUG = "cyberxai-url-module-b-training"
FILE_NAME = "url_voting_classifier.joblib"
TMP_PATH = Path("D:/tmp_scratch") / FILE_NAME


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_download_url() -> str:
    kaggle_token = KAGGLE_TOKEN_PATH.read_text().strip()
    from kagglehub.config import set_kaggle_api_token
    set_kaggle_api_token(kaggle_token)
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

    req = ApiDownloadKernelOutputRequest()
    req.owner_slug = USER_NAME
    req.kernel_slug = KERNEL_SLUG
    req.file_path = FILE_NAME
    client = build_kaggle_client()
    with client:
        redirect = client.kernels.kernels_api_client.download_kernel_output(req)
    return redirect.url


def download_with_retries(url: str, dest: Path, max_retries: int = 5) -> None:
    import requests

    for attempt in range(1, max_retries + 1):
        try:
            resume_from = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            mode = "ab" if resume_from else "wb"
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if resume_from and r.status_code == 200:
                    # Server ignored the Range request -- start clean.
                    resume_from = 0
                    mode = "wb"
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) + resume_from
                downloaded = resume_from
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (500 * 1024 * 1024) < len(chunk):
                            log(f"  {downloaded / 1e9:.2f} GB downloaded"
                                + (f" / {total / 1e9:.2f} GB" if total else ""))
            log(f"Download complete: {dest.stat().st_size / 1e9:.2f} GB")
            return
        except Exception as e:
            log(f"Attempt {attempt}/{max_retries} failed: {e}. Retrying...")
            time.sleep(5)
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts")


def upload_to_hf(local_path: Path) -> None:
    hf_token = None
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            hf_token = line.split("=", 1)[1].strip()
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    who = api.whoami()
    repo_id = f"{who['name']}/cyberxai-models"
    log(f"Uploading {local_path.name} to {repo_id} ...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=FILE_NAME,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )
    log("Upload complete.")


def main():
    log("Getting fresh download URL from Kaggle...")
    url = get_download_url()
    log("Downloading to temp file...")
    download_with_retries(url, TMP_PATH)
    upload_to_hf(TMP_PATH)
    TMP_PATH.unlink()
    log(f"Deleted temp file {TMP_PATH}. Done -- no local copy kept.")


if __name__ == "__main__":
    main()
