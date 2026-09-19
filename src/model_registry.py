"""
Ensures models/*.joblib exist locally before the API tries to load them,
downloading anything missing from the private Hugging Face Hub repo they
were uploaded to (src/upload_models.py). models/ is gitignored (multi-GB
files don't belong in git), so a fresh `git clone` of this repo has an
empty models/ directory -- without this, the API would just fail to start
on a machine that never ran training locally.

Not needed at all when models/ is already fully populated (the common
case on the machine that actually trained them): every ensure_model()
call is a no-op local-file-exists check first, so this adds effectively
zero cost when nothing's missing.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
HF_REPO_SUFFIX = "cyberxai-models"


def _load_hf_token() -> str | None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def ensure_model(filename: str) -> Path:
    """Returns the local path to `filename` under models/, downloading it
    from Hugging Face Hub first if it isn't already there. Raises the same
    kind of error callers already handle (FileNotFoundError) if it can't
    be found either locally or remotely, rather than silently returning a
    bad path."""
    local_path = MODEL_DIR / filename
    if local_path.exists():
        return local_path

    token = _load_hf_token()
    if not token:
        raise FileNotFoundError(
            f"{filename} not found in {MODEL_DIR} and no HF_TOKEN in .env to fetch it remotely"
        )

    from huggingface_hub import HfApi, hf_hub_download
    who = HfApi(token=token).whoami()
    repo_id = f"{who['name']}/{HF_REPO_SUFFIX}"

    print(f"[model_registry] {filename} missing locally -- downloading from {repo_id} ...", flush=True)
    MODEL_DIR.mkdir(exist_ok=True)
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename, token=token)

    import shutil
    shutil.copy2(downloaded, local_path)
    print(f"[model_registry] {filename} ready at {local_path}", flush=True)
    return local_path


def ensure_models(filenames: list[str]) -> None:
    for fn in filenames:
        ensure_model(fn)
