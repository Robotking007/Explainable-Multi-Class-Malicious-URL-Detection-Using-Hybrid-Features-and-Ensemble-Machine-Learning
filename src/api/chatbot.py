"""
Optional chatbot panel (currently hidden in the frontend behind a flag --
see frontend/code.html's CHATBOT_ENABLED constant). Explains a real scan
result in plain English rather than acting as a general-purpose assistant:
every call is grounded in the actual predicted_class/confidence/SHAP
top_features from a genuine /api/url/predict (or malware sample) response,
passed in by the frontend as `context`. This keeps it consistent with the
project's "explainable" premise -- the chatbot narrates real model output,
it doesn't invent its own verdicts.

Uses Gemini (gemini-3.6-flash) via a plain HTTPS call -- no SDK dependency,
just the REST API, since this is the only call site. The API key is read
from .env's GEMINI_API_KEY at call time (not cached at import time), same
handling as every other secret in this project: never logged, never sent
anywhere except Google's endpoint, never exposed to the browser (the
frontend only ever talks to this backend, never to Gemini directly).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

SYSTEM_PREAMBLE = (
    "You are the explanation assistant embedded in CyberXAI, a college project "
    "dashboard for explainable malicious URL / malware detection. You explain "
    "ONE real prediction the project's own trained ensemble (RandomForest + "
    "XGBoost + LogisticRegression + NaiveBayes, or the malware baselines) just "
    "made, using the real SHAP feature attributions provided below. Stay "
    "grounded in that data -- do not invent a different verdict, confidence, "
    "or features than what's given. If no scan context is provided, say so "
    "and suggest the user run a scan first. Keep answers concise (2-5 "
    "sentences unless asked for more detail)."
)


def _load_api_key() -> str | None:
    # Real env var first (how HF Spaces injects secrets at runtime -- no
    # .env file exists inside the container); falls back to .env locally.
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def _format_context(context: dict | None) -> str:
    if not context:
        return "No scan has been run yet in this session."
    lines = [
        f"Predicted class: {context.get('predicted_class')}",
        f"Confidence: {context.get('confidence')}",
    ]
    if context.get("url"):
        lines.append(f"URL scanned: {context['url']}")
    if context.get("probabilities"):
        lines.append(f"Class probabilities: {context['probabilities']}")
    top_features = context.get("top_features") or []
    if top_features:
        lines.append("Top SHAP-attributed features:")
        for f in top_features[:8]:
            lines.append(f"  - {f.get('feature')}: shap={f.get('shap_value')} "
                          f"(raw value={f.get('raw_value')}, pushes {f.get('direction')} the prediction)")
    return "\n".join(lines)


def ask_chatbot(message: str, context: dict | None, history: list[dict] | None = None) -> str:
    """`history` is a list of {"role": "user"|"model", "text": str} from
    earlier turns in this chat session, so follow-up questions have
    continuity without the frontend needing its own session store."""
    api_key = _load_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not configured in .env")

    contents = [
        {"role": "user", "parts": [{"text": SYSTEM_PREAMBLE + "\n\nScan context:\n" + _format_context(context)}]},
        {"role": "model", "parts": [{"text": "Understood, I'll explain this result grounded in that data."}]},
    ]
    for turn in (history or [])[-10:]:
        role = "model" if turn.get("role") == "model" else "user"
        contents.append({"role": role, "parts": [{"text": turn.get("text", "")}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = json.dumps({"contents": contents}).encode()
    req = urllib.request.Request(
        GEMINI_URL, data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        raise RuntimeError(f"Gemini API error {e.code}: {detail}") from e

    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {body}")
