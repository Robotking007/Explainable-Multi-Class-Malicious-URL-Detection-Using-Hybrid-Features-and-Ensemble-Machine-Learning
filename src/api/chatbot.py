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
DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_PREAMBLE = (
    "You are the CyberXAI Security Analyst Assistant powered by Google Gemini 3.5 Flash, "
    "embedded in an advanced Explainable AI (XAI) security dashboard for multi-class malicious URL "
    "and malware detection. You assist security operations center (SOC) analysts and researchers.\n\n"
    "Key capabilities:\n"
    "1. Explain real predictions from CyberXAI models (URLInfraGNN, HybridURLNet, classical Voting Ensemble, "
    "and CIC-MalMem-2022 malware classifiers) using quantitative SHAP and gradient attributions.\n"
    "2. If an active scan context is provided below, ground your response directly in its predicted class, "
    "confidence, feature values, and SHAP directions. Do not contradict the model's actual verdict.\n"
    "3. If no active scan context is provided, answer the user's cybersecurity, threat intelligence, "
    "machine learning, or SOC analysis query accurately, concisely, and authoritatively.\n"
    "4. Keep explanations clear, practical, and forensic (typically 2-5 concise sentences unless deeper analysis is requested)."
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


def _load_model_name() -> str:
    env_model = os.environ.get("GEMINI_MODEL")
    if env_model:
        return env_model
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GEMINI_MODEL="):
                return line.split("=", 1)[1].strip()
    return DEFAULT_MODEL


def _format_context(context: dict | None) -> str:
    if not context:
        return "No specific scan context provided for this prompt."
    lines = [
        f"Predicted class: {context.get('predicted_class') or context.get('classification') or 'Unknown'}",
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

    model_name = _load_model_name()
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    contents = [
        {"role": "user", "parts": [{"text": SYSTEM_PREAMBLE + "\n\nActive Scan Context:\n" + _format_context(context)}]},
        {"role": "model", "parts": [{"text": f"Understood. I am your CyberXAI Security Assistant powered by Gemini 3.5 Flash. I will evaluate queries grounded in this security telemetry."}]},
    ]
    for turn in (history or [])[-10:]:
        role = "model" if turn.get("role") in ("model", "assistant") else "user"
        turn_text = turn.get("text") or turn.get("content") or ""
        if turn_text:
            contents.append({"role": role, "parts": [{"text": turn_text}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = json.dumps({"contents": contents}).encode()
    req = urllib.request.Request(
        f"{gemini_url}?key={api_key}",
        data=payload,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        raise RuntimeError(f"Gemini API error {e.code}: {detail}") from e

    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Unexpected Gemini response shape: {body}")
