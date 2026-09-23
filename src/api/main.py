"""
Backend API for the CyberXAI frontend (frontend/code.html).

This is the "framework" layer the frontend calls into -- it loads the
already-trained models from models/ and exposes them over HTTP so the
static dashboard can show real predictions and real SHAP explanations
instead of the hardcoded demo data it originally shipped with. No web
app/chatbot product is being built here beyond what's needed to serve this
one page's real data; per project scope, the actual frontend design/UX was
supplied separately (Google AI Studio / "Stitch") and is not touched here
beyond wiring its existing DOM hooks up to real fetch() calls.

Endpoints:
  GET  /                      serves frontend/code.html (the dashboard)
  GET  /api/health            liveness + which models loaded
  POST /api/url/predict       {url} -> Module B prediction + SHAP, plus supplementary live
                               WHOIS + ASN/IP-prefix host-intelligence lookups (src/url_pipeline/explain.py)
  POST /api/url/escalate      {url} -> Module C dynamic/sandbox analysis (src/url_pipeline/dynamic_escalation.py)
  GET  /api/malware/sample    ?task=binary|family&row_index=<int> -> Module A prediction + SHAP on one
                               real CIC-MalMem-2022 row (src/malware_pipeline/explain.py)
  GET  /api/metrics           real, current model-evaluation numbers read live from reports/*.json,
                               so the dashboard's KPI strip is never stale hardcoded HTML

Run:
    uvicorn src.api.main:app --reload --port 8000
Then open http://127.0.0.1:8000/ in a browser -- same-origin, so no CORS
setup is required for local use. CORS is still enabled (permissive) in case
the frontend is opened from a different origin during development.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from url_pipeline.explain import predict_url, load_artifacts as load_url_artifacts  # noqa: E402
from malware_pipeline.explain import predict_sample, load_artifacts as load_malware_artifacts  # noqa: E402
from malware_pipeline.explain_static import predict_sample as predict_static_sample  # noqa: E402
from api.chatbot import ask_chatbot  # noqa: E402

app = FastAPI(title="CyberXAI Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_model_cache():
    print("[startup] Pre-loading Deep Learning model (HybridURLNet) and explainers...")
    try:
        from url_pipeline.explain_dl import load_dl_artifacts
        load_dl_artifacts()
        print("[startup] HybridURLNet DL model warm in <1s.")
    except Exception as e:
        print(f"[startup] Warning: DL model warm deferred: {e}")
    try:
        load_malware_artifacts()
    except Exception as e:
        print(f"[startup] Malware model warm deferred: {e}")
    print("[startup] Model cache ready to serve.")


class URLRequest(BaseModel):
    url: str
    model_type: Optional[str] = "auto"  # "auto", "dl", or "ensemble"


class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None
    history: Optional[list] = None


@app.get("/")
def index():
    page = ROOT / "frontend" / "code.html"
    if not page.exists():
        raise HTTPException(404, "frontend/code.html not found")
    return FileResponse(page)


@app.get("/api/health")
def health():
    models_present = {
        "url_voting_classifier": (ROOT / "models" / "url_voting_classifier.joblib").exists(),
        "url_dl_model": (ROOT / "models" / "url_dl_model.pt").exists(),
        "url_gnn_model": (ROOT / "models" / "url_gnn_model.pt").exists(),
        "malware_binary_rf": (ROOT / "models" / "malware_dynamic_rf_binary_benign_vs_malware.joblib").exists(),
        "malware_family_rf": (ROOT / "models" / "malware_dynamic_rf_multiclass_family.joblib").exists(),
    }
    return {"status": "ok", "models_present": models_present}


@app.post("/api/url/predict")
def url_predict(req: URLRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(400, "url must not be empty")
    cleaned_url = req.url.strip()
    mtype = (req.model_type or "auto").lower()

    if mtype in ("gnn", "graph"):
        try:
            from url_pipeline.explain_gnn import predict_url_gnn
            return predict_url_gnn(cleaned_url)
        except Exception as e:
            raise HTTPException(500, f"GNN prediction failed: {e}")
    elif mtype in ("dl", "deep_learning"):
        try:
            from url_pipeline.explain_dl import predict_url_dl
            return predict_url_dl(cleaned_url)
        except Exception as e:
            raise HTTPException(500, f"DL prediction failed: {e}")
    elif mtype in ("ensemble", "classical", "rf"):
        try:
            return predict_url(cleaned_url)
        except Exception as e:
            raise HTTPException(500, f"Ensemble prediction failed: {e}")
    else:  # auto: prefer DL if present, else fallback
        dl_exists = (ROOT / "models" / "url_dl_model.pt").exists()
        if dl_exists:
            try:
                from url_pipeline.explain_dl import predict_url_dl
                return predict_url_dl(cleaned_url)
            except Exception:
                return predict_url(cleaned_url)
        return predict_url(cleaned_url)


@app.post("/api/url/predict-dl")
def url_predict_dl_endpoint(req: URLRequest):
    """Direct endpoint strictly invoking the Deep Learning (HybridURLNet) pipeline."""
    if not req.url or not req.url.strip():
        raise HTTPException(400, "url must not be empty")
    try:
        from url_pipeline.explain_dl import predict_url_dl
        return predict_url_dl(req.url.strip())
    except Exception as e:
        raise HTTPException(500, f"Deep Learning prediction failed: {e}")


@app.post("/api/url/predict-gnn")
def url_predict_gnn_endpoint(req: URLRequest):
    """Direct endpoint strictly invoking the Graph Neural Network (URLInfraGNN) pipeline."""
    if not req.url or not req.url.strip():
        raise HTTPException(400, "url must not be empty")
    try:
        from url_pipeline.explain_gnn import predict_url_gnn
        return predict_url_gnn(req.url.strip())
    except Exception as e:
        raise HTTPException(500, f"Graph Neural Network prediction failed: {e}")


@app.post("/api/url/escalate")
def url_escalate(req: URLRequest):
    # A plain `def` endpoint: Starlette already runs these in its own worker
    # thread pool (anyio.to_thread), which is exactly the isolation
    # Playwright's sync API needs from any asyncio event loop -- no extra
    # nested ThreadPoolExecutor here. (An earlier version added one anyway;
    # it didn't provide real cancellation on timeout, since a genuinely
    # stuck call just keeps that inner thread -- and its Chromium process --
    # alive in the background regardless, so it added risk without benefit.)
    if not req.url or not req.url.strip():
        raise HTTPException(400, "url must not be empty")
    try:
        from url_pipeline.dynamic_escalation import analyze_url_dynamic
        return analyze_url_dynamic(req.url.strip())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"escalation failed: {e}")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/api/metrics")
def metrics():
    """Real, current model-evaluation numbers for the dashboard's KPI strip
    -- read live from reports/*.json on every call, rather than baked into
    the HTML, specifically so this can never go stale the way the original
    frontend mockup's hardcoded '89.8% / 100%' figures did (measured against
    an earlier, smaller training run, then never updated as the models were
    retrained)."""
    report_dir = ROOT / "reports"
    url_metrics = _read_json(report_dir / "url_module_b_metrics.json")
    url_dl_metrics = _read_json(report_dir / "url_dl_metrics.json")
    malware_dynamic = _read_json(report_dir / "malware_module_a_dynamic_baseline_metrics.json")
    malware_static = _read_json(report_dir / "malware_module_a_static_baseline_metrics.json")
    whois_ablation = _read_json(report_dir / "url_whois_ablation_metrics.json")

    result = {
        "url_module_b": None, "url_dl": None, "malware_module_a_dynamic": None,
        "malware_module_a_static": None, "whois_ablation": None,
    }

    if url_dl_metrics:
        result["url_dl"] = url_dl_metrics

    if url_metrics:
        result["url_module_b"] = {
            "accuracy": url_metrics.get("accuracy"),
            "macro_precision": url_metrics.get("macro_precision"),
            "macro_recall": url_metrics.get("macro_recall"),
            "macro_f1": url_metrics.get("macro_f1"),
            "macro_roc_auc": url_metrics.get("macro_roc_auc"),
            "n_train": url_metrics.get("n_train"),
            "n_test": url_metrics.get("n_test"),
        }
    if malware_dynamic:
        binary = malware_dynamic.get("binary_benign_vs_malware", {})
        family = malware_dynamic.get("multiclass_family", {})
        result["malware_module_a_dynamic"] = {
            "binary_accuracy": binary.get("accuracy"),
            "binary_roc_auc": binary.get("macro_roc_auc"),
            "family_accuracy": family.get("accuracy"),
            "family_macro_f1": family.get("macro_f1"),
        }
    if malware_static:
        internal = malware_static.get("internal_holdout", {})
        external = malware_static.get("external_ember_test", {})
        result["malware_module_a_static"] = {
            "internal_accuracy": internal.get("accuracy"),
            "external_accuracy": external.get("accuracy"),
            "external_roc_auc": external.get("roc_auc"),
            "n_train_pool": malware_static.get("n_train_pool"),
        }
    if whois_ablation:
        result["whois_ablation"] = {
            "sample_size": whois_ablation.get("sample_size"),
            "without_whois_f1": whois_ablation.get("without_whois", {}).get("macro_f1"),
            "with_whois_f1": whois_ablation.get("with_whois", {}).get("macro_f1"),
            "delta_macro_f1": whois_ablation.get("delta_macro_f1"),
            "whois_availability_rate": whois_ablation.get("whois_availability_rate"),
        }
    return result


@app.get("/api/malware/sample")
def malware_sample(
    task: str = Query("family", pattern="^(binary|family)$"),
    row_index: Optional[int] = Query(None, ge=0),
):
    try:
        return predict_sample(task=task, row_index=row_index)
    except Exception as e:
        raise HTTPException(500, f"malware sample failed: {e}")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Explains the current/last real scan result in plain English, grounded
    in that scan's actual predicted_class/confidence/SHAP data (see
    api/chatbot.py's docstring). Hidden in the frontend by default -- see
    frontend/code.html's CHATBOT_ENABLED flag -- until the project is fully
    tested, per explicit project decision."""
    if not req.message or not req.message.strip():
        raise HTTPException(400, "message must not be empty")
    try:
        reply = ask_chatbot(req.message.strip(), req.context, req.history)
        return {"reply": reply}
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"chat failed: {e}")


@app.get("/api/malware/static_sample")
def malware_static_sample(row_index: Optional[int] = Query(None, ge=0)):
    """Module A' (static PE features, EMBER/BODMAS) -- not eagerly warmed at
    startup like the other two branches: its artifacts (an unconstrained
    200-tree RF plus a ~1.9GB EMBER-test feature matrix) are large enough
    that loading them alongside the already-warmed URL and dynamic-malware
    models risked the same memory pressure that repeatedly crashed training
    itself. First request to this endpoint pays that load cost once
    (cached after); every request after is fast."""
    try:
        return predict_static_sample(row_index=row_index)
    except FileNotFoundError as e:
        raise HTTPException(503, f"static malware model not trained yet: {e}")
    except Exception as e:
        raise HTTPException(500, f"static malware sample failed: {e}")
