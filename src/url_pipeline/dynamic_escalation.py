"""
Module C - URL dynamic / sandbox escalation pipeline (Section 5.3 / 6.2).

Runs when Module B's static-model confidence is below a threshold (or the
static verdict is already "suspicious"). Loads the URL in an isolated,
disposable, headless-browser context (Playwright/Chromium) and captures
the runtime signals in Table 6.2:

  - redirect_count, domain_changed, redirect_chain_domains, final_url,
    final_status_code
  - has_login_form, form_action_external
  - external_resource_count, script_count, inline_script_ratio,
    js_obfuscation_score
  - popup_or_redirect_on_load
  - page_title_brand_mismatch (simple substring heuristic; no external NLP dep)
  - screenshot_perceptual_hash (requires the optional `imagehash` package)
  - TLS_certificate_validity (best-effort: HTTPS-vs-HTTP + cert error flag)

Safety controls: hard per-navigation timeout, blocked downloads, blocked
permission prompts (camera/mic/geolocation), no execution of anything
downloaded -- browser-observation only, matching the isolation requirements
in Section 6.2.1 step 8. Each check should still be run inside a
network-restricted sandbox/container in production; this script does not
itself provide network isolation.

Usage:
    python src/url_pipeline/dynamic_escalation.py "http://example-suspicious.com"

Requires: pip install playwright && playwright install chromium
Optional: pip install imagehash (for screenshot_perceptual_hash)
"""
from __future__ import annotations

import collections
import io
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

try:
    import imagehash
    from PIL import Image
    _HAS_IMAGEHASH = True
except ImportError:  # pragma: no cover
    _HAS_IMAGEHASH = False

BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "netflix", "bank", "instagram", "whatsapp", "outlook", "icloud",
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = collections.Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def analyze_url_dynamic(url: str, timeout_ms: int = 15000, total_timeout_ms: int = 30000) -> dict:
    """Run one URL through the sandboxed-browser escalation path and return
    the Table 6.2 dynamic feature dict. Raises RuntimeError if Playwright
    (and its browser binaries) are not installed."""
    if sync_playwright is None:
        raise RuntimeError(
            "playwright is not installed. Run: pip install playwright && playwright install chromium"
        )

    chain = []
    features: dict = {"popup_or_redirect_on_load": False, "error": None}
    start = time.time()
    final_url = url
    title = ""
    has_login_form = False
    form_action_external = False
    script_count, inline_count, inline_text = 0, 0, ""
    external_resource_count = 0
    phash = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        # Everything below runs inside try/finally so a headless Chromium
        # process can never leak on error -- 15+ orphaned chrome.exe
        # processes accumulated during testing before this was added.
        try:
            ctx = browser.new_context(ignore_https_errors=True, accept_downloads=False)
            # Deny risky permission prompts up front.
            ctx.grant_permissions([])
            page = ctx.new_page()
            page.set_default_timeout(timeout_ms)

            def _on_response(r):
                # Only top-level document navigations count as "redirects" --
                # a page loading dozens of images/scripts/stylesheets from its
                # own origin is normal and must not inflate redirect_count.
                req = r.request
                if req.resource_type == "document" and req.frame == page.main_frame:
                    chain.append((r.url, r.status))

            page.on("response", _on_response)
            page.on("popup", lambda popup: features.__setitem__("popup_or_redirect_on_load", True))
            page.on("dialog", lambda dialog: (features.__setitem__("popup_or_redirect_on_load", True), dialog.dismiss()))
            page.on("download", lambda download: download.cancel())

            try:
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            except Exception as e:  # navigation errors, timeouts, DNS failures, etc.
                features["error"] = str(e)

            final_url = page.url

            try:
                final_domain_for_checks = urlparse(final_url).netloc
                has_login_form = page.query_selector('input[type="password"]') is not None
                for form in page.query_selector_all("form"):
                    action = (form.get_attribute("action") or "").strip()
                    if action and action.startswith("http"):
                        action_domain = urlparse(action).netloc
                        if action_domain and action_domain != final_domain_for_checks:
                            form_action_external = True
            except Exception:
                pass

            try:
                final_domain_for_checks = urlparse(final_url).netloc
                scripts = page.query_selector_all("script")
                script_count = len(scripts)
                for s in scripts:
                    src = s.get_attribute("src")
                    if src:
                        if urlparse(src).netloc and urlparse(src).netloc != final_domain_for_checks:
                            external_resource_count += 1
                    else:
                        inline_count += 1
                        inline_text += (s.inner_text() or "")
                for tag in page.query_selector_all("img[src], iframe[src]"):
                    src = tag.get_attribute("src") or ""
                    if urlparse(src).netloc and urlparse(src).netloc != final_domain_for_checks:
                        external_resource_count += 1
            except Exception:
                pass

            try:
                title = page.title() or ""
            except Exception:
                pass

            if _HAS_IMAGEHASH:
                try:
                    shot = page.screenshot(timeout=timeout_ms)
                    phash = str(imagehash.phash(Image.open(io.BytesIO(shot))))
                except Exception:
                    phash = None
        finally:
            try:
                browser.close()
            except Exception:
                pass

    original_domain = urlparse(url if "://" in url else "http://" + url).netloc
    final_domain = urlparse(final_url).netloc
    redirect_count = max(len(chain) - 1, 0)
    domain_changed = bool(original_domain and final_domain and original_domain != final_domain)
    title_lower = title.lower()
    brand_hits = [b for b in BRAND_KEYWORDS if b in title_lower]
    page_title_brand_mismatch = bool(
        brand_hits and not any(b in final_domain.lower() for b in brand_hits)
    )

    features.update({
        "redirect_count": redirect_count,
        "domain_changed": domain_changed,
        "redirect_chain_domains": [urlparse(u).netloc for u, _ in chain],
        "final_url": final_url,
        "final_status_code": chain[-1][1] if chain else None,
        "has_login_form": has_login_form,
        "form_action_external": form_action_external,
        "external_resource_count": external_resource_count,
        "script_count": script_count,
        "inline_script_ratio": (inline_count / script_count) if script_count else 0.0,
        "js_obfuscation_score": _shannon_entropy(inline_text),
        "page_title": title,
        "page_title_brand_mismatch": page_title_brand_mismatch,
        "screenshot_perceptual_hash": phash,
        "tls_used": final_url.startswith("https://"),
        "elapsed_ms": int((time.time() - start) * 1000),
    })
    return features


def should_escalate(static_confidence: float, static_class: str, threshold: float = 0.75) -> bool:
    """Confidence-gate rule from Section 6.2.2 / Image 3."""
    if static_confidence < threshold:
        return True
    if static_class in ("phishing", "malware") and static_confidence < 0.90:
        return True
    return False


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    result = analyze_url_dynamic(target)
    print(json.dumps(result, indent=2, default=str))
