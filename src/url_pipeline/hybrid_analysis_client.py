"""
Hybrid Analysis (CrowdStrike Falcon Sandbox) API Client for CyberXAI.
Provides programmatic access to Hybrid Analysis API v2:
- API key & account quota verification
- Live sandbox threat feed streaming (latest detonated malware & URLs)
- Hash threat overview & multi-engine AV detection lookups
- Sandbox execution report summary lookups
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.hybrid-analysis.com/api/v2"


def _read_credentials() -> tuple[Optional[str], Optional[str]]:
    api_key = os.environ.get("HYBRID_ANALYSIS_API_KEY")
    api_id = os.environ.get("HYBRID_ANALYSIS_ID")

    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not api_key and line.startswith("HYBRID_ANALYSIS_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            if not api_id and line.startswith("HYBRID_ANALYSIS_ID="):
                api_id = line.split("=", 1)[1].strip()

    return api_key, api_id


class HybridAnalysisClient:
    """Client for Hybrid Analysis / Falcon Sandbox API v2."""

    def __init__(self, api_key: Optional[str] = None, api_id: Optional[str] = None):
        env_key, env_id = _read_credentials()
        self.api_key = api_key or env_key
        self.api_id = api_id or env_id

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise RuntimeError("HYBRID_ANALYSIS_API_KEY not configured in environment or .env")
        return {
            "api-key": self.api_key,
            "User-Agent": "Falcon Sandbox",
            "Accept": "application/json",
        }

    def _get(self, endpoint: str, timeout: int = 15) -> Dict[str, Any]:
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Hybrid Analysis API error {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"Hybrid Analysis request failed: {e}") from e

    def get_account_status(self) -> Dict[str, Any]:
        """Verify API key and retrieve account info."""
        data = self._get("key/current")
        version = self._get("system/version")
        return {
            "authenticated": True,
            "api_key_masked": f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key else None,
            "key_id": self.api_id,
            "user": data.get("user", {}),
            "auth_level": data.get("auth_level"),
            "auth_level_name": data.get("auth_level_name"),
            "system_version": version,
        }

    def get_environments(self) -> List[Dict[str, Any]]:
        """List supported sandbox guest OS environments."""
        res = self._get("system/environments")
        return res if isinstance(res, list) else []

    def get_live_threat_feed(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Retrieve recent detonations from the global live sandbox feed."""
        data = self._get("feed/latest", timeout=20)
        items = data.get("data", [])
        results = []
        for item in items[:limit]:
            results.append({
                "job_id": item.get("job_id"),
                "sha256": item.get("sha256"),
                "md5": item.get("md5"),
                "submit_name": item.get("submit_name"),
                "threat_score": item.get("threat_score"),
                "threat_level": item.get("threat_level"),
                "threat_level_human": item.get("threat_level_human") or "no specific threat",
                "verdict": item.get("threat_level_human") or "unknown",
                "av_detect": item.get("av_detect", 0),
                "environment": item.get("environment_description"),
                "analysis_start_time": item.get("analysis_start_time"),
                "report_url": f"https://www.hybrid-analysis.com{item.get('report_url')}" if item.get("report_url") else None,
            })
        return results

    def lookup_hash(self, sha256_hash: str) -> Dict[str, Any]:
        """Retrieve threat overview for a SHA256 file hash."""
        clean_hash = sha256_hash.strip().lower()
        data = self._get(f"overview/{clean_hash}/summary")
        return {
            "sha256": clean_hash,
            "threat_score": data.get("threat_score"),
            "verdict": data.get("verdict") or "no specific threat",
            "submitted_at": data.get("submitted_at"),
            "analysis_start_time": data.get("analysis_start_time"),
            "multiscan_result": data.get("multiscan_result"),
            "last_multi_scan": data.get("last_multi_scan"),
            "report_url": f"https://www.hybrid-analysis.com/sample/{clean_hash}",
        }

    def get_report_summary(self, job_id: str) -> Dict[str, Any]:
        """Fetch summary report by sandbox job ID."""
        clean_id = job_id.strip()
        data = self._get(f"report/{clean_id}/summary")
        return data


# Global singleton instance
client = HybridAnalysisClient()
