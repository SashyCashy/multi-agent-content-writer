#!/usr/bin/env python
"""
Check which Gemini models are usable right now.

Free-tier models fail in two distinct ways that look identical from inside the
app, and the right response differs:

  429 RESOURCE_EXHAUSTED  the daily cap (~20 requests per model per day) is
                          spent. Retrying will not help; it resets tomorrow.
                          Switch WRITER_MODEL to a model with quota left.

  503 UNAVAILABLE         Google's capacity, not your quota. Usually clears in
                          minutes. Nothing to fix on your side.

Run this BEFORE debugging the application code — it talks straight to the API
and skips every layer of the stack, so it tells you whether the problem is
yours at all.

    python scripts/check_models.py
    python scripts/check_models.py --all      # every model the key exposes
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx
from dotenv import load_dotenv

BASE = "https://generativelanguage.googleapis.com/v1beta"

# A trivial tool, so we also confirm function calling works — the tool loop
# depends on it, and a model that generates text but refuses tools is useless here.
TOOL = {
    "function_declarations": [
        {
            "name": "search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
}

# Candidates worth trying as WRITER_MODEL / ROUTER_MODEL, best first.
CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
]


def probe(key: str, model: str) -> tuple[str, str]:
    """Return (status label, detail) for one model."""
    started = time.time()
    try:
        r = httpx.post(
            f"{BASE}/models/{model}:generateContent",
            params={"key": key},
            json={
                "contents": [{"parts": [{"text": "Search for AI news. Use your tool."}]}],
                "tools": [TOOL],
            },
            timeout=45,
        )
    except Exception as exc:
        return "ERROR", f"{type(exc).__name__}: {exc}"[:80]

    elapsed = time.time() - started

    if r.status_code == 200:
        parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        tools_ok = any("functionCall" in p for p in parts)
        return "OK", f"{elapsed:.1f}s  tools={'yes' if tools_ok else 'NO'}"

    err = r.json().get("error", {})
    status = err.get("status", str(r.status_code))
    quota = ""
    for detail in err.get("details", []):
        for violation in detail.get("violations", []):
            quota = f"  cap={violation.get('quotaValue')}/day"
    if r.status_code == 429:
        return "QUOTA", f"daily cap spent{quota} — resets tomorrow"
    if r.status_code == 503:
        return "BUSY", "Google capacity, not your quota — retry shortly"
    return status, err.get("message", "")[:80]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Gemini model availability")
    parser.add_argument("--all", action="store_true",
                        help="probe every model the key exposes, not just the candidates")
    args = parser.parse_args()

    load_dotenv()
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not key:
        print("GOOGLE_API_KEY is not set. Add it to .env")
        return 1

    models = list(CANDIDATES)
    if args.all:
        r = httpx.get(f"{BASE}/models", params={"key": key}, timeout=30)
        models = sorted(
            m["name"].replace("models/", "")
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and not any(x in m["name"] for x in ("image", "tts", "embedding", "vision"))
        )

    configured = {
        "WRITER_MODEL": os.environ.get("WRITER_MODEL", "gemini-3.6-flash"),
        "ROUTER_MODEL": os.environ.get("ROUTER_MODEL", "gemini-3.1-flash-lite"),
    }
    print("\nConfigured:  " + "   ".join(f"{k}={v}" for k, v in configured.items()))
    print(f"Probing {len(models)} model(s)...\n")

    usable = []
    for model in models:
        status, detail = probe(key, model)
        marker = ""
        for role, name in configured.items():
            if name == model:
                marker = f"  <- {role}"
        print(f"  {status:6} {model:32} {detail}{marker}")
        if status == "OK":
            usable.append(model)

    print()
    if usable:
        print("Usable now: " + ", ".join(usable))
        print("\nTo switch the writer without redeploying code:")
        print(f"  local:  WRITER_MODEL={usable[0]} python -m src.main --query 'hi'")
        print("  cloud:  gcloud run services update multi-agent-content-writer \\")
        print(f"            --region=asia-south1 --update-env-vars WRITER_MODEL={usable[0]}")
    else:
        print("No usable model found. If everything shows QUOTA, wait for the daily")
        print("reset. If everything shows BUSY, it is Google's capacity — try later.")
    print()
    return 0 if usable else 2


if __name__ == "__main__":
    sys.exit(main())
