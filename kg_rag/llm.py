"""A dependency-free client for any OpenAI-compatible chat-completions API.

Works with OpenAI, Groq, OpenRouter, Google Gemini's OpenAI-compatible
endpoint and a local Ollama server - only ``LLM_BASE_URL``, ``LLM_MODEL`` and
``LLM_API_KEY`` change. Uses ``urllib`` so the project needs no SDK.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class LLMError(RuntimeError):
    pass


def available() -> bool:
    # Ollama needs no key, so a localhost base URL also counts as configured.
    return bool(config.LLM_API_KEY) or "localhost" in config.LLM_BASE_URL


def describe() -> str:
    return f"{config.LLM_MODEL} via {config.LLM_BASE_URL}" if available() else "not configured"


def chat(system: str, user: str, max_tokens: int = 400) -> str:
    if not available():
        raise LLMError("No LLM configured (set LLM_API_KEY).")
    body = json.dumps({
        "model": config.LLM_MODEL,
        "temperature": config.LLM_TEMPERATURE,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
    }).encode()
    request = urllib.request.Request(
        config.LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {config.LLM_API_KEY or 'ollama'}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_S) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise LLMError(f"LLM API returned {exc.code}: {exc.read()[:300]!r}") from exc
    except Exception as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected LLM response: {str(payload)[:300]}") from exc
