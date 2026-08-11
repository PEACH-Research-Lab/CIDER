"""OpenAI chat completions (OpenAI API, vLLM, gateways via ``base_url``)."""

from __future__ import annotations
from typing import Any
import requests
from constants import DEFAULT_BASE_URL

REQUEST_MODE = "openai"

def call_llm_with_raw(
    api_key: str,
    sys_prompt: str,
    user_prompt: str,
    model: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 120.0,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 1500,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    """Return ``(text, usage, raw_body, request_mode)`` for one completion."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code == 401:
        raise RuntimeError(f"{base_url} returned 401 Unauthorized; check the API key.")
    response.raise_for_status()
    body = response.json()
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    return text, usage, body, REQUEST_MODE


__all__ = ["call_llm_with_raw", "REQUEST_MODE"]
