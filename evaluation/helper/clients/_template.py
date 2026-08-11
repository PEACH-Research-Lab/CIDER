"""Generic LLM client template for evaluation"""

from __future__ import annotations
from typing import Any
import requests

REQUEST_MODE = "template_native"

def call_llm_with_raw(
    api_key: str,
    sys_prompt: str,
    user_prompt: str,
    model: str,
    *,
    base_url: str = "https://api.example.com/v1",
    timeout: float = 120.0,
    temperature: float = 0.0,
    max_tokens: int = 1500,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    """Send one chat request and unpack it into the runner's 4-tuple."""
    response = requests.post(
        f"{base_url.rstrip('/')}/messages",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            # Replace with the vendor's schema.
            "model": model,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    if response.status_code == 401:
        raise RuntimeError(f"{base_url} returned 401 Unauthorized; check the API key.")
    response.raise_for_status()
    body = response.json()

    # Point these at wherever the vendor puts the text and token counts.
    text = body["content"][0]["text"]
    raw_usage = body.get("usage") or {}
    usage = {
        "prompt_tokens": raw_usage.get("input_tokens", 0),
        "completion_tokens": raw_usage.get("output_tokens", 0),
        "reasoning_tokens": raw_usage.get("reasoning_tokens", 0),
        "total_tokens": (
            raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)
        ),
    }
    return text, usage, body, REQUEST_MODE


__all__ = ["call_llm_with_raw", "REQUEST_MODE"]
