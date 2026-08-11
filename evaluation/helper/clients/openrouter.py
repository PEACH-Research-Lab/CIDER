"""OpenRouter chat completions with token accounting and reasoning extras."""


from __future__ import annotations
from typing import Any
import requests
from constants import OPENROUTER_BASE_URL

_USAGE_ACCOUNTING = {"include": True}
_JSON_UNSUPPORTED_STATUS = (400, 422)

MODE_JSON_OBJECT = "openrouter_json_object"
MODE_PLAIN = "openrouter_plain"


def _normalize_extras(extras: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(extras or {})
    reasoning = out.get("reasoning")
    if reasoning is None:
        return out

    reasoning = dict(reasoning)
    if "enabled" in reasoning and not bool(reasoning["enabled"]):
        out["reasoning"] = {"enabled": False}
        return out
    if "enabled" in reasoning:
        reasoning.pop("enabled")
        reasoning.setdefault("effort", "medium")
    if str(reasoning.get("effort", "")).strip().lower() == "none":
        out["reasoning"] = {"enabled": False}
        return out

    out["reasoning"] = reasoning
    return out


def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    out = dict(usage or {})
    details = out.get("completion_tokens_details") or {}
    if "reasoning_tokens" not in out and "reasoning_tokens" in details:
        out["reasoning_tokens"] = details["reasoning_tokens"]
    return out


def _build_payload(
    sys_prompt: str,
    user_prompt: str,
    model: str,
    *,
    temperature: float,
    seed: int,
    max_tokens: int,
    json_object: bool,
    extras: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "usage": _USAGE_ACCOUNTING,
        **extras,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _unpack(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError(f"OpenRouter returned no choices: {body!r}")
    message = choices[0].get("message") or {}
    return message.get("content") or "", _normalize_usage(body.get("usage") or {})


def _post(
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """POST via the ``openai`` SDK when available, else ``requests``."""
    try:
        from openai import OpenAI
    except ImportError:
        response = requests.post(
            f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 401:
            raise RuntimeError(
                "OpenRouter returned 401 Unauthorized; check OPENROUTER_API_KEY "
                "(no trailing '\\' in .env)."
            ) from None
        if response.status_code in _JSON_UNSUPPORTED_STATUS:
            raise _PayloadRejected(response.status_code, response.text)
        response.raise_for_status()
        return response.json()

    client = OpenAI(
        api_key=api_key, base_url=OPENROUTER_BASE_URL, timeout=timeout
    )
    body = dict(payload)
    messages = body.pop("messages")
    model = body.pop("model")
    # OpenRouter-only fields (usage accounting, reasoning) are not parameters of
    # Completions.create(); send them in the JSON body via extra_body.
    sdk_keys = {
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "n",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "stream",
    }
    sdk_kwargs = {k: body.pop(k) for k in list(body) if k in sdk_keys}
    extra_body = body
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            **sdk_kwargs,
            **({"extra_body": extra_body} if extra_body else {}),
        )
    except Exception as err:  # SDK wraps HTTP errors in typed exceptions
        status = getattr(err, "status_code", None)
        if status in _JSON_UNSUPPORTED_STATUS:
            raise _PayloadRejected(status, str(err)) from err
        raise
    return completion.model_dump()


class _PayloadRejected(RuntimeError):
    """OpenRouter rejected the payload; retry without ``response_format``."""

    def __init__(self, status: int | None, detail: str) -> None:
        super().__init__(f"OpenRouter rejected payload (status={status}): {detail}")
        self.status = status


def call_llm_with_raw(
    api_key: str,
    sys_prompt: str,
    user_prompt: str,
    model: str,
    *,
    timeout: float = 120.0,
    temperature: float = 0.0,
    seed: int = 42,
    max_tokens: int = 1500,
    prefer_json_object: bool = True,
    openrouter_payload_extras: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
    """Return ``(text, usage, raw_body, request_mode)`` for one completion.

    With ``prefer_json_object`` the first attempt asks for a JSON object and
    falls back to a plain request if the model rejects that field;
    ``request_mode`` records which form produced the response.
    """
    extras = _normalize_extras(openrouter_payload_extras)

    modes = [MODE_JSON_OBJECT, MODE_PLAIN] if prefer_json_object else [MODE_PLAIN]
    for mode in modes:
        payload = _build_payload(
            sys_prompt,
            user_prompt,
            model,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            json_object=mode == MODE_JSON_OBJECT,
            extras=extras,
        )
        try:
            body = _post(api_key, payload, timeout)
        except _PayloadRejected as err:
            if mode is modes[-1]:
                raise
            print(f"  [openrouter] {err}; retrying without response_format")
            continue
        text, usage = _unpack(body)
        return text, usage, body, mode

    raise AssertionError("unreachable: mode loop always returns or raises")


__all__ = ["call_llm_with_raw"]
