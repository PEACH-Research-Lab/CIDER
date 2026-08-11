"""Pluggable chat backends for evaluation."""

from __future__ import annotations
import argparse
import os
from typing import Any, Callable
from constants import (
    DEFAULT_BASE_URL,
    DEFAULT_PROVIDER,
    PROVIDER_KEY_ENV,
    PROVIDERS,
)
from utils import ensure_env_loaded

_EXPERIMENT_FNS = {
    "openai": "run_icl_experiment_openai",
    "openrouter": "run_icl_experiment_openrouter",
}


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared ``--provider`` / ``--base-url`` / ``--api-key-env`` flags."""
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=DEFAULT_PROVIDER,
        help=(
            "Default chat backend: openai (default), openrouter "
            "(optional SDK), or external (score existing JSONL only). "
            "Individual models[] entries may override it."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=f"OpenAI API base URL (default: OPENAI_BASE_URL or {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Env var holding the API key (default depends on --provider)",
    )
    parser.add_argument(
        "--reasoning-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CoT/reasoning extras when supported (OpenRouter; or models[].reasoning)",
    )


def require_api_key(provider: str, api_key_env: str | None) -> str:
    """Return the env var name after verifying the key is set (live providers only)."""
    if provider == "external":
        raise ValueError("external provider does not use an API key")
    env = api_key_env or PROVIDER_KEY_ENV.get(provider)
    if not env:
        raise ValueError(f"No default API key env for provider={provider!r}")
    ensure_env_loaded()
    if not os.getenv(env):
        raise SystemExit(
            f"{env} is not set. Set it in the environment or .env, "
            f"or pass --api-key-env. For existing predictions only, use "
            f"--provider external or --metrics-only."
        )
    return env


def resolve_base_url(base_url: str | None) -> str:
    ensure_env_loaded()
    return base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL


def _resolve_run_backends(
    cfg: dict[str, Any],
    configs: list[dict],
    *,
    provider: str,
    api_key_env: str | None,
    base_url: str | None,
    reasoning_enabled: bool,
) -> list[dict]:
    """Fill provider / credentials / OpenRouter extras on each run row."""
    prefer_json_object = bool(cfg.get("openrouter_prefer_json_object", True))
    top_extras = dict(cfg.get("openrouter_payload_extras") or {})
    out: list[dict] = []
    for entry in configs:
        row = dict(entry)
        row_provider = str(row.get("provider") or provider)
        if row_provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {row_provider!r}")
        row["provider"] = row_provider
        if row_provider == "external":
            out.append(row)
            continue

        # CLI --base-url / --api-key-env only apply to rows that resolve to the
        # same provider as --provider, so mixed-backend configs can't leak keys.
        from_cli = row_provider == provider
        row["api_key_env"] = (
            row.get("api_key_env")
            or (api_key_env if from_cli else None)
            or PROVIDER_KEY_ENV.get(row_provider)
        )
        if row_provider == "openai":
            row["base_url"] = row.get("base_url") or resolve_base_url(
                base_url if from_cli else None
            )
        elif row_provider == "openrouter":
            extras = dict(top_extras)
            # ``reasoning`` may use ``{"enabled": bool}``; the client maps it to effort.
            extras["reasoning"] = (
                dict(row["reasoning"])
                if row.get("reasoning") is not None
                else {"effort": "medium" if reasoning_enabled else "none"}
            )
            row["openrouter_payload_extras"] = extras
            row["openrouter_prefer_json_object"] = prefer_json_object
        out.append(row)
    return out


def get_experiment_fn(provider: str) -> Callable[[dict, dict], dict]:
    """Return the ICL experiment runner for ``provider``."""
    if provider == "external":
        raise ValueError("external provider has no experiment runner; use metrics-only")
    try:
        factory = _EXPERIMENT_FNS[provider]
    except KeyError:
        raise ValueError(f"Unknown provider: {provider!r}") from None
    # Late import: evaluation/ is on sys.path for entry points.
    import runner

    return getattr(runner, factory)


def run_for_config(user_data: dict, config: dict) -> dict:
    """Dispatch one run config to the runner for its own ``provider``."""
    return get_experiment_fn(str(config["provider"]))(user_data, config)


def prepare_live_run(
    cfg: dict[str, Any],
    configs: list[dict],
    *,
    provider: str = DEFAULT_PROVIDER,
    api_key_env: str | None = None,
    base_url: str | None = None,
    reasoning_enabled: bool = False,
) -> tuple[list[dict], Callable[[dict, dict], dict]]:
    """Resolve backends + credentials; return ``(live_configs, experiment_fn)``.

    Rows that resolve to ``external`` are dropped (predictions assumed on disk).
    """
    prepared = _resolve_run_backends(
        cfg,
        configs,
        provider=provider,
        api_key_env=api_key_env,
        base_url=base_url,
        reasoning_enabled=reasoning_enabled,
    )
    live = [row for row in prepared if row["provider"] != "external"]
    if not live:
        raise ValueError(
            "No live runs to prepare: every model resolved to the external "
            "provider. Use --metrics-only to score existing predictions."
        )

    seen: set[tuple[str, str]] = set()
    for row in live:
        key = (row["provider"], row["api_key_env"])
        if key in seen:
            continue
        seen.add(key)
        require_api_key(*key)
    return live, run_for_config
