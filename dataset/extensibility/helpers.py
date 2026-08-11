"""Building blocks for the disclosure-variant generation pipeline."""

from __future__ import annotations
import csv
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import requests

_HERE = Path(__file__).resolve().parent
_HELPER = _HERE.parents[1] / "evaluation" / "helper"
for _path in (_HELPER, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from constants import (
    GENERATION_BASE_URL,
    GENERATION_MODEL,
    GENERATION_MODEL_OPENROUTER,
    OPENROUTER_BASE_URL,
    PRIVACYLENS_FIELDS,
    REQUIRED_ITEM_FIELDS,
)
from utils import clean_env_value as _clean, ensure_env_loaded


# --- Credentials -----------------------------------------------------------

def resolve_chat_credentials(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    env_file: str | Path | None = None,
) -> tuple[str, str, str]:
    """Return ``(api_key, base_url, model)`` for an OpenAI-compatible chat endpoint."""
    key = _clean(api_key)
    url = (base_url or "").strip()
    mdl = (model or "").strip()
    if key and url and mdl:
        return key, url, mdl

    ensure_env_loaded(env_file)

    if not key:
        key = _clean(os.getenv("OPENAI_API_KEY"))
        if key:
            url = url or (os.getenv("OPENAI_BASE_URL") or GENERATION_BASE_URL).strip()
            mdl = mdl or GENERATION_MODEL
        else:
            key = _clean(os.getenv("OPENROUTER_API_KEY"))
            if key:
                url = url or OPENROUTER_BASE_URL
                mdl = mdl or GENERATION_MODEL_OPENROUTER

    if not key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
            "(e.g. in the repo ``.env``), or pass api_key=..."
        )
    return key, url or GENERATION_BASE_URL, mdl or GENERATION_MODEL


# --- Scenario fields -------------------------------------------------------

def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from CSV/JSON input
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _require_fields(item: Mapping[str, Any], fields: Sequence[str]) -> None:
    missing = [field for field in fields if not as_text(item.get(field))]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def normalize_description(item: Mapping[str, Any]) -> dict[str, str]:
    """Validate and normalize a Choice A natural-language description."""
    _require_fields(item, REQUIRED_ITEM_FIELDS)
    return {field: as_text(item[field]) for field in REQUIRED_ITEM_FIELDS}


def privacy_lens_row_to_description(
    row: Mapping[str, Any],
    *,
    story: str,
) -> dict[str, str]:
    _require_fields(row, PRIVACYLENS_FIELDS)
    if not story.strip():
        raise ValueError("story must not be empty")

    data_type = f"{as_text(row['data_type'])}({as_text(row['data_type_concrete'])})"
    sender = f"{as_text(row['data_sender_name'])}({as_text(row['data_sender_concrete'])})"
    subject = (
        f"{as_text(row['data_subject_name'])}"
        f"({as_text(row['data_subject'])},{as_text(row['data_subject_concrete'])})"
    )
    recipient = (
        f"{as_text(row['data_recipient_name'])}"
        f"({as_text(row['data_recipient'])},{as_text(row['data_recipient_concrete'])})"
    )

    return {
        "name": as_text(row["name"]),
        "data_type": data_type,
        "sensitive_info_items": as_text(row["sensitive_info"]),
        "data_sender": sender,
        "data_subject": subject,
        "data_recipient": recipient,
        "transmission_principle": as_text(row["transmission_principle"]),
        "story": story.strip(),
    }


# --- Chat transport --------------------------------------------------------

def chat_completion(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 300.0,
) -> str:
    key, url, mdl = resolve_chat_credentials(
        api_key=api_key, base_url=base_url, model=model
    )

    response = requests.post(
        f"{url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": mdl,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    if response.status_code == 401:
        raise RuntimeError(
            f"Chat API returned 401 Unauthorized (base_url={url!r}, model={mdl!r}). "
            "Use OPENROUTER_API_KEY for OpenRouter or OPENAI_API_KEY for OpenAI, "
            "with no trailing '\\' in .env."
        ) from None
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# --- Record I/O ------------------------------------------------------------

def load_records(path: Path) -> list[Mapping[str, Any]]:
    """Read scenario rows from ``.json``, ``.jsonl``, or ``.csv``."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    value = json.loads(text)
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
        return value
    raise ValueError(f"{path} must hold a JSON object/list, JSONL lines, or CSV rows")


def write_results(path: Path, results: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
