#!/usr/bin/env python3
"""Generate disclosure variants from either supported input format with GPT-o3.
Choice A: Already normalized natural-language scenario seeds.
Choice B: A PrivacyLens seed with a third-person narrative (story).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from helpers import (
    as_text,
    chat_completion,
    load_records,
    normalize_description,
    privacy_lens_row_to_description,
    resolve_chat_credentials,
    write_results,
)
from prompts import build_generation_prompt


def generate_from_description(
    item: Mapping[str, Any],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    normalized = normalize_description(item)
    output = chat_completion(
        build_generation_prompt(normalized),
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    return {"name": normalized["name"], "output": output}


def generate_from_privacylens_row(
    row: Mapping[str, Any],
    *,
    story: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    item = privacy_lens_row_to_description(row, story=story)
    return generate_from_description(
        item,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def _generate_many(
    items: Iterable[Mapping[str, Any]],
    *,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    sleep_seconds: float,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if index and sleep_seconds:
            time.sleep(sleep_seconds)
        results.append(
            generate_from_description(
                item,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Chat model id (default: from .env / OPENAI or OpenRouter defaults)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: inferred from credentials)",
    )
    parser.add_argument("--sleep", type=float, default=1.0)
    subparsers = parser.add_subparsers(dest="choice", required=True)

    for name, aliases, help_text in (
        ("description", ["choice-a"], "Choice A: normalized natural-language descriptions"),
        ("privacylens", ["choice-b"], "Choice B: PrivacyLens rows plus a data type and story"),
    ):
        sub = subparsers.add_parser(name, aliases=aliases, help=help_text)
        sub.add_argument("--input", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        if name == "privacylens":
            sub.add_argument("--data-type", required=True)
            sub.add_argument("--story", required=True)
            sub.add_argument(
                "--name",
                action="append",
                help="Generate only matching scenario IDs; repeat for multiple IDs",
            )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    api_key, base_url, model = resolve_chat_credentials(
        base_url=args.base_url,
        model=args.model,
    )
    print(f"Using model={model} base_url={base_url}")

    records = load_records(args.input)
    if args.choice in {"description", "choice-a"}:
        items: list[Mapping[str, Any]] = list(records)
    else:
        if args.name:
            wanted = {str(name) for name in args.name}
            records = [row for row in records if as_text(row.get("name")) in wanted]
        items = [
            privacy_lens_row_to_description(
                row,
                story=args.story,
            )
            for row in records
        ]

    results = _generate_many(
        items,
        model=model,
        api_key=api_key,
        base_url=base_url,
        sleep_seconds=args.sleep,
    )
    write_results(args.output, results)
    print(f"Wrote {len(results)} result(s) to {args.output}")


if __name__ == "__main__":
    main()
