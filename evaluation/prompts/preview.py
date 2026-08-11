"""Build and display example prompts for selected evaluation setups."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from constants import SEED, SETUP_LABELS, SETUP_ORDER
from prompts.construction import build_prompts


def _first_example_case(user_data: dict) -> tuple[str, dict, dict]:
    """Return ``(user_id, user_info, test_scenario)`` for a stable sample case."""
    if not user_data:
        raise ValueError("user_data is empty; cannot build example prompts")
    for user_id, info in sorted(user_data.items(), key=lambda item: str(item[0])):
        test_cases = info.get("test_cases") or []
        if test_cases:
            return str(user_id), info, test_cases[0]
    raise ValueError("No users with test_cases; cannot build example prompts")


def build_example_prompts(
    user_data: dict,
    setups: list[str],
    *,
    variant_shuffle_seed: int | None = None,
    prompt_variant: str = "v0",
) -> list[dict[str, Any]]:
    """Build one example system/user prompt per setup using the same sample case."""
    user_id, info, test_case = _first_example_case(user_data)
    seed = SEED if variant_shuffle_seed is None else int(variant_shuffle_seed)
    order = {name: i for i, name in enumerate(SETUP_ORDER)}
    setups_sorted = sorted(set(setups), key=lambda s: order.get(s, len(SETUP_ORDER)))

    examples: list[dict[str, Any]] = []
    for setup in setups_sorted:
        system_prompt, user_prompt = build_prompts(
            user_id,
            info,
            test_case,
            prompt_pattern=setup,
            variant_shuffle_seed=seed,
            prompt_variant=prompt_variant,
        )
        examples.append(
            {
                "setup": setup,
                "label": SETUP_LABELS.get(setup, setup),
                "user_id": user_id,
                "scenario_id": str(test_case.get("scenario_id", "")),
                "role": info.get("role", test_case.get("role")),
                "condition": info.get("condition", test_case.get("condition")),
                "n_icl_examples": len(info.get("icl_examples") or []),
                "prompt_variant": prompt_variant,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
    return examples


def format_example_prompts(examples: list[dict[str, Any]]) -> str:
    """Render example prompts as a reviewable text document."""
    if not examples:
        return ""
    first = examples[0]
    lines = [
        "CIDER evaluation — example prompts (review before prediction)",
        f"Sample case: user={first['user_id']}  scenario={first['scenario_id']}  "
        f"role={first['role']}  condition={first['condition']}  "
        f"n_icl={first['n_icl_examples']}  prompt_variant={first['prompt_variant']}",
        "",
    ]
    for ex in examples:
        lines.extend(
            [
                "=" * 80,
                f"SETUP: {ex['setup']}  ({ex['label']})",
                "=" * 80,
                "",
                "----- SYSTEM -----",
                ex["system_prompt"],
                "",
                "----- USER -----",
                ex["user_prompt"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def print_example_prompts(examples: list[dict[str, Any]]) -> None:
    print(format_example_prompts(examples))


def save_example_prompts(examples: list[dict[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(format_example_prompts(examples), encoding="utf-8")
    return out


def display_example_prompts(examples: list[dict[str, Any]]) -> None:
    """Render one collapsible block per setup (IPython / Jupyter)."""
    from IPython.display import HTML, display

    if not examples:
        print("No example prompts to display.")
        return

    first = examples[0]
    header = (
        "<p><b>Sample case</b> — "
        f"user=<code>{html.escape(str(first['user_id']))}</code>, "
        f"scenario=<code>{html.escape(str(first['scenario_id']))}</code>, "
        f"role=<code>{html.escape(str(first['role']))}</code>, "
        f"condition=<code>{html.escape(str(first['condition']))}</code>, "
        f"n_icl={int(first['n_icl_examples'])}, "
        f"prompt_variant=<code>{html.escape(str(first['prompt_variant']))}</code>"
        "</p>"
    )
    blocks = [header]
    for ex in examples:
        setup = html.escape(str(ex["setup"]))
        label = html.escape(str(ex["label"]))
        system = html.escape(str(ex["system_prompt"]))
        user = html.escape(str(ex["user_prompt"]))
        blocks.append(
            f"""
<details open style="margin:1em 0;padding:0.5em 0.75em;border:1px solid #ccc;border-radius:6px;">
  <summary style="cursor:pointer;font-weight:600;">
    Setup <code>{setup}</code> — {label}
  </summary>
  <h4 style="margin:0.75em 0 0.25em;">System prompt</h4>
  <pre style="white-space:pre-wrap;word-break:break-word;background:#f7f7f7;padding:0.75em;border-radius:4px;">{system}</pre>
  <h4 style="margin:0.75em 0 0.25em;">User prompt</h4>
  <pre style="white-space:pre-wrap;word-break:break-word;background:#f7f7f7;padding:0.75em;border-radius:4px;">{user}</pre>
</details>
"""
        )
    display(HTML("\n".join(blocks)))
