"""Prompt construction for ``Baseline``, ``H``, ``HL``, and ``HC``."""
from __future__ import annotations

import hashlib
import random

from constants import VARIANT_SUFFIXES, normalize_setup
from prompts.templates import (
    ICL_SCENARIO_DESCRIPTION,
    LABEL_DEFINITION,
    PREDICTION_SCENARIO_DESCRIPTION,
    SYSTEM_TEMPLATE,
    USER_RATING_PROMPT_GENERAL,
    USER_RATING_PROMPT_USER,
    USER_TEMPLATE,
)
from prompts.sensitivity import resolve_prompt_variant
from load_data import rating_to_yes_no

_ROLE_KEY_MAP = {
    "data_sender": "sender",
    "data_subject": "subject",
    "data_recipient": "recipient",
}

_CONDITION_KEY_MAP = {
    "Human": "human",
    "AI": "ai",
}

_VARIANT_LABEL = {
    "11": "General, Not Identifiable",
    "12": "General, Partially Identifiable",
    "13": "General, Fully Identifiable",
    "21": "Moderately detailed, Not Identifiable",
    "22": "Moderately detailed, Partially Identifiable",
    "23": "Moderately detailed, Fully Identifiable",
    "31": "Very detailed, Not Identifiable",
    "32": "Very detailed, Partially Identifiable",
    "33": "Very detailed, Fully Identifiable",
}


def get_user_prompt(role: str, condition: str, data_sender: str, prompt_dict: dict) -> str:
    role_key = _ROLE_KEY_MAP[role]
    condition_key = _CONDITION_KEY_MAP[condition]
    key = f"{role_key}_{condition_key}".upper()
    return prompt_dict[key]["template"].format(data_sender=data_sender)


def get_random_order(scenario: dict, *, rng: random.Random | None = None) -> list[str]:
    order = scenario.get("randomized_suffix_order")
    if isinstance(order, list) and len(order) == len(VARIANT_SUFFIXES):
        return [str(s) for s in order]
    if rng is not None:
        return rng.sample(VARIANT_SUFFIXES, k=len(VARIANT_SUFFIXES))
    return random.sample(VARIANT_SUFFIXES, k=len(VARIANT_SUFFIXES))


def extract_variants(scenario: dict) -> dict[str, str]:
    vn = scenario.get("variant_narratives", {})
    return {s: vn.get(f"var{s}", "") for s in VARIANT_SUFFIXES}



def extract_ratings(scenario: dict) -> dict[str, str]:
    return {s: rating_to_yes_no(scenario[f"var{s}_rating"]) for s in VARIANT_SUFFIXES}


def _build_deterministic_rng(material: str) -> random.Random:
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_role_condition_description(
    condition: str, role: str, sender_label: str
) -> str:
    """AI-only extra line inside ``<scenario>`` (Human → empty)."""
    if condition != "AI":
        return ""
    if role == "data_sender":
        text = "Now you are using your AI assistant to share this information."
    else:
        text = f"Now {sender_label} is using their AI assistant to share this information."
    return f"\n{text}"


def _sender_for_history(history_key: str, sender_label: str) -> str:
    # Redacted history setups avoid leaking scenario-specific sender names.
    if history_key in {"H", "HL"}:
        return "data sender"
    return sender_label


def build_icl_block(
    scenario: dict,
    history_key: str,
    role: str,
    condition: str,
    *,
    rng: random.Random | None = None,
) -> tuple[str, list[str]]:
    sender_label = scenario.get("data_sender_label", "the data sender")
    sender_for_text = _sender_for_history(history_key, sender_label)
    user_prompt = get_user_prompt(
        role, condition, sender_for_text, USER_RATING_PROMPT_USER
    )
    order = get_random_order(scenario, rng=rng)
    variants = extract_variants(scenario)
    ratings = extract_ratings(scenario)
    kwargs = {
        "scenario_content": scenario.get("scenario_narrative", ""),
        "role_condition_description": build_role_condition_description(
            condition, role, sender_for_text
        ),
        "user_rating_prompt_user": user_prompt,
    }
    for i, s in enumerate(order, 1):
        kwargs[f"random_ordered_{i}_label"] = _VARIANT_LABEL[s]
        kwargs[f"random_ordered_{i}_content"] = variants[s]
        kwargs[f"random_ordered_{i}_rating"] = str(ratings[s])
    return ICL_SCENARIO_DESCRIPTION[history_key]["template"].format(**kwargs), order


def build_prediction_block(
    scenario: dict,
    role: str,
    condition: str,
    *,
    rng: random.Random | None = None,
    prediction_template: str | None = None,
) -> tuple[str, list[str]]:
    sender_label = scenario.get("data_sender_label", "the data sender")
    # Prediction always has full semantic context → use the real sender label.
    user_rating_prompt = get_user_prompt(
        role, condition, sender_label, USER_RATING_PROMPT_USER
    )
    order = get_random_order(scenario, rng=rng)
    variants = extract_variants(scenario)
    kwargs = {
        "scenario_content": scenario.get("scenario_narrative", ""),
        "role_condition_description": build_role_condition_description(
            condition, role, sender_label
        ),
        "user_rating_prompt": user_rating_prompt,
    }
    for i, s in enumerate(order, 1):
        kwargs[f"random_ordered_{i}_content"] = variants[s]
    template = prediction_template or PREDICTION_SCENARIO_DESCRIPTION["template"]
    return template.format(**kwargs), order


def build_prompts(
    user_id,
    user_info,
    test_scenario,
    *,
    prompt_pattern: str = "Baseline",
    return_order_mappings: bool = False,
    variant_shuffle_seed: int | None = None,
    prompt_variant: str = "v0",
):
    """Build system/user prompts for ``Baseline``, ``H``, ``HL``, and ``HC``."""
    setup = normalize_setup(prompt_pattern)
    history_key = "" if setup == "Baseline" else setup
    with_history = bool(history_key)

    overrides = resolve_prompt_variant(prompt_variant)
    role = user_info["role"]
    condition = user_info["condition"]
    icl_examples = user_info.get("icl_examples") or []
    test_sender = test_scenario.get("data_sender_label", "the data sender")

    # System prompt: use real sender when the setup exposes scenario semantics.
    if history_key in {"", "HC"} or not with_history:
        system_sender = test_sender if test_sender else "data sender"
        if system_sender == "the data sender":
            system_sender = "data sender"
    else:
        system_sender = "data sender"

    user_rating_prompt = get_user_prompt(
        role, condition, system_sender, USER_RATING_PROMPT_GENERAL
    )

    if with_history:
        override_key = f"system_{history_key}"
        system_template = overrides.get(
            override_key, SYSTEM_TEMPLATE[history_key]["template"]
        )
    else:
        system_template = overrides.get("system_NH", SYSTEM_TEMPLATE["NH"]["template"])

    format_kwargs = {
        "num_icl_examples": len(icl_examples),
        "user_rating_prompt": user_rating_prompt,
    }
    if with_history and history_key == "HL":
        format_kwargs["label_definition"] = LABEL_DEFINITION["template"]
    system_prompt = system_template.format(**format_kwargs).strip()

    icl_blocks: list[str] = []
    order_mappings: dict = {"historical": [], "prediction": {}}

    def _make_rng(scope: str, scenario_id: object) -> random.Random | None:
        if variant_shuffle_seed is None:
            return None
        scope_offset = 201 if scope == "history" else 202
        material = f"{variant_shuffle_seed + scope_offset}|{user_id}|{scenario_id}"
        return _build_deterministic_rng(material)

    if with_history:
        for i, ex in enumerate(icl_examples):
            ex_rng = _make_rng("history", ex.get("scenario_id"))
            block, order = build_icl_block(ex, history_key, role, condition, rng=ex_rng)
            icl_blocks.append(block)
            order_mappings["historical"].append(
                {
                    "example_index": i + 1,
                    "scenario_id": ex.get("scenario_id"),
                    "order": order,
                }
            )

    # Prediction always shows full scenario/variant text.
    pred_rng = _make_rng("prediction", test_scenario.get("scenario_id"))
    pred_block, pred_order = build_prediction_block(
        test_scenario,
        role,
        condition,
        rng=pred_rng,
        prediction_template=overrides.get("prediction_scenario_description"),
    )
    order_mappings["prediction"] = {
        "scenario_id": test_scenario.get("scenario_id"),
        "order": pred_order,
    }

    user_prompt = (
        USER_TEMPLATE["H" if with_history else "NH"]["template"]
        .format(
            icl_scenario_description="\n\n".join(icl_blocks),
            prediction_scenario_description=pred_block,
        )
        .strip()
    )

    if not return_order_mappings:
        return system_prompt, user_prompt
    return (system_prompt, user_prompt), order_mappings
