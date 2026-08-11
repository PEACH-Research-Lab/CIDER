"""Prompt paraphrases for HC sensitivity analysis."""
from __future__ import annotations

from typing import Dict


def _prediction_block(instruction_line: str) -> str:
    return (
        "<scenario>\n"
        "{scenario_content}"
        "{role_condition_description}\n"
        "</scenario>\n"
        f"{instruction_line}\n"
        "<variants>\n"
        'variant_1:"{random_ordered_1_content}"\n'
        'variant_2:"{random_ordered_2_content}"\n'
        'variant_3:"{random_ordered_3_content}"\n'
        'variant_4:"{random_ordered_4_content}"\n'
        'variant_5:"{random_ordered_5_content}"\n'
        'variant_6:"{random_ordered_6_content}"\n'
        'variant_7:"{random_ordered_7_content}"\n'
        'variant_8:"{random_ordered_8_content}"\n'
        'variant_9:"{random_ordered_9_content}"\n'
        "</variants>"
    )


_STEPS_BLOCK = (
    "Before making the prediction, follow these steps strictly: \n"
    "Step 1: Infer the user's preference patterns and summarize them concisely.\n"
    "Step 2: Based only on this inferred preference, predict answers for the new scenario."
)

_JSON_H = """Return only valid JSON, exactly in this format:
{{
 "reasoning": "your key reasoning in exactly 2 sentences (1st sentence is about your reasoning of the user's historical disclosure preference; 2nd sentence is about your reasoning for applying this preference in the current scenario contexts to predict the answers).",
 "predictions": {{
  "variant_1": "[YES or NO]",
  "variant_2": "[YES or NO]",
  "variant_3": "[YES or NO]",
  "variant_4": "[YES or NO]",
  "variant_5": "[YES or NO]",
  "variant_6": "[YES or NO]",
  "variant_7": "[YES or NO]",
  "variant_8": "[YES or NO]",
  "variant_9": "[YES or NO]"
  }}
}}"""


_V1_SYSTEM_HC = (
    """
You will see how a user answered ({user_rating_prompt}) across {num_icl_examples} past interpersonal communication scenarios. Infer their disclosure preferences from those answers and, where possible, use them to predict the same user's answers for a new scenario.

Two sections are provided.

<History>
{num_icl_examples} past interpersonal communication scenarios in shuffled order, each with the user's answer for every variant.
For each scenario:
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants with the user's answers (YES or NO), presented in shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed. 


<Prediction>
A new interpersonal communication scenario the user has not seen, containing:
1. <scenario> Who is sharing what information, with whom, through which transmission principle, plus any additional context.
2. <variants> 9 disclosure variants in shuffled order — alternative ways the same underlying information could be shared, differing only in how it is disclosed.

"""
    + _STEPS_BLOCK
    + "\n\n"
    + _JSON_H
    + "\n"
)

_V2_SYSTEM_HC = (
    """
Overview: This task concerns a single user's information-sharing decisions. From a record of how this user previously answered ({user_rating_prompt}) across {num_icl_examples} different interpersonal communication scenarios, your job is to characterize the user's underlying disclosure preferences and, whenever those preferences are clear enough, to use them to anticipate how the same user would answer in a scenario they have not encountered before.

The input is organized into two clearly separated sections, described below.

Section 1 — <History>:
This section lists {num_icl_examples} independent interpersonal communication scenarios. They appear in a shuffled order, and each one is paired with the user's recorded answer for every disclosure variant of that scenario.
For each scenario:
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants with the user's answers (YES or NO), presented in shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed. 


Section 2 — <Prediction>:
This section introduces a single, previously unseen interpersonal communication scenario. It is made up of:
1. <scenario> A description of the context — specifically, who is sharing what information, with whom, and through which transmission principle. When additional context is relevant, it is included as well.
2. <variants> A set of 9 disclosure variants, presented in a shuffled order. Each variant is simply a different way of sharing the very same underlying information; the variants differ only in how that information is disclosed.

"""
    + _STEPS_BLOCK
    + "\n\n"
    + _JSON_H
    + "\n"
)

PROMPT_VARIANTS: Dict[str, Dict[str, str]] = {
    "v0": {},
    "v1": {
        "system_HC": _V1_SYSTEM_HC,
        "prediction_scenario_description": _prediction_block(
            "For each of the 9 variants, predict the user's answer to {user_rating_prompt} (YES or NO). "
        ),
    },
    "v2": {
        "system_HC": _V2_SYSTEM_HC,
        "prediction_scenario_description": _prediction_block(
            "For every one of the 9 variants listed below, predict the user's answer to {user_rating_prompt}; each answer must be exactly YES or NO. "
        ),
    },
}


def resolve_prompt_variant(prompt_variant: str | None) -> Dict[str, str]:
    key = "v0" if prompt_variant is None else str(prompt_variant)
    if key not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant {key!r}; known: {sorted(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[key]
