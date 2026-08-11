from __future__ import annotations

from typing import Any, Dict, TypedDict


class TemplateEntry(TypedDict):
    template: Any
    internal_description: str


LABEL_DEFINITION: TemplateEntry = {
    "template": (
        "\n"
        "Granularity:\n"
        "1. General: The disclosure is a high-level abstraction of the information "
        "without mentioning fine details about the action, processes, or context.\n"
        "2. Moderately detailed: The disclosure elaborates some details about the "
        "information, but is still abstract and not exhaustive.\n"
        "3. Very detailed: The disclosure covers the comprehensive and fine-grained "
        "details of the information.\n"
        "Identifiability:\n"
        "1. Not Identifiable: The disclosure anonymizes or omits all personal "
        "identifiers of the data subject that could be used to directly or "
        "indirectly trace back to them.\n"
        "2. Partially Identifiable: The disclosure contains attributes or contextual "
        "references that cannot be directly used to identify the data subject, but "
        "can be combined with other attributes, contextual metadata, or publicly "
        "available information to trace back to them.\n"
        "3. Fully Identifiable: The disclosure contains direct identifiers that can "
        "uniquely identify the data subject - such as their name, role, or other "
        "specific identifiers.\n"
    ),
    "internal_description": "G×I label definitions for HL history.",
}


# Per-setup system prompts (history setups are not interchangeable).
SYSTEM_TEMPLATE: Dict[str, TemplateEntry] = {
    "H": {
        "template": """
Your task is to infer a user's information disclosure preferences from their historical answers across {num_icl_examples} independent interpersonal communication scenarios, and, if possible, use these preferences to predict their answers ({user_rating_prompt}) in a new scenario.
 
You will be provided with two sections:
 
<History>
This part contains {num_icl_examples} independent interpersonal communication scenarios, presented in shuffled order, along with the user's answers for each variant for the corresponding scenario.
For each scenario:
1. <scenario> The scenario content is redacted. 
2. <variants> 9 disclosure variants with the user's answers (YES or NO) and variant semantic content redacted, presented in shuffled order. These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed. 
 
<Prediction>
This section contains a new interpersonal communication scenario the user has not seen before. It includes:
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants, presentedin shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed.

Before making the prediction, follow these steps strictly: 
Step 1: Infer the user's preference patterns and summarize them concisely.
Step 2: Based only on this inferred preference, predict answers for the new scenario.
 
Return JSON exactly in this format:
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
}}
""",
        "internal_description": "H (history, redacted)",
    },
    "HL": {
        "template": """
Your task is to infer a user's information disclosure preferences from their historical answers across {num_icl_examples} independent interpersonal communication scenarios, and, if possible, use these preferences to predict their answers ({user_rating_prompt}) in a new scenario.
 
You will be provided with two sections:
 
<History>
This part contains {num_icl_examples} independent interpersonal communication scenarios, presented in shuffled order, along with the user's answers for each scenario’s variants.
For each scenario:
1. <scenario> The scenario content is redacted.
2. <variants> 9 disclosure variants with the user's answers (YES or NO) and variant semantic content redacted, presented in shuffled order. These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed. Each variant corresponds to a unique combination of granularity (level of detail) and identifiability (how identifiable the data subject is), described as a label and defined as follows: 
{label_definition}
<Prediction>
This section contains a new interpersonal communication scenario the user has not seen before. It includes:
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants, presented in shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed.

Before making the prediction, follow these steps strictly: 
Step 1: Infer the user's preference patterns and summarize them concisely.
Step 2: Based only on this inferred preference, predict answers for the new scenario.
 
Return JSON exactly in this format:
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
}}
""",
        "internal_description": "HL (history + labels)",
    },
    "HC": {
        "template": """
Your task is to infer a user's information disclosure preferences from their historical answers across {num_icl_examples} independent interpersonal communication scenarios, and, if possible, use these preferences to predict their answers ({user_rating_prompt}) in a new scenario.

You will be provided with two sections:

<History>
This part contains {num_icl_examples} independent interpersonal communication scenarios, presented in shuffled order, along with the user's answers for each variant for the corresponding scenario.

For each scenario:
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants with the user's answers (YES or NO), presented in shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed. 

<Prediction>
This section contains a new interpersonal communication scenario the user has not seen before. It includes:
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants, presented in shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed.

Before making the prediction, follow these steps strictly: 
Step 1: Infer the user's preference patterns and summarize them concisely.
Step 2: Based only on this inferred preference, predict answers for the new scenario.
 
Return JSON exactly in this format:
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
}}
""",
        "internal_description": "HC (history + context)",
    },
    "NH": {
        "template": """
Your task is to predict the user's answer to ({user_rating_prompt}) for 9 disclosure variants given an interpersonal communication scenario.

You will be provided with the following section: 
<Prediction>
This section contains an interpersonal communication scenario. It includes: 
1. <scenario> The scenario context, including who is sharing what information, with whom, through which transmission principle. Additional context will be included if any.
2. <variants> 9 disclosure variants, presented in shuffled order: These variants are alternative ways the same underlying information could be shared, differing in how the information is disclosed.

Before making the prediction, reason about the current scenario and then predict answers.

Return JSON exactly in this format:
{{
"reasoning": "your 1-sentence reasoning about your prediction for the answers in the given scenario.",
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
}}
""",
        "internal_description": "_F (no history)",
    },
}


# System-prompt wording (may keep "the data sender" when context is redacted).
USER_RATING_PROMPT_GENERAL: Dict[str, TemplateEntry] = {
    "SENDER_HUMAN": {
        "template": "whether they feel comfortable sharing the information in this way",
        "internal_description": "Sender + Human",
    },
    "SUBJECT_HUMAN": {
        "template": (
            "whether they feel comfortable if the {data_sender} shared their "
            "information in this way"
        ),
        "internal_description": "Subject + Human",
    },
    "RECIPIENT_HUMAN": {
        "template": (
            "whether they feel comfortable if the {data_sender} shared the "
            "information with them in this way"
        ),
        "internal_description": "Recipient + Human",
    },
    "SENDER_AI": {
        "template": (
            "whether they feel comfortable if their AI assistant shared the "
            "information on their behalf in this way"
        ),
        "internal_description": "Sender + AI",
    },
    "SUBJECT_AI": {
        "template": (
            "whether they feel comfortable with an AI assistant sharing their "
            "information on behalf of the data sender"
        ),
        "internal_description": "Subject + AI (system)",
    },
    "RECIPIENT_AI": {
        "template": (
            "whether they feel comfortable with an AI assistant sharing the "
            "information with them on behalf of the data sender"
        ),
        "internal_description": "Recipient + AI (system)",
    },
}


# User-prompt wording (sender name when semantic context is available).
USER_RATING_PROMPT_USER: Dict[str, TemplateEntry] = {
    "SENDER_HUMAN": {
        "template": "whether they feel comfortable sharing the information in this way",
        "internal_description": "Sender + Human",
    },
    "SUBJECT_HUMAN": {
        "template": (
            "whether they feel comfortable if the {data_sender} shared their "
            "information in this way"
        ),
        "internal_description": "Subject + Human",
    },
    "RECIPIENT_HUMAN": {
        "template": (
            "whether they feel comfortable if the {data_sender} shared the "
            "information with them in this way"
        ),
        "internal_description": "Recipient + Human",
    },
    "SENDER_AI": {
        "template": (
            "whether they feel comfortable if their AI assistant shared the "
            "information on their behalf in this way"
        ),
        "internal_description": "Sender + AI",
    },
    "SUBJECT_AI": {
        "template": (
            "whether they feel comfortable if {data_sender}'s AI assistant shared "
            "their information in this way"
        ),
        "internal_description": "Subject + AI (user)",
    },
    "RECIPIENT_AI": {
        "template": (
            "whether they feel comfortable if {data_sender}'s AI assistant shared "
            "the information with them in this way"
        ),
        "internal_description": "Recipient + AI (user)",
    },
}


USER_TEMPLATE: Dict[str, TemplateEntry] = {
    "H": {
        "template": (
            "<History>\n"
            "{icl_scenario_description}\n"
            "</History>\n\n"
            "<Prediction>\n"
            "{prediction_scenario_description}\n"
            "</Prediction>"
        ),
        "internal_description": "history + prediction",
    },
    "NH": {
        "template": (
            "<Prediction>\n"
            "{prediction_scenario_description}\n"
            "</Prediction>"
        ),
        "internal_description": "prediction only",
    },
}


PREDICTION_SCENARIO_DESCRIPTION: TemplateEntry = {
    "template": (
        "<scenario>\n"
        "{scenario_content}"
        "{role_condition_description}\n"
        "</scenario>\n"
        "Predict the user's answer to {user_rating_prompt} for each of the 9 variants. "
        "Each answer is YES or NO. \n"
        "<variants>\n"
        "{random_ordered_1_content}:\n"
        "{random_ordered_2_content}:\n"
        "{random_ordered_3_content}:\n"
        "{random_ordered_4_content}:\n"
        "{random_ordered_5_content}:\n"
        "{random_ordered_6_content}:\n"
        "{random_ordered_7_content}:\n"
        "{random_ordered_8_content}:\n"
        "{random_ordered_9_content}:\n"
        "</variants>"
    ),
    "internal_description": "prediction block (full scenario + content)",
}


ICL_SCENARIO_DESCRIPTION: Dict[str, TemplateEntry] = {
    "HC": {
        "template": (
            "<scenario>\n"
            "{scenario_content}"
            "{role_condition_description}\n"
            "</scenario>\n"
            "The user answered YES or NO to {user_rating_prompt_user} for each of "
            "the 9 variants.\n"
            "<variants>\n"
            "{random_ordered_1_content}: {random_ordered_1_rating}\n"
            "{random_ordered_2_content}: {random_ordered_2_rating}\n"
            "{random_ordered_3_content}: {random_ordered_3_rating}\n"
            "{random_ordered_4_content}: {random_ordered_4_rating}\n"
            "{random_ordered_5_content}: {random_ordered_5_rating}\n"
            "{random_ordered_6_content}: {random_ordered_6_rating}\n"
            "{random_ordered_7_content}: {random_ordered_7_rating}\n"
            "{random_ordered_8_content}: {random_ordered_8_rating}\n"
            "{random_ordered_9_content}: {random_ordered_9_rating}\n"
            "</variants>\n"
        ),
        "internal_description": "HC history example",
    },
    "HL": {
        "template": (
            "<scenario>\n"
            "(scenario redacted)"
            "{role_condition_description}\n"
            "</scenario>\n"
            "The user answered YES or NO to {user_rating_prompt_user} for each of "
            "the 9 variants.\n"
            "<variants>\n"
            "[{random_ordered_1_label}]: {random_ordered_1_rating}\n"
            "[{random_ordered_2_label}]: {random_ordered_2_rating}\n"
            "[{random_ordered_3_label}]: {random_ordered_3_rating}\n"
            "[{random_ordered_4_label}]: {random_ordered_4_rating}\n"
            "[{random_ordered_5_label}]: {random_ordered_5_rating}\n"
            "[{random_ordered_6_label}]: {random_ordered_6_rating}\n"
            "[{random_ordered_7_label}]: {random_ordered_7_rating}\n"
            "[{random_ordered_8_label}]: {random_ordered_8_rating}\n"
            "[{random_ordered_9_label}]: {random_ordered_9_rating}\n"
            "</variants>\n"
        ),
        "internal_description": "HL history example",
    },
    "H": {
        "template": (
            "<scenario>\n"
            "(scenario content redacted)"
            "{role_condition_description}\n"
            "</scenario>\n"
            "The user answered YES or NO to {user_rating_prompt_user} for each of "
            "the 9 variants.\n"
            "<variants>\n"
            "(variant content redacted):{random_ordered_1_rating}\n"
            "(variant content redacted):{random_ordered_2_rating}\n"
            "(variant content redacted):{random_ordered_3_rating}\n"
            "(variant content redacted):{random_ordered_4_rating}\n"
            "(variant content redacted):{random_ordered_5_rating}\n"
            "(variant content redacted):{random_ordered_6_rating}\n"
            "(variant content redacted):{random_ordered_7_rating}\n"
            "(variant content redacted):{random_ordered_8_rating}\n"
            "(variant content redacted):{random_ordered_9_rating}\n"
            "</variants>\n"
        ),
        "internal_description": "H history example",
    },
}
