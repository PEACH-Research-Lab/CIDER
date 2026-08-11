"""Shared constants and default dataset paths (dataset + evaluation)."""

from __future__ import annotations
from pathlib import Path

# ------------------------------
# CIDER paths, dataset info, extensibility.
# ------------------------------
# evaluation/helper/constants.py → parents[2] is the repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = REPO_ROOT / "dataset" / "original"
RATINGS_PATH = ORIGINAL_DIR / "user_ratings.jsonl"
SCENARIOS_PATH = ORIGINAL_DIR / "scenarios.jsonl"
VISUAL_CARDS_DIR = ORIGINAL_DIR / "visual_card_artifacts"

# Drop unused demographics for prediction.
COLUMNS_TO_DROP = [
    "Age",
    "Sex",
    "education_level",
    "AI_use",
    "AI_q1",
    "AI_q2",
    "AI_q3",
    "AI_q4",
    "AI_attitude",
    "Privacy_q1",
    "Privacy_q2",
    "Privacy_q3",
    "Privacy_q4",
    "Privacy_q5",
    "Privacy_q6",
    "Privacy_q7",
    "Privacy_q8",
    "Privacy_q9",
    "Privacy_q10",
    "Privacy_q11",
    "Privacy_q12",
]

USER_ID_COL = "Participant_ID"
VARIANT_SUFFIXES = ["11", "12", "13", "21", "22", "23", "31", "32", "33"]
ROLE_TO_IMAGE_KEY = {
    "data_sender": "image_description_sender",
    "data_subject": "image_description_subject",
    "data_recipient": "image_description_recipient",
}

VARIANT_LABEL_MAP = {
    "11": "G1-I1",
    "12": "G1-I2",
    "13": "G1-I3",
    "21": "G2-I1",
    "22": "G2-I2",
    "23": "G2-I3",
    "31": "G3-I1",
    "32": "G3-I2",
    "33": "G3-I3",
}

# Variant generation choice A: Natural language description of norm-violating case.
REQUIRED_ITEM_FIELDS = (
    "name",
    "data_type",
    "sensitive_info_items",
    "data_sender",
    "data_subject",
    "data_recipient",
    "transmission_principle",
    "story",  # third-person narrative
)

# Variant generation choice B: PrivacyLens.
PRIVACYLENS_FIELDS = (
    "name",
    "data_sender",
    "data_sender_name",
    "data_sender_concrete",
    "data_subject",
    "data_subject_name",
    "data_subject_concrete",
    "data_recipient",
    "data_recipient_name",
    "data_recipient_concrete",
    "transmission_principle",
    "sensitive_info",
)

GENERATION_BASE_URL = "https://api.openai.com/v1"
GENERATION_MODEL = "o3"
GENERATION_MODEL_OPENROUTER = "openai/o3"

# ------------------------------
# Evaluation task
# ------------------------------

SEED = 42
N_ICL_EXAMPLES = 6
MIN_SCENARIOS = 8

DEFAULT_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROVIDERS = ("openai", "openrouter", "external")
DEFAULT_PROVIDER = "openai"
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
# Optional per-model overrides in JSON configs / run rows.
MODEL_BACKEND_FIELDS = ("provider", "base_url", "api_key_env")

# Setups (canonical tags; old ``*_F`` aliases still accepted)
SETUP_DEFAULT = "Baseline"
SETUP_ORDER = ["Baseline", "H", "HL", "HC"]
# Predictions that do not depend on ICL k: one file per model, reused across ks.
K_INDEPENDENT_SETUPS = frozenset({"Baseline"})
SETUP_LABELS = {
    "Baseline": "Baseline (no history)",
    "H": "H (history, redacted)",
    "HL": "HL (history + labels)",
    "HC": "HC (history + context)",
}
SETUP_ALIASES = {
    "baseline": "Baseline",
    "h": "H",
    "hl": "HL",
    "hc": "HC",
}

def normalize_setup(raw: str) -> str:
    key = raw.strip().lower()
    if key not in SETUP_ALIASES:
        known = ", ".join(dict.fromkeys(SETUP_ALIASES.values()))
        raise ValueError(
            f"Unknown setup {raw!r}; expected one of: {known}"
        )
    return SETUP_ALIASES[key]

# ------------------------------
# Plot and analysis
# ------------------------------
DEFAULT_KS = (1, 4, 5, 6)
MODEL_ORDER: list[tuple[str, str]] = [
    ("gpt5.4", "GPT-5.4"),
    ("claude-sonnet-4.6", "Claude Sonnet 4.6"),
    ("deepseek-3.2", "DeepSeek-V3.2"),
    ("gpt5.4-nano", "GPT-5.4 nano"),
    ("llama4-maverick", "Llama 4 Maverick"),
    ("llama4-scout", "Llama 4 Scout"),
    ("qwen3.5-9", "Qwen3.5-9B"),
    ("qwen3-32", "Qwen3-32B"),
    ("qwen3-14", "Qwen3-14B"),
    ("qwen3-8", "Qwen3-8B"),
    ("ministral3-8b", "Ministral 3 8B"),
    ("llama3.1", "Llama 3.1 8B"),
]

MODEL_NAMES = [m for m, _ in MODEL_ORDER]
MODEL_LABEL_MAP = {m: label for m, label in MODEL_ORDER}

# Optional $/token rates for ``usage_cost_table_usd`` (keys = API model ids).
# When empty, table cost is left null; OpenRouter-reported cost is still stored.
COST: dict[str, dict[str, float]] = {}
MODEL_DISPLAY_COLORS = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#984EA3",
    "#E6AB02",
    "#FF7F00",
    "#A65628",
    "#666666",
    "#80B1D3",
    "#FCCDE5",
    "#F781BF",
    "#1B9E77",
]