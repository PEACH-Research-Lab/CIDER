#!/usr/bin/env python3
"""Dataset quick start: explore CIDER tables and optionally generate variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent
ROOT = DATASET_DIR.parent
sys.path.insert(0, str(ROOT / "evaluation" / "helper"))
sys.path.insert(0, str(DATASET_DIR / "extensibility"))
sys.path.insert(0, str(DATASET_DIR))

from constants import (
    ORIGINAL_DIR,
    RATINGS_PATH,
    SCENARIOS_PATH,
    VARIANT_LABEL_MAP,
    VARIANT_SUFFIXES,
    VISUAL_CARDS_DIR,
)
from explore import plot_inter_user_agreement, plot_variant_level_yes_rates
from load_data import rating_to_yes_no, read_table

# Example seeds matching dataset/quick_start.ipynb §3
CHOICE_A_EXAMPLE = {
    "name": "quick-start-example-1",
    "data_type": "the client's personal therapy experience details",
    "sensitive_info_items": [
        "Emily is a college student with severe test anxiety.",
        "Before and during exams, Emily experiences a rapid heartbeat, excessive sweating, and overwhelming thoughts of failing.",
        "Her therapy focuses on grounding and anxiety-management techniques.",
    ],
    "data_sender": "Alex (Emily's music therapist)",
    "data_subject": "Emily (client and college student)",
    "data_recipient": "Alex's social-media followers",
    "transmission_principle": "a Facebook post",
    "story": (
        "A music therapist shares a client's personal therapy experience "
        "with social-media followers in a Facebook post."
    ),
}

PRIVACYLENS_SEED = {
    "name": "main404",
    "data_type": "work history",
    "data_subject": "a famous client",
    "data_sender": "music therapist",
    "data_sender_name": "Jane",
    "data_recipient": "social media followers",
    "transmission_principle": "send a facebook post",
    "data_type_concrete": (
        "Detailed notes on sessions with a famous client, including their "
        "extensive and confidential work history in the music industry."
    ),
    "data_subject_concrete": "A famous client",
    "data_sender_concrete": "Jane",
    "data_recipient_concrete": "Social media followers",
    "data_subject_name": "David Melrose",
    "data_recipient_name": "Social media followers",
    "sensitive_info": [
        "David Melrose is a famous pop star.",
        "David has struggled with the pressures of public performances.",
        "Our sessions have been focused on developing personalized strategies to enhance his stage presence and confidence.",
        "Since beginning therapy, there has been a measurable improvement in his ability to perform under pressure.",
        "David has reported a higher level of satisfaction with his musical engagements.",
    ],
}

PRIVACYLENS_STORY = (
    "A music therapist shares a famous client's personal therapy experience "
    "with social-media followers in a Facebook post."
)


def _heading(title: str) -> None:
    print(f"\n=== {title} ===")


def _to_yes_no(series: pd.Series) -> pd.Series:
    out = []
    for value in series.dropna():
        try:
            out.append(rating_to_yes_no(value))
        except ValueError:
            continue
    return pd.Series(out, dtype=str)


def _role_condition_cols(ratings: pd.DataFrame) -> tuple[str, str]:
    role_col = "Role" if "Role" in ratings.columns else "role"
    cond_col = "Condition" if "Condition" in ratings.columns else "condition"
    return role_col, cond_col


def explore() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tables, print shapes, and peek at one scenario's 9 variants."""
    assert RATINGS_PATH.is_file() and SCENARIOS_PATH.is_file()
    _heading("1. Explore CIDER dataset")
    print(f"original dir: {ORIGINAL_DIR}")

    ratings = read_table(RATINGS_PATH)
    scenarios = read_table(SCENARIOS_PATH)

    print(f"ratings:   {ratings.shape[0]} rows × {ratings.shape[1]} cols")
    print(f"scenarios: {scenarios.shape[0]} rows × {scenarios.shape[1]} cols")
    print(f"users:     {ratings['Participant_ID'].nunique()}")
    print(f"scenario_ids in scenarios: {scenarios['scenario_id'].nunique()}")
    print(f"visual cards: {len(list(VISUAL_CARDS_DIR.glob('*.png')))}")
    print(f"loaded from: {RATINGS_PATH.name}, {SCENARIOS_PATH.name}")

    print("\nRatings (head):")
    print(ratings.head(3).to_string())

    scenario_preview_cols = [
        c
        for c in [
            "scenario_id",
            "sender_format",
            "subject_format",
            "recipient_format",
            "transmission_principle",
            "3rd_narrative",
        ]
        if c in scenarios.columns
    ]
    print("\nScenarios (selected columns):")
    print(scenarios[scenario_preview_cols].head(3).to_string())

    sid = scenarios["scenario_id"].iloc[0]
    row = scenarios.loc[scenarios["scenario_id"] == sid].iloc[0]
    print(f"\nScenario `{sid}`")
    print(row.get("3rd_narrative", ""))
    print()
    for suffix in VARIANT_SUFFIXES:
        col = f"variant_{suffix}_cleaned"
        if col not in row.index:
            col = f"variant_{suffix}"
        label = VARIANT_LABEL_MAP[suffix]
        text = row[col] if col in row.index else ""
        print(f"[{label} / var{suffix}] {text}")

    print("\nRole visual cards:")
    for role in ("data_sender", "data_subject", "data_recipient"):
        png = VISUAL_CARDS_DIR / f"{sid}_{role}.png"
        status = "found" if png.is_file() else "missing"
        print(f"  {role}: {png.name} ({status})")

    return ratings, scenarios


def analyze(ratings: pd.DataFrame, *, plot: bool = True) -> None:
    """Role×condition counts, slots per user, Yes rates, and G×I heatmap."""
    _heading("2. Dataset analysis")

    role_col, cond_col = _role_condition_cols(ratings)

    print("\nRole × condition counts:")
    print(pd.crosstab(ratings[role_col], ratings[cond_col], margins=True).to_string())

    slot_cols = [c for c in ratings.columns if c.startswith("s") and c.endswith("_id")]
    n_slots = ratings[slot_cols].notna().sum(axis=1)
    print("\nScenarios rated per user:")
    print(n_slots.value_counts().sort_index().rename("n_users").to_frame().to_string())

    flat_parts = []
    for si in range(10):
        id_col = f"s{si}_id"
        if id_col not in ratings.columns:
            continue
        mask = ratings[id_col].notna()
        for suffix in VARIANT_SUFFIXES:
            col = f"s{si}_var{suffix}_rating"
            if col not in ratings.columns:
                continue
            flat_parts.append(_to_yes_no(ratings.loc[mask, col]))

    flat = pd.concat(flat_parts, ignore_index=True) if flat_parts else pd.Series(dtype=str)
    print("\nOverall Yes/No (slots with non-null scenario id only):")
    print(flat.value_counts(normalize=True).rename("share").to_frame().to_string())
    print(f"Overall Yes rate: {(flat == 'YES').mean():.2%}  (n={len(flat)})")

    rows = []
    for suffix in VARIANT_SUFFIXES:
        parts = []
        for si in range(10):
            id_col = f"s{si}_id"
            col = f"s{si}_var{suffix}_rating"
            if id_col not in ratings.columns or col not in ratings.columns:
                continue
            mask = ratings[id_col].notna()
            parts.append(_to_yes_no(ratings.loc[mask, col]))
        vals = pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=str)
        yes = (vals == "YES").mean() if len(vals) else float("nan")
        rows.append(
            {
                "variant": suffix,
                "label": VARIANT_LABEL_MAP[suffix],
                "n": int(len(vals)),
                "yes_rate": float(yes),
            }
        )
    print("\nYes-rate by disclosure variant (G×I):")
    print(pd.DataFrame(rows).to_string(index=False))

    if plot:
        print("\nAverage Yes rate heatmap by variant (G×I), role × condition:")
        plot_variant_level_yes_rates(ratings)
        print("\nInter-user agreement heatmaps (condition × scenario, per role):")
        plot_inter_user_agreement(ratings)


def generate_choice_a() -> None:
    """Choice A: generate 9 variants from a natural-language description."""
    from script import generate_from_description, normalize_description, resolve_chat_credentials

    _heading("3.1 Choice A: natural-language description")
    api_key, base_url, model = resolve_chat_credentials(env_file=ROOT / ".env")
    print(f"Generation ready: model={model} base_url={base_url}")
    print(normalize_description(CHOICE_A_EXAMPLE))
    result = generate_from_description(
        CHOICE_A_EXAMPLE,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    print("\n" + result["output"])


def generate_choice_b() -> None:
    """Choice B: generate 9 variants from a PrivacyLens seed + story."""
    from script import (
        generate_from_privacylens_row,
        privacy_lens_row_to_description,
        resolve_chat_credentials,
    )

    _heading("3.2 Choice B: PrivacyLens seed")
    api_key, base_url, model = resolve_chat_credentials(env_file=ROOT / ".env")
    print(f"Generation ready: model={model} base_url={base_url}")
    print(
        privacy_lens_row_to_description(
            PRIVACYLENS_SEED,
            story=PRIVACYLENS_STORY,
        )
    )
    result = generate_from_privacylens_row(
        PRIVACYLENS_SEED,
        story=PRIVACYLENS_STORY,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    print("\n" + result["output"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explore-only",
        action="store_true",
        help="Load and peek at tables only (skip analysis)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Run analysis only (still loads tables)",
    )
    parser.add_argument(
        "--generate-a",
        action="store_true",
        help="Also run Choice A example generation (needs API key)",
    )
    parser.add_argument(
        "--generate-b",
        action="store_true",
        help="Also run Choice B PrivacyLens example generation (needs API key)",
    )
    args = parser.parse_args()

    if args.explore_only and args.analyze_only:
        raise SystemExit("Use only one of --explore-only / --analyze-only")

    ratings, _scenarios = explore() if not args.analyze_only else (
        read_table(RATINGS_PATH),
        read_table(SCENARIOS_PATH),
    )
    if args.analyze_only:
        _heading("1. Explore CIDER dataset")
        print("(skipped; --analyze-only)")

    if not args.explore_only:
        analyze(ratings)

    if args.generate_a:
        generate_choice_a()
    if args.generate_b:
        generate_choice_b()


if __name__ == "__main__":
    main()
