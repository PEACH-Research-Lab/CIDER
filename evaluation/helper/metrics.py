"""Metrics: accuracy, trivial baselines, and variant-level FP/FN."""

from __future__ import annotations

import numpy as np
import pandas as pd

from constants import SEED, USER_ID_COL, VARIANT_LABEL_MAP, VARIANT_SUFFIXES
from load_data import binary_rating_to_numeric


PREDICTION_USER_ID_COLS = ("user_id", USER_ID_COL)
SCENARIO_ID_COL = "scenario_id"


def _user_id_column(df: pd.DataFrame) -> str:
    for col in PREDICTION_USER_ID_COLS:
        if col in df.columns:
            return col
    raise KeyError(
        "Prediction table has no user id column; expected one of "
        f"{list(PREDICTION_USER_ID_COLS)}"
    )


def _scored_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form (user, scenario, variant, truth, pred) rows with numeric labels."""
    user_col = _user_id_column(df)
    scenario = (
        df[SCENARIO_ID_COL].astype(str)
        if SCENARIO_ID_COL in df.columns
        else pd.Series("", index=df.index)
    )
    parts: list[pd.DataFrame] = []
    for s in VARIANT_SUFFIXES:
        true_col, pred_col = f"true_var{s}", f"pred_var{s}"
        if true_col not in df.columns or pred_col not in df.columns:
            continue
        truth = df[true_col].map(binary_rating_to_numeric)
        pred = df[pred_col].map(binary_rating_to_numeric)
        keep = truth.notna() & pred.notna()
        parts.append(
            pd.DataFrame(
                {
                    "user": df[user_col],
                    "scenario": scenario,
                    "variant": s,
                    "truth": truth,
                    "pred": pred,
                }
            )[keep]
        )

    if not parts:
        raise ValueError(
            "Prediction table has no true_var*/pred_var* column pairs to score "
            f"(expected suffixes: {', '.join(VARIANT_SUFFIXES)})"
        )
    pairs = pd.concat(parts, ignore_index=True)
    if pairs.empty:
        raise ValueError("No ground-truth ratings to score in prediction table")
    return pairs


def _canonical_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = _scored_pairs(df)
    keys = pairs[["user", "scenario", "variant"]].astype(str)
    order = keys.sort_values(["user", "scenario", "variant"], kind="stable").index
    return pairs.loc[order].reset_index(drop=True)


def _user_macro(correct: pd.Series, users: pd.Series) -> pd.Series:
    """Per-user accuracy. Averaging these weights every user equally."""
    return correct.groupby(users).mean()


# ----------------------------------------------------
# 1. Accuracy
# ----------------------------------------------------

def compute_dataset_metrics(df: pd.DataFrame) -> dict[str, float]:
    pairs = _scored_pairs(df)
    correct = pairs["pred"] == pairs["truth"]
    per_user = _user_macro(correct, pairs["user"])
    return {
        "accuracy": float(per_user.mean()),
        "n_users": int(per_user.size),
        "n_scored": int(len(pairs)),
        "n_correct": int(correct.sum()),
    }


# ----------------------------------------------------
# 2. Trivial baselines (Always YES / Always NO / Random)
# ----------------------------------------------------

def constant_baseline_accuracies(df: pd.DataFrame) -> dict[str, float]:
    pairs = _scored_pairs(df)
    return {
        label: float(_user_macro(pairs["truth"] == rating, pairs["user"]).mean())
        for label, rating in (("Always YES", 1.0), ("Always NO", 2.0))
    }


def random_baseline_accuracy(
    df: pd.DataFrame, *, seed: int = SEED, n_repeats: int = 1
) -> float:
    pairs = _canonical_pairs(df)
    truth = pairs["truth"].to_numpy()
    rng = np.random.default_rng(seed)
    scores = [
        float(
            _user_macro(
                pd.Series(rng.choice((1.0, 2.0), size=len(pairs)) == truth),
                pairs["user"],
            ).mean()
        )
        for _ in range(max(1, int(n_repeats)))
    ]
    return float(np.mean(scores))


def baseline_accuracies(
    df: pd.DataFrame, *, seed: int = SEED, n_repeats: int = 1
) -> dict[str, float]:
    if df is None:
        return {}
    return {
        **constant_baseline_accuracies(df),
        "Random": random_baseline_accuracy(df, seed=seed, n_repeats=n_repeats),
    }


# ----------------------------------------------------
# 3. Variant-level FP / FN
# ----------------------------------------------------
def variant_fp_fn_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-variant TP/FP/FN/TN counts and FP/FN rates."""
    pairs = _scored_pairs(df)
    rows: list[dict] = []
    for s in VARIANT_SUFFIXES:
        sub = pairs[pairs["variant"] == s]
        if sub.empty:
            continue
        true, pred = sub["truth"], sub["pred"]
        tp = int(((pred == 1) & (true == 1)).sum())
        fp = int(((pred == 1) & (true == 2)).sum())
        fn = int(((pred == 2) & (true == 1)).sum())
        tn = int(((pred == 2) & (true == 2)).sum())
        rows.append(
            {
                "variant": s,
                "variant_label": VARIANT_LABEL_MAP.get(s, s),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "fp_rate": float(fp) / float(fp + tn) if (fp + tn) else 0.0,
                "fn_rate": float(fn) / float(fn + tp) if (fn + tp) else 0.0,
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)
