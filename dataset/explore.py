"""Dataset exploration, plotting acceptance and inter-user agreement)."""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator, PercentFormatter

DATASET_DIR = Path(__file__).resolve().parent
ROOT = DATASET_DIR.parent
sys.path.insert(0, str(ROOT / "evaluation" / "helper"))

from constants import RATINGS_PATH, VARIANT_SUFFIXES
from load_data import rating_to_yes_no, read_table

CONDITIONS = ("AI", "Human")
ROLES = ("data_sender", "data_subject", "data_recipient")
OUTPUT_DIR = ROOT / "outputs" / "dataset"
DEFAULT_HEATMAP_PATH = OUTPUT_DIR / "variant_level_yes_rates.png"
AGREEMENT_HEATMAP_STEM = "inter_user_agreement_heatmap_{role}.png"
FONT_SIZE_LARGE = 18
FONT_SIZE_SMALL = 13
_FONT_RC = {
    "font.size": FONT_SIZE_LARGE,
    "axes.titlesize": FONT_SIZE_LARGE,
    "axes.labelsize": FONT_SIZE_LARGE,
    "xtick.labelsize": FONT_SIZE_LARGE,
    "ytick.labelsize": FONT_SIZE_LARGE,
    "legend.fontsize": FONT_SIZE_LARGE,
    "figure.titlesize": FONT_SIZE_LARGE,
}


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


def variant_yes_rate_matrix(
    ratings: pd.DataFrame,
    *,
    role: str,
    condition: str,
) -> np.ndarray:
    role_col, cond_col = _role_condition_cols(ratings)
    sub = ratings[(ratings[role_col] == role) & (ratings[cond_col] == condition)]
    mat = np.full((3, 3), np.nan, dtype=float)
    for suffix in VARIANT_SUFFIXES:
        g = int(suffix[0]) - 1
        i = int(suffix[1]) - 1
        parts = []
        for si in range(10):
            id_col = f"s{si}_id"
            col = f"s{si}_var{suffix}_rating"
            if id_col not in sub.columns or col not in sub.columns:
                continue
            mask = sub[id_col].notna()
            parts.append(_to_yes_no(sub.loc[mask, col]))
        vals = pd.concat(parts, ignore_index=True) if parts else pd.Series(dtype=str)
        if len(vals):
            mat[g, i] = float((vals == "YES").mean())
    return mat


def plot_variant_level_yes_rates(
    ratings: pd.DataFrame,
    output_path: str | Path | None = None,
    *,
    show: bool = False,
    close: bool = True,
) -> Path:
    output_path = Path(output_path) if output_path is not None else DEFAULT_HEATMAP_PATH
    n = 3
    x = np.arange(n + 1) - 0.5
    y = np.arange(n + 1) - 0.5

    with plt.rc_context(_FONT_RC):
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0), sharex=True, sharey=True)
        pcm = None
        for r, condition in enumerate(CONDITIONS):
            for c, role in enumerate(ROLES):
                ax = axes[r, c]
                mat = variant_yes_rate_matrix(ratings, role=role, condition=condition)
                pcm = ax.pcolormesh(
                    x,
                    y,
                    np.ma.masked_invalid(mat),
                    cmap="viridis",
                    vmin=0.0,
                    vmax=1.0,
                    shading="flat",
                    edgecolors="white",
                    linewidth=2.25,
                )
                for gi in range(n):
                    for ii in range(n):
                        v = mat[gi, ii]
                        if np.isnan(v):
                            continue
                        ax.text(
                            ii,
                            gi,
                            f"{100.0 * v:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=FONT_SIZE_SMALL,
                            color="white",
                            fontweight="medium",
                        )
                ax.set_ylim(n - 0.5, -0.5)
                ax.set_aspect("equal")
                ax.set_xticks(range(n), [f"I{i + 1}" for i in range(n)])
                ax.set_yticks(range(n), [f"G{g + 1}" for g in range(n)])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(False)
                if r == 0:
                    ax.set_title(role, fontsize=FONT_SIZE_LARGE)
                if c == 0:
                    ax.set_ylabel(condition, fontsize=FONT_SIZE_LARGE)

        fig.suptitle(
            "Average Yes rate by variant (G×I) aggregated over contexts",
            fontsize=FONT_SIZE_LARGE,
        )
        fig.subplots_adjust(right=0.90, top=0.90)
        cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label("Yes Rate", fontsize=FONT_SIZE_LARGE)
        cb.ax.tick_params(labelsize=FONT_SIZE_LARGE)
        cb.ax.yaxis.set_major_locator(MultipleLocator(0.25))
        cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        for spine in cb.ax.spines.values():
            spine.set_visible(False)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {output_path}")
        if show:
            plt.show()
        if close:
            plt.close(fig)
    return output_path


def _complete_boundary_vector(row: pd.Series, slot: int) -> np.ndarray | None:
    """Length-9 binary vector (1=YES, 0=NO), or None if any variant is missing."""
    vec: list[int] = []
    for suffix in VARIANT_SUFFIXES:
        col = f"s{slot}_var{suffix}_rating"
        if col not in row.index or pd.isna(row[col]):
            return None
        try:
            label = rating_to_yes_no(row[col])
        except ValueError:
            return None
        vec.append(1 if label == "YES" else 0)
    return np.asarray(vec, dtype=np.int8)


def _mean_pairwise_agreement(mat: np.ndarray) -> float:
    """Mean over unordered user pairs of fraction of matching variants."""
    n = int(mat.shape[0])
    if n < 2:
        return float("nan")
    scores = [
        float((mat[i] == mat[j]).mean())
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(scores))


def collect_complete_boundaries(
    ratings: pd.DataFrame,
) -> dict[tuple[str, str, str], list[np.ndarray]]:
    """Map (scenario_id, role, condition) → list of complete 9-variant vectors."""
    role_col, cond_col = _role_condition_cols(ratings)
    contexts: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    for _, row in ratings.iterrows():
        role = str(row[role_col])
        condition = str(row[cond_col])
        for si in range(10):
            id_col = f"s{si}_id"
            if id_col not in row.index or pd.isna(row[id_col]):
                continue
            vec = _complete_boundary_vector(row, si)
            if vec is None:
                continue
            contexts[(str(row[id_col]), role, condition)].append(vec)
    return contexts


def inter_user_agreement_matrix(
    ratings: pd.DataFrame,
    *,
    role: str,
) -> tuple[np.ndarray, list[str]]:
    """Return (2 × n_scenarios) agreement matrix and sorted scenario_id columns.

    Rows follow ``CONDITIONS`` (AI, Human). Cells with fewer than 2 users are NaN.
    """
    contexts = collect_complete_boundaries(ratings)
    scenario_ids = sorted({sid for (sid, r, _cond) in contexts if r == role})
    mat = np.full((len(CONDITIONS), len(scenario_ids)), np.nan, dtype=float)
    for c, condition in enumerate(CONDITIONS):
        for s, sid in enumerate(scenario_ids):
            vecs = contexts.get((sid, role, condition), [])
            if len(vecs) < 2:
                continue
            mat[c, s] = _mean_pairwise_agreement(np.stack(vecs, axis=0))
    return mat, scenario_ids


def plot_inter_user_agreement(
    ratings: pd.DataFrame,
    output_dir: str | Path | None = None,
    *,
    show: bool = False,
    close: bool = True,
) -> list[Path]:
    """One inter-user agreement heatmap per role (condition × scenario)."""
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for role in ROLES:
        mat, scenario_ids = inter_user_agreement_matrix(ratings, role=role)
        n_scenarios = len(scenario_ids)
        if n_scenarios == 0:
            print(f"Skipping {role}: no contexts with ≥2 complete boundaries")
            continue

        # Fixed height; only width scales with the number of scenarios.
        fig_h = 5.0
        fig_w = max(10.0, n_scenarios * 0.42)
        with plt.rc_context(_FONT_RC):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
                cmap="magma",
            )
            ax.set_yticks(range(len(CONDITIONS)), list(CONDITIONS))
            ax.set_xticks(
                range(n_scenarios),
                scenario_ids,
                rotation=75,
                ha="right",
                fontsize=FONT_SIZE_LARGE,
            )
            ax.tick_params(axis="both", labelsize=FONT_SIZE_LARGE)
            ax.set_ylabel("AI-mediated condition", fontsize=FONT_SIZE_LARGE)
            ax.set_xlabel("Scenario ID", fontsize=FONT_SIZE_LARGE)
            ax.set_title(f"Inter-user agreement — {role}", fontsize=FONT_SIZE_LARGE)
            cb = fig.colorbar(im, ax=ax, fraction=0.02)
            cb.set_label("Agreement", fontsize=FONT_SIZE_LARGE)
            cb.ax.tick_params(labelsize=FONT_SIZE_LARGE)
            # Lock axes vertical margins so exported height stays constant across roles.
            fig.subplots_adjust(left=0.10, right=0.92, bottom=0.38, top=0.88)

            path = output_dir / AGREEMENT_HEATMAP_STEM.format(role=role)
            # Avoid bbox_inches="tight" so canvas height stays exactly fig_h for every role.
            fig.savefig(path, dpi=200)
            print(f"Wrote {path}")
            written.append(path)
            if show:
                plt.show()
            if close:
                plt.close(fig)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Directory for plot PNGs (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--yes-rates-only",
        action="store_true",
        help="Only write the G×I P(Yes) heatmap",
    )
    parser.add_argument(
        "--agreement-only",
        action="store_true",
        help="Only write the inter-user agreement heatmaps",
    )
    args = parser.parse_args()

    if args.yes_rates_only and args.agreement_only:
        raise SystemExit("Use only one of --yes-rates-only / --agreement-only")

    assert RATINGS_PATH.is_file(), f"Missing ratings table: {RATINGS_PATH}"
    ratings = read_table(RATINGS_PATH)
    out = Path(args.output_dir)

    if not args.agreement_only:
        plot_variant_level_yes_rates(ratings, output_path=out / DEFAULT_HEATMAP_PATH.name)
    if not args.yes_rates_only:
        plot_inter_user_agreement(ratings, output_dir=out)


if __name__ == "__main__":
    main()
