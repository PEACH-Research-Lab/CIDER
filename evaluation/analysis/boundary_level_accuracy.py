"""Boundary-level accuracy analysis and visualization."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR / "helper"))
sys.path.insert(0, str(EVAL_DIR))

from config import load_json_config
from constants import (
    DEFAULT_KS,
    K_INDEPENDENT_SETUPS,
    SEED,
    SETUP_ORDER,
)
from load_data import list_k_values, load_runs
from metrics import baseline_accuracies, compute_dataset_metrics
from plotting import (
    BAR_FILL_ALPHAS,
    BAR_HATCHES,
    bar_face_edge,
    despine,
    draw_baselines,
    model_color,
    model_label,
    padded_limits,
    save_figure,
    setup_label,
    setup_slug,
    sort_models,
)
from utils import write_analysis_csv


# change these to the accuracy range you want to plot
_LINE_FLOOR, _LINE_CEIL = 48.0, 76.0
_BAR_FLOOR, _BAR_CEIL = 40.0, 80.0


# ---------------------------------------------------------------------------
# Calculations (metrics)
# ---------------------------------------------------------------------------
def score_runs(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Score loaded runs with ``compute_dataset_metrics``; keep one df for baselines."""
    rows: list[dict] = []
    gt_df: pd.DataFrame | None = None

    for run in runs:
        df = run["df"]
        if gt_df is None:
            gt_df = df
        metrics = compute_dataset_metrics(df)
        k = run.get("k")
        row: dict = {"model": run["model"], "setup": str(run["setup"])}
        if k is not None:
            row["k"] = int(k)
        row.update(
            accuracy=float(metrics["accuracy"]),
            n_users=int(metrics["n_users"]),
            n_scored=int(metrics["n_scored"]),
            n_rows=int(len(df)),
        )
        rows.append(row)
        print(
            f"{f'k={k} ' if k is not None else ''}{row['model']} [{row['setup']}]: "
            f"accuracy={row['accuracy']:.4f} "
            f"(n_users={row['n_users']}, n_scored={row['n_scored']})"
        )

    out = pd.DataFrame(rows)
    out.attrs["gt_df"] = gt_df
    return out, gt_df


def expand_accuracy_grid(
    long_df: pd.DataFrame,
    *,
    models: list[str],
    setups: list[str],
    ks: list[int],
) -> pd.DataFrame:
    """Full model × setup × k table; missing runs stay blank (NaN)."""
    if not models or not setups:
        return long_df.copy()

    cols = ["model", "setup", "k", "accuracy", "n_users", "n_scored", "n_rows"]
    if long_df.empty:
        scored = pd.DataFrame(columns=cols)
    else:
        scored = long_df.copy()
        if "k" not in scored.columns:
            scored["k"] = pd.NA
        scored = scored.reindex(columns=cols)

    if not ks:
        # No k axis (flat layout): one row per model × setup.
        index = pd.MultiIndex.from_product([models, setups], names=["model", "setup"])
        out = (
            scored.drop(columns=["k"], errors="ignore")
            .drop_duplicates(subset=["model", "setup"], keep="first")
            .set_index(["model", "setup"])
            .reindex(index)
            .reset_index()
        )
        return out

    index = pd.MultiIndex.from_product(
        [models, setups, ks], names=["model", "setup", "k"]
    )
    out = (
        scored.dropna(subset=["k"])
        .assign(k=lambda d: d["k"].astype(int))
        .drop_duplicates(subset=["model", "setup", "k"], keep="first")
        .set_index(["model", "setup", "k"])
        .reindex(index)
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _spread_label_ys(
    ys: list[float],
    *,
    min_gap: float,
    y_lo: float,
    y_hi: float,
) -> list[float]:
    """Push label y-positions apart by at least *min_gap*, clamped to [y_lo, y_hi]."""
    if not ys:
        return []
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [float(ys[i]) for i in order]
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    if placed[-1] > y_hi:
        placed[-1] = y_hi
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = min(placed[i], placed[i + 1] - min_gap)
    if placed[0] < y_lo:
        placed[0] = y_lo
        for i in range(1, len(placed)):
            placed[i] = max(placed[i], placed[i - 1] + min_gap)
    out = [0.0] * len(ys)
    for rank, i in enumerate(order):
        out[i] = placed[rank]
    return out


def plot_accuracy_across_setups(
    long_df: pd.DataFrame,
    baselines: dict[str, float],
    output_dir: Path,
    *,
    k: int,
    annotate_n: bool = True,
) -> Path | None:
    """One line per model across Baseline → H → HL → HC at fixed *k*."""

    setups = [s for s in SETUP_ORDER if s in set(long_df["setup"].astype(str))]
    models = sort_models(long_df["model"])
    if not setups or not models:
        return None

    acc: dict[tuple[str, str], float] = {}
    for r in long_df.itertuples():
        if pd.isna(r.accuracy):
            continue
        key = (str(r.model), str(r.setup))
        acc[key] = float(r.accuracy) * 100.0

    x_index = {s: i for i, s in enumerate(setups)}
    plotted: list[float] = []
    # Endpoint (near HC / last condition) used for right-side model labels.
    end_y: dict[str, float] = {}

    height = 6.0 if len(models) <= 8 else 9.0
    rc_paper = {
        "font.size": 19,
        "axes.titlesize": 19,
        "axes.labelsize": 19,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
    }
    label_x_offset = 0.22
    label_min_gap = 1.55
    leader_linewidth = 0.8
    leader_alpha = 0.55

    with plt.rc_context(rc_paper):
        fig, ax = plt.subplots(figsize=(9.0, height))
        for m in models:
            pts = [(x_index[s], acc[(m, s)]) for s in setups if (m, s) in acc]
            if not pts:
                continue
            xs, ys = zip(*pts)
            plotted.extend(ys)
            end_y[m] = float(ys[-1])
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=6,
                linewidth=1.8,
                linestyle="-",
                color=model_color(m),
                zorder=2,
            )
            if annotate_n:
                for s in setups:
                    key = (m, s)
                    if key not in acc:
                        continue
                    ax.annotate(
                        f"{acc[key]:.1f}",
                        (x_index[s], acc[key]),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                        color=model_color(m),
                        alpha=0.85,
                        zorder=3,
                    )
        if not plotted:
            plt.close(fig)
            return None

        ax.set_xticks(range(len(setups)))
        ax.set_xticklabels(setups)
        ax.set_xlim(-0.3, len(setups) - 0.7)
        ax.set_ylim(_LINE_FLOOR, _LINE_CEIL)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy across conditions — all models (k = {k})")
        ax.grid(alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
        draw_baselines(
            ax,
            baselines,
            fontsize=19,
            label_min_gap=1.35,
            color="#aaa",
            label_color="#555555",
            linestyle=":",
            linewidth=0.8,
            label_side="right",
        )

        # Right-side model labels near the last condition, with leader lines.
        labeled = [m for m in models if m in end_y]
        x_end = len(setups) - 1
        x_label = x_end + label_x_offset
        label_ys = _spread_label_ys(
            [end_y[m] for m in labeled],
            min_gap=label_min_gap,
            y_lo=_LINE_FLOOR + 0.4,
            y_hi=_LINE_CEIL - 0.4,
        )
        for m, y_lab in zip(labeled, label_ys):
            y_data = end_y[m]
            color = model_color(m)
            ax.plot(
                [x_end, x_label],
                [y_data, y_lab],
                color=color,
                linewidth=leader_linewidth,
                alpha=leader_alpha,
                solid_capstyle="round",
                clip_on=False,
                zorder=4,
            )
            ax.text(
                x_label,
                y_lab,
                model_label(m),
                ha="left",
                va="center",
                fontsize=19,
                color=color,
                clip_on=False,
                zorder=5,
            )

        fig.tight_layout()
        fig.subplots_adjust(right=0.70)

        png = save_figure(
            fig,
            output_dir / f"accuracy_setups_k{k}.png",
            dpi=300,
            also_pdf=True,
            dpi_pdf=400,
        )
        plt.close(fig)
    return png


def plot_accuracy_across_ks(
    long_df: pd.DataFrame,
    baselines: dict[str, float],
    output_dir: Path,
    *,
    ks: list[int],
    setup: str,
    annotate_n: bool = True,
) -> Path | None:

    sub = long_df[long_df["setup"].astype(str) == setup]
    if sub.empty:
        return None
    models = sort_models(sub["model"])
    acc: dict[tuple[str, int], float] = {}
    for r in sub.itertuples():
        if pd.isna(getattr(r, "k", np.nan)) or pd.isna(r.accuracy):
            continue
        key = (str(r.model), int(r.k))
        acc[key] = float(r.accuracy) * 100.0

    plotted = list(acc.values())
    if not plotted:
        return None

    n_k = len(ks)
    x = np.arange(len(models))
    bar_w = 0.70 / max(n_k, 1)
    offsets = (np.arange(n_k) - (n_k - 1) / 2.0) * bar_w
    floor, ceil = padded_limits(
        plotted + [v * 100.0 for v in baselines.values()],
        floor=_BAR_FLOOR,
        ceil=_BAR_CEIL,
    )

    rc_bar = {
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
    with plt.rc_context(rc_bar):
        fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * len(models) + 2.2), 5.6))
        for j, k in enumerate(ks):
            for mi, m in enumerate(models):
                v = acc.get((m, int(k)))
                if v is None:
                    continue
                face, edge = bar_face_edge(model_color(m), j)
                ax.bar(
                    x[mi] + offsets[j],
                    v - floor,
                    bar_w,
                    bottom=floor,
                    color=face,
                    hatch=BAR_HATCHES[j % len(BAR_HATCHES)],
                    edgecolor=edge,
                    linewidth=0.4,
                    zorder=2,
                )
                if annotate_n:
                    ax.text(
                        x[mi] + offsets[j],
                        v,
                        f"{v:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                        color="0.25",
                        zorder=3,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [model_label(m) for m in models], rotation=30, ha="right"
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(floor, ceil)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="-", alpha=0.35, zorder=0)
        despine(ax)
        draw_baselines(ax, baselines, fontsize=15)
        ax.legend(
            handles=[
                Patch(
                    facecolor=(
                        0.6, 0.6, 0.6, BAR_FILL_ALPHAS[j % len(BAR_FILL_ALPHAS)]
                    ),
                    edgecolor=(0.4, 0.4, 0.4, 0.9),
                    hatch=BAR_HATCHES[j % len(BAR_HATCHES)],
                    linewidth=0.4,
                    label=f"k={ks[j]}",
                )
                for j in range(n_k)
            ],
            loc="upper right",
            framealpha=0.95,
        )
        ax.set_title(
            f"Accuracy by model across k = {', '.join(str(k) for k in ks)} "
            f"({setup_label(setup)})",
            fontweight="bold",
        )
        fig.tight_layout()
        fig.subplots_adjust(right=0.82)

        k_tag = "_".join(str(k) for k in ks)
        out = save_figure(
            fig,
            output_dir / f"accuracy_by_k_{k_tag}_{setup_slug(setup)}.png",
            dpi=220,
        )
        plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _split_csv(raw: str | None) -> list[str] | None:
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else None


def _parse_ks(raw: str | None) -> list[int] | None:
    """Parse comma-separated k values, e.g. ``1,4,5,6``."""
    parts = _split_csv(raw)
    if parts is None:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(
            f"--ks must be comma-separated integers (e.g. 1,4,5,6); got {raw!r}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Run analysis with config models and setups",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Run analysis with given directories of prediction .jsonl files",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Run analysis for given batch of prediction .jsonl files, "
        "i.e. <runs-root>/<model>/<batch-id>/k{N}_pr{R}/",
    )
    parser.add_argument(
        "--ks",
        default=None,
        help="Comma-separated k values (n_icl_examples) to score, e.g. 1,4,5,6; "
        "required with --runs-root (with --config, defaults to dataset.n_icl_examples)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="icl_reference_k (prediction reference); required with --runs-root "
        "(with --config, defaults to dataset.icl_reference_k)",
    )
    parser.add_argument(
        "--setups",
        default=None,
        help="Run analysis for given setups",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Run analysis for given models",
    )
    parser.add_argument(
        "--annotate-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate each point/bar with accuracy (%%) on the figure (default: on)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Score and write CSV only; skip all figures",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <runs-root>/analysis)",
    )
    parser.add_argument(
        "--no-filter-pr-cases",
        action="store_true",
        help="Skip filtering prediction rows to the icl_reference_k case set",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve data source
    # ------------------------------------------------------------------
    cfg = None

    if args.runs_root is None:
        cfg = load_json_config(
            args.config or str(EVAL_DIR / "configs" / "sample.json")
        )
        runs_root = Path(cfg["output_root"]).expanduser().resolve()
        batch_id = str(cfg["batch_id"])

        config_models = [str(m["model"]) for m in cfg["models"]]
        config_setups = sorted(
            {str(setup) for model in cfg["models"] for setup in model["setups"]},
            key=lambda s: SETUP_ORDER.index(s) if s in SETUP_ORDER else len(SETUP_ORDER),
        )
        config_k = int(cfg["dataset"]["n_icl_examples"])
        config_pr = int(cfg["dataset"]["icl_reference_k"])
        ratings_path = cfg.get("ratings_path")
        scenarios_path = cfg.get("scenarios_path")
        min_scenarios = int(cfg["dataset"].get("min_scenarios", 8))
        random_seed = int(cfg["dataset"].get("seed", SEED))
    else:
        runs_root = args.runs_root.expanduser().resolve()
        batch_id = args.batch_id
        config_models = None
        config_setups = None
        config_k = None
        config_pr = None
        ratings_path = None
        scenarios_path = None
        min_scenarios = 8
        random_seed = SEED
        if args.ks is None or args.pr is None:
            raise SystemExit(
                "--runs-root requires both --ks and --pr "
                "(e.g. --ks 1,4,5,6 --pr 6)"
            )

    if not runs_root.is_dir():
        raise SystemExit(f"Not a directory: {runs_root}")
    if not batch_id:
        raise SystemExit("--runs-root requires --batch-id")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else runs_root / "analysis"
    )

    # ------------------------------------------------------------------
    # Resolve models, setups, and k values
    # ------------------------------------------------------------------
    models = _split_csv(args.models)
    if models is None:
        models = config_models  # None → discover all under runs-root

    setups = _split_csv(args.setups)
    if setups is None:
        setups = config_setups if config_setups is not None else list(SETUP_ORDER)

    parsed_ks = _parse_ks(args.ks)
    if parsed_ks is not None:
        ks: list[int | None] | None = list(parsed_ks)
    elif config_k is not None:
        ks = [config_k]
    else:
        ks = None  # every k{N}_pr{R}/ found

    pr = args.pr if args.pr is not None else config_pr

    # ------------------------------------------------------------------
    # Load and score runs
    # ------------------------------------------------------------------
    try:
        runs = load_runs(
            runs_root,
            batch_id,
            ks=ks,
            pr=pr,
            setups=setups,
            models=models,
            filter_to_pr_cases=not args.no_filter_pr_cases,
            ratings_path=ratings_path,
            scenarios_path=scenarios_path,
            min_scenarios=min_scenarios,
            random_seed=random_seed,
        )
    except FileNotFoundError as exc:
        raise SystemExit(exc) from exc

    long_df, gt_df = score_runs(runs)

    scored_models = sort_models(long_df["model"]) if not long_df.empty else []
    table_models = sort_models(models) if models else scored_models
    if models:
        missing = [m for m in models if m not in set(scored_models)]
        if missing:
            print(f"  [warn] no predictions for: {missing}")

    scored_ks = (
        sorted({int(v) for v in long_df["k"].dropna()})
        if "k" in long_df.columns and long_df["k"].notna().any()
        else []
    )
    if ks is not None:
        table_ks = [int(x) for x in ks if x is not None]
    elif scored_ks:
        table_ks = scored_ks
    else:
        table_ks = list_k_values(runs_root, batch_id)

    table_setups = [s for s in SETUP_ORDER if s in setups] + [
        s for s in setups if s not in SETUP_ORDER
    ]

    # ------------------------------------------------------------------
    # Save accuracy table (full grid; blanks where a run is missing)
    # ------------------------------------------------------------------
    table = expand_accuracy_grid(
        long_df,
        models=table_models,
        setups=table_setups,
        ks=table_ks,
    )
    k_tag = "_".join(str(k) for k in table_ks)
    if cfg is not None and len(table_ks) == 1:
        csv_name = f"accuracy_{batch_id}_k{table_ks[0]}"
    else:
        csv_name = f"accuracy{f'_k{k_tag}' if k_tag else ''}"

    write_analysis_csv(
        table,
        output_dir,
        csv_name,
        label="Boundary-level accuracy",
    )

    # ------------------------------------------------------------------
    # Plot figures (default on: setup-trend line + k-trend bars)
    # ------------------------------------------------------------------
    if args.no_plots:
        return

    baselines = (
        baseline_accuracies(gt_df, seed=SEED) if gt_df is not None else {}
    )
    annotate_n = bool(args.annotate_n)

    # setup-trend lines: accuracy across setups, one figure per k
    if parsed_ks:
        setup_ks = [int(k) for k in parsed_ks if k is not None]
    elif scored_ks:
        setup_ks = list(scored_ks)
    elif table_ks:
        setup_ks = [int(k) for k in table_ks]
    else:
        print("  [warn] no setup-trend plot (no k available)")
        setup_ks = []

    for setup_k in setup_ks:
        if "k" in long_df.columns:
            is_indep = long_df["setup"].astype(str).isin(K_INDEPENDENT_SETUPS)
            sub = long_df.loc[is_indep | (long_df["k"] == setup_k)]
        else:
            sub = long_df
        path = plot_accuracy_across_setups(
            sub,
            baselines,
            output_dir,
            k=setup_k,
            annotate_n=annotate_n,
        )
        if path is None:
            print(f"  [warn] no setup-trend plot for k={setup_k}")

    # k-trend bars: accuracy across k per setup (skip k-independent setups)
    trend_ks = table_ks or list(DEFAULT_KS)
    if len(trend_ks) < 2:
        print("  [warn] k-trend needs at least two k values; skipping")
    else:
        for setup in table_setups:
            if setup in K_INDEPENDENT_SETUPS:
                print(
                    f"  skip k-trend for {setup} "
                    "(k-independent; one file held out for setup comparisons)"
                )
                continue
            path = plot_accuracy_across_ks(
                long_df,
                baselines,
                output_dir,
                ks=trend_ks,
                setup=setup,
                annotate_n=annotate_n,
            )
            if path is None:
                print(f"  [warn] no k-trend plot for setup={setup}")


if __name__ == "__main__":
    main()
