"""Variant-level FP/FN rates and setup-shift visualizations."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from adjustText import adjust_text as _adjust_text_labels

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR / "helper"))
sys.path.insert(0, str(EVAL_DIR))

from config import load_json_config
from constants import (
    K_INDEPENDENT_SETUPS,
    SEED,
    SETUP_ORDER,
    VARIANT_LABEL_MAP,
    VARIANT_SUFFIXES,
)
from load_data import load_runs
from metrics import variant_fp_fn_rates
from plotting import (
    SETUP_MARKER,
    SETUP_SCATTER_S,
    despine,
    model_label,
    rate_axis_limits,
    save_figure,
    setup_label,
    setup_slug,
    sort_models,
)
from utils import write_analysis_csv


DEFAULT_SHIFT_FROM = "Baseline"
DEFAULT_SHIFT_TO = "HC"
DEFAULT_SHIFT_K = 6


# ---------------------------------------------------------------------------
# Calculations (metrics)
# ---------------------------------------------------------------------------
def score_runs(runs: list[dict]) -> pd.DataFrame:
    """Score loaded runs with ``variant_fp_fn_rates``."""
    frames: list[pd.DataFrame] = []
    for run in runs:
        rates = variant_fp_fn_rates(run["df"])
        if rates.empty:
            print(f"  no true_var*/pred_var* pairs to score: {run['save_path']}")
            continue
        k_value = run.get("k")
        rates = rates.copy()
        rates.insert(0, "k", k_value)
        rates.insert(0, "setup", str(run["setup"]))
        rates.insert(0, "model", str(run["model"]))
        frames.append(rates)
        print(
            f"{f'k={k_value} ' if k_value is not None else ''}{run['model']} [{run['setup']}]: "
            f"FP={rates['fp_rate'].mean():.4f} FN={rates['fn_rate'].mean():.4f} "
            f"(mean over {len(rates)} variants)"
        )
    if not frames:
        raise ValueError("No scored FP/FN tables from loaded runs")
    out = pd.concat(frames, ignore_index=True)
    out["k"] = out["k"].astype("Int64")
    return out


def analyzed_ks(rates_df: pd.DataFrame) -> list[int]:
    """k values present in a rates table; empty when the runs carry no k."""
    if "k" not in rates_df.columns:
        return []
    values = pd.to_numeric(rates_df["k"], errors="coerce").dropna()
    return sorted({int(v) for v in values})


def _at_k(rates_df: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Rows scored at *k*. K-independent setups (Baseline) are always included."""
    if k is None or "k" not in rates_df.columns:
        return rates_df
    is_indep = rates_df["setup"].astype(str).isin(K_INDEPENDENT_SETUPS)
    at = rates_df.loc[~is_indep & (rates_df["k"] == k)]
    indep = rates_df.loc[is_indep]
    if indep.empty:
        return at
    return pd.concat([at, indep], ignore_index=True)


def _broadcast_k_independent_setup(
    rates_df: pd.DataFrame, setup: str
) -> pd.DataFrame:
    """Copy a k-independent setup onto every k present in other setups."""
    if setup not in K_INDEPENDENT_SETUPS or "k" not in rates_df.columns:
        return rates_df
    base = rates_df[rates_df["setup"].astype(str) == setup]
    other = rates_df[rates_df["setup"].astype(str) != setup]
    if base.empty or other.empty:
        return rates_df
    target_ks = sorted({int(v) for v in other["k"].dropna()})
    if not target_ks:
        return rates_df

    pieces: list[pd.DataFrame] = [other]
    for _, g in base.groupby("model", sort=False):
        src = g
        if g["k"].notna().any():
            src_k = int(g["k"].dropna().iloc[0])
            src = g[g["k"] == src_k]
        for k in target_ks:
            piece = src.copy()
            piece["k"] = k
            pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


# ---------------------------------------------------------------------------
# Shift table
# ---------------------------------------------------------------------------
def _shift_direction(base: float, other: float, *, eps: float = 1e-9) -> str:
    d = float(other) - float(base)
    if abs(d) <= eps:
        return "unchanged"
    return "increase" if d > 0 else "decrease"


def calculate_shift_table(
    rates_df: pd.DataFrame, from_setup: str, to_setup: str
) -> pd.DataFrame:
    """Per (k, variant, model): both setups' FP/FN rates and how they moved."""
    if rates_df.empty:
        return pd.DataFrame()

    # Reuse the single Baseline (etc.) slice across every k for pairing.
    rates_df = _broadcast_k_independent_setup(rates_df, from_setup)
    rates_df = _broadcast_k_independent_setup(rates_df, to_setup)

    present = set(rates_df["setup"].astype(str))
    if from_setup not in present or to_setup not in present:
        return pd.DataFrame()

    from_label = setup_label(from_setup)
    to_label = setup_label(to_setup)
    group_cols = (["k"] if "k" in rates_df.columns else []) + ["model", "variant"]
    sub = (
        rates_df[rates_df["setup"].astype(str).isin([from_setup, to_setup])]
        .drop_duplicates(subset=group_cols + ["setup"], keep="first")
        .copy()
    )
    rows: list[dict] = []
    for keys, g in sub.groupby(group_cols, sort=False, dropna=False):
        key = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        g = g.set_index("setup")
        if from_setup not in g.index or to_setup not in g.index:
            continue
        f_fp = float(g.at[from_setup, "fp_rate"])
        f_fn = float(g.at[from_setup, "fn_rate"])
        t_fp = float(g.at[to_setup, "fp_rate"])
        t_fn = float(g.at[to_setup, "fn_rate"])
        rows.append(
            {
                "k": key.get("k"),
                "variant_name": str(g.at[from_setup, "variant_label"]),
                "model_name": str(key["model"]),
                f"{from_label} FP": f_fp,
                f"{from_label} FN": f_fn,
                f"{to_label} FP": t_fp,
                f"{to_label} FN": t_fn,
                "FP shifting direction": _shift_direction(f_fp, t_fp),
                "FN shifting direction": _shift_direction(f_fn, t_fn),
            }
        )
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    variant_order = [VARIANT_LABEL_MAP[s] for s in VARIANT_SUFFIXES]
    out["variant_name"] = pd.Categorical(
        out["variant_name"], categories=variant_order, ordered=True
    )
    sort_cols = ["variant_name", "model_name"]
    if out["k"].notna().any():
        sort_cols.insert(0, "k")
    else:
        out = out.drop(columns="k")
    return out.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def prediction_size(df: pd.DataFrame, *, setup: str | None = None) -> int | None:
    """Typical prediction count behind a slice's rates, for labelling only."""
    if "n" not in df.columns:
        return None
    sub = df if setup is None else df[df["setup"].astype(str) == str(setup)]
    vals = pd.to_numeric(sub["n"], errors="coerce").dropna()
    return int(vals.mode().iloc[0]) if len(vals) else None


def _place_variant_labels_at_source(
    ax: plt.Axes,
    items: list[tuple[float, float, str, tuple]],
    *,
    label_fontsize: float = 18.0,
) -> None:
    if not items:
        return
    texts: list = []
    anchor_xs: list[float] = []
    anchor_ys: list[float] = []
    n_items = len(items)
    y_step = 0.55 if n_items <= 5 else 0.70
    x_step = 0.35 if n_items <= 5 else 0.45
    for idx, (x_b, y_b, variant_name, color) in enumerate(items):
        # Seed a few %-points away from the Baseline marker (axis span ~80).
        x_off = 3.2 + (idx % 3) * x_step
        y_off = -2.4 - (idx % 4) * y_step
        anchor_xs.append(float(x_b))
        anchor_ys.append(float(y_b))
        texts.append(
            ax.text(
                x_b + x_off,
                y_b + y_off,
                variant_name,
                fontsize=label_fontsize,
                color=color,
                fontweight="medium",
                va="top",
                ha="left",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.38",
                    "facecolor": (1.0, 1.0, 1.0, 0.55),
                    "edgecolor": color,
                    "linewidth": 1.05,
                },
            )
        )
    _adjust_text_labels(
        texts,
        x=np.asarray(anchor_xs, dtype=float),
        y=np.asarray(anchor_ys, dtype=float),
        ax=ax,
        only_move={"texts": "xy"},
        expand=(1.35, 1.45),
        force_text=(0.25, 0.40),
        force_static=(0.70, 0.90),
        force_pull=(0.50, 0.65),
        iter_lim=600,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.42, alpha=0.5),
    )


def _draw_model_shift_ax(
    ax: plt.Axes,
    mdf: pd.DataFrame,
    *,
    from_setup: str,
    to_setup: str,
    k: int | None,
    title: str,
    label_fontsize: float = 18.0,
    legend_fontsize: float = 18.0,
    annotate_n: bool = True,
) -> bool:
    """Draw one model's linked FP/FN scatter. Returns False if nothing plotted."""
    cmap = plt.get_cmap("tab10")
    variant_colors = {v: cmap(i % 10) for i, v in enumerate(VARIANT_SUFFIXES)}
    n_from = prediction_size(mdf, setup=from_setup) if annotate_n else None
    n_to = prediction_size(mdf, setup=to_setup) if annotate_n else None
    model_fp: list[float] = []
    model_fn: list[float] = []
    label_items: list[tuple[float, float, str, tuple]] = []
    for v in VARIANT_SUFFIXES:
        vc = variant_colors.get(str(v), "#333333")
        xs: list[float] = []
        ys: list[float] = []
        setups_hit: list[str] = []
        for setup in (from_setup, to_setup):
            row = mdf[(mdf["variant"] == v) & (mdf["setup"] == setup)]
            if row.empty:
                continue
            xs.append(float(row["fp_rate"].iloc[0]) * 100.0)
            ys.append(float(row["fn_rate"].iloc[0]) * 100.0)
            setups_hit.append(setup)
        model_fp.extend(xs)
        model_fn.extend(ys)
        if len(xs) >= 2:
            ax.plot(
                xs,
                ys,
                linestyle="--",
                color=vc,
                linewidth=1.1,
                alpha=0.55,
                zorder=1,
            )
            label_items.append(
                (float(xs[0]), float(ys[0]), VARIANT_LABEL_MAP.get(str(v), str(v)), vc)
            )
        for x, y, su in zip(xs, ys, setups_hit):
            ax.scatter(
                [x],
                [y],
                s=SETUP_SCATTER_S.get(su, 68),
                marker=SETUP_MARKER.get(su, "o"),
                color=vc,
                edgecolor="white",
                linewidth=0.9,
                zorder=6,  # above variant tags so markers stay visible
            )
    if not model_fp:
        return False

    xlim, ylim = rate_axis_limits(model_fp, model_fn)
    # Put k in the title (not legend) so legend labels stay short and don't cover points.
    titled = f"{title} (k={k})" if k is not None else title
    if annotate_n and n_from is not None and n_to is not None and n_from == n_to:
        full_title = f"{titled}  (n={n_from} predictions)"
    elif annotate_n and (n_from is not None or n_to is not None):
        parts = []
        if n_from is not None:
            parts.append(f"{setup_label(from_setup)} n={n_from}")
        if n_to is not None:
            parts.append(f"{setup_label(to_setup)} n={n_to}")
        full_title = f"{titled}  ({', '.join(parts)})"
    else:
        full_title = titled
    ax.set_title(full_title, fontsize=18, fontweight=600)
    ax.set_xlabel("FP rate (%)", fontsize=18)
    ax.set_ylabel("FN rate (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(alpha=0.28, linestyle="-", linewidth=0.6)
    despine(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _place_variant_labels_at_source(ax, label_items, label_fontsize=label_fontsize)

    def _legend_label(setup: str, n: int | None) -> str:
        base = setup_label(setup)
        return f"{base} (n={n})" if annotate_n and n is not None else base

    from_mk = SETUP_MARKER.get(from_setup, "o")
    to_mk = SETUP_MARKER.get(to_setup, "^")
    # Same dark gray for face and edge (white edges disappear on a white legend).
    legend_gray = "0.35"
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=from_mk,
            color=legend_gray,
            linestyle="none",
            markersize=12,
            markeredgecolor=legend_gray,
            markeredgewidth=1.0,
            markerfacecolor=legend_gray,
            label=_legend_label(from_setup, n_from),
        ),
        Line2D(
            [0],
            [0],
            marker=to_mk,
            color=legend_gray,
            linestyle="none",
            markersize=12,
            markeredgecolor=legend_gray,
            markeredgewidth=1.0,
            markerfacecolor=legend_gray,
            label=_legend_label(to_setup, n_to),
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=legend_fontsize,
        framealpha=0.35,
        fancybox=False,
        edgecolor="0.7",
    )
    if annotate_n:
        ax.text(
            0.02,
            0.02,
            "n = # predicted scenarios",
            transform=ax.transAxes,
            fontsize=legend_fontsize,
            color="#555555",
            va="bottom",
            ha="left",
        )
    return True


def plot_fp_fn_scatter_setup_shift(
    rates_df: pd.DataFrame,
    output_dir: Path,
    *,
    from_setup: str = DEFAULT_SHIFT_FROM,
    to_setup: str = DEFAULT_SHIFT_TO,
    k: int | None = None,
    annotate_n: bool = True,
) -> list[Path]:
    """One linked FP/FN scatter per model (plus a grid) for one pair at one k."""

    if rates_df.empty or "setup" not in rates_df.columns:
        return []
    sub = _at_k(rates_df, k)
    present = set(sub["setup"].astype(str))
    if from_setup not in present or to_setup not in present:
        return []
    sub = sub[sub["setup"].isin([from_setup, to_setup])].copy()
    models = sort_models(sub["model"].unique())
    if not models:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fp_fn_shift_{setup_slug(from_setup)}_to_{setup_slug(to_setup)}"
    if k is not None:
        stem = f"{stem}_k{k}"
    written: list[Path] = []

    rc = {
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    }
    with plt.rc_context(rc):
        for model in models:
            mdf = sub[sub["model"] == model]
            fig, ax = plt.subplots(figsize=(8.2, 6.4))
            ok = _draw_model_shift_ax(
                ax,
                mdf,
                from_setup=from_setup,
                to_setup=to_setup,
                k=k,
                title=model_label(model),
                label_fontsize=18.0,
                legend_fontsize=18.0,
                annotate_n=annotate_n,
            )
            if not ok:
                plt.close(fig)
                continue
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            safe = model.replace("/", "__").replace(":", "_")
            path = save_figure(
                fig,
                output_dir / f"{stem}_{safe}.png",
                dpi=240,
                also_pdf=True,
                dpi_pdf=300,
                pad_inches=0.14,
            )
            plt.close(fig)
            written.append(path)

    return written


def shift_pairs(
    rates_df: pd.DataFrame,
    *,
    from_setup: str,
    to_setup: str,
    k: int | None = None,
) -> list[tuple[str, str]]:
    """Return ``[(from_setup, to_setup)]`` when both setups exist at *k*."""
    present = set(_at_k(rates_df, k)["setup"].astype(str))
    if from_setup in present and to_setup in present:
        return [(from_setup, to_setup)]
    return []


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _print_shift_summary(shift: pd.DataFrame) -> None:
    """Direction tallies, split per k once the table spans more than one."""
    groups = (
        [(f"k={k} ", g) for k, g in shift.groupby("k", sort=True)]
        if "k" in shift.columns and shift["k"].nunique() > 1
        else [("", shift)]
    )
    for prefix, g in groups:
        for column in ("FP shifting direction", "FN shifting direction"):
            counts = ", ".join(
                f"{name}={n}" for name, n in g[column].value_counts().items()
            )
            print(f"  {prefix}{column}: {counts}")


def report_fp_fn(
    rates_df: pd.DataFrame,
    output_dir: Path,
    *,
    from_setup: str = DEFAULT_SHIFT_FROM,
    to_setup: str = DEFAULT_SHIFT_TO,
    shift_k: int | None = None,
    no_plots: bool = False,
    annotate_n: bool = True,
    stem_prefix: str = "",
) -> None:
    """Write the rates table, the shift table, and the shift figures.

    Rates and the shift table cover every scored k; the figures are drawn for a
    single *shift_k*, since one scatter can only show one pair of setups.
    """
    ks = analyzed_ks(rates_df)
    tag = stem_prefix + (f"_k{'_'.join(str(k) for k in ks)}" if ks else "")
    write_analysis_csv(rates_df, output_dir, f"variant_fp_fn_rates{tag}")
    print(
        f"\nVariant FP/FN rates: {len(rates_df)} rows "
        f"({rates_df['model'].nunique()} models × {rates_df['setup'].nunique()} setups "
        f"× {len(ks) or 1} k × {rates_df['variant'].nunique()} variants); "
        "FP=FP/(FP+TN), FN=FN/(FN+TP)"
    )

    pair = f"{setup_label(from_setup)}→{setup_label(to_setup)}"
    shift = calculate_shift_table(rates_df, from_setup, to_setup)
    if shift.empty:
        print(
            f"\nNo {pair} shift table "
            f"(need both {from_setup!r} and {to_setup!r} among the scored runs)."
        )
    else:
        stem = f"fp_fn_shift_{setup_slug(from_setup)}_to_{setup_slug(to_setup)}{tag}"
        write_analysis_csv(shift, output_dir, stem)
        print(f"\n{pair} shift ({len(shift)} rows):")
        _print_shift_summary(shift)

    if no_plots:
        return
    figures: list[Path] = []
    for f_setup, t_setup in shift_pairs(
        rates_df, from_setup=from_setup, to_setup=to_setup, k=shift_k
    ):
        figures.extend(
            plot_fp_fn_scatter_setup_shift(
                rates_df,
                output_dir,
                from_setup=f_setup,
                to_setup=t_setup,
                k=shift_k,
                annotate_n=annotate_n,
            )
        )
    if figures:
        print(f"\nWrote {len(figures)} figure(s) → {output_dir}")
        for path in figures:
            print(f"  {path.name}")
    else:
        where = f"at k={shift_k}" if shift_k is not None else "in the scored runs"
        print(f"\nNo shift figures (need {from_setup!r} and {to_setup!r} {where}).")


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


def _figure_k(
    requested_ks: list[int] | None, available: list[int]
) -> int | None:
    """k for shift figures: last of ``--ks`` if scored, else default/max available."""
    if not available:
        return None
    if requested_ks:
        for k in reversed(requested_ks):
            if int(k) in available:
                return int(k)
    if DEFAULT_SHIFT_K in available:
        return DEFAULT_SHIFT_K
    return max(available)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=None,
        help="Use output_root + models + batch_id from config",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Prediction tree (e.g. outputs/existing or outputs/eval)",
    )
    parser.add_argument(
        "--batch-id",
        default=None,
        help="Batch to score, i.e. <runs-root>/<model>/<batch-id>/k{N}_pr{R}/ "
        "(required with --runs-root; --config takes it from the config)",
    )
    parser.add_argument(
        "--ks",
        default=None,
        help="Comma-separated k values (n_icl_examples) to score, e.g. 1,4,5,6; "
        "required with --runs-root (with --config, defaults to dataset.n_icl_examples); "
        "shift figures use the last listed k that was scored",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="icl_reference_k (prediction reference); required with --runs-root "
        "(with --config, defaults to dataset.icl_reference_k)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model folder names (default: all under runs-root)",
    )
    parser.add_argument(
        "--setups",
        default=None,
        help="Comma-separated setups (default: from config, else all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <runs-root>/analysis)",
    )
    parser.add_argument(
        "--from-setup",
        default=DEFAULT_SHIFT_FROM,
        help="Setup the shift starts from (default: Baseline)",
    )
    parser.add_argument(
        "--to-setup",
        default=DEFAULT_SHIFT_TO,
        help="Setup the shift ends at (default: HC)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write the tables only, no scatters",
    )
    parser.add_argument(
        "--annotate-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate n= prediction counts in titles/legends (default: on)",
    )
    parser.add_argument(
        "--no-filter-pr-cases",
        action="store_true",
        help="Skip filtering prediction rows to the icl_reference_k case set",
    )
    args = parser.parse_args()

    if args.config:
        cfg = load_json_config(args.config)
        runs_root = Path(cfg["output_root"])
        batch_id = str(cfg["batch_id"])
        models = [str(m["model"]) for m in cfg["models"]]
        setups = _split_csv(args.setups) or sorted(
            {s for m in cfg["models"] for s in m["setups"]}
        )
        requested_ks = _parse_ks(args.ks) or [int(cfg["dataset"]["n_icl_examples"])]
        pr = args.pr if args.pr is not None else int(cfg["dataset"]["icl_reference_k"])
        ratings_path = cfg.get("ratings_path")
        scenarios_path = cfg.get("scenarios_path")
        min_scenarios = int(cfg["dataset"].get("min_scenarios", 8))
        random_seed = int(cfg["dataset"].get("seed", SEED))
        stem_prefix = f"_{batch_id}"
    else:
        if args.runs_root is None:
            raise SystemExit("Provide --config or --runs-root")
        runs_root = args.runs_root.expanduser().resolve()
        if not runs_root.is_dir():
            raise SystemExit(f"Not a directory: {runs_root}")
        if not args.batch_id:
            raise SystemExit("--runs-root requires --batch-id")
        if args.ks is None or args.pr is None:
            raise SystemExit(
                "--runs-root requires both --ks and --pr "
                "(e.g. --ks 1,4,5,6 --pr 6)"
            )
        batch_id = str(args.batch_id)
        models = _split_csv(args.models)
        setups = _split_csv(args.setups) or list(SETUP_ORDER)
        requested_ks = _parse_ks(args.ks)
        assert requested_ks is not None  # guarded above
        pr = int(args.pr)
        ratings_path = None
        scenarios_path = None
        min_scenarios = 8
        random_seed = SEED
        stem_prefix = ""

    try:
        runs = load_runs(
            runs_root,
            batch_id,
            ks=requested_ks,
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

    try:
        rates = score_runs(runs)
    except ValueError as exc:
        raise SystemExit(exc) from exc
    if models:
        missing = [m for m in models if m not in set(rates["model"].astype(str))]
        if missing:
            print(f"  [warn] no predictions for: {missing}")

    scored = analyzed_ks(rates)
    explicit_ks = _parse_ks(args.ks)
    report_fp_fn(
        rates,
        Path(args.output_dir or runs_root / "analysis"),
        from_setup=args.from_setup,
        to_setup=args.to_setup,
        shift_k=_figure_k(explicit_ks, scored),
        no_plots=bool(args.no_plots),
        annotate_n=bool(args.annotate_n),
        stem_prefix=stem_prefix,
    )


if __name__ == "__main__":
    main()
