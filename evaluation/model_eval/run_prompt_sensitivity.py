"""Prompt-sensitivity runs for the ``HC`` setup only (v0/v1/v2 paraphrases)."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR.parent / "utils"))
sys.path.insert(0, str(EVAL_DIR / "helper"))
sys.path.insert(0, str(EVAL_DIR))

from config import (
    build_run_configs,
    load_json_config,
    run_subdir,
)
from constants import REPO_ROOT, VARIANT_SUFFIXES
from load_data import binary_rating_to_numeric, load_user_data, load_prediction_jsonl, resolve_prediction_path
from metrics import compute_dataset_metrics
from prompts.sensitivity import PROMPT_VARIANTS
from providers import add_provider_args, prepare_live_run
from runner import run_icl_experiments_parallel

METRIC_COLS = ["accuracy"]
SENSITIVITY_SETUP = "HC"


def _variants(cfg: dict) -> list[str]:
    variants = cfg.get("prompt_variants") or list(PROMPT_VARIANTS)
    if isinstance(variants, str):
        variants = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in PROMPT_VARIANTS]
    if unknown:
        raise ValueError(f"Unknown prompt_variants {unknown}; known: {sorted(PROMPT_VARIANTS)}")
    return list(variants)


def _force_hc_setup(cfg: dict) -> dict:
    """Prompt sensitivity is defined for HC only."""
    cfg = dict(cfg)
    cfg["setups"] = [SENSITIVITY_SETUP]
    models = []
    for model_cfg in cfg.get("models") or []:
        row = dict(model_cfg)
        row["setups"] = [SENSITIVITY_SETUP]
        models.append(row)
    cfg["models"] = models
    return cfg


def analyze(cfg: dict, variants: list[str]) -> None:
    root = Path(cfg["output_root"])
    batch_id = str(cfg["batch_id"])
    k = int(cfg["dataset"]["n_icl_examples"])
    out_dir = root / "analysis_prompt_sensitivity" / f"{batch_id}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows, frames, notes = [], {}, []
    for model_cfg in cfg["models"]:
        model = model_cfg["model"]
        for variant in variants:
            variant_dir = root / model / run_subdir(cfg, prompt_variant=variant)
            path = resolve_prediction_path(variant_dir, SENSITIVITY_SETUP)
            if path is None:
                # Report the canonical path even when nothing was written.
                path = variant_dir / f"{SENSITIVITY_SETUP}.jsonl"
            df = load_prediction_jsonl(path)
            if df is None:
                notes.append(f"missing: {path}")
                continue
            m = compute_dataset_metrics(df)
            metric_rows.append(
                {
                    "model": model,
                    "setup": SENSITIVITY_SETUP,
                    "prompt_variant": variant,
                    **{k: float(m[k]) for k in METRIC_COLS},
                }
            )
            frames[(model, variant)] = df

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metrics_by_variant.csv", index=False)

    summary_rows = []
    if not metrics_df.empty:
        for model, g in metrics_df.groupby("model"):
            row = {
                "model": model,
                "setup": SENSITIVITY_SETUP,
                "n_variants": len(g),
            }
            for col in METRIC_COLS:
                vals = g[col].astype(float)
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_sd"] = float(vals.std(ddof=0))
                row[f"{col}_range"] = float(vals.max() - vals.min())
            summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "sensitivity_summary.csv", index=False)

    agree_rows = []
    by_model: dict[str, list[str]] = {}
    for model, variant in frames:
        by_model.setdefault(model, []).append(variant)
    for model, vs in by_model.items():
        for a, b in combinations(sorted(vs), 2):
            da, db = frames[(model, a)], frames[(model, b)]
            merged = da.merge(db, on=["user_id", "scenario_id"], suffixes=("_a", "_b"))
            cell = []
            for s in VARIANT_SUFFIXES:
                ca, cb = f"pred_var{s}_a", f"pred_var{s}_b"
                if ca not in merged or cb not in merged:
                    continue
                va = merged[ca].map(binary_rating_to_numeric)
                vb = merged[cb].map(binary_rating_to_numeric)
                mask = va.notna() & vb.notna()
                if mask.any():
                    cell.append(float((va[mask] == vb[mask]).mean()))
            agree_rows.append(
                {
                    "model": model,
                    "setup": SENSITIVITY_SETUP,
                    "variant_a": a,
                    "variant_b": b,
                    "agreement": float(np.mean(cell)) if cell else float("nan"),
                    "n_rows": int(len(merged)),
                }
            )
    pd.DataFrame(agree_rows).to_csv(out_dir / "pairwise_prediction_agreement.csv", index=False)
    (out_dir / "notes.json").write_text(
        json.dumps({"missing": notes, "setup": SENSITIVITY_SETUP}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote sensitivity tables → {out_dir}")
    if notes:
        print(f"({len(notes)} missing run files)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt sensitivity (HC only; v0/v1/v2)"
    )
    parser.add_argument(
        "--config",
        default=str(EVAL_DIR / "configs" / "sample.json"),
    )
    add_provider_args(parser)
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    metrics_only = bool(args.metrics_only or args.provider == "external")
    cfg = _force_hc_setup(load_json_config(args.config))
    if Path(cfg["output_root"]).name == "eval":
        cfg["output_root"] = str(REPO_ROOT / "outputs" / "prompt_sensitivity")
    variants = _variants(cfg)

    if not metrics_only:
        live_configs, experiment_fn = prepare_live_run(
            cfg,
            build_run_configs(cfg, prompt_variants=variants),
            provider=args.provider,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            reasoning_enabled=bool(args.reasoning_enabled),
        )
        assert all(c["run_tag"] == SENSITIVITY_SETUP for c in live_configs)
        user_data = load_user_data(cfg)
        providers_used = sorted({c["provider"] for c in live_configs})
        print(
            f"Sensitivity runs: providers={providers_used} "
            f"{len(live_configs)} (setup={SENSITIVITY_SETUP}, variants={variants})"
        )
        run_icl_experiments_parallel(
            user_data,
            live_configs,
            max_workers=int(cfg["max_workers"]),
            experiment_fn=experiment_fn,
        )

    analyze(cfg, variants)


if __name__ == "__main__":
    main()
