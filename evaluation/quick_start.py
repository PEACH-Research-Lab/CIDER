#!/usr/bin/env python3
"""Small-batch evaluation + analysis entry point."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "utils"))
sys.path.insert(0, str(EVAL_DIR / "helper"))
sys.path.insert(0, str(EVAL_DIR))
REPO = EVAL_DIR.parent

from config import load_json_config, run_subdir
from load_data import load_user_data
from prompts.preview import (
    build_example_prompts,
    print_example_prompts,
    save_example_prompts,
)
from providers import DEFAULT_PROVIDER, add_provider_args

DEFAULT_CONFIG = EVAL_DIR / "configs" / "sample.json"
ANALYSIS_SCRIPTS = ("boundary_level_accuracy.py", "variant_level_fp_fn.py")


def _run(script: Path, *args: str) -> None:
    command = [sys.executable, str(script), *args]
    print("+", " ".join(command))
    subprocess.run(command, cwd=REPO, check=True)


def _preview_prompts(cfg: dict, *, assume_yes: bool) -> None:
    setups = sorted({s for m in cfg["models"] for s in m["setups"]})
    print("\nLoading a sample case to preview prompts for selected setups…")
    user_data = load_user_data(cfg)
    examples = build_example_prompts(
        user_data,
        setups,
        variant_shuffle_seed=int(cfg["dataset"]["seed"]),
    )
    print_example_prompts(examples)

    out_dir = Path(cfg["output_root"]) / "prompts" / run_subdir(cfg)
    saved = save_example_prompts(examples, out_dir / "example_prompts.txt")
    print(f"Saved example prompts → {saved}")

    if assume_yes:
        return
    try:
        reply = input(
            "\nProceed with model prediction using these setups? [y/N] "
        ).strip().lower()
    except EOFError:
        reply = ""
    if reply not in {"y", "yes"}:
        raise SystemExit("Aborted before prediction. Edit setups/config and re-run.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    add_provider_args(parser)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip API calls; score existing prediction JSONL",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip interactive confirmation after prompt preview",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Print example prompts for selected setups and exit",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Also run boundary-level accuracy + variant-level FP/FN",
    )
    args = parser.parse_args()

    metrics_only = bool(args.metrics_only or args.provider == "external")
    cfg = load_json_config(args.config)
    frac = float(cfg["dataset"]["sample_frac"])
    models = [entry["model"] for entry in cfg["models"]]
    setups = sorted({s for m in cfg["models"] for s in m["setups"]})
    print(f"Models: {models}")
    print(f"Setups: {setups}")
    print(f"Provider: {args.provider} (default={DEFAULT_PROVIDER})")
    print(f"Held-out user sample: {frac:.2%} (seed={cfg['dataset']['seed']})")
    print(f"Output root: {cfg['output_root']}")
    print(f"Batch: {cfg['batch_id']} (reuse it to extend this batch)")

    # Previewing is about the prompts a prediction would send, so it happens
    # even when the provider makes no calls.
    if args.preview_only:
        _preview_prompts(cfg, assume_yes=True)
        return

    config_path = str(Path(args.config).resolve())
    if not metrics_only:
        _preview_prompts(cfg, assume_yes=bool(args.yes))
        eval_args = ["--config", config_path, "--provider", args.provider]
        if args.base_url:
            eval_args.extend(["--base-url", args.base_url])
        if args.api_key_env:
            eval_args.extend(["--api-key-env", args.api_key_env])
        if args.reasoning_enabled:
            eval_args.append("--reasoning-enabled")
        _run(EVAL_DIR / "model_eval" / "run_eval.py", *eval_args)

    # Prediction and scoring are separate steps; with no prediction to run,
    # analysis is the only thing left to do.
    if args.analyze or metrics_only:
        for script in ANALYSIS_SCRIPTS:
            _run(EVAL_DIR / "analysis" / script, "--config", config_path)
        print(f"\nAnalysis tables and figures → {Path(cfg['output_root']) / 'analysis'}")


if __name__ == "__main__":
    main()
