"""Run model predictions for the setups named in a JSON config."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR.parent / "utils"))
sys.path.insert(0, str(EVAL_DIR / "helper"))
sys.path.insert(0, str(EVAL_DIR))

from config import build_run_configs, load_json_config
from load_data import load_user_data
from providers import add_provider_args, prepare_live_run
from runner import run_icl_experiments_parallel

def main() -> None:
    parser = argparse.ArgumentParser(description="CIDER prediction runs")
    parser.add_argument(
        "--config",
        default=str(EVAL_DIR / "configs" / "sample.json"),
        help="JSON config (default: evaluation/configs/sample.json)",
    )
    add_provider_args(parser)
    args = parser.parse_args()

    cfg = load_json_config(args.config)
    configs = build_run_configs(cfg)
    output_root = Path(cfg["output_root"])

    if args.provider == "external":
        raise SystemExit(
            "--provider external makes no API calls, and this script only "
            "predicts. Score existing predictions with:\n"
            f"  python evaluation/analysis/boundary_level_accuracy.py --config {args.config}"
        )

    live_configs, experiment_fn = prepare_live_run(
        cfg,
        configs,
        provider=args.provider,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        reasoning_enabled=bool(args.reasoning_enabled),
    )
    live_paths = {c["save_path"] for c in live_configs}
    external = sorted({c["model"] for c in configs if c["save_path"] not in live_paths})
    if external:
        print(f"Skipping prediction for external models: {external}")

    user_data = load_user_data(cfg)
    setups = sorted({c["run_tag"] for c in live_configs})
    providers_used = sorted({c["provider"] for c in live_configs})
    print(
        f"Running providers={providers_used} setups={setups} "
        f"for {len(live_configs)} run(s) → {output_root} "
        f"(batch {cfg['batch_id']})"
    )
    run_icl_experiments_parallel(
        user_data,
        live_configs,
        max_workers=int(cfg["max_workers"]),
        experiment_fn=experiment_fn,
    )

    print(
        "\nPrediction done."
    )


if __name__ == "__main__":
    main()
