"""Loading and parsing config helpers for evaluation"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from constants import (
    MIN_SCENARIOS,
    MODEL_BACKEND_FIELDS,
    N_ICL_EXAMPLES,
    PROVIDERS,
    RATINGS_PATH,
    REPO_ROOT,
    SCENARIOS_PATH,
    SEED,
    SETUP_DEFAULT,
    SETUP_ORDER,
    normalize_setup,
)

def _resolve_repo_path(raw: str | Path) -> str:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def _validate_batch_id(raw: Any) -> str:
    """A batch id is one directory name, so reject anything path-like."""
    batch_id = str(raw or "").strip()
    if not batch_id:
        raise ValueError(
            "Config must include a non-empty 'batch_id'. Reusing a batch id "
            "resumes that batch: already-predicted (user, scenario) pairs are "
            "skipped, so you can widen dataset.sample_frac and only pay for "
            "the new users."
        )
    if batch_id != Path(batch_id).name or batch_id in {".", ".."}:
        raise ValueError(f"batch_id must be a bare directory name; got {batch_id!r}")
    return batch_id


def run_dirname(n_icl_examples: int, icl_reference_k: int) -> str:
    """Prediction folder name: ``k{n_icl}_pr{icl_reference_k}``."""
    return f"k{int(n_icl_examples)}_pr{int(icl_reference_k)}"


def run_subdir(cfg: dict[str, Any], *, prompt_variant: str | None = None) -> str:
    """Per-run directory below ``<output_root>/<model>/``.

    Layout: ``<batch_id>/k{n_icl_examples}_pr{icl_reference_k}[/prompt_variant]``.
    ``pr`` = prediction reference: which k's held-out case set this run targets.
    """
    ds = cfg["dataset"]
    n_icl = int(ds["n_icl_examples"])
    reference_k = ds.get("icl_reference_k")
    pr = n_icl if reference_k is None else int(reference_k)
    sub = f"{cfg['batch_id']}/{run_dirname(n_icl, pr)}"
    return f"{sub}/{prompt_variant}" if prompt_variant else sub


def config_to_directory(
    cfg: dict[str, Any], *, prompt_variant: str | None = None
) -> dict[str, Path]:
    """Return prediction directories a config addresses, keyed by model."""
    root = Path(cfg["output_root"])
    subdir = run_subdir(cfg, prompt_variant=prompt_variant)
    return {str(m["model"]): root / str(m["model"]) / subdir for m in cfg["models"]}


def _parse_setups(raw: Any, *, default: list[str] | None = None) -> list[str]:
    """Parse setups from a list or comma-separated string into canonical tags."""
    if raw is None:
        values = list(default or [SETUP_DEFAULT])
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        values = [str(part).strip() for part in raw if str(part).strip()]
    else:
        raise ValueError("setups must be a list or comma-separated string")
    if not values:
        values = list(default or [SETUP_DEFAULT])
    return [normalize_setup(value) for value in values]


def _normalize_model_backend(model: dict[str, Any]) -> None:
    """Strip/validate optional ``provider`` / ``base_url`` / ``api_key_env`` on a JSON model row.

    Load-time only — does not apply CLI/env defaults. Those are resolved later in
    ``providers.prepare_live_run`` when a live prediction run starts.
    """
    provider = model.get("provider")
    if provider is not None:
        provider = str(provider).strip()
        if provider not in PROVIDERS:
            raise ValueError(
                f"models[].provider must be one of {list(PROVIDERS)}; got {provider!r}"
            )
        model["provider"] = provider

    for field in ("base_url", "api_key_env"):
        value = model.get(field)
        if value is None:
            continue
        value = str(value).strip()
        if not value:
            raise ValueError(f"models[].{field} must be a non-empty string.")
        model[field] = value


def load_json_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("ratings_path", str(RATINGS_PATH))
    cfg.setdefault("scenarios_path", str(SCENARIOS_PATH))
    cfg.setdefault("output_root", str(REPO_ROOT / "outputs" / "eval"))
    cfg.setdefault("dataset", {})
    cfg["dataset"].setdefault("seed", SEED)
    cfg["dataset"].setdefault("n_icl_examples", N_ICL_EXAMPLES)
    cfg["dataset"].setdefault("min_scenarios", MIN_SCENARIOS)
    cfg["dataset"].setdefault("sample_frac", 1.0)
    cfg["dataset"].setdefault("align_to_k", True)
    cfg.setdefault("max_workers", 2)
    cfg.setdefault("prediction_temperature", 0.0)
    cfg.setdefault("prediction_seed", SEED)
    cfg.setdefault("prediction_timeout_seconds", 120.0)
    cfg.setdefault("prediction_max_retries", 2)
    cfg.setdefault("prediction_max_tokens", 1500)
    cfg.setdefault("save_interval", 1)
    cfg.setdefault("return_usage", False)
    cfg.setdefault("save_raw_response", False)
    cfg["batch_id"] = _validate_batch_id(cfg.get("batch_id"))
    if "models" not in cfg or not cfg["models"]:
        raise ValueError("Config must include non-empty 'models'.")

    ds = cfg["dataset"]
    sample_frac = float(ds["sample_frac"])
    if not 0.0 < sample_frac <= 1.0:
        raise ValueError("dataset.sample_frac must be in the interval (0, 1].")
    ds["sample_frac"] = sample_frac

    n_icl = int(ds["n_icl_examples"])
    min_scenarios = int(ds["min_scenarios"])
    if n_icl < 0:
        raise ValueError("dataset.n_icl_examples must be non-negative.")
    if min_scenarios < 1:
        raise ValueError("dataset.min_scenarios must be at least 1.")

    # Default prediction reference to this run's k when omitted.
    if ds.get("icl_reference_k") is None:
        ds["icl_reference_k"] = n_icl
    reference_k = int(ds["icl_reference_k"])
    ds["icl_reference_k"] = reference_k
    if reference_k < n_icl:
        raise ValueError(
            "dataset.icl_reference_k must be >= dataset.n_icl_examples "
            f"(got icl_reference_k={reference_k}, n_icl_examples={n_icl})."
        )
    if reference_k >= min_scenarios:
        raise ValueError(
            "The test split would be empty for users with min_scenarios; "
            "require icl_reference_k < min_scenarios."
        )

    # Default True; treat missing/null as True (no None state after load).
    align_to_k = ds.get("align_to_k", True)
    if align_to_k is None:
        align_to_k = True
    elif isinstance(align_to_k, str):
        align_to_k = align_to_k.strip().lower() in {"1", "true", "yes", "y"}
    else:
        align_to_k = bool(align_to_k)
    ds["align_to_k"] = align_to_k

    cfg["prediction_temperature"] = float(cfg["prediction_temperature"])
    cfg["prediction_seed"] = int(cfg["prediction_seed"])
    cfg["prediction_timeout_seconds"] = float(cfg["prediction_timeout_seconds"])
    cfg["prediction_max_retries"] = int(cfg["prediction_max_retries"])
    cfg["prediction_max_tokens"] = int(cfg["prediction_max_tokens"])
    cfg["save_interval"] = int(cfg["save_interval"])
    if cfg["prediction_max_retries"] < 1:
        raise ValueError("prediction_max_retries must be >= 1.")
    if cfg["prediction_max_tokens"] < 1:
        raise ValueError("prediction_max_tokens must be >= 1.")
    if cfg["prediction_timeout_seconds"] <= 0:
        raise ValueError("prediction_timeout_seconds must be > 0.")
    if cfg["save_interval"] < 1:
        raise ValueError("save_interval must be >= 1 (flush every N new rows).")

    # Top-level default setups; each model may override.
    cfg["setups"] = _parse_setups(cfg.get("setups"), default=[SETUP_DEFAULT])
    for model in cfg["models"]:
        model_id = str(model.get("model", "")).strip()
        if not model_id:
            raise ValueError("Every models entry must include a non-empty 'model'.")
        model["model"] = model_id
        model["setups"] = _parse_setups(model.get("setups"), default=cfg["setups"])
        _normalize_model_backend(model)

    model_ids = [model["model"] for model in cfg["models"]]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Config contains duplicate model IDs.")

    cfg["ratings_path"] = _resolve_repo_path(cfg["ratings_path"])
    cfg["scenarios_path"] = _resolve_repo_path(cfg["scenarios_path"])
    cfg["output_root"] = _resolve_repo_path(cfg["output_root"])
    return cfg


def build_run_configs(
    cfg: dict[str, Any],
    *,
    prompt_variants: list[str] | None = None,
) -> list[dict]:
    """Expand (model × setup × optional prompt variant) into per-run configs."""

    configs: list[dict] = []
    output_root = Path(cfg["output_root"])
    ds = cfg["dataset"]
    sample_frac = float(ds["sample_frac"])
    split_seed = int(ds["seed"])
    n_icl = int(ds["n_icl_examples"])
    reference_k = int(ds["icl_reference_k"])
    align_to_k = True if ds.get("align_to_k") is None else bool(ds["align_to_k"])
    variants = prompt_variants if prompt_variants is not None else [None]
    for model_cfg in cfg["models"]:
        model = model_cfg["model"]
        setups = list(model_cfg["setups"])
        for setup in setups:
            model_setup = setup
            for variant in variants:
                prompt_variant = "v0" if variant is None else variant
                save_dir = output_root / model / run_subdir(
                    cfg, prompt_variant=variant
                )
                save_path = save_dir / f"{model_setup}.jsonl"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                configs.append(
                    {
                        "model": model,
                        # Optional overrides; CLI/env filled in by providers.prepare_live_run.
                        **{f: model_cfg.get(f) for f in MODEL_BACKEND_FIELDS},
                        # OpenRouter-only; ignored by other backends.
                        "reasoning": model_cfg.get("reasoning"),
                        "save_path": str(save_path),
                        "with_history": setup != "Baseline",
                        "with_context": setup == "HC",
                        "with_label": setup == "HL",
                        "prediction_reasoning_mode": "F",
                        "run_tag": setup,
                        "model_setup": model_setup,
                        "prompt_variant": prompt_variant,
                        "batch_id": cfg["batch_id"],
                        "n_icl_examples": n_icl,
                        "icl_reference_k": reference_k,
                        "align_to_k": align_to_k,
                        "sample_frac": sample_frac,
                        "split_seed": split_seed,
                        "prediction_temperature": cfg["prediction_temperature"],
                        "prediction_seed": cfg["prediction_seed"],
                        "prediction_timeout_seconds": cfg["prediction_timeout_seconds"],
                        "prediction_max_retries": cfg["prediction_max_retries"],
                        "prediction_max_tokens": cfg["prediction_max_tokens"],
                        "save_interval": cfg["save_interval"],
                        "return_usage": cfg["return_usage"],
                        "save_raw_response": cfg["save_raw_response"],
                    }
                )
    # Stable order for logs / analysis tables
    order = {name: i for i, name in enumerate(SETUP_ORDER)}
    configs.sort(
        key=lambda row: (
            row["model"],
            order.get(row["run_tag"], len(SETUP_ORDER)),
            row.get("prompt_variant") or "",
        )
    )
    return configs
