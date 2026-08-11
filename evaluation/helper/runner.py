from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from clients.openai import call_llm_with_raw as call_openai_with_raw
from clients.openrouter import call_llm_with_raw as call_openrouter_with_raw
from constants import COST, VARIANT_SUFFIXES
from load_data import rating_to_yes_no
from prompts.construction import build_prompts
from utils import (
    brief_parse_shape_hint,
    format_parse_failure_diagnostic,
    parse_prediction_output,
    normalize_predictions,
    append_jsonl_rows
)

# ------------------------------
# USAGE / COST (per prediction row)
# ------------------------------

def build_usage_and_cost_record(usage: dict, model: str) -> dict:
    """Token counts + cost fields matching ``outputs/existing`` JSONL rows."""
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int(usage.get("reasoning_tokens") or 0)
    output_tokens = completion_tokens + reasoning_tokens
    total_tokens = int(usage.get("total_tokens") or 0)
    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + output_tokens

    openrouter_cost = usage.get("cost")
    if openrouter_cost is not None:
        try:
            openrouter_cost = float(openrouter_cost)
        except (TypeError, ValueError):
            openrouter_cost = None

    rates = COST.get(model)
    table_cost: float | None = None
    if rates is not None:
        table_cost = (
            prompt_tokens * float(rates["input"])
            + output_tokens * float(rates["output"])
        )

    return {
        "usage_tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "reasoning_tokens": reasoning_tokens,
        "usage_tokens_output_tokens": output_tokens,
        "usage_original": dict(usage),
        "usage_cost_openrouter_usd": openrouter_cost,
        "usage_cost_table_usd": table_cost,
    }

# ------------------------------
# LLM CALL
# ------------------------------

def _complete_with_parse_retries(
    complete_fn: Callable[[], tuple[str, dict[str, Any], dict[str, Any], str]],
    *,
    max_retries: int = 2,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Call ``complete_fn`` until parse succeeds or retries are exhausted."""

    last_error: BaseException | None = None
    last_completion_text: str | None = None
    for attempt in range(1, max_retries + 1):
        try:
            text, usage, raw_body, mode_label = complete_fn()
            last_completion_text = text
            outcome = parse_prediction_output(text)
            assert outcome.predictions, "Empty parsed predictions"
            merged = {
                **outcome.predictions,
                "reasoning": outcome.reasoning or "",
                "_prediction_parse_meta": {
                    "raw": text,
                    "raw_response": raw_body,
                    "request_mode": mode_label,
                    "path": outcome.parse_path,
                    "detail": outcome.detail,
                },
            }
            return merged, usage
        except Exception as e:
            last_error = e
            hint = brief_parse_shape_hint(last_completion_text or "")
            print(f"[attempt {attempt}/{max_retries}] error: {e}")
            print(f"  parse_shape_hint: {hint}")
            if attempt < max_retries:
                time.sleep(1)

    if last_completion_text is not None and last_error is not None:
        print(
            f"\n[parse_diagnostic] giving up after {max_retries} attempts; "
            f"last_error={last_error!r}\n"
            + format_parse_failure_diagnostic(last_completion_text)
            + "\n"
        )
    err_msg = str(last_error) if last_error is not None else "unknown"
    return None, {"error": err_msg}


def build_predict_fn(
    call_llm_with_raw: Callable[..., tuple[str, dict[str, Any], dict[str, Any], str]],
    config: dict,
    **client_kwargs: Any,
) -> Callable[[str, str, str], tuple[dict[str, Any] | None, dict[str, Any]]]:
    """Bind a client's ``call_llm_with_raw`` to the shared retry/parse loop."""
    
    api_key_env = str(config["api_key_env"])
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is not set")
    max_retries = int(config["prediction_max_retries"])
    shared = {
        "timeout": float(config["prediction_timeout_seconds"]),
        "temperature": float(config["prediction_temperature"]),
        "seed": int(config["prediction_seed"]),
        "max_tokens": int(config["prediction_max_tokens"]),
    }

    def predict_fn(sys_prompt: str, user_prompt: str, model: str):
        return _complete_with_parse_retries(
            lambda: call_llm_with_raw(
                api_key, sys_prompt, user_prompt, model, **shared, **client_kwargs
            ),
            max_retries=max_retries,
        )

    return predict_fn

def load_existing_prediction_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"Warning: skip invalid JSONL line {line_no} in {path}: {err}")
                continue
            if not isinstance(value, dict):
                continue
            user_id = value.get("user_id")
            scenario_id = value.get("scenario_id")
            if user_id is None or scenario_id is None:
                continue
            keys.add((str(user_id), str(scenario_id)))
    return keys

# ------------------------------
# EXPERIMENT
# ------------------------------

def _run_icl_experiment_common(
    user_data: dict,
    config: dict,
    predict_fn: Callable[[str, str, str], tuple[Any, dict]],
) -> dict:
    model = config["model"]
    save_path = config["save_path"]
    reasoning_mode = config["prediction_reasoning_mode"]
    setup_key = str(config["run_tag"])
    model_setup = str(config.get("model_setup"))
    prompt_variant = str(config["prompt_variant"])
    save_interval = int(config["save_interval"])
    split_seed = int(config["split_seed"])
    sample_frac = float(config["sample_frac"])
    batch_id = str(config["batch_id"])
    n_icl_examples = int(config["n_icl_examples"])
    icl_reference_k = config["icl_reference_k"]

    save_p = Path(save_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_prediction_keys(save_p)

    user_items = sorted(user_data.items(), key=lambda item: str(item[0]))

    total = 0
    skipped = 0
    success = 0
    pending_rows: list[dict] = []
    print(f"\n[{model_setup}] model={model} → {save_path}")
    if existing_keys:
        print(f"[{model_setup}] resume mode: {len(existing_keys)} existing predictions detected")
    if save_interval > 1:
        print(f"[{model_setup}] save_interval={save_interval}")

    def _flush_pending() -> None:
        nonlocal pending_rows
        if pending_rows:
            append_jsonl_rows(save_path, pending_rows)
            pending_rows = []

    try:
        for user_id, info in user_items:
            test_cases = info["test_cases"]
            n_cases = len(test_cases)
            for case_idx, test_case in enumerate(test_cases, start=1):
                scenario_id = str(test_case["scenario_id"])
                key = (str(user_id), scenario_id)
                total += 1
                if key in existing_keys:
                    skipped += 1
                    print(
                        f"  [{model}] [{model_setup}] user={user_id} "
                        f"case={case_idx}/{n_cases} scenario={scenario_id} -> skipped"
                    )
                    continue
                (sys_prompt, user_prompt), order_mappings = build_prompts(
                    user_id,
                    info,
                    test_case,
                    prompt_pattern=setup_key,
                    return_order_mappings=True,
                    variant_shuffle_seed=split_seed,
                    prompt_variant=prompt_variant,
                )
                prediction_started_at = time.perf_counter()
                pred, usage = predict_fn(sys_prompt, user_prompt, model)
                prediction_elapsed_seconds = time.perf_counter() - prediction_started_at

                if pred is None:
                    print(
                        f"❌ [{model}] [{model_setup}] user={user_id} case={case_idx}/{n_cases} "
                        f"scenario={scenario_id} -> failed ({prediction_elapsed_seconds:.2f}s)"
                    )
                    continue

                parse_meta = pred.pop("_prediction_parse_meta", {})
                pred = normalize_predictions(
                    pred, order_mappings["prediction"]["order"]
                )

                row = {
                    "user_id": user_id,
                    "scenario_id": test_case["scenario_id"],
                    "role": info.get("role", test_case.get("role")),
                    "condition": info.get("condition", test_case.get("condition")), # condition refers to AI-mediated condition.
                    "model": model,
                    "model_setup": model_setup, # model_setup refers to the setup of the model (Baseline, H, HL, or HC), the "condition" as described in paper.
                    "prompt_variant": prompt_variant,
                    "with_history": bool(config["with_history"]),
                    "with_context": bool(config["with_context"]),
                    "with_label": bool(config["with_label"]),
                    "prediction_reasoning_mode": reasoning_mode,
                    "run_tag": model_setup,
                    "batch_id": batch_id,
                    "predicted_at": datetime.now(timezone.utc).isoformat(),
                    "n_icl_examples": n_icl_examples,
                    "icl_reference_k": icl_reference_k,
                    "align_to_k": (
                        True
                        if config.get("align_to_k") is None
                        else bool(config["align_to_k"])
                    ),
                    "sample_frac": sample_frac,
                    "split_seed": split_seed,
                    "prediction_seed": int(config["prediction_seed"]),
                    "system_prompt": sys_prompt,
                    "user_prompt": user_prompt,
                    "randomized_mappings": order_mappings,
                    "reasoning": pred["reasoning"],
                    "model_completion_raw": parse_meta.get("raw", ""),
                    "prediction_parse_path": parse_meta.get("path", ""),
                    "prediction_parse_detail": parse_meta.get("detail", ""),
                    "request_elapsed_seconds": prediction_elapsed_seconds,
                    **{
                        f"true_var{s}": rating_to_yes_no(test_case[f"var{s}_rating"])
                        for s in VARIANT_SUFFIXES
                    },
                    **{f"pred_var{s}": pred[f"var{s}_rating"] for s in VARIANT_SUFFIXES},
                }
                if config["save_raw_response"]:
                    row["raw_response"] = parse_meta.get("raw_response")
                    row["request_mode"] = parse_meta.get("request_mode")
                row.update(build_usage_and_cost_record(usage, model))
                if config["return_usage"]:
                    # The provider's own usage block, before normalization.
                    row["usage"] = usage

                pending_rows.append(row)
                success += 1
                if len(pending_rows) >= save_interval:
                    _flush_pending()
                    flushed = "saved"
                else:
                    flushed = f"buffered ({len(pending_rows)}/{save_interval})"
                print(
                    f"  [{model}] [{model_setup}] user={user_id} case={case_idx}/{n_cases} "
                    f"scenario={scenario_id} -> {flushed} ({prediction_elapsed_seconds:.2f}s)"
                )
    finally:
        _flush_pending()

    print(f"Done [{model_setup}]: new={success}, skipped={skipped}, total={total}")
    return {
        "save_path": save_path,
        "run_tag": model_setup,
        "model_setup": model_setup,
        "success": success,
        "skipped": skipped,
        "total": total,
    }


def run_icl_experiment_openai(user_data: dict, config: dict) -> dict:
    """Batch ICL run via OpenAI ``/chat/completions``."""
    return _run_icl_experiment_common(
        user_data,
        config,
        build_predict_fn(
            call_openai_with_raw, config, base_url=str(config["base_url"])
        ),
    )


def run_icl_experiment_openrouter(user_data: dict, config: dict) -> dict:
    """Batch ICL run via OpenRouter (reasoning extras, provider-reported cost)."""
    return _run_icl_experiment_common(
        user_data,
        config,
        build_predict_fn(
            call_openrouter_with_raw,
            config,
            openrouter_payload_extras=config["openrouter_payload_extras"],
            prefer_json_object=bool(config["openrouter_prefer_json_object"]),
        ),
    )


# ------------------------------
# PARALLEL EXPERIMENT
# ------------------------------
def run_icl_experiments_parallel(
    user_data: dict,
    configs: list,
    max_workers: int = 2,
    *,
    experiment_fn: Callable[[dict, dict], dict],
) -> list:
    """Run configs in parallel. ``experiment_fn`` must be set explicitly by the caller."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_cfg = {pool.submit(experiment_fn, user_data, c): c for c in configs}
        for fut in as_completed(future_to_cfg):
            cfg = future_to_cfg[fut]
            try:
                results.append(fut.result())
            except Exception as err:
                print(f"ICL experiment failed run_tag={cfg.get('run_tag')!r}: {err}")
                raise
    return results
