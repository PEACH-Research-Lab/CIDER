""" other util and helpers functions"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd
from pathlib import Path
import os
from constants import REPO_ROOT, SETUP_ORDER

# ----------------------------------------------------
# Loading and cleaning env
# ----------------------------------------------------
_loaded = False


def clean_env_value(value: str | None) -> str:
    return (value or "").strip().rstrip("\\").strip()


def _load_env_file(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = clean_env_value(value).strip('"').strip("'")
    return True


def ensure_env_loaded(env_file: str | Path | None = None) -> None:
    global _loaded
    if env_file is not None:
        _load_env_file(env_file)
        _loaded = True
        return
    if _loaded:
        return
    _loaded = True
    _load_env_file(REPO_ROOT / ".env")


def write_analysis_csv(
    df: pd.DataFrame,
    output_dir: str | Path,
    stem: str,
    *,
    label: str | None = None,
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path}")
    if label is not None:
        print(f"\n{label} ({len(df)} rows):")
        print(df.to_string(index=False))
    return path

# ----------------------------------------------------
# Parsing prediction output and failures
# ---------------------------------------------------- 

def _balanced_brace_slice(s: str, start: int) -> str | None:
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None

def extract_json_object_candidate(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty model output")
    for m in re.finditer(r"```(?:json)?\s*(\{)", text, re.IGNORECASE | re.DOTALL):
        span = _balanced_brace_slice(text, m.start(1))
        if span:
            return span
    start = text.find("{")
    if start != -1:
        span = _balanced_brace_slice(text, start)
        if span:
            return span
    raise ValueError("No JSON found")

def brief_parse_shape_hint(text: str) -> str:
    """One-line summary of what the completion looks like after JSON extraction (for logs)."""
    if not isinstance(text, str) or not text.strip():
        return "empty_completion"
    try:
        cand = extract_json_object_candidate(text)
        data = json.loads(cand)
        if not isinstance(data, dict):
            return f"json_root_type={type(data).__name__}"
        keys = list(data.keys())
        pred = data.get("predictions")
        if pred is None:
            return f"root_keys={keys!r} predictions_key=absent"
        if isinstance(pred, dict):
            pk = list(pred.keys())
            return (
                f"root_keys={keys!r} predictions=dict len={len(pk)} "
                f"keys_sample={pk[:15]!r}"
            )
        if isinstance(pred, list):
            elem_types = [type(x).__name__ for x in pred[:5]]
            return (
                f"root_keys={keys!r} predictions=list len={len(pred)} "
                f"first_elem_types={elem_types!r}"
            )
        return f"root_keys={keys!r} predictions_type={type(pred).__name__!r}"
    except json.JSONDecodeError as err:
        return f"extracted_blob_json_decode_error={err}"
    except ValueError as err:
        return f"extract_json_failed={err}"

def format_parse_failure_diagnostic(text: str, *, max_snippet_chars: int = 2500) -> str:
    """
    Multi-line diagnostic: length, preview, and structured shape hint for a failed parse.
    """
    lines: list[str] = [
        f"[parse_diagnostic] completion_length_chars={len(text)}",
        "[parse_diagnostic] shape_hint:",
        brief_parse_shape_hint(text),
        f"[parse_diagnostic] first_{max_snippet_chars}_chars:",
        text[:max_snippet_chars],
    ]
    if len(text) > max_snippet_chars:
        lines.append(f"[parse_diagnostic] ... {len(text) - max_snippet_chars} more chars omitted")
    return "\n".join(lines)



def _repair_unescaped_controls_in_json_strings(blob: str) -> str:
    out: list[str] = []
    i = 0
    in_str = False
    esc = False
    while i < len(blob):
        c = blob[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        if esc:
            out.append(c)
            esc = False
            i += 1
            continue
        if c == "\\":
            out.append(c)
            esc = True
            i += 1
            continue
        if c == '"':
            out.append(c)
            in_str = False
            i += 1
            continue
        o = ord(c)
        if o < 32:
            if c == "\t":
                out.append("\\t")
            elif c == "\n":
                out.append("\\n")
            elif c == "\r":
                out.append("\\r")
            else:
                out.append(f"\\u{o:04x}")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)

def _strip_trailing_commas(blob: str) -> str:
    prev = None
    cur = blob
    while cur != prev:
        prev = cur
        cur = re.sub(r",(\s*})", r"\1", cur)
        cur = re.sub(r",(\s*])", r"\1", cur)
    return cur

def _normalize_variant_keys(pred: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pred.items():
        if isinstance(k, int) and 1 <= k <= 9:
            out[f"variant_{k}"] = v
            continue
        ks = str(k).strip()
        m = re.match(r"variant[_\s-]?(\d+)$", ks, re.I)
        if m:
            out[f"variant_{int(m.group(1))}"] = v
            continue
        m = re.match(r"variant\s+(\d+)$", ks, re.I)
        if m:
            out[f"variant_{int(m.group(1))}"] = v
            continue
        if re.fullmatch(r"\d+", ks):
            n = int(ks)
            if 1 <= n <= 9:
                out[f"variant_{n}"] = v
                continue
        m = re.match(r"v\s*(\d+)$", ks, re.I)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 9:
                out[f"variant_{n}"] = v
                continue
        m = re.match(r"var[_\s-]?(\d+)$", ks, re.I)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 9:
                out[f"variant_{n}"] = v
                continue
        out[k] = v
    return out

def _only_variant_slots(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if re.fullmatch(r"variant_[1-9]", k)}

def _predictions_value_to_flat(pred: Any) -> dict[str, Any] | None:
    if isinstance(pred, list):
        if len(pred) == 9 and all(not isinstance(x, dict) for x in pred):
            return {f"variant_{i + 1}": pred[i] for i in range(9)}
        flat: dict[str, Any] = {}
        for item in pred:
            if not isinstance(item, dict):
                continue
            merged = _normalize_variant_keys(item)
            for kk, vv in merged.items():
                if re.fullmatch(r"variant_[1-9]", kk):
                    flat[kk] = vv
            idx: int | None = None
            num_key_used: str | None = None
            for num_key in ("variant", "index", "order", "slot", "i", "n"):
                if num_key not in item:
                    continue
                try:
                    idx = int(str(item[num_key]).strip())
                except (TypeError, ValueError):
                    idx = None
                if idx is not None:
                    num_key_used = num_key
                    break
            if idx is not None and 1 <= idx <= 9:
                val = (
                    item.get("answer")
                    or item.get("value")
                    or item.get("rating")
                    or item.get("prediction")
                    or item.get("yes_no")
                )
                if val is None:
                    skip = {num_key_used} if num_key_used else set()
                    skip.update({"variant", "index", "order", "slot", "i", "n"})
                    rest = {kk: vv for kk, vv in item.items() if kk not in skip}
                    if len(rest) == 1:
                        val = next(iter(rest.values()))
                if val is not None:
                    flat[f"variant_{idx}"] = val
        picked = _only_variant_slots(flat)
        if len(picked) >= 9:
            return picked
        return None
    if isinstance(pred, dict) and pred:
        norm = _normalize_variant_keys(pred)
        picked = _only_variant_slots(norm)
        if len(picked) >= 9:
            return picked
        # Models sometimes key predictions by the variant *text* (same strings as in <variants>),
        # in shuffled prompt order — JSON object order matches that order; map position → slot.
        if len(pred) == 9 and len(picked) == 0:
            return {f"variant_{i + 1}": v for i, v in enumerate(pred.values())}
        return None
    return None

def _coerce_top_level_predictions(data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    reasoning = str(data.get("reasoning", "") or "")
    pred = data.get("predictions")
    if pred is not None:
        flat_pred = _predictions_value_to_flat(pred)
        if flat_pred is not None:
            return reasoning, flat_pred
    flat = {
        k: v
        for k, v in data.items()
        if k != "reasoning" and k != "predictions"
    }
    flat = _normalize_variant_keys(flat)
    only_variants = _only_variant_slots(flat)
    if len(only_variants) >= 9:
        return reasoning, only_variants
    raise ValueError("Missing 'predictions' field or 9 variant_* keys")

def _require_all_variants(preds: dict[str, Any]) -> dict[str, Any]:
    for i in range(1, 10):
        k = f"variant_{i}"
        if k not in preds:
            raise ValueError(f"missing {k}")
        value = str(preds[k]).strip().upper()
        if value not in {"YES", "NO"}:
            raise ValueError(f"{k} must be YES or NO; got {preds[k]!r}")
        preds[k] = value
    return preds

def _parse_dict_to_predictions(data: dict[str, Any], path: str, detail: str) -> tuple[str, dict[str, Any], str, str]:
    reasoning, preds = _coerce_top_level_predictions(data)
    if not preds:
        raise ValueError("Empty predictions")
    _require_all_variants(preds)
    return reasoning, preds, path, detail

def _try_regex_variant_scrape(text: str) -> tuple[str, dict[str, Any], str, str] | None:
    pat = re.compile(
        r'["\']?variant[_\s-]?(\d+)["\']?\s*[:=]\s*["\']?(YES|NO)["\']?',
        re.I,
    )
    preds: dict[str, Any] = {}
    for m in pat.finditer(text):
        preds[f"variant_{int(m.group(1))}"] = m.group(2).upper()
    if len(preds) < 9:
        return None
    for i in range(1, 10):
        if f"variant_{i}" not in preds:
            return None
    return "", preds, "regex_variants", "scraped variant_1..variant_9 YES/NO from raw text; reasoning not recovered"

@dataclass(frozen=True)
class ParseOutcome:
    reasoning: str
    predictions: dict[str, Any]
    parse_path: str
    detail: str

def parse_prediction_output(text: str) -> ParseOutcome:
    """
    Parse model output into reasoning + predictions dict (variant_1..variant_9 values).
    Tries strict JSON, repaired JSON, Python literal dict, then regex scrape.
    """
    errors: list[str] = []

    def attempt(blob: str, path: str, detail: str) -> ParseOutcome | None:
        try:
            data = json.loads(blob)
            if not isinstance(data, dict):
                raise TypeError("root must be object")
            r, p, path_f, det = _parse_dict_to_predictions(data, path, detail)
            return ParseOutcome(r, p, path_f, det)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            errors.append(f"{path}: {e}")
            return None

    try:
        candidate = extract_json_object_candidate(text)
    except ValueError as e:
        errors.append(f"extract: {e}")
        regex_out = _try_regex_variant_scrape(text)
        if regex_out:
            r, p, path, detail = regex_out
            return ParseOutcome(r, p, path, detail + "; " + "; ".join(errors))
        raise ValueError("no JSON object and regex fallback failed: " + "; ".join(errors)) from e

    blobs_to_try: list[tuple[str, str, str]] = [
        ("json_strict", candidate, ""),
        (
            "json_repaired_controls",
            _repair_unescaped_controls_in_json_strings(candidate),
            "escaped raw control chars inside strings",
        ),
        ("json_repaired_trailing_comma", _strip_trailing_commas(candidate), "stripped trailing commas"),
        (
            "json_repaired_both",
            _repair_unescaped_controls_in_json_strings(_strip_trailing_commas(candidate)),
            "stripped trailing commas; escaped raw control chars",
        ),
    ]
    for path, blob, note in blobs_to_try:
        out = attempt(blob, path, note)
        if out is not None:
            if path != "json_strict" and note:
                return ParseOutcome(out.reasoning, out.predictions, out.parse_path, note)
            return out

    try:
        data = ast.literal_eval(candidate)
        if isinstance(data, dict):
            r, p, path_f, det = _parse_dict_to_predictions(data, "python_literal", "ast.literal_eval on extracted object")
            return ParseOutcome(r, p, path_f, det)
    except (ValueError, SyntaxError, TypeError) as e:
        errors.append(f"python_literal: {e}")

    regex_out = _try_regex_variant_scrape(text)
    if regex_out:
        r, p, path, detail = regex_out
        return ParseOutcome(r, p, path, detail + "; prior: " + "; ".join(errors))

    raise ValueError("all parse strategies failed: " + "; ".join(errors))


def normalize_predictions(predictions: dict, prediction_order: list) -> dict:
    """Map positional model outputs onto semantic ``var{suffix}_rating`` keys."""
    
    missing = [i for i in range(1, 10) if f"variant_{i}" not in predictions]
    assert not missing, f"Prediction is missing positional keys: {missing}"

    out = {
        f"var{label}_rating": predictions[f"variant_{position}"]
        for position, label in enumerate(prediction_order, start=1)
    }
    out["reasoning"] = predictions.get("reasoning", "")
    return out


def append_jsonl_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")