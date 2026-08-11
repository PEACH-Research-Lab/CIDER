"""Data loading and preprocessing for evaluation and analysis."""

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from constants import (
    COLUMNS_TO_DROP,
    K_INDEPENDENT_SETUPS,
    RATINGS_PATH,
    ROLE_TO_IMAGE_KEY,
    SCENARIOS_PATH,
    SEED,
    SETUP_ORDER,
    USER_ID_COL,
    VARIANT_SUFFIXES,
)

# ----------------------------------------------------
# Preprocessing and loading CIDER dataset
# ---------------------------------------------------- 
def binary_rating_to_numeric(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if value == 1.0:
            return 1.0
        if value == 2.0:
            return 2.0

    s = str(value).strip().lower()
    if s in {"1", "1.0", "yes", "y"}:
        return 1.0
    if s in {"2", "2.0", "no", "n"}:
        return 2.0

    return float("nan")

def rating_to_yes_no(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return "YES"
        if float(value) == 2.0:
            return "NO"
    s = str(value).strip().upper()
    if s in {"1", "1.0", "YES", "Y"}:
        return "YES"
    if s in {"2", "2.0", "NO", "N"}:
        return "NO"
    raise ValueError(f"Unrecognized rating for prompt text: {value!r}")

def read_table(path: str | Path) -> pd.DataFrame:
    """Load a flat ratings/scenarios JSONL table."""
    return pd.read_json(Path(path), lines=True)


def load_for_prediction(
    ratings_path: str | Path | None = None,
    scenarios_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ratings with demographics dropped (prediction) + scenarios."""
    ratings = read_table(ratings_path or RATINGS_PATH)
    scenarios = read_table(scenarios_path or SCENARIOS_PATH)
    drop = [c for c in COLUMNS_TO_DROP if c in ratings.columns] # drop unused demographics.
    return ratings.drop(columns=drop), scenarios


def load_prediction_jsonl(path: str | Path) -> pd.DataFrame | None:
    """Load a prediction JSONL file; return None if missing or empty."""
    path = Path(path)
    if not path.is_file():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows) if rows else None

# ----------------------------------------------------
# Evaluation data preparation: ICL splits, sample fraction.
# ---------------------------------------------------- 

def prepare_icl_splits(
    ratings_df: pd.DataFrame,
    scenarios_df: pd.DataFrame,
    n_examples: int = 5,
    min_scenarios: int = 8,
    random_seed: int | None = None,
    icl_reference_k: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    base_seed = int(SEED if random_seed is None else random_seed)
    if icl_reference_k is not None and int(n_examples) > int(icl_reference_k):
        raise ValueError(
            f"n_examples ({n_examples}) must be <= icl_reference_k ({icl_reference_k}) "
            "to avoid ICL/test overlap (label leakage)."
        )
    if USER_ID_COL not in ratings_df.columns:
        raise KeyError(f"Ratings table missing user id column {USER_ID_COL!r}")
    scenario_cols = [f"s{i}_id" for i in range(10)]
    ratings_df = ratings_df.copy()
    ratings_df["scenario_count"] = ratings_df[scenario_cols].notna().sum(axis=1)
    filtered_df = ratings_df[ratings_df["scenario_count"] >= min_scenarios].copy()

    print(f"Users included: {len(filtered_df)}")
    for n in (10, 9, 8):
        print(f"  - {n} scenarios: {len(filtered_df[filtered_df['scenario_count'] == n])}")
    if icl_reference_k is not None:
        print(
            f"ICL reference k={int(icl_reference_k)}: test_cases = perm[{int(icl_reference_k)}:], "
            f"icl_examples = perm[:{n_examples}]"
        )

    scenario_lookup: dict = {}
    for _, srow in scenarios_df.iterrows():
        sid = srow["scenario_id"]
        scenario_lookup[sid] = srow

    user_data: dict = {}

    for user_idx, (_, row) in enumerate(filtered_df.iterrows()):
        user_id = row[USER_ID_COL]
        user_role = row["Role"]
        image_key = ROLE_TO_IMAGE_KEY[user_role]
        rng = np.random.default_rng(base_seed + user_idx)

        user_scenarios: list = []
        for i in range(10):
            scenario_id = row[f"s{i}_id"]
            if pd.isna(scenario_id):
                continue
            srow = scenario_lookup.get(scenario_id)
            if srow is None:
                raise ValueError(
                    f"Scenario {scenario_id!r} for user {user_id!r} "
                    "is missing from scenarios"
                )
            narrative = srow.get(image_key, "") or ""
            sender = srow.get("sender")
            data_sender_label = (
                str(sender).strip()
                if pd.notna(sender) and str(sender).strip()
                else "the data sender"
            )
            ratings_map = {}
            for s in VARIANT_SUFFIXES:
                col = f"s{i}_var{s}_rating"
                val = row[col]
                if pd.isna(val):
                    raise ValueError(f"Missing rating {col} for user {user_id}, slot s{i}")
                ratings_map[f"var{s}_rating"] = binary_rating_to_numeric(val)
            user_scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_idx": i,
                    "narrative": narrative,
                    "scenario_narrative": narrative,
                    "variant_narratives": {
                        f"var{s}": srow.get(f"variant_{s}_cleaned", "") or ""
                        for s in VARIANT_SUFFIXES
                    },
                    "role": user_role,
                    "condition": row["Condition"],
                    "data_sender_label": data_sender_label,
                    **ratings_map,
                }
            )

        perm = rng.permutation(len(user_scenarios))
        icl_examples = [user_scenarios[j] for j in perm[:n_examples]]
        if icl_reference_k is not None:
            k = int(icl_reference_k)
            test_cases = [user_scenarios[j] for j in perm[k:]]
        else:
            test_cases = [user_scenarios[j] for j in perm[n_examples:]]

        user_data[user_id] = {
            "role": user_role,
            "condition": row["Condition"],
            "icl_examples": icl_examples,
            "test_cases": test_cases,
            "n_icl": len(icl_examples),
            "n_test": len(test_cases),
            "need_for_privacy": row.get("Need_for_Privacy"),
        }

    test_counts: dict[int, int] = {}
    for d in user_data.values():
        n = d["n_test"]
        test_counts[n] = test_counts.get(n, 0) + 1
    for n_test in sorted(test_counts):
        print(f"  {n_test} test cases: {test_counts[n_test]} users")

    return user_data, filtered_df

def sample_test_cases(
    user_data: dict,
    *,
    sample_frac: float,
    random_seed: int,
) -> dict:
    """Deterministically sample users; keep every held-out test case for each."""

    frac = float(sample_frac)
    if not 0.0 < frac <= 1.0:
        raise ValueError("sample_frac must be in the interval (0, 1].")
    if frac == 1.0:
        return user_data

    user_ids = sorted(user_data, key=str)
    if not user_ids:
        return {}

    n_keep = max(1, math.ceil(len(user_ids) * frac))
    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(len(user_ids))
    selected = {user_ids[int(position)] for position in order[:n_keep]}

    sampled = {user_id: user_data[user_id] for user_id in user_ids if user_id in selected}
    n_tasks = sum(info["n_test"] for info in sampled.values())
    print(
        f"Sampled {len(sampled)}/{len(user_ids)} users "
        f"({frac:.2%}, seed={int(random_seed)}); "
        f"{n_tasks} held-out evaluation tasks (all test cases per user)"
    )
    return sampled


def load_user_data(cfg: dict[str, Any]) -> dict:
    """build ICL user_data from a validated eval config dict."""

    ratings, scenarios = load_for_prediction(cfg["ratings_path"], cfg["scenarios_path"])
    scenario_cols = [f"s{i}_id" for i in range(10)]
    incomplete = int((ratings[scenario_cols].notna().sum(axis=1) != 10).sum())
    if incomplete:
        print(f"Warning: {incomplete} users missing some scenario slots")

    ds = cfg["dataset"]
    n_icl = int(ds["n_icl_examples"])
    # Missing/null icl_reference_k → this run's n_icl (same as load_json_config).
    reference_k = (
        n_icl if ds.get("icl_reference_k") is None else int(ds["icl_reference_k"])
    )
    # Default True; missing/null → True.
    align_raw = ds.get("align_to_k", True)
    align_to_k = True if align_raw is None else bool(align_raw)
    # align_to_k=True  → only predict the icl_reference_k held-out cases
    # align_to_k=False → predict all remaining cases for this n_icl (perm[n_icl:])
    split_kw: dict[str, Any] = {
        "n_examples": n_icl,
        "min_scenarios": ds["min_scenarios"],
        "random_seed": int(ds["seed"]),
    }
    if align_to_k:
        split_kw["icl_reference_k"] = reference_k
        print(
            f"align_to_k=True: predict cases for icl_reference_k={reference_k} "
            f"(n_icl_examples={n_icl})"
        )
    else:
        print(
            f"align_to_k=False: predict all remaining cases for "
            f"n_icl_examples={n_icl} (icl_reference_k={reference_k} recorded in path)"
        )
    user_data, _ = prepare_icl_splits(ratings, scenarios, **split_kw)
    user_data = sample_test_cases(
        user_data,
        sample_frac=float(ds["sample_frac"]),
        random_seed=int(ds["seed"]),
    )
    print(f"Users: {len(user_data)}")
    return user_data


# ----------------------------------------------------
# Result analysis data preparation.
# ----------------------------------------------------

# ``k{n_icl}_pr{icl_reference_k}`` — pr = prediction_reference case set.
_K_PR_DIR_RE = re.compile(r"^k(\d+)_pr(\d+)$")
_SKIP_DIR_RE = re.compile(r"^(analysis|__pycache__|\.)", re.I)


def parse_run_dirname(name: str) -> tuple[int, int] | None:
    """Parse ``k{N}_pr{R}`` → ``(n_icl_examples, icl_reference_k)``."""
    m = _K_PR_DIR_RE.match(str(name).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def run_dirname(n_icl_examples: int, icl_reference_k: int) -> str:
    """Prediction folder name: ``k{n_icl}_pr{icl_reference_k}``."""
    return f"k{int(n_icl_examples)}_pr{int(icl_reference_k)}"


def list_model_dirs(runs_root: Path) -> list[str]:
    """List model directory names under *runs_root*."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    names: list[str] = []
    for p in root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if _SKIP_DIR_RE.match(p.name) or parse_run_dirname(p.name) is not None:
            continue
        names.append(p.name)
    return sorted(names)


def _run_dirs(
    batch_dir: Path,
    *,
    k: int | None = None,
    pr: int | None = None,
) -> list[tuple[int, int, Path]]:
    """``(n_icl, pr, path)`` children of one ``<model>/<batch_id>/``."""
    if not batch_dir.is_dir():
        return []
    out: list[tuple[int, int, Path]] = []
    for c in batch_dir.iterdir():
        if not c.is_dir():
            continue
        parsed = parse_run_dirname(c.name)
        if parsed is None:
            continue
        n_icl, ref = parsed
        if k is not None and n_icl != int(k):
            continue
        if pr is not None and ref != int(pr):
            continue
        out.append((n_icl, ref, c))
    return sorted(out, key=lambda t: (t[0], t[1]))


def list_k_values(
    runs_root: Path,
    batch_id: str,
    *,
    pr: int | None = None,
) -> list[int]:
    """List distinct ``n_icl`` (k) values for a batch, optionally filtered by pr."""
    root = Path(runs_root)
    ks: set[int] = set()
    for model in list_model_dirs(root):
        for n_icl, _, _ in _run_dirs(root / model / batch_id, pr=pr):
            ks.add(n_icl)
    return sorted(ks)


def list_pr_values(runs_root: Path, batch_id: str) -> list[int]:
    """List distinct ``icl_reference_k`` (pr) values for a batch."""
    root = Path(runs_root)
    prs: set[int] = set()
    for model in list_model_dirs(root):
        for _, ref, _ in _run_dirs(root / model / batch_id):
            prs.add(ref)
    return sorted(prs)


def _setup_from_prediction_stem(stem: str, setups: list[str]) -> str | None:
    if stem in setups:
        return stem
    for setup in setups:
        if setup == stem:
            return setup
    for setup in sorted(setups, key=len, reverse=True):
        if stem.endswith(f"_{setup}"):
            return setup
    return None


def resolve_prediction_path(directory: Path, setup: str) -> Path | None:
    directory = Path(directory)
    setup = str(setup)
    candidates = [directory / f"{setup}.jsonl"]
    for exact in candidates:
        if exact.is_file():
            return exact
    if not directory.is_dir():
        return None

    known = list(SETUP_ORDER)
    if setup not in known:
        known.append(setup)
    matches = [
        p
        for p in sorted(directory.glob("*.jsonl"))
        if p.is_file() and _setup_from_prediction_stem(p.stem, known) == setup
    ]
    return matches[0] if matches else None


def load_results(
    runs_root: str | Path,
    model: str,
    batch_id: str,
    *,
    k: int | None = None,
    pr: int | None = None,
    setups: str | list[str] | None = None,
) -> list[dict]:
    """Locate prediction files for one model/batch (paths only, no file I/O).

    Directories are ``k{n_icl}_pr{icl_reference_k}/``.
    """
    batch_dir = Path(runs_root) / str(model) / str(batch_id)
    if setups is None:
        wanted = list(SETUP_ORDER)
    else:
        wanted = [setups] if isinstance(setups, str) else list(setups)

    levels: list[tuple[int | None, int | None, Path]] = list(
        _run_dirs(batch_dir, k=k, pr=pr)
    )
    if not levels and k is None and pr is None:
        levels = [(None, None, batch_dir)]

    rows: list[dict] = []
    for n_icl, ref, directory in levels:
        if not directory.is_dir():
            continue
        for setup in wanted:
            path = resolve_prediction_path(directory, setup)
            if path is None:
                continue
            rows.append(
                {
                    "model": str(model),
                    "batch_id": str(batch_id),
                    "k": n_icl,
                    "pr": ref,
                    "setup": setup,
                    "save_path": str(path),
                }
            )
    return rows


def _prediction_case_key_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Return (user_col, scenario_col) for a prediction table."""
    user_col = "user_id" if "user_id" in df.columns else USER_ID_COL
    if user_col not in df.columns:
        raise KeyError(
            f"Prediction table missing user id column "
            f"(tried 'user_id' and {USER_ID_COL!r})"
        )
    if "scenario_id" not in df.columns:
        raise KeyError("Prediction table missing 'scenario_id' column")
    return user_col, "scenario_id"


def prediction_case_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Unique ``(user_id, scenario_id)`` pairs in a prediction table."""
    user_col, scenario_col = _prediction_case_key_columns(df)
    return {
        (str(u), str(s))
        for u, s in zip(df[user_col].tolist(), df[scenario_col].tolist())
    }


def filter_df_to_case_keys(
    df: pd.DataFrame, keys: set[tuple[str, str]]
) -> pd.DataFrame:
    """Keep only rows whose ``(user_id, scenario_id)`` is in *keys*."""
    if df.empty or not keys:
        return df.iloc[0:0].copy()
    user_col, scenario_col = _prediction_case_key_columns(df)
    mask = [
        (str(u), str(s)) in keys
        for u, s in zip(df[user_col].tolist(), df[scenario_col].tolist())
    ]
    return df.loc[mask].reset_index(drop=True)


def expected_case_keys_for_pr(
    icl_reference_k: int,
    *,
    ratings_path: str | Path | None = None,
    scenarios_path: str | Path | None = None,
    min_scenarios: int = 8,
    random_seed: int | None = None,
) -> set[tuple[str, str]]:
    """Held-out ``(user, scenario)`` keys for a given ``icl_reference_k``."""
    ratings, scenarios = load_for_prediction(ratings_path, scenarios_path)
    user_data, _ = prepare_icl_splits(
        ratings,
        scenarios,
        n_examples=int(icl_reference_k),
        min_scenarios=int(min_scenarios),
        random_seed=random_seed,
        icl_reference_k=int(icl_reference_k),
    )
    keys: set[tuple[str, str]] = set()
    for user_id, info in user_data.items():
        for case in info["test_cases"]:
            keys.add((str(user_id), str(case["scenario_id"])))
    return keys


def filter_runs_to_pr_cases(
    runs: list[dict],
    *,
    ratings_path: str | Path | None = None,
    scenarios_path: str | Path | None = None,
    min_scenarios: int = 8,
    random_seed: int | None = None,
) -> list[dict]:
    """Keep only rows that belong to each run's ``pr`` (icl_reference_k) case set."""
    cache: dict[int, set[tuple[str, str]]] = {}
    out: list[dict] = []
    for run in runs:
        pr = run.get("pr")
        if pr is None:
            out.append(run)
            continue
        pr_i = int(pr)
        if pr_i not in cache:
            cache[pr_i] = expected_case_keys_for_pr(
                pr_i,
                ratings_path=ratings_path,
                scenarios_path=scenarios_path,
                min_scenarios=min_scenarios,
                random_seed=random_seed,
            )
            print(
                f"  pr={pr_i}: expected {len(cache[pr_i])} held-out "
                f"(user, scenario) cases"
            )
        keys = cache[pr_i]
        df = run["df"]
        before = len(df)
        filtered = filter_df_to_case_keys(df, keys)
        after = len(filtered)
        dropped = before - after
        # if dropped:
        #     print(
        #         f"  score filter pr={pr_i}: {run['model']} [{run['setup']}] "
        #         f"k={run.get('k')}: kept {after}/{before} cases "
        #         f"(dropped {dropped} not in icl_reference_k set)"
        #     )
        out.append({**run, "df": filtered})
    return out


def _k_independent_result(
    root: Path,
    model: str,
    batch_id: str,
    setup: str,
    *,
    pr: int | None,
    prefer_k: int | None,
    candidate_ks: list[int | None],
) -> dict | None:
    """Locate one Baseline (etc.) file for a given prediction reference."""
    ordered: list[int | None] = []
    if prefer_k is not None:
        ordered.append(int(prefer_k))
    for k in sorted((k for k in candidate_ks if k is not None), reverse=True):
        if k not in ordered:
            ordered.append(k)
    batch_dir = root / str(model) / str(batch_id)
    for n_icl, _, _ in reversed(_run_dirs(batch_dir, pr=pr)):
        if n_icl not in ordered:
            ordered.append(n_icl)
    for k in ordered:
        rows = load_results(
            root, model, batch_id, k=k, pr=pr, setups=[setup]
        )
        if rows:
            return rows[0]
    return None


def load_runs(
    runs_root: str | Path,
    batch_id: str,
    *,
    ks: list[int | None] | None = None,
    pr: int | None = None,
    models: list[str] | None = None,
    setups: str | list[str] | None = None,
    filter_to_pr_cases: bool = True,
    ratings_path: str | Path | None = None,
    scenarios_path: str | Path | None = None,
    min_scenarios: int = 8,
    random_seed: int | None = None,
) -> list[dict]:
    """Discover prediction files and load their tables for analysis.

    Each item: ``model``, ``batch_id``, ``k``, ``pr``, ``setup``, ``save_path``, ``df``.

    Directories are ``k{n_icl}_pr{icl_reference_k}/``. Baseline is loaded once per
    ``(model, pr)``. When ``filter_to_pr_cases`` is True (default), every table is
    restricted to the held-out case set for its ``pr`` before scoring.
    """
    root = Path(runs_root)
    wanted_models = list(models) if models is not None else list_model_dirs(root)
    if setups is None:
        wanted_setups = list(SETUP_ORDER)
    else:
        wanted_setups = [setups] if isinstance(setups, str) else list(setups)

    k_independent = [s for s in wanted_setups if s in K_INDEPENDENT_SETUPS]
    k_dependent = [s for s in wanted_setups if s not in K_INDEPENDENT_SETUPS]

    if pr is None:
        found_prs = list_pr_values(root, batch_id)
        if len(found_prs) == 1:
            pr = found_prs[0]
            print(f"  using pr={pr} (only prediction reference found)")
        elif len(found_prs) > 1:
            print(
                f"  [warn] multiple pr values {found_prs}; "
                "pass pr= to select one (loading all)"
            )

    if ks is None:
        found = list_k_values(root, batch_id, pr=pr)
        ks_list: list[int | None] = list(found) if found else [None]
    else:
        ks_list = list(ks)
        available = list_k_values(root, batch_id, pr=pr)
        unknown = [
            k for k in ks_list if k is not None and available and k not in available
        ]
        if unknown:
            print(f"  [warn] no k{unknown} under {root} (have {available})")

    loaded: list[dict] = []
    for k in ks_list:
        if not k_dependent:
            continue
        runs = [
            run
            for model in wanted_models
            for run in load_results(
                root, model, batch_id, k=k, pr=pr, setups=k_dependent
            )
        ]
        if not runs:
            print(
                "  [warn] no runs found"
                + (f" for k={k}" if k is not None else "")
                + (f" pr={pr}" if pr is not None else "")
            )
            continue

        scored: set[tuple[str, str]] = set()
        for run in runs:
            path = Path(run["save_path"])
            df = load_prediction_jsonl(path)
            if df is None:
                print(f"  missing predictions: {path}")
                continue
            loaded.append({**run, "df": df})
            scored.add((str(run["model"]), str(run["setup"])))

        for model, setup in sorted(
            {(m, s) for m in wanted_models for s in k_dependent} - scored
        ):
            print(
                f"  [warn] no predictions for {model} [{setup}]"
                + (f" k={k}" if k is not None else "")
                + (f" pr={pr}" if pr is not None else "")
            )

    prefer_k = int(pr) if pr is not None else None
    for setup in k_independent:
        for model in wanted_models:
            run = _k_independent_result(
                root,
                model,
                batch_id,
                setup,
                pr=pr,
                prefer_k=prefer_k,
                candidate_ks=ks_list,
            )
            if run is None:
                print(f"  [warn] no predictions for {model} [{setup}] (k-independent)")
                continue
            path = Path(run["save_path"])
            df = load_prediction_jsonl(path)
            if df is None:
                print(f"  missing predictions: {path}")
                continue
            loaded.append({**run, "df": df})
            print(
                f"  {model} [{setup}]: using k-independent file "
                f"(k={run.get('k')}_pr{run.get('pr')}) → {path.name}"
            )

    if not loaded:
        raise FileNotFoundError(
            f"No prediction files under {root} (batch {batch_id!r}) "
            f"(ks={ks_list}, pr={pr}, setups={wanted_setups}, models={wanted_models})"
        )

    if filter_to_pr_cases:
        loaded = filter_runs_to_pr_cases(
            loaded,
            ratings_path=ratings_path,
            scenarios_path=scenarios_path,
            min_scenarios=min_scenarios,
            random_seed=random_seed,
        )

    return loaded
