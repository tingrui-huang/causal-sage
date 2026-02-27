"""
Batch-generate multi-sample-size datasets for:
  alarm, child, sachs, insurance, hailfinder, win95pts

Outputs for each dataset and each N:
  - Neuro-Symbolic-Reasoning/data/<dataset>/<dataset>_data_<N>.csv
  - <project_root>/<dataset>_data_variable_<N>.csv
  - Neuro-Symbolic-Reasoning/data/<dataset>/metadata_<N>.json

Compatibility (default enabled):
  If N == 10000, also writes legacy filenames used by older configs/scripts.

Usage examples:
  python Neuro-Symbolic-Reasoning/scripts/generate_multi_sample_size_data.py
  python Neuro-Symbolic-Reasoning/scripts/generate_multi_sample_size_data.py --datasets alarm child sachs insurance hailfinder win95pts --sizes 1000 2000 5000 10000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["alarm"]
DEFAULT_SIZES = [1000, 2000, 5000, 10000]


def _parse_bif_variable_order(bif_path: Path) -> List[str]:
    """Parse variable declaration order from BIF text."""
    text = bif_path.read_text(encoding="utf-8", errors="ignore")
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^variable\s+([A-Za-z0-9_]+)\s*\{", line)
        if m:
            out.append(m.group(1))
    if not out:
        raise ValueError(f"Failed to parse variable order from BIF: {bif_path}")
    return out


def _load_bif_model_with_states(dataset: str, ns_root: Path):
    """
    Load pgmpy model from BIF and return:
      model, variable_order, var_to_states
    """
    try:
        from pgmpy.readwrite import BIFReader
    except ImportError as e:
        raise RuntimeError("pgmpy is required. Install with: pip install pgmpy") from e

    bif_path = ns_root / "data" / dataset / f"{dataset}.bif"
    if not bif_path.exists():
        raise FileNotFoundError(f"Missing BIF file: {bif_path}")

    reader = BIFReader(str(bif_path))
    model = reader.get_model()
    variable_order = _parse_bif_variable_order(bif_path)

    var_to_states: Dict[str, List[str]] = {}
    for var in variable_order:
        cpd = model.get_cpds(var)
        if cpd is None:
            raise ValueError(f"Missing CPD for variable '{var}' in dataset '{dataset}'")
        names = cpd.state_names.get(var)
        if names is None:
            raise ValueError(f"Missing state names for variable '{var}' in dataset '{dataset}'")
        var_to_states[var] = [str(x) for x in names]

    return model, variable_order, var_to_states


def _sample_from_bif_model(model, n_samples: int, seed: int, variable_order: List[str]) -> pd.DataFrame:
    try:
        from pgmpy.sampling import BayesianModelSampling
    except ImportError as e:
        raise RuntimeError("pgmpy is required. Install with: pip install pgmpy") from e

    np.random.seed(int(seed))
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=int(n_samples), show_progress=False)
    return df[variable_order].copy()


def _load_sachs_reference(seed: int) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    """
    Load Sachs observational reference (854 rows), preserving the original script behavior.
    """
    try:
        import bnlearn as bn
    except ImportError as e:
        raise RuntimeError("bnlearn is required for sachs. Install with: pip install bnlearn") from e

    np.random.seed(int(seed))
    df = bn.import_example("sachs")
    obs_df = df.head(854).copy()
    variable_order = list(obs_df.columns)
    var_to_states = {v: [str(x) for x in sorted(obs_df[v].astype(str).unique())] for v in variable_order}
    return obs_df, variable_order, var_to_states


def _sample_sachs(obs_df: pd.DataFrame, n_samples: int, seed: int, variable_order: List[str]) -> pd.DataFrame:
    """
    For N <= 854: sample without replacement.
    For N > 854: sample with replacement (bootstrap) to reach target N.
    """
    rng = np.random.default_rng(int(seed))
    n_ref = len(obs_df)
    replace = int(n_samples) > n_ref
    idx = rng.choice(n_ref, size=int(n_samples), replace=replace)
    out = obs_df.iloc[idx].reset_index(drop=True).copy()
    return out[variable_order]


def _build_onehot_and_variable(
    df_raw: pd.DataFrame,
    variable_order: List[str],
    var_to_states: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, str]]]:
    """
    Build deterministic one-hot and variable-level DataFrames from raw categorical data.
    """
    n_rows = len(df_raw)
    onehot_cols: List[str] = []
    for v in variable_order:
        onehot_cols.extend([f"{v}_{s}" for s in var_to_states[v]])

    onehot_out = np.zeros((n_rows, len(onehot_cols)), dtype=np.uint8)
    variable_out = np.zeros((n_rows, len(variable_order)), dtype=np.int16)

    state_mappings: Dict[str, Dict[str, str]] = {}
    col_offset = 0
    for vi, var in enumerate(variable_order):
        states = var_to_states[var]
        s2i = {str(s): i for i, s in enumerate(states)}
        state_mappings[var] = {str(i): f"{var}_{states[i]}" for i in range(len(states))}

        vals = df_raw[var].astype(str).to_numpy()
        idx = np.fromiter((s2i[x] for x in vals), dtype=np.int32, count=len(vals))
        variable_out[:, vi] = idx.astype(np.int16)
        onehot_out[np.arange(n_rows), col_offset + idx] = 1
        col_offset += len(states)

    onehot_df = pd.DataFrame(onehot_out, columns=onehot_cols)
    variable_df = pd.DataFrame(variable_out, columns=variable_order)
    return onehot_df, variable_df, state_mappings


def _write_metadata(
    *,
    dataset: str,
    variable_order: List[str],
    state_mappings: Dict[str, Dict[str, str]],
    n_samples: int,
    onehot_cols: int,
    source_file: str,
    out_path: Path,
) -> None:
    meta = {
        "dataset_name": dataset,
        "n_variables": len(variable_order),
        "n_states": int(onehot_cols),
        "variable_names": variable_order,
        "state_mappings": state_mappings,
        "data_format": "one_hot_csv",
        "source_file": source_file,
        "n_samples": int(n_samples),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _legacy_paths_for_10k(dataset: str, project_root: Path, ns_root: Path):
    """
    Return legacy paths for N=10000 compatibility.
    """
    ds_dir = ns_root / "data" / dataset
    # One-hot legacy names in current codebase
    onehot_legacy_map = {
        "alarm": ds_dir / "alarm_data_10000.csv",
        "insurance": ds_dir / "insurance_data_10000.csv",
        "child": ds_dir / "child_data.csv",
        "sachs": ds_dir / "sachs_data.csv",
        "hailfinder": ds_dir / "hailfinder_data.csv",
        "win95pts": ds_dir / "win95pts_data.csv",
        "andes": ds_dir / "andes_data.csv",
    }
    fci_legacy_map = {
        "alarm": project_root / "alarm_data.csv",
        "insurance": project_root / "insurance_data.csv",
        "child": project_root / "child_data_variable.csv",
        "sachs": project_root / "sachs_data_variable.csv",
        "hailfinder": project_root / "hailfinder_data_variable.csv",
        "win95pts": project_root / "win95pts_data_variable.csv",
        "andes": project_root / "andes_data_variable.csv",
    }
    return onehot_legacy_map[dataset], fci_legacy_map[dataset], ds_dir / "metadata.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seed_offset_by_size",
        action="store_true",
        help="If set, use (seed + sample_size) for each N. Default uses the same seed for all N.",
    )
    ap.add_argument("--write_legacy_for_10000", action="store_true", default=True)
    ap.add_argument("--no_write_legacy_for_10000", action="store_true")
    args = ap.parse_args()

    if args.no_write_legacy_for_10000:
        write_legacy_for_10000 = False
    else:
        write_legacy_for_10000 = bool(args.write_legacy_for_10000)

    script_path = Path(__file__).resolve()
    ns_root = script_path.parents[1]
    project_root = script_path.parents[2]

    datasets = [str(d).lower() for d in args.datasets]
    sizes = [int(n) for n in args.sizes]
    seed = int(args.seed)
    use_seed_offset_by_size = bool(args.seed_offset_by_size)

    print("=" * 90)
    print("GENERATE MULTI SAMPLE-SIZE DATASETS")
    print("=" * 90)
    print(f"Datasets: {datasets}")
    print(f"Sizes:    {sizes}")
    print(f"Seed:     {seed}")
    print(f"Seed strategy: {'seed+N' if use_seed_offset_by_size else 'fixed seed'}")
    print(f"Legacy 10k outputs: {write_legacy_for_10000}")
    print("=" * 90)

    for dataset in datasets:
        if dataset not in set(DEFAULT_DATASETS):
            print(f"\n[SKIP] Unsupported dataset: {dataset}")
            continue

        print("\n" + "-" * 90)
        print(f"DATASET: {dataset}")
        print("-" * 90)

        if dataset == "sachs":
            ref_df, variable_order, var_to_states = _load_sachs_reference(seed=seed)
            sampler = lambda n, s: _sample_sachs(ref_df, n, s, variable_order)  # noqa: E731
            print(f"[INFO] Sachs reference rows: {len(ref_df)} (obs-only, bootstrap if N>{len(ref_df)})")
        else:
            model, variable_order, var_to_states = _load_bif_model_with_states(dataset, ns_root)
            sampler = lambda n, s: _sample_from_bif_model(model, n, s, variable_order)  # noqa: E731

        ds_dir = ns_root / "data" / dataset
        ds_dir.mkdir(parents=True, exist_ok=True)

        for n in sizes:
            run_seed = seed + int(n) if use_seed_offset_by_size else seed
            print(f"\n[N={n}] sampling with seed={run_seed} ...")
            df_raw = sampler(int(n), run_seed)

            onehot_df, variable_df, state_mappings = _build_onehot_and_variable(
                df_raw=df_raw,
                variable_order=variable_order,
                var_to_states=var_to_states,
            )

            onehot_path = ds_dir / f"{dataset}_data_{n}.csv"
            variable_path = project_root / f"{dataset}_data_variable_{n}.csv"
            metadata_n_path = ds_dir / f"metadata_{n}.json"

            onehot_df.to_csv(onehot_path, index=False)
            variable_df.to_csv(variable_path, index=False)
            _write_metadata(
                dataset=dataset,
                variable_order=variable_order,
                state_mappings=state_mappings,
                n_samples=int(n),
                onehot_cols=len(onehot_df.columns),
                source_file=onehot_path.name,
                out_path=metadata_n_path,
            )

            print(f"  [OK] one-hot:   {onehot_path}")
            print(f"  [OK] variable:  {variable_path}")
            print(f"  [OK] metadata:  {metadata_n_path}")

            if write_legacy_for_10000 and int(n) == 10000:
                onehot_legacy, fci_legacy, meta_legacy = _legacy_paths_for_10k(
                    dataset=dataset,
                    project_root=project_root,
                    ns_root=ns_root,
                )
                onehot_legacy.parent.mkdir(parents=True, exist_ok=True)
                fci_legacy.parent.mkdir(parents=True, exist_ok=True)
                meta_legacy.parent.mkdir(parents=True, exist_ok=True)

                if onehot_path.resolve() != onehot_legacy.resolve():
                    shutil.copy2(onehot_path, onehot_legacy)
                shutil.copy2(variable_path, fci_legacy)
                shutil.copy2(metadata_n_path, meta_legacy)

                print(f"  [OK] legacy one-hot:  {onehot_legacy}")
                print(f"  [OK] legacy fci:      {fci_legacy}")
                print(f"  [OK] legacy metadata: {meta_legacy}")

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

