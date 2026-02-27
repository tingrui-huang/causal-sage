"""
Experiment: LLM as "Intelligent Guide" vs "Blind Perturbation"

Question: Does LLM direction matter, or just any asymmetric initialization?

Setup:
    1. Baseline (LLM): Use LLM direction prior (normal training)
    2. Control (Random): Use random direction prior (same magnitude, random direction)

Expected Results:
    - If LLM is just "blind perturbation":
        Both should have similar Unresolved Ratio AND Orientation Accuracy
    
    - If LLM is "intelligent guide":
        Random: Unresolved Ratio -> 0%, but Orientation Accuracy ~50% (random guessing)
        LLM: Unresolved Ratio -> 0%, and Orientation Accuracy >> 50% (guided)

UPDATED: Now uses weaker initialization (0.6/0.4 instead of 0.7/0.3)
         Tests on multiple datasets: Alarm, Sachs
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Import config to get dataset paths
import config
from train_complete import train_complete

# Optional postprocess: v-structure hard mask (scheme A)
from modules.vstructure_postprocess import postprocess_vstructure_on_run_dir


def run_experiment_for_dataset(dataset_name: str, 
                               high_conf: float = 0.6,
                               low_conf: float = 0.4,
                               n_epochs: int = 200,
                               run_mode: str = 'both',
                               sample_size: Optional[int] = None,
                               random_seed: Optional[int] = None,
                               auto_generate_constraint: bool = True,
                               vstructure_postprocess: bool = False,
                               vstructure_fci_csv_path: Optional[str] = None,
                               run_id: Optional[str] = None,
                               vstructure_in_mask: bool = False,
                               dag_check: bool = False,
                               dag_project_on_cycle: bool = False,
                               tie_blocks: bool = False,
                               tie_method: str = "mean",
                               reconstruction_mode: str = "bce",
                               pred_mode: Optional[str] = None,
                               lambda_group_override: Optional[float] = None,
                               lambda_cycle_override: Optional[float] = None,
                               lambda_skeleton_override: Optional[float] = None,
                               batch_size: Optional[int] = None):
    """
    Run LLM vs Random experiment for a specific dataset
    
    Args:
        dataset_name: Name of dataset ('alarm', 'sachs', etc.)
        high_conf: High confidence weight (default: 0.6, weaker than 0.7)
        low_conf: Low confidence weight (default: 0.4, weaker than 0.3)
        n_epochs: Number of training epochs
        run_mode: 'both' | 'llm' | 'random'
        sample_size: Optional sample size override for sample-sweep datasets.
                    Ignored by pigs/link and other fixed-size datasets.
        random_seed: Random seed for reproducibility (training + random prior)
        auto_generate_constraint: If True, auto-run FCI/RFCI when constraint outputs are missing.
        vstructure_postprocess: Whether to apply v-structure hard mask after training (scheme A)
        vstructure_fci_csv_path: Optional PAG CSV path (must include edge_type). If None, uses pure skeleton path.
        run_id: Optional run identifier used for outputs folder naming. If None, uses a timestamp.
        vstructure_in_mask: If True, enforce inferred v-structure constraints directly on the skeleton mask during training.
        dag_check: If True, run DAG check (directedness + acyclicity) after training and save complete_dag_check.json.
        dag_project_on_cycle: If True, when cyclic, cut weakest edge(s) on cycles until acyclic; saves complete_*_dag artifacts.
        tie_blocks: If True, tie all state-to-state weights within each variable-pair block (ablation: remove state-level DoF).
        tie_method: Aggregation method used to tie a block ("mean" or "max").
        reconstruction_mode: Reconstruction loss mode ("bce" or "group_ce").
        pred_mode: Optional model prediction mode override. If None, chosen based on reconstruction_mode.
        lambda_group_override: Optional override for lambda_group (group lasso weight).
        lambda_cycle_override: Optional override for lambda_cycle (cycle consistency weight).
        lambda_skeleton_override: Optional override for lambda_skeleton (skeleton preservation weight).
        batch_size: Optional mini-batch size. If None, uses full-batch training.
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {dataset_name.upper()} DATASET")
    print("=" * 80)
    print(f"Prior weights: High={high_conf}, Low={low_conf}")
    print("=" * 80)
    
    # Default: follow unified config (single source of truth)
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    random_seed = int(random_seed)

    # Output run id (timestamp by default) to avoid overwriting prior runs.
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    run_id = str(run_id)
    
    # Get dataset configuration (sample-size aware for eligible datasets)
    base_dataset_cfg = config.DATASET_CONFIGS[dataset_name]
    dataset_config = config.resolve_dataset_paths(
        dataset_name,
        base_dataset_cfg,
        sample_size=sample_size,
    )
    resolved_sample_size = dataset_config.get("sample_size")
    n_tag = f"n_{resolved_sample_size}" if resolved_sample_size is not None else "n_fixed"
    
    # Auto-detect latest skeleton and (optional) LLM direction files.
    #
    # IMPORTANT:
    # - pigs/link may use RFCI instead of FCI (edges_RFCI_*.csv).
    # - Some datasets/runs may NOT have LLM outputs at all. In that case we should still
    #   be able to run Random Prior only.
    # NOTE: edges_FCI_*.csv would also match edges_FCI_LLM_*.csv.
    # For "pure" constraint skeleton we must EXCLUDE any file containing "LLM".
    def _auto_detect_latest_non_llm(patterns, directory):
        from pathlib import Path
        d = Path(directory)
        if not d.exists():
            return None
        for pat in patterns:
            hits = [p for p in d.glob(pat) if "LLM" not in p.name.upper()]
            if hits:
                latest = max(hits, key=lambda p: p.stat().st_mtime)
                return str(latest)
        return None

    def _run_constraint_discovery_for_dataset() -> bool:
        """
        Run dataset-appropriate constraint discovery (FCI or RFCI) to generate
        edges files under refactored/outputs/<dataset>/[n_<size>/].
        """
        try:
            project_root = Path(__file__).parent.parent
            refactored_dir = project_root / "refactored"

            if not refactored_dir.exists():
                print(f"[ERROR] Missing refactored directory: {refactored_dir}")
                return False

            constraint_algo = str(dataset_config.get("constraint_algo", "fci")).lower()
            script_name = "main_rfci.py" if constraint_algo == "rfci" else "main_fci.py"
            script_path = refactored_dir / script_name

            env = os.environ.copy()
            env["DATASET"] = str(dataset_name)
            if resolved_sample_size is not None:
                env["SAMPLE_SIZE"] = str(int(resolved_sample_size))

            print("\n" + "=" * 80)
            print("AUTO-GENERATE CONSTRAINT OUTPUTS")
            print("=" * 80)
            print(f"Dataset: {dataset_name}")
            print(f"Sample size: {resolved_sample_size}")
            print(f"Algorithm: {constraint_algo.upper()}")
            print(f"Script: {script_path}")
            print("=" * 80)

            res = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(refactored_dir),
                env=env,
                check=False,
            )
            if res.returncode != 0:
                print(f"[ERROR] Constraint discovery failed with code: {res.returncode}")
                return False
            return True
        except Exception as e:
            print(f"[ERROR] Failed to auto-generate constraint outputs: {e}")
            return False

    generated_constraint_once = False

    # IMPORTANT FOR SAMPLE-SIZE EXPERIMENTS:
    # If sample_size is provided, we must bind skeleton/LLM files to outputs/<dataset>/n_<size>/.
    # This prevents accidental mixing with "latest" files from other sample sizes.
    base_constraint_dir = Path(config.FCI_OUTPUT_DIR) / dataset_name
    if resolved_sample_size is not None:
        constraint_dir = base_constraint_dir / n_tag
        if not constraint_dir.exists():
            if auto_generate_constraint:
                print(f"\n[WARN] Missing constraint dir: {constraint_dir}")
                print("[INFO] Attempting to auto-run FCI/RFCI for this dataset + sample size...")
                if not _run_constraint_discovery_for_dataset():
                    print(f"[ERROR] Auto-generation failed for {dataset_name} @ {n_tag}")
                    return None
                generated_constraint_once = True
            else:
                print(f"\n[ERROR] Sample-size-specific constraint outputs dir not found: {constraint_dir}")
                print("For strict N-to-N comparison, generate/run FCI(RFCI)+LLM for this sample size first.")
                print(f"Expected dir: outputs/{dataset_name}/{n_tag}/")
                return None
    else:
        constraint_dir = base_constraint_dir

    pure_skeleton_path = _auto_detect_latest_non_llm(
        ["edges_RFCI_*.csv", "edges_FCI_*.csv"],
        constraint_dir,
    )
    llm_skeleton_path = config._auto_detect_latest_file_any(
        ["edges_RFCI_LLM_*.csv", "edges_FCI_LLM_*.csv"],
        constraint_dir,
    )

    # If directory exists but skeleton is still missing, optionally auto-generate once and retry.
    if not pure_skeleton_path and auto_generate_constraint and not generated_constraint_once:
        print(f"\n[WARN] No constraint skeleton found under {constraint_dir}")
        print("[INFO] Attempting to auto-run FCI/RFCI...")
        if _run_constraint_discovery_for_dataset():
            pure_skeleton_path = _auto_detect_latest_non_llm(
                ["edges_RFCI_*.csv", "edges_FCI_*.csv"],
                constraint_dir,
            )
            llm_skeleton_path = config._auto_detect_latest_file_any(
                ["edges_RFCI_LLM_*.csv", "edges_FCI_LLM_*.csv"],
                constraint_dir,
            )

    # -------------------------------------------------------------------------
    # Training cache: if an experiment output_dir already contains saved results,
    # skip re-training and just load metrics/history.
    # -------------------------------------------------------------------------
    def _try_load_cached_training(output_dir: Union[str, Path]) -> Optional[Dict]:
        out = Path(output_dir)
        metrics_path = out / "complete_metrics.json"
        history_path = out / "complete_history.json"
        if metrics_path.exists() and history_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                history = json.loads(history_path.read_text(encoding="utf-8"))
                # Optional cached v-structure postprocess results
                v_metrics_path = out / "complete_metrics_vstructure.json"
                v_stats_path = out / "complete_vstructure_stats.json"
                v_metrics = None
                v_stats = None
                if v_metrics_path.exists():
                    try:
                        v_metrics = json.loads(v_metrics_path.read_text(encoding="utf-8"))
                    except Exception:
                        v_metrics = None
                if v_stats_path.exists():
                    try:
                        v_stats = json.loads(v_stats_path.read_text(encoding="utf-8"))
                    except Exception:
                        v_stats = None
                return {
                    "metrics": metrics,
                    "history": history,
                    "model": None,  # not needed for evaluation reporting here
                    "fci_baseline_unresolved_ratio": None,
                    "vstructure_metrics": v_metrics,
                    "vstructure_stats": v_stats,
                    "cached": True,
                    "output_dir": str(out),
                }
            except Exception as e:
                print(f"[WARN] Failed to load cached results from {out}: {e}")
                return None
        return None

    def _maybe_run_vstructure_postprocess(*, out_dir: Union[str, Path], pag_csv_path: Optional[str], edge_threshold_val: float) -> Optional[Dict]:
        if not vstructure_postprocess:
            return None
        if not pag_csv_path:
            return None
        return postprocess_vstructure_on_run_dir(
            run_dir=str(out_dir),
            data_path=str(dataset_config["data_path"]),
            metadata_path=str(dataset_config["metadata_path"]),
            ground_truth_path=str(dataset_config["ground_truth_path"]),
            ground_truth_type=str(dataset_config.get("ground_truth_type", "bif")),
            pag_csv_path=str(pag_csv_path),
            edge_threshold=float(edge_threshold_val),
            force=False,
        )

    requested_mode = run_mode
    effective_mode = run_mode

    # If LLM files are missing, automatically fall back to random prior.
    if effective_mode in ["both", "llm"] and not llm_skeleton_path:
        if effective_mode == "llm":
            print(f"\n[WARN] No LLM skeleton found for {dataset_name}. Falling back to RANDOM prior only.")
        else:
            print(f"\n[WARN] No LLM skeleton found for {dataset_name}. Running RANDOM prior only.")
        effective_mode = "random"

    # For random/both we must have at least a constraint skeleton (RFCI/FCI).
    if effective_mode in ["both", "random"] and not pure_skeleton_path:
        print(f"\n[ERROR] Missing constraint skeleton for {dataset_name}.")
        print(f"Expected one of: edges_RFCI_*.csv or edges_FCI_*.csv under {constraint_dir}")
        print("Please run the pipeline first:")
        print(f"  1. Set DATASET = '{dataset_name}' in config.py")
        if resolved_sample_size is not None:
            print(f"  2. Set SAMPLE_SIZE = {resolved_sample_size} in config.py")
        print("  3. Run: python run_pipeline.py  (or run refactored/main_rfci.py or refactored/main_fci.py)")
        return None

    print("\nUsing files:")
    print(f"  Data path:       {dataset_config['data_path']}")
    print(f"  Metadata path:   {dataset_config['metadata_path']}")
    print(f"  Sample size tag: {n_tag}")
    print(f"  Constraint dir:  {constraint_dir}")
    print(f"  Skeleton (for Random Prior): {pure_skeleton_path}")
    print(f"  LLM skeleton (for LLM Prior): {llm_skeleton_path}")
    print(f"  Ground truth:    {dataset_config['ground_truth_path']}")
    print(f"  Requested mode:  {requested_mode} -> Effective mode: {effective_mode}")

    # Get dataset-specific hyperparameters
    if dataset_name == 'sachs':
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.1
    elif dataset_name == 'alarm':
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.1
    elif dataset_name == 'andes':
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.1
    elif dataset_name == 'child':
        lambda_group = 0.005
        lambda_cycle = 5
        edge_threshold = 0.1
    elif dataset_name == 'hailfinder':
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.08
    elif dataset_name == 'win95pts':
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.1
    elif dataset_name == 'insurance':
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.1
    else:
        lambda_group = 0.01
        lambda_cycle = 5
        edge_threshold = 0.1

    # Shared configuration (without skeleton paths - will be set per experiment)
    # Choose prediction mode automatically unless overridden.
    if pred_mode is None:
        if str(reconstruction_mode) == "group_ce":
            pred_mode = "paper_logits"
        else:
            pred_mode = "propagate"

    base_config = {
        'dataset_name': dataset_name,
        'sample_size': resolved_sample_size,
        'data_path': str(dataset_config['data_path']),
        'metadata_path': str(dataset_config['metadata_path']),
        'ground_truth_path': str(dataset_config['ground_truth_path']),
        'ground_truth_type': dataset_config.get('ground_truth_type', 'bif'),
        'n_epochs': n_epochs,
        'learning_rate': 0.01,
        'n_hops': 1,
        'lambda_group': lambda_group,
        'lambda_cycle': lambda_cycle,
        'lambda_skeleton': float(lambda_skeleton_override) if lambda_skeleton_override is not None else 0.1,
        'monitor_interval': 20,
        'edge_threshold': edge_threshold,
        'high_confidence': high_conf,  # Pass to prior builder
        'low_confidence': low_conf,    # Pass to prior builder
        'random_seed': random_seed,    # For reproducibility (training + random prior)
        'dag_check': bool(dag_check),
        'dag_project_on_cycle': bool(dag_project_on_cycle),
        # Ablation knobs (passed through to CausalDiscoveryModel via train_complete)
        'tie_blocks': bool(tie_blocks),
        'tie_method': str(tie_method),
        # Paper-vs-code reconstruction modes
        'reconstruction_mode': str(reconstruction_mode),
        'pred_mode': str(pred_mode),
        # Mini-batch
        'batch_size': int(batch_size) if batch_size is not None else None,
    }

    # Optional lambda overrides (for quick ablations)
    if lambda_group_override is not None:
        base_config['lambda_group'] = float(lambda_group_override)
    if lambda_cycle_override is not None:
        base_config['lambda_cycle'] = float(lambda_cycle_override)

    # ============================================================================
    # Run experiments based on run_mode
    # ============================================================================
    results_llm = None
    results_random = None

    if effective_mode in ['both', 'llm']:
        # Experiment 1: LLM Prior (Baseline)
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {dataset_name.upper()} - LLM PRIOR (Intelligent Guide)")
        print("=" * 80)

        config_llm = base_config.copy()
        config_llm.update({
            # IMPORTANT: For fair comparison, hard skeleton must come from PURE FCI/RFCI (no LLM).
            # The only difference vs random should be the *direction prior* initialization.
            'fci_skeleton_path': str(pure_skeleton_path),
            'llm_direction_path': str(llm_skeleton_path),     # LLM CSV used ONLY for soft direction prior
            'use_llm_prior': True,
            'use_random_prior': False,
            # Optional: training-time v-structure hard mask (must be identical across LLM vs Random runs)
            'enforce_vstructure_mask': bool(vstructure_in_mask),
            'vstructure_pag_csv_path': str(vstructure_fci_csv_path) if vstructure_fci_csv_path else str(pure_skeleton_path),
            'output_dir': f'results/experiment_llm_vs_random/{dataset_name}/{n_tag}/seed_{random_seed}/{run_id}/llm_prior',
            'run_id': run_id,
        })

        cached = _try_load_cached_training(config_llm["output_dir"])
        if cached:
            print(f"\n[CACHE] Found existing LLM-prior training results at: {cached['output_dir']}")
            results_llm = cached
        else:
            results_llm = train_complete(config_llm)
        # Optional: v-structure postprocess (scheme A)
        # Prefer pure FCI/RFCI PAG (non-LLM) if available; otherwise fall back to llm_skeleton_path.
        pag_csv_for_v = vstructure_fci_csv_path or pure_skeleton_path or llm_skeleton_path
        vout = _maybe_run_vstructure_postprocess(out_dir=config_llm["output_dir"], pag_csv_path=pag_csv_for_v, edge_threshold_val=config_llm["edge_threshold"])
        if vout is not None:
            results_llm["vstructure_metrics"] = vout["metrics"]
            results_llm["vstructure_stats"] = vout["stats"]

    if effective_mode in ['both', 'random']:
        # Experiment 2: Random Prior (Control)
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {dataset_name.upper()} - RANDOM PRIOR (Blind Perturbation)")
        print("=" * 80)

        config_random = base_config.copy()
        config_random.update({
            'fci_skeleton_path': str(pure_skeleton_path),      # Use constraint skeleton (RFCI/FCI)
            'llm_direction_path': None,                        # Not used for random prior
            'use_llm_prior': False,
            'use_random_prior': True,
            'enforce_vstructure_mask': bool(vstructure_in_mask),
            # If caller provided a PAG CSV path use it; otherwise default to pure skeleton path.
            'vstructure_pag_csv_path': str(vstructure_fci_csv_path) if vstructure_fci_csv_path else str(pure_skeleton_path),
            'output_dir': f'results/experiment_llm_vs_random/{dataset_name}/{n_tag}/seed_{random_seed}/{run_id}/random_prior',
            'run_id': run_id,
        })
        
        cached = _try_load_cached_training(config_random["output_dir"])
        if cached:
            print(f"\n[CACHE] Found existing RANDOM-prior training results at: {cached['output_dir']}")
            results_random = cached
        else:
            results_random = train_complete(config_random)
        # Optional: v-structure postprocess (scheme A)
        pag_csv_for_v = vstructure_fci_csv_path or pure_skeleton_path
        vout = _maybe_run_vstructure_postprocess(out_dir=config_random["output_dir"], pag_csv_path=pag_csv_for_v, edge_threshold_val=config_random["edge_threshold"])
        if vout is not None:
            results_random["vstructure_metrics"] = vout["metrics"]
            results_random["vstructure_stats"] = vout["stats"]
    
    # ============================================================================
    # Comparison (only if both experiments were run)
    # ============================================================================
    if results_llm and results_random:
        print("\n" + "=" * 80)
        print(f"RESULTS COMPARISON - {dataset_name.upper()}")
        print("=" * 80)
        
        print("\n[Symmetry Breaking] Unresolved Ratio:")
        print(f"  LLM Prior:    {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%")
        print(f"  Random Prior: {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%")
        print(f"  Difference:   {(results_llm['history']['unresolved_ratio'][-1] - results_random['history']['unresolved_ratio'][-1])*100:+5.1f}%")
        
        print("\n[Orientation Accuracy] Direction Correctness:")
        print(f"  LLM Prior:    {results_llm['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"  Random Prior: {results_random['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"  Difference:   {(results_llm['metrics']['orientation_accuracy'] - results_random['metrics']['orientation_accuracy'])*100:+5.1f}%")

        # Optional: postprocess comparison
        if results_llm.get("vstructure_metrics") and results_random.get("vstructure_metrics"):
            llm_vm = results_llm["vstructure_metrics"]
            rnd_vm = results_random["vstructure_metrics"]
            print("\n[V-Structure Postprocess] (Scheme A)")
            print(f"  LLM OA:       {llm_vm.get('orientation_accuracy', 0)*100:5.1f}%")
            print(f"  Random OA:    {rnd_vm.get('orientation_accuracy', 0)*100:5.1f}%")
            print(f"  LLM full SHD: {llm_vm.get('full_shd', llm_vm.get('shd'))}")
            print(f"  Rnd full SHD: {rnd_vm.get('full_shd', rnd_vm.get('shd'))}")
        
        print("\n[Edge Metrics]:")
        print(f"  Edge F1:")
        print(f"    LLM:    {results_llm['metrics']['edge_f1']*100:5.1f}%")
        print(f"    Random: {results_random['metrics']['edge_f1']*100:5.1f}%")
        print(f"  Directed F1:")
        print(f"    LLM:    {results_llm['metrics']['directed_f1']*100:5.1f}%")
        print(f"    Random: {results_random['metrics']['directed_f1']*100:5.1f}%")
    elif results_llm:
        print("\n" + "=" * 80)
        print(f"RESULTS - {dataset_name.upper()} - LLM PRIOR ONLY")
        print("=" * 80)
        print(f"\nOrientation Accuracy: {results_llm['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"Edge F1:              {results_llm['metrics']['edge_f1']*100:5.1f}%")
        print(f"Directed F1:          {results_llm['metrics']['directed_f1']*100:5.1f}%")
        print(f"Unresolved Ratio:     {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%")
        if results_llm.get("vstructure_metrics"):
            m = results_llm["vstructure_metrics"]
            print("\n[V-Structure Postprocess] (Scheme A)")
            print(f"  OA:                 {m.get('orientation_accuracy', 0)*100:5.1f}%")
            print(f"  Full SHD:           {m.get('full_shd', m.get('shd'))}")
    elif results_random:
        print("\n" + "=" * 80)
        print(f"RESULTS - {dataset_name.upper()} - RANDOM PRIOR ONLY")
        print("=" * 80)
        print(f"\nOrientation Accuracy: {results_random['metrics']['orientation_accuracy']*100:5.1f}%")
        print(f"Edge F1:              {results_random['metrics']['edge_f1']*100:5.1f}%")
        print(f"Directed F1:          {results_random['metrics']['directed_f1']*100:5.1f}%")
        print(f"Unresolved Ratio:     {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%")
        if results_random.get("vstructure_metrics"):
            m = results_random["vstructure_metrics"]
            print("\n[V-Structure Postprocess] (Scheme A)")
            print(f"  OA:                 {m.get('orientation_accuracy', 0)*100:5.1f}%")
            print(f"  Full SHD:           {m.get('full_shd', m.get('shd'))}")
    
    # ============================================================================
    # Comparison summary (only if both experiments were run)
    # ============================================================================
    orientation_diff = 0
    unresolved_diff = 0
    
    if results_llm and results_random:
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        
        unresolved_diff = abs(results_llm['history']['unresolved_ratio'][-1] - 
                              results_random['history']['unresolved_ratio'][-1])
        orientation_diff = results_llm['metrics']['orientation_accuracy'] - \
                           results_random['metrics']['orientation_accuracy']
        print(f"\nUnresolved ratio diff (abs): {unresolved_diff*100:.1f}%")
        print(f"Orientation accuracy diff:    {orientation_diff*100:+.1f}%")
        print("Sign convention: positive Orientation diff means LLM-prior run is higher.")
    
    # ============================================================================
    # Save comparison report (only if both experiments were run)
    # ============================================================================
    if results_llm and results_random:
        output_dir = Path(f"results/experiment_llm_vs_random/{dataset_name}/{n_tag}/seed_{random_seed}/{run_id}")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        with open(output_dir / 'comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"LLM vs Random Prior Experiment - {dataset_name.upper()}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Prior Configuration:\n")
            f.write(f"  High confidence: {high_conf}\n")
            f.write(f"  Low confidence:  {low_conf}\n\n")
            
            f.write("Comparison Notes:\n")
            f.write("  This report compares two prior configurations within the same method family.\n")
            f.write("  Positive Orientation diff means LLM-prior run is higher.\n\n")
            
            f.write("Results:\n")
            f.write(f"  Symmetry Breaking (Unresolved Ratio):\n")
            f.write(f"    LLM:    {results_llm['history']['unresolved_ratio'][-1]*100:5.1f}%\n")
            f.write(f"    Random: {results_random['history']['unresolved_ratio'][-1]*100:5.1f}%\n")
            f.write(f"    Diff:   {(results_llm['history']['unresolved_ratio'][-1] - results_random['history']['unresolved_ratio'][-1])*100:+5.1f}%\n\n")
            
            f.write(f"  Orientation Accuracy:\n")
            f.write(f"    LLM:    {results_llm['metrics']['orientation_accuracy']*100:5.1f}%\n")
            f.write(f"    Random: {results_random['metrics']['orientation_accuracy']*100:5.1f}%\n")
            f.write(f"    Diff:   {orientation_diff*100:+5.1f}%\n\n")
            
            f.write(f"  Edge F1:\n")
            f.write(f"    LLM:    {results_llm['metrics']['edge_f1']*100:5.1f}%\n")
            f.write(f"    Random: {results_random['metrics']['edge_f1']*100:5.1f}%\n\n")
            
            f.write(f"  Directed F1:\n")
            f.write(f"    LLM:    {results_llm['metrics']['directed_f1']*100:5.1f}%\n")
            f.write(f"    Random: {results_random['metrics']['directed_f1']*100:5.1f}%\n\n")
            
            f.write("Summary:\n")
            f.write(f"  unresolved_diff_abs={unresolved_diff}\n")
            f.write(f"  orientation_diff={orientation_diff}\n")
        
        print(f"\nComparison report saved to: {output_dir / 'comparison_report.txt'}")
        print("\n" + "=" * 80)
    
    # ============================================================================
    # Save run report (ALWAYS, even for random-only)
    # ============================================================================
    seed_dir = Path(f"results/experiment_llm_vs_random/{dataset_name}/{n_tag}/seed_{random_seed}/{run_id}")
    seed_dir.mkdir(exist_ok=True, parents=True)
    report_path = seed_dir / "run_report.txt"

    def _safe_get_final_unresolved(res):
        try:
            return res["history"]["unresolved_ratio"][-1]
        except Exception:
            return None

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset_name}\n")
        f.write(f"sample_size={resolved_sample_size}\n")
        f.write(f"sample_size_tag={n_tag}\n")
        f.write(f"seed={random_seed}\n")
        f.write(f"run_id={run_id}\n")
        f.write(f"requested_mode={requested_mode}\n")
        f.write(f"effective_mode={effective_mode}\n")
        f.write(f"skeleton_path={pure_skeleton_path}\n")
        f.write(f"llm_skeleton_path={llm_skeleton_path}\n")
        f.write(f"data_path={dataset_config['data_path']}\n")
        f.write(f"metadata_path={dataset_config['metadata_path']}\n")
        f.write(f"ground_truth_path={dataset_config['ground_truth_path']}\n")
        f.write(f"high_confidence={high_conf}\n")
        f.write(f"low_confidence={low_conf}\n")
        f.write(f"n_epochs={n_epochs}\n")
        f.write("\n")

        if results_llm:
            f.write("[LLM PRIOR]\n")
            f.write(f"orientation_accuracy={results_llm['metrics'].get('orientation_accuracy')}\n")
            f.write(f"edge_f1={results_llm['metrics'].get('edge_f1')}\n")
            f.write(f"directed_f1={results_llm['metrics'].get('directed_f1')}\n")
            f.write(f"unresolved_ratio_final={_safe_get_final_unresolved(results_llm)}\n")
            f.write("\n")
        if results_random:
            f.write("[RANDOM PRIOR]\n")
            f.write(f"orientation_accuracy={results_random['metrics'].get('orientation_accuracy')}\n")
            f.write(f"edge_f1={results_random['metrics'].get('edge_f1')}\n")
            f.write(f"directed_f1={results_random['metrics'].get('directed_f1')}\n")
            f.write(f"unresolved_ratio_final={_safe_get_final_unresolved(results_random)}\n")
            f.write("\n")

        if results_llm and results_random:
            f.write("[COMPARISON]\n")
            unresolved_diff = abs((_safe_get_final_unresolved(results_llm) or 0) - (_safe_get_final_unresolved(results_random) or 0))
            orientation_diff = (results_llm["metrics"].get("orientation_accuracy") or 0) - (results_random["metrics"].get("orientation_accuracy") or 0)
            f.write(f"unresolved_diff={unresolved_diff}\n")
            f.write(f"orientation_diff={orientation_diff}\n")

    print(f"\n[OK] Saved run report: {report_path}")

    return {
        'dataset': dataset_name,
        'sample_size': resolved_sample_size,
        'sample_size_tag': n_tag,
        'seed': random_seed,
        'llm': results_llm,
        'random': results_random,
        'requested_mode': requested_mode,
        'effective_mode': effective_mode,
        'skeleton_path': str(pure_skeleton_path) if pure_skeleton_path else None,
        'llm_skeleton_path': str(llm_skeleton_path) if llm_skeleton_path else None,
        'report_path': str(report_path),
        'orientation_diff': orientation_diff,
        'unresolved_diff': unresolved_diff,
    }


def main():
    """Run experiments on multiple datasets"""
    # Optional CLI (mirrors run_multi_seed_random_prior.py style)
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--datasets", nargs="+", type=str, default=None,
                    help="Datasets to run (e.g. --datasets alarm sachs). If omitted, uses the defaults below.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Random seeds to run (e.g. --seeds 0 1 2). If omitted, uses the defaults below.")
    ap.add_argument("--sample_size", type=int, default=None,
                    help="Optional sample size override (e.g. --sample_size 2000).")
    ap.add_argument("--run_mode", type=str, default=None, choices=["both", "llm", "random"],
                    help="Which experiments to run: both | llm | random. If omitted, uses the defaults below.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Training epochs. If omitted, uses the defaults below.")
    ap.add_argument("--high_conf", type=float, default=None,
                    help="High confidence prior weight. If omitted, uses the defaults below.")
    ap.add_argument("--low_conf", type=float, default=None,
                    help="Low confidence prior weight. If omitted, uses the defaults below.")
    ap.add_argument("--vstructure_postprocess", action="store_true",
                    help="If set, apply v-structure hard mask postprocess after training (scheme A).")
    ap.add_argument("--vstructure_fci_csv", type=str, default=None,
                    help="Optional PAG CSV path (must include edge_type) for v-structure inference. Defaults to pure skeleton path if available.")
    ap.add_argument("--run_id", type=str, default=None,
                    help="Optional run id for outputs folder naming. If omitted, a timestamp is used.")
    tie_group = ap.add_mutually_exclusive_group()
    tie_group.add_argument(
        "--tie_blocks",
        action="store_true",
        help="Enable block-tied ablation: within each variable-pair block, tie all allowed state-to-state weights to one scalar.",
    )
    tie_group.add_argument(
        "--no_tie_blocks",
        action="store_true",
        help="Disable block-tied ablation (default behavior).",
    )
    ap.add_argument(
        "--tie_method",
        type=str,
        default=None,
        choices=["mean", "max"],
        help="Block tying aggregation method: mean | max. If omitted, uses the defaults below.",
    )
    ap.add_argument(
        "--reconstruction_mode",
        type=str,
        default=None,
        choices=["bce", "group_ce"],
        help="Reconstruction loss mode: bce (current code) | group_ce (paper-style per-variable softmax CE).",
    )
    ap.add_argument("--lambda_group", type=float, default=None,
                    help="Override: lambda_group (weighted group lasso coefficient).")
    ap.add_argument("--lambda_cycle", type=float, default=None,
                    help="Override: lambda_cycle (cycle consistency coefficient).")
    ap.add_argument("--lambda_skeleton", type=float, default=None,
                    help="Override: lambda_skeleton (skeleton preservation coefficient).")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Mini-batch size. If omitted, uses full-batch training.")
    vmask_group = ap.add_mutually_exclusive_group()
    vmask_group.add_argument(
        "--vstructure_in_mask",
        action="store_true",
        help="Override: enable training-time v-structure hard mask (in-skeleton). If omitted, uses the defaults below.",
    )
    vmask_group.add_argument(
        "--no_vstructure_in_mask",
        action="store_true",
        help="Override: disable training-time v-structure hard mask (in-skeleton). If omitted, uses the defaults below.",
    )
    dag_group = ap.add_mutually_exclusive_group()
    dag_group.add_argument(
        "--dag_check",
        action="store_true",
        help="Override: enable DAG check (directedness + acyclicity) after training. If omitted, uses the defaults below.",
    )
    dag_group.add_argument(
        "--no_dag_check",
        action="store_true",
        help="Override: disable DAG check. If omitted, uses the defaults below.",
    )
    proj_group = ap.add_mutually_exclusive_group()
    proj_group.add_argument(
        "--dag_project_on_cycle",
        action="store_true",
        help="Override: if a directed cycle exists, cut weakest edge(s) on cycles until acyclic; saves complete_*_dag artifacts. If omitted, uses defaults below.",
    )
    proj_group.add_argument(
        "--no_dag_project_on_cycle",
        action="store_true",
        help="Override: disable DAG projection-on-cycle. If omitted, uses defaults below.",
    )
    args = ap.parse_args()

    # 'alarm','insurance'
    datasets = ['insurance']

    run_mode = 'both'

    seeds: Union[int, List[int]] = [5]

    high_confidence = 0.9
    low_confidence = 0.1

    n_epochs = 140

    reconstruction_mode = "group_ce"

    lambda_group_override = None
    lambda_cycle_override = 5.0
    lambda_skeleton_override = None
    
    # Optional postprocess
    use_vstructure_postprocess = False
    vstructure_fci_csv_path = None
    run_id = None
    
    # Training-time v-structure hard mask (in-skeleton).
    # True  = enforce inferred colliders directly in the skeleton mask during training.
    # False = do not enforce (baseline).
    vstructure_in_mask = True
    
    # DAG check (post-training): saves complete_dag_check.json in each run dir.
    dag_check = True
    
    # If cyclic, optionally project to DAG by cutting weakest edge(s) on cycles.
    dag_project_on_cycle = True
    
    # Ablation: block-tied adjacency (removes state-level degrees of freedom while staying in one-hot space)
    tie_blocks = False
    tie_method = "mean"
    
    # Mini-batch size (None = full-batch training)
    batch_size = None
    sample_size = None
    # ============================================================================

    # Apply CLI overrides (if provided)
    if args.datasets is not None:
        datasets = args.datasets
    if args.run_mode is not None:
        run_mode = args.run_mode
    if args.seeds is not None:
        seeds = args.seeds
    if args.epochs is not None:
        n_epochs = int(args.epochs)
    if args.high_conf is not None:
        high_confidence = float(args.high_conf)
    if args.low_conf is not None:
        low_confidence = float(args.low_conf)
    if args.vstructure_postprocess:
        use_vstructure_postprocess = True
    if args.vstructure_fci_csv is not None:
        vstructure_fci_csv_path = str(args.vstructure_fci_csv)
    if args.run_id is not None:
        run_id = str(args.run_id)
    if args.tie_blocks:
        tie_blocks = True
    if args.no_tie_blocks:
        tie_blocks = False
    if args.tie_method is not None:
        tie_method = str(args.tie_method)
    if args.reconstruction_mode is not None:
        reconstruction_mode = str(args.reconstruction_mode)
    if args.lambda_group is not None:
        lambda_group_override = float(args.lambda_group)
    if args.lambda_cycle is not None:
        lambda_cycle_override = float(args.lambda_cycle)
    if args.lambda_skeleton is not None:
        lambda_skeleton_override = float(args.lambda_skeleton)
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    if args.sample_size is not None:
        sample_size = int(args.sample_size)
    # CLI override (optional): otherwise keep the default configured above.
    if args.vstructure_in_mask:
        vstructure_in_mask = True
    if args.no_vstructure_in_mask:
        vstructure_in_mask = False
    if args.dag_check:
        dag_check = True
    if args.no_dag_check:
        dag_check = False
    if args.dag_project_on_cycle:
        dag_project_on_cycle = True
    if args.no_dag_project_on_cycle:
        dag_project_on_cycle = False

    # Normalize seeds to list[int]
    if isinstance(seeds, int):
        seeds_list = [int(seeds)]
    else:
        seeds_list = [int(s) for s in seeds]

    total_start = time.time()
    all_results: Dict[str, Dict[int, Dict]] = {}
    
    for dataset_name in datasets:
        all_results[dataset_name] = {}
        for seed in seeds_list:
            try:
                result = run_experiment_for_dataset(
                    dataset_name=dataset_name,
                    high_conf=high_confidence,
                    low_conf=low_confidence,
                    n_epochs=n_epochs,
                    run_mode=run_mode,
                    sample_size=sample_size,
                    random_seed=seed,
                    vstructure_postprocess=use_vstructure_postprocess,
                    vstructure_fci_csv_path=vstructure_fci_csv_path,
                    run_id=run_id,
                    vstructure_in_mask=vstructure_in_mask,
                    # Make train_complete emit complete_dag_check.json (and classify not_directed vs not_acyclic)
                    dag_check=dag_check,
                    dag_project_on_cycle=dag_project_on_cycle,
                    tie_blocks=tie_blocks,
                    tie_method=tie_method,
                    reconstruction_mode=reconstruction_mode,
                    lambda_group_override=lambda_group_override,
                    lambda_cycle_override=lambda_cycle_override,
                    lambda_skeleton_override=lambda_skeleton_override,
                    batch_size=batch_size,
                )
                if result:
                    all_results[dataset_name][int(seed)] = result
            except Exception as e:
                print(f"\n[ERROR] Failed to run experiment for {dataset_name} (seed={seed}): {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # ============================================================================
    # Overall Summary
    # ============================================================================
    if any(all_results.get(ds) for ds in all_results):
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY - ALL DATASETS")
        print("=" * 80)

        def _extract_shd_from_result(run_result: Optional[Dict]) -> Optional[float]:
            """
            Prefer full_shd (structural hamming distance on fully-directed graph),
            then fall back to shd if needed.
            """
            if not run_result:
                return None
            metrics = run_result.get("metrics", {})
            if not isinstance(metrics, dict):
                return None
            shd_val = metrics.get("full_shd", metrics.get("shd"))
            if shd_val is None:
                return None
            try:
                return float(shd_val)
            except (TypeError, ValueError):
                return None
        
        for dataset_name, seed_map in all_results.items():
            if not seed_map:
                continue
            print(f"\n{dataset_name.upper()}:")
            llm_shd_values: List[float] = []
            random_shd_values: List[float] = []
            for seed, result in seed_map.items():
                # Only print diffs when both runs exist
                if result.get("llm") and result.get("random"):
                    print(f"  seed_{seed}:")
                    print(f"    Orientation Diff: {result['orientation_diff']*100:+5.1f}%")
                    print(f"    LLM Orient Acc:   {result['llm']['metrics']['orientation_accuracy']*100:5.1f}%")
                    print(f"    Random Orient Acc:{result['random']['metrics']['orientation_accuracy']*100:5.1f}%")
                    print("    Note: positive Orientation Diff means LLM-prior run is higher.")
                elif result.get("llm"):
                    print(f"  seed_{seed}: LLM-only run completed")
                elif result.get("random"):
                    print(f"  seed_{seed}: Random-only run completed")

                llm_shd = _extract_shd_from_result(result.get("llm"))
                if llm_shd is not None:
                    llm_shd_values.append(llm_shd)
                rnd_shd = _extract_shd_from_result(result.get("random"))
                if rnd_shd is not None:
                    random_shd_values.append(rnd_shd)

            # Multi-seed SHD variance evaluation (separate for LLM vs Random)
            print("  SHD variance across seeds:")
            if llm_shd_values:
                llm_mean = statistics.mean(llm_shd_values)
                llm_var = statistics.pvariance(llm_shd_values) if len(llm_shd_values) > 1 else 0.0
                print(f"    LLM:    n={len(llm_shd_values)}, mean={llm_mean:.4f}, variance={llm_var:.4f}")
            else:
                print("    LLM:    no SHD values found")
            if random_shd_values:
                rnd_mean = statistics.mean(random_shd_values)
                rnd_var = statistics.pvariance(random_shd_values) if len(random_shd_values) > 1 else 0.0
                print(f"    Random: n={len(random_shd_values)}, mean={rnd_mean:.4f}, variance={rnd_var:.4f}")
            else:
                print("    Random: no SHD values found")
        
        print("\n" + "=" * 80)
        print(f"Total runtime (this script): {time.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
