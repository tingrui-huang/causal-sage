"""
Complete Training Script - Phase 4 & 5

Full integration of all components:
- Data loading
- Prior building (FCI + LLM)
- Dense matrix model
- Weighted Group Lasso + Cycle Consistency
- Comprehensive monitoring metrics

Focus: Direction learning and sparsity tracking
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json
import numpy as np
import time
import os
import sys

from src.refinement.modules.data_loader import CausalDataLoader
from src.refinement.modules.prior_builder import PriorBuilder
from src.refinement.modules.model import CausalDiscoveryModel
from src.refinement.modules.loss import LossComputer
from src.refinement.modules.evaluator import CausalGraphEvaluator
from src.refinement.modules.metrics import compute_unresolved_ratio, compute_sparsity_metrics
from src.refinement.modules.dag_check import (
    find_one_cycle,
    is_dag_kahn,
    cycle_to_string,
    project_to_dag_cut_weakest_in_cycle,
)

def train_complete(config: dict):
    """
    Complete training with full monitoring
    
    Args:
        config: Training configuration
    """
    # ---------------------------------------------------------------------
    # Reproducibility: seed Python/NumPy/PyTorch as early as possible.
    # ---------------------------------------------------------------------
    try:
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from reproducibility import set_global_seed
        
        seed = config.get("random_seed")
        if seed is None:
            # Fall back to unified config if the caller didn't pass a seed
            try:
                import config as unified_config
                seed = unified_config.RANDOM_SEED
            except Exception:
                raise ValueError("Missing config['random_seed'] and could not import unified config.RANDOM_SEED")

        set_global_seed(
            int(seed),
            deterministic_torch=bool(config.get("deterministic_torch", False)),
        )
    except Exception as e:
        # Best-effort: training can still run without the helper
        print(f"[WARN] Could not set global seed: {e}")

    # Start timing
    start_time = time.time()

    
    # === 1. LOAD DATA ===
    print("\n[1/6] Loading Data...")
    data_loader = CausalDataLoader(
        data_path=config['data_path'],
        metadata_path=config['metadata_path']
    )
    observations = data_loader.load_data()
    var_structure = data_loader.get_variable_structure()
    
    # === 2. BUILD PRIORS ===
    print("\n[2/6] Building Priors...")
    dataset_name = data_loader.metadata.get('dataset_name', 'Unknown')
    prior_builder = PriorBuilder(var_structure, dataset_name=dataset_name)
    priors = prior_builder.get_all_priors(
        fci_skeleton_path=config['fci_skeleton_path'],  # Pure FCI for hard mask
        llm_direction_path=config.get('llm_direction_path'),  # FCI+LLM for soft direction (optional)
        use_llm_prior=config.get('use_llm_prior', True),  # Whether to use LLM prior
        use_random_prior=config.get('use_random_prior', False),  # Whether to use random prior (control experiment)
        random_seed=config.get('random_seed'),  # Random seed for reproducibility (recommended)
        high_confidence=config.get('high_confidence', 0.7),  # High confidence weight (customizable)
        low_confidence=config.get('low_confidence', 0.3),  # Low confidence weight (customizable)
        enforce_vstructure=bool(config.get("enforce_vstructure_mask", False)),
        vstructure_pag_csv_path=config.get("vstructure_pag_csv_path"),
    )
    
    # === 3. INITIALIZE MODEL ===
    print("\n[3/6] Initializing Model...")
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior'],
        blocks=priors.get('blocks'),
        tie_blocks=bool(config.get('tie_blocks', False)),
        tie_method=str(config.get('tie_method', 'mean')),
    )
    
    # === 4. INITIALIZE LOSS COMPUTER ===
    print("\n[4/6] Initializing Loss Computer...")
    # Variable groups for per-variable softmax CE reconstruction (paper-style option).
    variable_groups = None
    if str(config.get("reconstruction_mode", "bce")) == "group_ce":
        variable_groups = [list(var_structure["var_to_states"][v]) for v in var_structure.get("variable_names", [])]

    loss_computer = LossComputer(
        block_structure=priors['blocks'],
        penalty_weights=priors['penalty_weights'],
        skeleton_mask=priors['skeleton_mask'],  # Add skeleton mask for preservation loss
        reconstruction_mode=str(config.get("reconstruction_mode", "bce")),
        variable_groups=variable_groups,
    )
    
    # === 5. SETUP TRAINING ===
    print("\n[5/6] Setting up Training...")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Initialize evaluator
    gt_type = config.get('ground_truth_type', 'bif')
    evaluator = CausalGraphEvaluator(
        ground_truth_path=config['ground_truth_path'],
        var_structure=var_structure,
        ground_truth_type=gt_type
    )
    
    # === BASELINE: FCI UNRESOLVED RATIO ===
    # Load FCI results to compute baseline unresolved ratio
    # Unresolved = all edges where FCI didn't determine a unique direction
    # This includes: bidirected, partial, undirected, tail-tail (everything except directed)
    fci_baseline_unresolved_ratio = None
    if config.get('fci_skeleton_path'):
        try:
            from src.constraints.evaluate_fci import parse_fci_csv
            
            # Parse FCI CSV to get edge type breakdown
            fci_directed, fci_undirected, edge_counts = parse_fci_csv(config['fci_skeleton_path'])
            
            # Calculate unresolved ratio
            # Unresolved = ALL non-directed edges (bidirected + partial + undirected + tail-tail)
            total_edges = sum(edge_counts.values())
            directed_edges = edge_counts.get('directed', 0)
            unresolved_edges = total_edges - directed_edges  # Everything except directed
            fci_baseline_unresolved_ratio = unresolved_edges / total_edges if total_edges > 0 else 0
            
            print("\n" + "=" * 70)
            print("FCI BASELINE (No LLM, No Training)")
            print("=" * 70)
            print(f"Total FCI edges: {total_edges}")
            print(f"  Directed (->):       {directed_edges:3d}  ({directed_edges/total_edges*100:.1f}%) [direction resolved]")
            print(f"  Unresolved:          {unresolved_edges:3d}  ({fci_baseline_unresolved_ratio*100:.1f}%) [direction NOT resolved]")
            print(f"    - Bidirected (<->): {edge_counts.get('bidirected', 0):3d}")
            print(f"    - Partial (o->):    {edge_counts.get('partial', 0):3d}")
            print(f"    - Undirected (o-o): {edge_counts.get('undirected', 0):3d}")
            print(f"    - Tail-tail (--):   {edge_counts.get('tail-tail', 0):3d}")
            print(f"\nFCI Unresolved Ratio (Baseline): {fci_baseline_unresolved_ratio*100:.1f}%")
            print("=" * 70)
        except Exception as e:
            print(f"\n[WARN] Could not compute FCI baseline: {e}")
    
    # === 6. TRAINING LOOP WITH MONITORING ===
    print("\n[6/6] Starting Training Loop...")
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"Epochs: {config['n_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Lambda Group Lasso: {config['lambda_group']}")
    print(f"Lambda Cycle: {config['lambda_cycle']}")
    print(f"N hops: {config['n_hops']}")
    print("-" * 70)
    
    # === INITIAL EVALUATION (EPOCH 0) ===
    print("\n" + "=" * 70)
    print("INITIAL STATE (Before Training, After LLM Prior)")
    print("=" * 70)
    
    with torch.no_grad():
        adjacency = model.get_adjacency()
        
        # Initial unresolved ratio
        unresolved_stats_init = compute_unresolved_ratio(
            adjacency, 
            priors['blocks'],
            threshold=config['edge_threshold']
        )
        
        # Initial sparsity
        sparsity_stats_init = compute_sparsity_metrics(
            adjacency,
            priors['skeleton_mask'],
            priors['blocks'],
            threshold=config['edge_threshold']
        )
        
        print(f"\nInitial Unresolved (Symmetric) Ratio: {unresolved_stats_init['unresolved_ratio']*100:.1f}%")
        print(f"  Unresolved pairs (symmetric): {unresolved_stats_init['unresolved']}")
        print(f"  Resolved pairs (one direction): {unresolved_stats_init['resolved']}")
        print(f"  No direction pairs: {unresolved_stats_init['no_direction']}")
        print(f"  Total pairs: {unresolved_stats_init['total_pairs']}")
        
        print(f"\nInitial Sparsity:")
        print(f"  Overall: {sparsity_stats_init['overall_sparsity']*100:.1f}%")
        print(f"  Active connections: {sparsity_stats_init['active_connections']}")
        print(f"  Active blocks: {sparsity_stats_init['active_blocks']}/{sparsity_stats_init['total_blocks']}")
    
    print("=" * 70)
    
    # Tracking history
    history = {
        'epoch': [],
        'loss_total': [],
        'loss_reconstruction': [],
        'loss_group_lasso': [],
        'loss_cycle': [],
        'loss_skeleton': [],  # NEW: Skeleton preservation loss
        'unresolved_ratio': [],  # Renamed from bidirectional_ratio
        'overall_sparsity': [],
        'block_sparsity': [],
        'active_connections': [],
        'active_blocks': []
    }

    # Always record the initial (epoch 0) state so summary/reporting works even when
    # n_epochs < monitor_interval.
    try:
        with torch.no_grad():
            pred_mode = str(config.get("pred_mode", "propagate"))
            predictions0 = model(
                observations,
                n_hops=config['n_hops'],
                pred_mode=pred_mode,
            )
            adjacency0 = model.get_adjacency()
            losses0 = loss_computer.compute_total_loss(
                predictions=predictions0,
                targets=observations,
                adjacency=adjacency0,
                lambda_group=config['lambda_group'],
                lambda_cycle=config['lambda_cycle'],
                lambda_skeleton=config.get('lambda_skeleton', 0.1),
            )
        history['epoch'].append(0)
        history['loss_total'].append(losses0['total'].item())
        history['loss_reconstruction'].append(losses0['reconstruction'].item())
        history['loss_group_lasso'].append(losses0['weighted_group_lasso'].item())
        history['loss_cycle'].append(losses0['cycle_consistency'].item())
        history['loss_skeleton'].append(losses0['skeleton_preservation'].item())
        history['unresolved_ratio'].append(float(unresolved_stats_init.get('unresolved_ratio', 0.0)))
        history['overall_sparsity'].append(float(sparsity_stats_init.get('overall_sparsity', 0.0)))
        history['block_sparsity'].append(float(sparsity_stats_init.get('block_sparsity', 0.0)))
        history['active_connections'].append(int(sparsity_stats_init.get('active_connections', 0)))
        history['active_blocks'].append(int(sparsity_stats_init.get('active_blocks', 0)))
    except Exception as e:
        print(f"[WARN] Failed to record initial history (epoch 0): {e}")
    
    batch_size = config.get("batch_size")
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            batch_size = None

    dataset = None
    loader = None
    if batch_size is not None and batch_size < int(observations.shape[0]):
        dataset = TensorDataset(observations)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=bool(config.get("shuffle", True)),
            drop_last=False,
            num_workers=0,  # Windows-friendly
            pin_memory=False,
        )
        print(f"[INFO] Mini-batch training enabled: batch_size={batch_size} | batches/epoch={len(loader)}")
    else:
        print(f"[INFO] Full-batch training (single batch) | N={int(observations.shape[0])}")

    losses = None
    pred_mode_cfg = str(config.get("pred_mode", "propagate"))
    for epoch in range(config['n_epochs']):
        optimizer.zero_grad()
        if loader is None:
            # === FORWARD PASS ===
            predictions = model(
                observations,
                n_hops=config['n_hops'],
                pred_mode=pred_mode_cfg,
            )
            adjacency = model.get_adjacency()
            losses = loss_computer.compute_total_loss(
                predictions=predictions,
                targets=observations,
                adjacency=adjacency,
                lambda_group=config['lambda_group'],
                lambda_cycle=config['lambda_cycle'],
                lambda_skeleton=config.get('lambda_skeleton', 0.1)
            )
            losses['total'].backward()
        else:
            # Mini-batch: step once per epoch via gradient accumulation to keep objective scaling stable.
            # Structural losses depend only on adjacency, so we backprop them once per epoch.
            adjacency = model.get_adjacency()
            loss_group = loss_computer.weighted_group_lasso_loss(adjacency)
            loss_cycle = loss_computer.cycle_consistency_loss(adjacency)
            loss_skeleton = loss_computer.skeleton_preservation_loss(adjacency)
            struct_total = (
                config['lambda_group'] * loss_group
                + config['lambda_cycle'] * loss_cycle
                + config.get('lambda_skeleton', 0.1) * loss_skeleton
            )
            struct_total.backward()

            # Recon gradients: average over all samples (micro-batch accumulation)
            n_total = float(observations.shape[0])
            for (xb,) in loader:
                pred_b = model(xb, n_hops=config['n_hops'], pred_mode=pred_mode_cfg)
                recon_b = loss_computer.reconstruction_loss(pred_b, xb)
                (recon_b * (xb.shape[0] / n_total)).backward()
            # For logging, compute a representative losses dict on full data every monitor interval below.
            losses = None

        optimizer.step()
        
        # Clamp weights
        with torch.no_grad():
            model.raw_adj.data.clamp_(-5.0, 5.0)
        
        # === MONITORING METRICS ===
        if (epoch + 1) % config['monitor_interval'] == 0:
            with torch.no_grad():
                # Recompute predictions/losses for reporting (avoid relying on per-batch partials).
                pred_rep = model(observations, n_hops=config['n_hops'], pred_mode=pred_mode_cfg)
                adjacency_rep = model.get_adjacency()
                # Unresolved ratio
                unresolved_stats = compute_unresolved_ratio(
                    adjacency_rep, 
                    priors['blocks'],
                    threshold=config['edge_threshold']
                )
                
                # Sparsity metrics
                sparsity_stats = compute_sparsity_metrics(
                    adjacency_rep,
                    priors['skeleton_mask'],
                    priors['blocks'],
                    threshold=config['edge_threshold']
                )

                # Losses for reporting
                losses = loss_computer.compute_total_loss(
                    predictions=pred_rep,
                    targets=observations,
                    adjacency=adjacency_rep,
                    lambda_group=config['lambda_group'],
                    lambda_cycle=config['lambda_cycle'],
                    lambda_skeleton=config.get('lambda_skeleton', 0.1),
                )
                
                # Record history
                history['epoch'].append(epoch + 1)
                history['loss_total'].append(losses['total'].item())
                history['loss_reconstruction'].append(losses['reconstruction'].item())
                history['loss_group_lasso'].append(losses['weighted_group_lasso'].item())
                history['loss_cycle'].append(losses['cycle_consistency'].item())
                history['loss_skeleton'].append(losses['skeleton_preservation'].item())
                history['unresolved_ratio'].append(unresolved_stats['unresolved_ratio'])
                history['overall_sparsity'].append(sparsity_stats['overall_sparsity'])
                history['block_sparsity'].append(sparsity_stats['block_sparsity'])
                history['active_connections'].append(sparsity_stats['active_connections'])
                history['active_blocks'].append(sparsity_stats['active_blocks'])
                
                # Print progress
                print(f"\nEpoch {epoch + 1:3d}/{config['n_epochs']}")
                print(f"  Losses:")
                print(f"    Total:        {losses['total'].item():8.4f}")
                print(f"    Recon:        {losses['reconstruction'].item():8.4f}")
                print(f"    Group Lasso:  {losses['weighted_group_lasso'].item():8.4f}")
                print(f"    Cycle:        {losses['cycle_consistency'].item():8.4f}")
                print(f"    Skeleton:     {losses['skeleton_preservation'].item():8.4f}")
                print(f"  Symmetry Breaking (Direction Learning):")
                print(f"    Unresolved (Symmetric): {unresolved_stats['unresolved']:3d} / {unresolved_stats['total_pairs']:3d} "
                      f"({unresolved_stats['unresolved_ratio']*100:5.1f}%) [Want: DOWN]")
                print(f"    Resolved (One Direction): {unresolved_stats['resolved']:3d} / {unresolved_stats['total_pairs']:3d} "
                      f"({unresolved_stats['resolved']/unresolved_stats['total_pairs']*100:5.1f}%) [Want: UP]")
                print(f"  Sparsity:")
                print(f"    Overall:      {sparsity_stats['overall_sparsity']*100:5.1f}% "
                      f"({sparsity_stats['active_connections']}/{sparsity_stats['total_allowed']})")
                print(f"    Block-level:  {sparsity_stats['block_sparsity']*100:5.1f}% "
                      f"({sparsity_stats['active_blocks']}/{sparsity_stats['total_blocks']})")

    # Ensure final epoch is recorded even if monitor_interval doesn't divide n_epochs.
    if history.get('epoch') and history['epoch'][-1] != int(config['n_epochs']):
        try:
            with torch.no_grad():
                adjacency_f = model.get_adjacency()
                unresolved_f = compute_unresolved_ratio(adjacency_f, priors['blocks'], threshold=config['edge_threshold'])
                sparsity_f = compute_sparsity_metrics(adjacency_f, priors['skeleton_mask'], priors['blocks'], threshold=config['edge_threshold'])
            # losses holds the last epoch's losses
            if losses is not None:
                history['epoch'].append(int(config['n_epochs']))
                history['loss_total'].append(losses['total'].item())
                history['loss_reconstruction'].append(losses['reconstruction'].item())
                history['loss_group_lasso'].append(losses['weighted_group_lasso'].item())
                history['loss_cycle'].append(losses['cycle_consistency'].item())
                history['loss_skeleton'].append(losses['skeleton_preservation'].item())
                history['unresolved_ratio'].append(float(unresolved_f.get('unresolved_ratio', 0.0)))
                history['overall_sparsity'].append(float(sparsity_f.get('overall_sparsity', 0.0)))
                history['block_sparsity'].append(float(sparsity_f.get('block_sparsity', 0.0)))
                history['active_connections'].append(int(sparsity_f.get('active_connections', 0)))
                history['active_blocks'].append(int(sparsity_f.get('active_blocks', 0)))
        except Exception as e:
            print(f"[WARN] Failed to record final history: {e}")
    
    # Calculate training time
    training_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"TRAINING TIME: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("=" * 70)
    
    # === FINAL GRAPH SELECTION (optional DAG projection) + FINAL EVALUATION ===
    print("\n" + "=" * 70)
    print("FINAL GRAPH + EVALUATION")
    print("=" * 70)
    
    adjacency = model.get_adjacency()
    learned_edges_raw = evaluator.extract_learned_edges(adjacency, threshold=config['edge_threshold'])
    learned_edges = set(learned_edges_raw)

    # Optional: DAG check + optional projection BEFORE evaluation so reported metrics match the final graph.
    dag_check = None
    dag_check_raw = None
    dag_projection = None

    if bool(config.get("dag_check", False)) or bool(config.get("dag_project_on_cycle", False)):
        try:
            # Directedness proxy: unresolved_ratio_final (symmetric pairs) > 0 => not fully directed.
            if history.get("unresolved_ratio"):
                unresolved_ratio_final = float(history["unresolved_ratio"][-1])
            else:
                unresolved_ratio_final = float(
                    compute_unresolved_ratio(adjacency, priors["blocks"], threshold=config.get("edge_threshold", 0.3)).get(
                        "unresolved_ratio", 0.0
                    )
                )

            nodes = set(var_structure.get("variable_names", []))
            for u, v in learned_edges:
                nodes.add(u)
                nodes.add(v)

            is_dag_raw = bool(is_dag_kahn(nodes, learned_edges))
            cyc_raw = None if is_dag_raw else find_one_cycle(nodes, learned_edges)
            dag_check_raw = {
                "is_dag": is_dag_raw,
                "unresolved_ratio_final": unresolved_ratio_final,
                "not_directed": bool(unresolved_ratio_final > 0.0),
                "not_acyclic": bool(not is_dag_raw),
                "cycle_example": cycle_to_string(cyc_raw) if cyc_raw else None,
                "num_nodes": int(len(nodes)),
                "num_edges": int(len(learned_edges)),
            }

            # Greedy DAG projection: cut weakest edge(s) on cycles until acyclic.
            if (not is_dag_raw) and bool(config.get("dag_project_on_cycle", False)):
                var_to_states = var_structure["var_to_states"]

                def _edge_strength(u: str, v: str) -> float:
                    idx_u = var_to_states[u]
                    idx_v = var_to_states[v]
                    block = adjacency[idx_u][:, idx_v]
                    return float(block.mean().item())

                dag_edges, proj = project_to_dag_cut_weakest_in_cycle(
                    edges=set(learned_edges),
                    edge_strength=_edge_strength,
                    nodes=set(nodes),
                )
                learned_edges = set(dag_edges)

                # Also zero out cut edges in the state-level adjacency so saved adjacency aligns with final edge set.
                # (We keep the rest of adjacency unchanged.)
                cuts = list(proj.cuts)
                for c in cuts:
                    u, v = c.u, c.v
                    if u not in var_to_states or v not in var_to_states:
                        continue
                    iu = torch.as_tensor(var_to_states[u], dtype=torch.long, device=adjacency.device)
                    iv = torch.as_tensor(var_to_states[v], dtype=torch.long, device=adjacency.device)
                    adjacency[iu[:, None], iv[None, :]] = 0.0

                dag_projection = {
                    "enabled": True,
                    "is_dag_after": bool(proj.is_dag),
                    "num_cuts": int(len(proj.cuts)),
                    "cuts": [
                        {
                            "u": c.u,
                            "v": c.v,
                            "strength": float(c.strength),
                            "cycle": c.cycle,
                        }
                        for c in proj.cuts
                    ],
                    "num_edges_in": int(proj.num_edges_in),
                    "num_edges_out": int(proj.num_edges_out),
                }

            # Final DAG check after optional projection
            is_dag_final = bool(is_dag_kahn(nodes, learned_edges))
            cyc_final = None if is_dag_final else find_one_cycle(nodes, learned_edges)
            dag_check = {
                "is_dag": is_dag_final,
                "unresolved_ratio_final": unresolved_ratio_final,
                "not_directed": bool(unresolved_ratio_final > 0.0),
                "not_acyclic": bool(not is_dag_final),
                "cycle_example": cycle_to_string(cyc_final) if cyc_final else None,
                "num_nodes": int(len(nodes)),
                "num_edges": int(len(learned_edges)),
            }
        except Exception as e:
            print(f"[WARN] DAG check/projection failed: {e}")

    # Now evaluate the FINAL graph (possibly DAG-projected).
    metrics = evaluator.evaluate(set(learned_edges))
    evaluator.print_metrics(metrics)

    # Write DAG artifacts (raw + final) if enabled
    try:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(exist_ok=True, parents=True)
        if dag_check_raw is not None:
            with open(output_dir / "complete_dag_check_raw.json", "w", encoding="utf-8") as f:
                json.dump(dag_check_raw, f, indent=2)
        if dag_check is not None:
            with open(output_dir / "complete_dag_check.json", "w", encoding="utf-8") as f:
                json.dump(dag_check, f, indent=2)
        if dag_projection is not None:
            with open(output_dir / "complete_dag_projection.json", "w", encoding="utf-8") as f:
                json.dump(dag_projection, f, indent=2)
    except Exception:
        pass

    if dag_check is not None:
        print("\n" + "-" * 70)
        print("DAG CHECK (FINAL GRAPH)")
        print("-" * 70)
        print(f"is_dag: {dag_check['is_dag']}")
        if not dag_check["is_dag"]:
            print(f"reason.not_directed: {dag_check['not_directed']} (unresolved_ratio_final={dag_check['unresolved_ratio_final']:.4f})")
            print(f"reason.not_acyclic:  {dag_check['not_acyclic']}")
            if dag_check["cycle_example"]:
                print(f"cycle_example: {dag_check['cycle_example']}")
        else:
            print(f"unresolved_ratio_final={dag_check['unresolved_ratio_final']:.4f}")
        if dag_projection is not None:
            print(f"dag_projection.cuts: {dag_projection.get('num_cuts')} | is_dag_after: {dag_projection.get('is_dag_after')}")
        print("-" * 70)
    
    # === SAVE RESULTS ===
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'history': history
    }, output_dir / 'complete_model.pt')
    
    # Save adjacency (may have cut-edge blocks zeroed if dag_project_on_cycle was enabled)
    torch.save(adjacency, output_dir / 'complete_adjacency.pt')
    
    # Save learned edges (FINAL)
    with open(output_dir / 'complete_edges.txt', 'w') as f:
        f.write("Learned Causal Edges (Complete Training)\n")
        f.write("=" * 70 + "\n\n")
        for var_a, var_b in sorted(learned_edges):
            # Get block strength
            block = None
            for b in priors['blocks']:
                if b['var_pair'] == (var_a, var_b):
                    block = b
                    break
            if block:
                block_weights = adjacency[block['row_indices']][:, block['col_indices']]
                strength = block_weights.mean().item()
                f.write(f"{var_a} -> {var_b} (strength: {strength:.4f})\n")

    # Save raw learned edges for debugging if projection changed anything
    if dag_projection is not None:
        try:
            with open(output_dir / "complete_edges_raw.txt", "w", encoding="utf-8") as f:
                f.write("Learned Causal Edges (RAW, before DAG projection)\n")
                f.write("=" * 70 + "\n\n")
                for u, v in sorted(learned_edges_raw):
                    f.write(f"{u} -> {v}\n")
        except Exception:
            pass
    
    # Save metrics (FINAL)
    with open(output_dir / 'complete_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save raw metrics for debugging if projection changed anything
    if dag_projection is not None:
        try:
            metrics_raw = evaluator.evaluate(set(learned_edges_raw))
            with open(output_dir / "complete_metrics_raw.json", "w", encoding="utf-8") as f:
                json.dump(metrics_raw, f, indent=2)
        except Exception:
            pass
    
    # Save training history
    with open(output_dir / 'complete_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save summary report
    with open(output_dir / 'complete_summary.txt', 'w') as f:
        f.write("Complete Training Summary\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Training Time:\n")
        f.write(f"  Total: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n")
        f.write(f"  Per epoch: {training_time/config['n_epochs']:.3f} seconds\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Epochs: {config['n_epochs']}\n")
        f.write(f"  Learning rate: {config['learning_rate']}\n")
        f.write(f"  Lambda Group Lasso: {config['lambda_group']}\n")
        f.write(f"  Lambda Cycle: {config['lambda_cycle']}\n\n")
        
        f.write("Final Losses:\n")
        f.write(f"  Total: {history['loss_total'][-1]:.4f}\n")
        f.write(f"  Reconstruction: {history['loss_reconstruction'][-1]:.4f}\n")
        f.write(f"  Group Lasso: {history['loss_group_lasso'][-1]:.4f}\n")
        f.write(f"  Cycle Consistency: {history['loss_cycle'][-1]:.4f}\n\n")
        
        f.write("Symmetry Breaking (Direction Learning):\n")
        if fci_baseline_unresolved_ratio is not None:
            f.write(f"  FCI Baseline (edges without direction):     {fci_baseline_unresolved_ratio*100:.1f}%\n")
            # f.write(f"  After LLM Prior (Epoch 0):        {history['unresolved_ratio'][0]*100:.1f}% (symmetric pairs)\n")
            f.write(f"  After Training (Final, symmetric pairs):    {history['unresolved_ratio'][-1]*100:.1f}%\n")
            total_improvement = (fci_baseline_unresolved_ratio - history['unresolved_ratio'][-1]) * 100
            f.write(f"  Total Improvement (FCI → Final): {total_improvement:+.1f}%\n\n")
        else:
            f.write(f"  Initial Unresolved Ratio: {history['unresolved_ratio'][0]*100:.1f}%\n")
            f.write(f"  Final Unresolved Ratio: {history['unresolved_ratio'][-1]*100:.1f}%\n")
            f.write(f"  Change: {(history['unresolved_ratio'][-1] - history['unresolved_ratio'][0])*100:+.1f}%\n\n")
        
        f.write("Sparsity:\n")
        f.write(f"  Initial Overall: {history['overall_sparsity'][0]*100:.1f}%\n")
        f.write(f"  Final Overall: {history['overall_sparsity'][-1]*100:.1f}%\n")
        f.write(f"  Initial Block: {history['block_sparsity'][0]*100:.1f}%\n")
        f.write(f"  Final Block: {history['block_sparsity'][-1]*100:.1f}%\n\n")
        
        f.write("Evaluation Metrics:\n")
        f.write(f"  Edge Precision: {metrics['edge_precision']*100:.1f}%\n")
        f.write(f"  Edge Recall: {metrics['edge_recall']*100:.1f}%\n")
        f.write(f"  Orientation Accuracy: {metrics['orientation_accuracy']*100:.1f}%\n")
        f.write(f"  Learned Edges: {metrics['learned_edges']}\n")
        f.write(f"  Ground Truth Edges: {metrics['ground_truth_edges']}\n")
        if dag_check is not None:
            f.write("\nDAG Check:\n")
            f.write(f"  is_dag: {dag_check.get('is_dag')}\n")
            if not dag_check.get("is_dag"):
                f.write(f"  reason.not_directed: {dag_check.get('not_directed')} (unresolved_ratio_final={dag_check.get('unresolved_ratio_final')})\n")
                f.write(f"  reason.not_acyclic:  {dag_check.get('not_acyclic')}\n")
                if dag_check.get("cycle_example"):
                    f.write(f"  cycle_example: {dag_check.get('cycle_example')}\n")
            else:
                f.write(f"  unresolved_ratio_final: {dag_check.get('unresolved_ratio_final')}\n")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Return results including FCI baseline for comparison
    results = {
        'model': model,
        'metrics': metrics,
        'history': history,
        'fci_baseline_unresolved_ratio': fci_baseline_unresolved_ratio
    }
    if dag_check is not None:
        results["dag_check"] = dag_check
    if dag_projection is not None:
        results["dag_projection"] = dag_projection
    
    return results


if __name__ == "__main__":
    # Configuration
    config = {
        # Data paths (updated to new structure)
        'data_path': 'data/alarm/alarm_data_10000.csv',
        'metadata_path': 'data/alarm/metadata.json',
        'fci_skeleton_path': 'data/alarm/edges_FCI_20251207_230824.csv',  # Pure FCI for HARD skeleton mask
        'llm_direction_path': 'data/alarm/edges_Hybrid_FCI_LLM_20251207_230956.csv',  # FCI+LLM for SOFT direction prior
        'ground_truth_path': 'data/alarm/alarm.bif',
        
        # Training parameters
        'n_epochs': 200,  # Pilot training: 200 epochs
        'learning_rate': 0.01,
        'n_hops': 1,  # Single-hop reasoning
        
        # Loss hyperparameters (Conservative start)
        'lambda_group': 0.01,    # Group Lasso weight
        'lambda_cycle': 0.001,   # Cycle Consistency weight
        
        # Monitoring
        'monitor_interval': 20,  # Monitor every 20 epochs
        'edge_threshold': 0.1,   # Threshold for active edge
        
        # Output
        'output_dir': 'results/complete'
    }
    
    # Train
    print("\n" + "=" * 70)
    print("PILOT TRAINING: 200 EPOCHS")
    print("=" * 70)
    print("Goal: Verify loss convergence and direction learning")
    print()
    
    results = train_complete(config)
    model = results['model']
    metrics = results['metrics']
    history = results['history']
    fci_baseline_unresolved_ratio = results['fci_baseline_unresolved_ratio']
    
    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)
    
    print("\nLoss Convergence:")
    print(f"  Reconstruction: {history['loss_reconstruction'][0]:.4f} -> {history['loss_reconstruction'][-1]:.4f}")
    print(f"  Group Lasso:    {history['loss_group_lasso'][0]:.4f} -> {history['loss_group_lasso'][-1]:.4f}")
    print(f"  Cycle:          {history['loss_cycle'][0]:.4f} -> {history['loss_cycle'][-1]:.4f}")
    
    print("\nSymmetry Breaking (Direction Learning):")
    if fci_baseline_unresolved_ratio is not None:
        print(f"  FCI Baseline (edges without direction):     {fci_baseline_unresolved_ratio*100:.1f}%")
        # print(f"  After LLM Prior (Epoch 0):        {history['unresolved_ratio'][0]*100:.1f}% (symmetric pairs)")
        print(f"  After Training (Final, symmetric pairs):    {history['unresolved_ratio'][-1]*100:.1f}%")
        total_improvement = (fci_baseline_unresolved_ratio - history['unresolved_ratio'][-1]) * 100
        print(f"  Total Improvement (FCI → Final): {total_improvement:+.1f}%")
    else:
        print(f"  Unresolved Ratio: {history['unresolved_ratio'][0]*100:.1f}% -> {history['unresolved_ratio'][-1]*100:.1f}%")
        change = (history['unresolved_ratio'][-1] - history['unresolved_ratio'][0]) * 100
        print(f"  Change: {change:+.1f}% {'[GOOD]' if change < 0 else '[NEEDS TUNING]'}")
    
    print("\nSparsity:")
    print(f"  Overall: {history['overall_sparsity'][-1]*100:.1f}% ({history['active_connections'][-1]}/{416})")
    print(f"  Block:   {history['block_sparsity'][-1]*100:.1f}% ({history['active_blocks'][-1]}/{1332})")
    
    print("\nEvaluation:")
    print(f"  Precision: {metrics['edge_precision']*100:.1f}%")
    print(f"  Recall: {metrics['edge_recall']*100:.1f}%")
    print(f"  Orientation: {metrics['orientation_accuracy']*100:.1f}%")
    print(f"  Edges: {metrics['learned_edges']} / {metrics['ground_truth_edges']}")

