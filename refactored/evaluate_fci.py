"""
Evaluate FCI Results Against Ground Truth

Automatically evaluates the latest FCI outputs and generates a report.
This script is called after FCI runs to assess skeleton quality.
"""

import re
import pandas as pd
from pathlib import Path
from datetime import datetime


def parse_ground_truth(gt_path):
    """
    Parse ground truth edges from file
    
    Supports:
    - BIF format (Bayesian Network)
    - Edge list format (simple text: source -> target)
    
    Args:
        gt_path: Path to ground truth file
    
    Returns:
        Set of (source, target) tuples
    """
    ground_truth_edges = set()
    gt_path = Path(gt_path)
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try BIF format first
    prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'
    matches = list(re.finditer(prob_pattern, content))
    
    if matches:
        # BIF format detected
        for match in matches:
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(',')]
            
            for parent in parents:
                if parent:
                    ground_truth_edges.add((parent, child))
    else:
        # Try edge list format: "source -> target"
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse edge: "A -> B"
            if '->' in line:
                parts = line.split('->')
                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    if source and target:
                        ground_truth_edges.add((source, target))
    
    return ground_truth_edges


def parse_fci_csv(fci_csv_path):
    """Parse FCI edges from CSV file"""
    df = pd.read_csv(fci_csv_path)
    
    # Determine column names (case-insensitive compatibility)
    if 'Source' in df.columns and 'Target' in df.columns:
        source_col, target_col = 'Source', 'Target'
    elif 'source' in df.columns and 'target' in df.columns:
        source_col, target_col = 'source', 'target'
    else:
        # Use first two columns as fallback
        source_col, target_col = df.columns[0], df.columns[1]
        print(f"[WARN] Standard columns not found, using: {source_col} -> {target_col}")
    
    # Determine edge type column
    if 'edge_type' in df.columns:
        edge_type_col = 'edge_type'
    elif 'Edge_Type' in df.columns:
        edge_type_col = 'Edge_Type'
    elif 'type' in df.columns:
        edge_type_col = 'type'
    else:
        edge_type_col = None
        print("[WARN] No edge_type column found, assuming all edges are directed")
    
    fci_directed = set()
    fci_undirected = set()
    
    edge_counts = {
        'directed': 0,
        'undirected': 0,
        'partial': 0,
        'tail-tail': 0,
        'bidirected': 0
    }
    
    for _, row in df.iterrows():
        source = row[source_col]
        target = row[target_col]
        edge_type = row[edge_type_col] if edge_type_col else 'directed'
        
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
        
        if edge_type == 'directed':
            fci_directed.add((source, target))
        elif edge_type in ['undirected', 'partial', 'tail-tail', 'bidirected']:
            # For undirected/partial edges, consider both directions
            fci_undirected.add(tuple(sorted([source, target])))
    
    return fci_directed, fci_undirected, edge_counts


def compute_fci_unresolved_ratio(fci_csv_path):
    """
    Compute unresolved ratio from FCI edge list
    
    Unresolved ratio = (# of non-directed edges) / (# of total edges)
    
    Unresolved edges are those where FCI didn't determine a unique direction:
    - Bidirected (<->): latent confounders
    - Partial (o->): ambiguous direction
    - Undirected (o-o): completely ambiguous
    - Tail-tail (--): no clear direction
    
    This is the FCI baseline that LLM and neural training aim to reduce.
    
    Args:
        fci_csv_path: Path to FCI edges CSV
    
    Returns:
        dict: Statistics including unresolved_ratio
    """
    # Parse FCI CSV to get edge type breakdown
    fci_directed, fci_undirected, edge_counts = parse_fci_csv(fci_csv_path)
    
    # Calculate unresolved ratio
    # Unresolved = ALL non-directed edges (bidirected + partial + undirected + tail-tail)
    total_edges = sum(edge_counts.values())
    directed_edges = edge_counts.get('directed', 0)
    unresolved_edges = total_edges - directed_edges  # Everything except directed
    
    unresolved_ratio = unresolved_edges / total_edges if total_edges > 0 else 0
    
    return {
        'unresolved_count': unresolved_edges,
        'resolved_count': directed_edges,
        'total_edges': total_edges,
        'unresolved_ratio': unresolved_ratio,
        'edge_type_breakdown': edge_counts
    }


def compute_shd(fci_csv_path, ground_truth_path):
    """
    Compute Structural Hamming Distance (SHD)
    
    Two types of SHD:
    1. Skeleton SHD = E_add + E_del (undirected, only edge existence)
    2. Full SHD = E_add + E_del + E_rev (directed, standard NeurIPS/ICLR metric)
    
    Where:
    - E_add (additions): edges in FCI but not in GT (undirected)
    - E_del (deletions): edges in GT but not in FCI (undirected)
    - E_rev (reversals): edges with correct skeleton but wrong direction
    
    Args:
        fci_csv_path: Path to FCI edges CSV
        ground_truth_path: Path to ground truth file
    
    Returns:
        dict: SHD statistics with both skeleton_shd and full_shd
    """
    # Load ground truth
    gt_edges = parse_ground_truth(ground_truth_path)
    
    # Load FCI edges (only directed ones for SHD)
    fci_directed, fci_undirected, edge_counts = parse_fci_csv(fci_csv_path)
    
    # Convert to undirected for skeleton comparison
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    fci_all_undirected = {tuple(sorted([e[0], e[1]])) for e in fci_directed}
    fci_all_undirected.update(fci_undirected)
    
    # Edge additions (FCI has, GT doesn't) - undirected
    additions = len(fci_all_undirected - gt_undirected)
    
    # Edge deletions (GT has, FCI doesn't) - undirected
    deletions = len(gt_undirected - fci_all_undirected)
    
    # Edge reversals and unresolved edges
    # For fair comparison with Neural Network:
    # - Directed edges with wrong direction count as reversals
    # - Undirected edges (o-o, o->, <->, --) also count as reversals
    #   because they fail to determine the correct direction
    
    reversals = 0
    unresolved_as_errors = 0
    
    # Count reversals in directed FCI edges
    for fci_edge in fci_directed:
        undirected_edge = tuple(sorted([fci_edge[0], fci_edge[1]]))
        if undirected_edge in gt_undirected:
            # Edge exists in GT, check direction
            reversed_edge = (fci_edge[1], fci_edge[0])
            if reversed_edge in gt_edges and fci_edge not in gt_edges:
                reversals += 1
    
    # Count undirected edges as errors (strict mode for fair comparison)
    # These edges exist in both FCI and GT, but FCI failed to orient them
    for fci_undir_edge in fci_undirected:
        if fci_undir_edge in gt_undirected:
            # This edge exists in GT with a direction, but FCI left it undirected
            # Count as reversal (direction error)
            unresolved_as_errors += 1
    
    # Skeleton SHD: only edge existence (undirected)
    skeleton_shd = additions + deletions
    
    # Full SHD: edge existence + direction (standard metric)
    # Include unresolved edges as direction errors for fair comparison
    full_shd = additions + deletions + reversals + unresolved_as_errors
    
    return {
        'skeleton_shd': skeleton_shd,
        'full_shd': full_shd,
        'shd': full_shd,  # Default to full_shd for backward compatibility
        'additions': additions,
        'deletions': deletions,
        'reversals': reversals,
        'unresolved_as_errors': unresolved_as_errors,
        'total_direction_errors': reversals + unresolved_as_errors
    }


def evaluate_fci(fci_csv_path, ground_truth_path, output_dir=None):
    """
    Evaluate FCI skeleton against ground truth
    
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "=" * 80)
    print("FCI EVALUATION AGAINST GROUND TRUTH")
    print("=" * 80)
    
    # Load data
    gt_edges = parse_ground_truth(ground_truth_path)
    fci_directed, fci_undirected, edge_counts = parse_fci_csv(fci_csv_path)
    
    print(f"\nGround Truth: {len(gt_edges)} directed edges")
    print(f"FCI Edges: {len(fci_directed)} directed + {len(fci_undirected)} undirected/partial")
    print(f"  - Directed: {edge_counts['directed']}")
    print(f"  - Undirected: {edge_counts.get('undirected', 0)}")
    print(f"  - Partial: {edge_counts.get('partial', 0)}")
    print(f"  - Tail-tail: {edge_counts.get('tail-tail', 0)}")
    print(f"  - Bidirected: {edge_counts.get('bidirected', 0)}")
    
    # === 1. UNDIRECTED SKELETON METRICS ===
    # Convert GT to undirected
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in gt_edges}
    
    # All FCI edges (both directed and undirected)
    fci_all_undirected = set()
    for edge in fci_directed:
        fci_all_undirected.add(tuple(sorted([edge[0], edge[1]])))
    fci_all_undirected.update(fci_undirected)
    
    # Calculate undirected metrics
    undirected_tp = len(fci_all_undirected & gt_undirected)
    undirected_fp = len(fci_all_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - fci_all_undirected)
    
    edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
    edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
    
    print("\n" + "=" * 80)
    print("EDGE DISCOVERY (Undirected Skeleton)")
    print("=" * 80)
    print(f"True Positives (TP):  {undirected_tp}")
    print(f"False Positives (FP): {undirected_fp}")
    print(f"False Negatives (FN): {undirected_fn}")
    print(f"\nPrecision: {edge_precision*100:.1f}%")
    print(f"Recall:    {edge_recall*100:.1f}%")
    print(f"F1 Score:  {edge_f1*100:.1f}%")
    
    # === 2. ORIENTATION ACCURACY ===
    correctly_oriented = 0
    incorrectly_oriented = 0
    
    for fci_edge in fci_directed:
        undirected_edge = tuple(sorted([fci_edge[0], fci_edge[1]]))
        
        # Check if this edge exists in GT (undirected)
        if undirected_edge in gt_undirected:
            # Check if direction is correct
            if fci_edge in gt_edges:
                correctly_oriented += 1
            else:
                reversed_edge = (fci_edge[1], fci_edge[0])
                if reversed_edge in gt_edges:
                    incorrectly_oriented += 1
    
    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
    
    print("\n" + "=" * 80)
    print("ORIENTATION ACCURACY (Directed Edges Only)")
    print("=" * 80)
    print(f"FCI Directed Edges: {len(fci_directed)}")
    print(f"Correctly Oriented: {correctly_oriented}")
    print(f"Incorrectly Oriented: {incorrectly_oriented}")
    print(f"\nOrientation Accuracy: {orientation_accuracy*100:.1f}%")
    
    # === 3. UNRESOLVED RATIO (FCI BASELINE) ===
    unresolved_stats = compute_fci_unresolved_ratio(fci_csv_path)
    
    print("\n" + "=" * 80)
    print("UNRESOLVED RATIO (FCI Baseline)")
    print("=" * 80)
    print(f"Total FCI edges: {unresolved_stats['total_edges']}")
    print(f"  Directed (->):       {unresolved_stats['resolved_count']:3d}  ({unresolved_stats['resolved_count']/unresolved_stats['total_edges']*100:.1f}%) [direction resolved]")
    print(f"  Unresolved:          {unresolved_stats['unresolved_count']:3d}  ({unresolved_stats['unresolved_ratio']*100:.1f}%) [direction NOT resolved]")
    
    breakdown = unresolved_stats['edge_type_breakdown']
    print(f"    - Bidirected (<->): {breakdown.get('bidirected', 0):3d}")
    print(f"    - Partial (o->):    {breakdown.get('partial', 0):3d}")
    print(f"    - Undirected (o-o): {breakdown.get('undirected', 0):3d}")
    print(f"    - Tail-tail (--):   {breakdown.get('tail-tail', 0):3d}")
    print(f"\nFCI Unresolved Ratio (Baseline): {unresolved_stats['unresolved_ratio']*100:.1f}%")
    print("  ↑ This is what LLM and neural training aim to reduce")
    
    # === 4. STRUCTURAL HAMMING DISTANCE (SHD) ===
    shd_stats = compute_shd(fci_csv_path, ground_truth_path)
    
    print("\n" + "=" * 80)
    print("STRUCTURAL HAMMING DISTANCE (SHD)")
    print("=" * 80)
    print(f"Skeleton SHD: {shd_stats['skeleton_shd']}  (E_add + E_del, undirected)")
    print(f"  E_add (FP):   {shd_stats['additions']}  (edges added)")
    print(f"  E_del (FN):   {shd_stats['deletions']}  (edges missing)")
    print(f"\nFull SHD:     {shd_stats['full_shd']}  (E_add + E_del + E_rev, directed)")
    print(f"  E_add (FP):   {shd_stats['additions']}  (edges added)")
    print(f"  E_del (FN):   {shd_stats['deletions']}  (edges missing)")
    print(f"  E_rev:        {shd_stats['total_direction_errors']}  (direction errors)")
    print(f"    - Reversed: {shd_stats['reversals']}  (wrong direction)")
    print(f"    - Unresolved: {shd_stats['unresolved_as_errors']}  (undirected in FCI)")
    print(f"\n[NOTE] Unresolved edges count as direction errors for fair comparison with Neural Network")
    
    # === 5. UNDIRECTED RATIO ===
    undirected_ratio = len(fci_undirected) / len(fci_all_undirected) if len(fci_all_undirected) > 0 else 0
    
    print("\n" + "=" * 80)
    print("UNDIRECTED / PARTIAL EDGES")
    print("=" * 80)
    print(f"Total FCI edges: {len(fci_all_undirected)}")
    print(f"Directed: {len(fci_directed)}")
    print(f"Undirected/Partial: {len(fci_undirected)}")
    print(f"Undirected Ratio: {undirected_ratio*100:.1f}%")
    
    # === 6. SUMMARY ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Skeleton SHD:         {shd_stats['skeleton_shd']}  (undirected)")
    print(f"Full SHD:             {shd_stats['full_shd']}  (directed, standard metric)")
    print(f"Edge F1:              {edge_f1*100:.1f}%")
    print(f"Precision:            {edge_precision*100:.1f}%")
    print(f"Recall:               {edge_recall*100:.1f}%")
    print(f"Orient. Accuracy:     {orientation_accuracy*100:.1f}%")
    print(f"Unresolved Ratio:     {unresolved_stats['unresolved_ratio']*100:.1f}%  <- FCI Only (Baseline)")
    print(f"Undirected Ratio:     {undirected_ratio*100:.1f}%")
    print("=" * 80)
    
    # Save evaluation report
    if output_dir:
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"evaluation_FCI_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FCI EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"FCI CSV: {fci_csv_path.name}\n")
            f.write(f"Ground Truth: {ground_truth_path.name}\n\n")
            
            f.write("METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Edge F1:              {edge_f1*100:.1f}%\n")
            f.write(f"Edge Precision:       {edge_precision*100:.1f}%\n")
            f.write(f"Edge Recall:          {edge_recall*100:.1f}%\n")
            f.write(f"Orientation Accuracy: {orientation_accuracy*100:.1f}%\n")
            f.write(f"Undirected Ratio:     {undirected_ratio*100:.1f}%\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 80 + "\n")
            f.write(f"True Positives:  {undirected_tp}\n")
            f.write(f"False Positives: {undirected_fp}\n")
            f.write(f"False Negatives: {undirected_fn}\n\n")
            
            f.write("ORIENTATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Correctly Oriented:   {correctly_oriented}\n")
            f.write(f"Incorrectly Oriented: {incorrectly_oriented}\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n[OK] Evaluation report saved to: {report_path}")
    
    return {
        'skeleton_shd': shd_stats['skeleton_shd'],
        'full_shd': shd_stats['full_shd'],
        'shd': shd_stats['shd'],  # Default to full_shd
        'shd_additions': shd_stats['additions'],
        'shd_deletions': shd_stats['deletions'],
        'shd_reversals': shd_stats['reversals'],
        'edge_f1': edge_f1,
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'orientation_accuracy': orientation_accuracy,
        'unresolved_ratio': unresolved_stats['unresolved_ratio'],
        'unresolved_count': unresolved_stats['unresolved_count'],
        'resolved_count': unresolved_stats['resolved_count'],
        'undirected_ratio': undirected_ratio,
        'undirected_tp': undirected_tp,
        'undirected_fp': undirected_fp,
        'undirected_fn': undirected_fn,
        'correctly_oriented': correctly_oriented,
        'incorrectly_oriented': incorrectly_oriented
    }


def find_latest_fci_csv(output_dir='outputs'):
    """
    Find the most recent constraint-based skeleton CSV file.

    Backward-compatible name: historically this returned edges_FCI_*.csv.
    Now it also accepts edges_RFCI_*.csv and prefers RFCI when both exist.
    """
    from config import DATASET
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    # Try dataset-specific directory first
    dataset_dir = output_path / DATASET
    if dataset_dir.exists():
        rfci_csvs = list(dataset_dir.glob('edges_RFCI_*.csv'))
        if rfci_csvs:
            return max(rfci_csvs, key=lambda p: p.stat().st_mtime)
        fci_csvs = list(dataset_dir.glob('edges_FCI_*.csv'))
        if fci_csvs:
            return max(fci_csvs, key=lambda p: p.stat().st_mtime)
    
    # Fall back to root outputs directory
    rfci_csvs = list(output_path.glob('edges_RFCI_*.csv'))
    if rfci_csvs:
        return max(rfci_csvs, key=lambda p: p.stat().st_mtime)

    fci_csvs = list(output_path.glob('edges_FCI_*.csv'))
    if not fci_csvs:
        return None

    return max(fci_csvs, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    from config import GROUND_TRUTH_PATH, OUTPUT_DIR
    
    # Find latest FCI outputs
    latest_fci = find_latest_fci_csv(OUTPUT_DIR)
    
    if not latest_fci:
        print(f"[ERROR] No FCI CSV files found in {OUTPUT_DIR}/")
        print("Run main_fci.py first to generate FCI results.")
        exit(1)
    
    print(f"Found latest FCI outputs: {latest_fci.name}")
    
    # Ground truth path
    gt_path = Path(GROUND_TRUTH_PATH)
    
    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        print("Please update GROUND_TRUTH_PATH in config.py")
        exit(1)
    
    # Evaluate
    metrics = evaluate_fci(latest_fci, gt_path, output_dir=OUTPUT_DIR)

