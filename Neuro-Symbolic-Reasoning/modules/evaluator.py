"""
Evaluator Module

Evaluate learned causal graph against ground truth with comprehensive metrics
and metadata tracking.

Supports:
- Single dataset evaluation
- Tuebingen batch evaluation with aggregated metrics
- Auto-detection of Tuebingen dataset for aggregate statistics
"""

import re
import torch
import json
import time
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Set, Tuple, Optional, List
from pathlib import Path


class CausalGraphEvaluator:
    """
    Evaluate learned causal structure against ground truth
    
    Metrics:
    1. Edge-level (undirected): Precision, Recall, F1
    2. Directed edge-level: Precision, Recall, F1
    3. Orientation accuracy: Among correct edges, how many have correct direction
    4. Structural Hamming Distance (SHD)
    """
    
    def __init__(self, ground_truth_path: str, var_structure: Dict, ground_truth_type: str = 'bif'):
        """
        Args:
            ground_truth_path: Path to ground truth file
            var_structure: Variable structure from DataLoader
            ground_truth_type: Type of ground truth file ('bif', 'edge_list', etc.)
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.var_structure = var_structure
        self.ground_truth_type = ground_truth_type
        
        # Parse ground truth based on type
        if ground_truth_type == 'bif':
            self.ground_truth_edges, self.all_variables = self._parse_bif()
        elif ground_truth_type == 'edge_list':
            self.ground_truth_edges, self.all_variables = self._parse_edge_list()
        else:
            raise ValueError(f"Unsupported ground truth type: {ground_truth_type}")
        
        print("=" * 70)
        print("EVALUATOR INITIALIZED")
        print("=" * 70)
        print(f"Ground truth type: {ground_truth_type}")
        print(f"Ground truth edges: {len(self.ground_truth_edges)}")
        print(f"Variables: {len(self.all_variables)}")
    
    def _parse_bif(self) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """
        Parse ground truth from BIF file
        
        Returns:
            (ground_truth_edges, all_variables)
            ground_truth_edges: Set of (parent, child) tuples
            all_variables: Set of variable names
        """
        ground_truth_edges = set()
        all_variables = set()
        
        with open(self.ground_truth_path, 'r') as f:
            content = f.read()
        
        # Extract variable declarations
        var_pattern = r'variable\s+(\w+)\s*\{'
        variables = re.findall(var_pattern, content)
        all_variables = set(variables)
        
        # Extract probability declarations (define causal structure)
        # Format: probability ( CHILD | PARENT1, PARENT2, ... )
        prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'
        
        for match in re.finditer(prob_pattern, content):
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(',')]
            
            for parent in parents:
                if parent:  # Skip empty strings
                    ground_truth_edges.add((parent, child))
        
        return ground_truth_edges, all_variables
    
    def _parse_edge_list(self) -> Tuple[Set[Tuple[str, str]], Set[str]]:
        """
        Parse ground truth from edge list file
        
        Format:
            # Comments start with #
            source1 -> target1
            source2 -> target2
            ...
        
        Returns:
            (ground_truth_edges, all_variables)
            ground_truth_edges: Set of (source, target) tuples
            all_variables: Set of variable names (extracted from edges)
        """
        ground_truth_edges = set()
        all_variables = set()
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            for line in f:
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
                            all_variables.add(source)
                            all_variables.add(target)
        
        return ground_truth_edges, all_variables
    
    def extract_learned_edges(self, adjacency: torch.Tensor, 
                              threshold: float = 0.3) -> Set[Tuple[str, str]]:
        """
        Extract variable-level edges from learned adjacency matrix
        
        Args:
            adjacency: Learned adjacency matrix (105, 105)
            threshold: Threshold for considering a block as having an edge
        
        Returns:
            Set of (var_a, var_b) directed edges
        """
        learned_edges = set()
        
        thr = float(threshold) + 1e-6
        for var_a in self.var_structure['variable_names']:
            for var_b in self.var_structure['variable_names']:
                if var_a == var_b:
                    continue
                
                # Get block for this variable pair
                states_a = self.var_structure['var_to_states'][var_a]
                states_b = self.var_structure['var_to_states'][var_b]
                
                # Extract block
                block = adjacency[states_a][:, states_b]
                
                # Compute block strength (max)
                block_strength = block.max().item()
                
                # If block strength exceeds threshold, add edge
                if block_strength > thr:
                    learned_edges.add((var_a, var_b))
        
        return learned_edges
    
    def evaluate(self, learned_edges: Set[Tuple[str, str]]) -> Dict:
        """
        Compute all evaluation metrics
        
        Args:
            learned_edges: Set of learned directed edges
        
        Returns:
            Dictionary with all metrics
        """
        # Convert to undirected edges
        learned_undirected = {tuple(sorted([e[0], e[1]])) for e in learned_edges}
        gt_undirected = {tuple(sorted([e[0], e[1]])) for e in self.ground_truth_edges}
        
        # === 1. UNDIRECTED EDGE METRICS ===
        undirected_tp = len(learned_undirected & gt_undirected)
        undirected_fp = len(learned_undirected - gt_undirected)
        undirected_fn = len(gt_undirected - learned_undirected)
        
        edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
        edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
        
        # === 2. DIRECTED EDGE METRICS ===
        directed_tp = len(learned_edges & self.ground_truth_edges)
        directed_fp = len(learned_edges - self.ground_truth_edges)
        directed_fn = len(self.ground_truth_edges - learned_edges)
        
        directed_precision = directed_tp / (directed_tp + directed_fp) if (directed_tp + directed_fp) > 0 else 0
        directed_recall = directed_tp / (directed_tp + directed_fn) if (directed_tp + directed_fn) > 0 else 0
        directed_f1 = 2 * directed_precision * directed_recall / (directed_precision + directed_recall) if (directed_precision + directed_recall) > 0 else 0
        
        # === 3. ORIENTATION ACCURACY ===
        correctly_oriented = 0
        incorrectly_oriented = 0
        
        for learned_edge in learned_edges:
            undirected_edge = tuple(sorted([learned_edge[0], learned_edge[1]]))
            if undirected_edge in gt_undirected:
                # We found this edge, check if direction is correct
                if learned_edge in self.ground_truth_edges:
                    correctly_oriented += 1
                else:
                    # Check if it's reversed
                    reversed_edge = (learned_edge[1], learned_edge[0])
                    if reversed_edge in self.ground_truth_edges:
                        incorrectly_oriented += 1
        
        orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
        
        # === 4. STRUCTURAL HAMMING DISTANCE (SHD) ===
        # According to standard definition :
        # 
        # Skeleton SHD: Only considers edge existence (undirected)
        #   SHD_skeleton = E_add + E_del
        #   E_add (False Positive): edges in learned but not in GT (undirected)
        #   E_del (False Negative): edges in GT but not in learned (undirected)
        #
        # Full SHD: Considers both edge existence and direction
        #   SHD_full = E_add + E_del + E_rev
        #   E_add: edges added (not in GT at all)
        #   E_del: edges missing (in GT but not learned at all)
        #   E_rev: edges with correct skeleton but wrong direction
        
        # Count reversals (edges with correct skeleton but wrong direction)
        reversals = 0
        for learned_edge in learned_edges:
            reversed_edge = (learned_edge[1], learned_edge[0])
            if reversed_edge in self.ground_truth_edges and learned_edge not in self.ground_truth_edges:
                reversals += 1
        
        # Skeleton SHD: undirected edge errors
        skeleton_shd = undirected_fp + undirected_fn
        
        # Full SHD: directed edge errors (standard NeurIPS/ICLR metric)
        # E_add = edges added (not in GT undirected graph)
        # E_del = edges deleted (in GT undirected graph but not learned)
        # E_rev = edges with correct skeleton but wrong direction
        full_shd = undirected_fp + undirected_fn + reversals
        
        # Compile metrics
        metrics = {
            # Undirected (skeleton) metrics
            'edge_precision': edge_precision,
            'edge_recall': edge_recall,
            'edge_f1': edge_f1,
            'undirected_tp': undirected_tp,
            'undirected_fp': undirected_fp,
            'undirected_fn': undirected_fn,
            
            # Directed metrics
            'directed_precision': directed_precision,
            'directed_recall': directed_recall,
            'directed_f1': directed_f1,
            'directed_tp': directed_tp,
            'directed_fp': directed_fp,
            'directed_fn': directed_fn,
            
            # Orientation
            'orientation_accuracy': orientation_accuracy,
            'correctly_oriented': correctly_oriented,
            'incorrectly_oriented': incorrectly_oriented,
            
            # SHD (Structural Hamming Distance)
            'skeleton_shd': skeleton_shd,  # Only edge existence (undirected)
            'full_shd': full_shd,          # Edge existence + direction (standard metric)
            'shd': full_shd,               # Default to full_shd for backward compatibility
            'reversals': reversals,        # Number of edges with wrong direction
            
            # Counts
            'learned_edges': len(learned_edges),
            'ground_truth_edges': len(self.ground_truth_edges)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way"""
        print("\n" + "=" * 70)
        print("EVALUATION METRICS")
        print("=" * 70)
        
        print("\n--- EDGE-LEVEL (Ignoring Direction) ---")
        print(f"Edge Precision:     {metrics['edge_precision']:.1%}")
        print(f"Edge Recall:        {metrics['edge_recall']:.1%}")
        print(f"Edge F1 Score:      {metrics['edge_f1']:.1%}")
        print(f"True Positives:     {metrics['undirected_tp']}")
        print(f"False Positives:    {metrics['undirected_fp']}")
        print(f"False Negatives:    {metrics['undirected_fn']}")
        
        print("\n--- DIRECTED EDGE-LEVEL (With Direction) ---")
        print(f"Directed Precision: {metrics['directed_precision']:.1%}")
        print(f"Directed Recall:    {metrics['directed_recall']:.1%}")
        print(f"Directed F1 Score:  {metrics['directed_f1']:.1%}")
        print(f"True Positives:     {metrics['directed_tp']}")
        print(f"False Positives:    {metrics['directed_fp']}")
        print(f"False Negatives:    {metrics['directed_fn']}")
        
        print("\n--- ORIENTATION ACCURACY ---")
        print(f"Orientation Accuracy: {metrics['orientation_accuracy']:.1%}")
        print(f"Correctly Oriented:   {metrics['correctly_oriented']}")
        print(f"Incorrectly Oriented: {metrics['incorrectly_oriented']}")
        
        print("\n--- STRUCTURAL HAMMING DISTANCE (SHD) ---")
        print(f"Skeleton SHD:  {metrics['skeleton_shd']}  (E_add + E_del, undirected)")
        print(f"  E_add (FP):  {metrics['undirected_fp']}  (edges added)")
        print(f"  E_del (FN):  {metrics['undirected_fn']}  (edges missing)")
        print(f"\nFull SHD:      {metrics['full_shd']}  (E_add + E_del + E_rev, directed)")
        print(f"  E_add (FP):  {metrics['undirected_fp']}  (edges added)")
        print(f"  E_del (FN):  {metrics['undirected_fn']}  (edges missing)")
        print(f"  E_rev:       {metrics['reversals']}  (edges reversed)")
        
        print("\n--- SUMMARY ---")
        print(f"Learned Edges:      {metrics['learned_edges']}")
        print(f"Ground Truth Edges: {metrics['ground_truth_edges']}")
    
    def save_results(self, 
                     metrics: Dict, 
                     learned_edges: Set[Tuple[str, str]],
                     output_dir: str,
                     config: Dict,
                     timing_info: Optional[Dict] = None):
        """
        Save evaluation results with comprehensive metadata
        
        Args:
            metrics: Evaluation metrics dictionary
            learned_edges: Set of learned edges
            output_dir: Directory to save results
            config: Training configuration with metadata
            timing_info: Dictionary with timing information (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare comprehensive results
        results = {
            # Metadata
            'metadata': {
                'dataset': config.get('dataset_name', 'Unknown'),
                'llm_model': config.get('llm_model', 'None'),
                'use_llm_prior': config.get('use_llm_prior', False),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'run_id': config.get('run_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
            },
            
            # Configuration
            'config': {
                'learning_rate': config.get('learning_rate', None),
                'lambda_group_lasso': config.get('lambda_group_lasso', None),
                'lambda_cycle': config.get('lambda_cycle', None),
                'n_epochs': config.get('n_epochs', None),
                'threshold': config.get('threshold', 0.3),
                'fci_skeleton_path': config.get('fci_skeleton_path', None),
                'llm_direction_path': config.get('llm_direction_path', None)
            },
            
            # Timing information
            'timing': timing_info or {},
            
            # Metrics
            'metrics': metrics,
            
            # Learned edges
            'learned_edges': [{'source': e[0], 'target': e[1]} for e in sorted(learned_edges)],
            
            # Ground truth edges
            'ground_truth_edges': [{'source': e[0], 'target': e[1]} for e in sorted(self.ground_truth_edges)]
        }
        
        # Save as JSON
        json_path = output_path / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as human-readable text
        txt_path = output_path / 'evaluation_results.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CAUSAL DISCOVERY EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            f.write("METADATA\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset:        {results['metadata']['dataset']}\n")
            f.write(f"LLM Model:      {results['metadata']['llm_model']}\n")
            f.write(f"Use LLM Prior:  {results['metadata']['use_llm_prior']}\n")
            f.write(f"Timestamp:      {results['metadata']['timestamp']}\n")
            f.write(f"Run ID:         {results['metadata']['run_id']}\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            for key, value in results['config'].items():
                if value is not None:
                    f.write(f"{key:25s}: {value}\n")
            f.write("\n")
            
            # Timing
            if timing_info:
                f.write("TIMING INFORMATION\n")
                f.write("-" * 80 + "\n")
                for key, value in timing_info.items():
                    if isinstance(value, float):
                        f.write(f"{key:25s}: {value:.2f} seconds\n")
                    else:
                        f.write(f"{key:25s}: {value}\n")
                f.write("\n")
            
            # Metrics
            f.write("EVALUATION METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Edge-Level (Undirected Skeleton)\n")
            f.write(f"  Precision:      {metrics['edge_precision']:.1%}\n")
            f.write(f"  Recall:         {metrics['edge_recall']:.1%}\n")
            f.write(f"  F1 Score:       {metrics['edge_f1']:.1%}\n")
            f.write(f"  True Positives: {metrics['undirected_tp']}\n")
            f.write(f"  False Positives:{metrics['undirected_fp']}\n")
            f.write(f"  False Negatives:{metrics['undirected_fn']}\n\n")
            
            f.write("Directed Edge-Level\n")
            f.write(f"  Precision:      {metrics['directed_precision']:.1%}\n")
            f.write(f"  Recall:         {metrics['directed_recall']:.1%}\n")
            f.write(f"  F1 Score:       {metrics['directed_f1']:.1%}\n")
            f.write(f"  True Positives: {metrics['directed_tp']}\n")
            f.write(f"  False Positives:{metrics['directed_fp']}\n")
            f.write(f"  False Negatives:{metrics['directed_fn']}\n\n")
            
            f.write("Orientation Accuracy\n")
            f.write(f"  Accuracy:       {metrics['orientation_accuracy']:.1%}\n")
            f.write(f"  Correct:        {metrics['correctly_oriented']}\n")
            f.write(f"  Incorrect:      {metrics['incorrectly_oriented']}\n\n")
            
            f.write("Structural Hamming Distance (SHD)\n")
            f.write(f"  Skeleton SHD:   {metrics['skeleton_shd']} (E_add + E_del, undirected)\n")
            f.write(f"    E_add (FP):   {metrics['undirected_fp']}\n")
            f.write(f"    E_del (FN):   {metrics['undirected_fn']}\n")
            f.write(f"  Full SHD:       {metrics['full_shd']} (E_add + E_del + E_rev, directed)\n")
            f.write(f"    E_add (FP):   {metrics['undirected_fp']}\n")
            f.write(f"    E_del (FN):   {metrics['undirected_fn']}\n")
            f.write(f"    E_rev:        {metrics['reversals']}\n\n")
            
            f.write("Summary\n")
            f.write(f"  Learned Edges:  {metrics['learned_edges']}\n")
            f.write(f"  GT Edges:       {metrics['ground_truth_edges']}\n\n")
            
            # Learned edges
            f.write("=" * 80 + "\n")
            f.write("LEARNED EDGES\n")
            f.write("=" * 80 + "\n")
            for edge in sorted(learned_edges):
                status = "✓" if edge in self.ground_truth_edges else ("↔" if (edge[1], edge[0]) in self.ground_truth_edges else "✗")
                f.write(f"{status} {edge[0]:20s} → {edge[1]:20s}\n")
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")
        
        # Auto-aggregate for Tuebingen dataset
        dataset_name = config.get('dataset_name', '')
        if dataset_name.startswith('tuebingen_pair'):
            self._auto_aggregate_tuebingen_results()
    
    def _auto_aggregate_tuebingen_results(self):
        """
        Auto-aggregate all Tuebingen results when a new pair completes
        
        This method:
        1. Finds all tuebingen_pair* result directories
        2. Loads the latest result for each pair
        3. Computes aggregate statistics
        4. Saves benchmark CSV and summary
        """
        try:
            # Get results directory (parent of current evaluator's ground truth path)
            results_base = Path(__file__).parent.parent / 'results'
            
            if not results_base.exists():
                return
            
            # Find all tuebingen pair results
            pair_results = []
            
            for i in range(1, 109):
                pair_id = f"pair{i:04d}"
                pair_result_dir = results_base / f"tuebingen_{pair_id}"
                
                if not pair_result_dir.exists():
                    continue
                
                # Find all subdirectories (different runs)
                subdirs = [d for d in pair_result_dir.iterdir() if d.is_dir()]
                
                if not subdirs:
                    continue
                
                # Get the most recent run
                latest_dir = max(subdirs, key=lambda x: x.stat().st_mtime)
                eval_json = latest_dir / 'evaluation_results.json'
                
                if not eval_json.exists():
                    continue
                
                try:
                    with open(eval_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metrics = data.get('metrics', {})
                    timing = data.get('timing', {})
                    
                    result = {
                        'pair_id': pair_id,
                        'pair_number': i,
                        'run_dir': latest_dir.name,
                        'edge_precision': metrics.get('edge_precision', 0),
                        'edge_recall': metrics.get('edge_recall', 0),
                        'edge_f1': metrics.get('edge_f1', 0),
                        'directed_precision': metrics.get('directed_precision', 0),
                        'directed_recall': metrics.get('directed_recall', 0),
                        'directed_f1': metrics.get('directed_f1', 0),
                        'orientation_accuracy': metrics.get('orientation_accuracy', 0),
                        'correctly_oriented': metrics.get('correctly_oriented', 0),
                        'incorrectly_oriented': metrics.get('incorrectly_oriented', 0),
                        'skeleton_shd': metrics.get('skeleton_shd', 0),
                        'full_shd': metrics.get('full_shd', 0),
                        'reversals': metrics.get('reversals', 0),
                        'learned_edges': metrics.get('learned_edges', 0),
                        'ground_truth_edges': metrics.get('ground_truth_edges', 0),
                        'time_seconds': timing.get('total', 0),
                    }
                    
                    pair_results.append(result)
                
                except Exception:
                    continue
            
            if len(pair_results) < 2:
                # Need at least 2 pairs to compute meaningful statistics
                return
            
            # Create DataFrame
            df = pd.DataFrame(pair_results)
            
            # Save detailed CSV
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = results_base / f'tuebingen_benchmark_{timestamp}.csv'
            df.to_csv(csv_path, index=False, float_format='%.4f')
            
            # Compute aggregate statistics
            metrics_to_aggregate = [
                'edge_precision', 'edge_recall', 'edge_f1',
                'directed_precision', 'directed_recall', 'directed_f1',
                'orientation_accuracy',
                'skeleton_shd', 'full_shd', 'reversals',
                'time_seconds'
            ]
            
            summary = {}
            for metric in metrics_to_aggregate:
                values = df[metric]
                summary[f'{metric}_mean'] = values.mean()
                summary[f'{metric}_std'] = values.std()
                summary[f'{metric}_min'] = values.min()
                summary[f'{metric}_max'] = values.max()
            
            # Save summary report
            summary_path = results_base / f'tuebingen_summary_{timestamp}.txt'
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("TUEBINGEN BENCHMARK SUMMARY (AUTO-GENERATED)\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"Total Pairs Completed: {len(pair_results)}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("AGGREGATE METRICS (Mean ± Std)\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Edge-Level (Undirected Skeleton)\n")
                f.write(f"  Precision:  {summary['edge_precision_mean']:.1%} ± {summary['edge_precision_std']:.1%}\n")
                f.write(f"  Recall:     {summary['edge_recall_mean']:.1%} ± {summary['edge_recall_std']:.1%}\n")
                f.write(f"  F1 Score:   {summary['edge_f1_mean']:.1%} ± {summary['edge_f1_std']:.1%}\n\n")
                
                f.write("Directed Edge-Level\n")
                f.write(f"  Precision:  {summary['directed_precision_mean']:.1%} ± {summary['directed_precision_std']:.1%}\n")
                f.write(f"  Recall:     {summary['directed_recall_mean']:.1%} ± {summary['directed_recall_std']:.1%}\n")
                f.write(f"  F1 Score:   {summary['directed_f1_mean']:.1%} ± {summary['directed_f1_std']:.1%}\n\n")
                
                f.write("Orientation Accuracy\n")
                f.write(f"  Accuracy:   {summary['orientation_accuracy_mean']:.1%} ± {summary['orientation_accuracy_std']:.1%}\n\n")
                
                f.write("Structural Hamming Distance (SHD)\n")
                f.write(f"  Skeleton:   {summary['skeleton_shd_mean']:.2f} ± {summary['skeleton_shd_std']:.2f}\n")
                f.write(f"  Full:       {summary['full_shd_mean']:.2f} ± {summary['full_shd_std']:.2f}\n")
                f.write(f"  Reversals:  {summary['reversals_mean']:.2f} ± {summary['reversals_std']:.2f}\n\n")
                
                f.write("Timing\n")
                total_time = summary['time_seconds_mean'] * len(pair_results)
                f.write(f"  Total:      {total_time:.1f}s ({total_time/60:.1f} min)\n")
                f.write(f"  Per Pair:   {summary['time_seconds_mean']:.1f}s ± {summary['time_seconds_std']:.1f}s\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("RANGE STATISTICS\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Edge F1 Score\n")
                f.write(f"  Min:  {summary['edge_f1_min']:.1%}\n")
                f.write(f"  Max:  {summary['edge_f1_max']:.1%}\n\n")
                
                f.write("Orientation Accuracy\n")
                f.write(f"  Min:  {summary['orientation_accuracy_min']:.1%}\n")
                f.write(f"  Max:  {summary['orientation_accuracy_max']:.1%}\n\n")
                
                f.write("Full SHD\n")
                f.write(f"  Min:  {summary['full_shd_min']:.0f}\n")
                f.write(f"  Max:  {summary['full_shd_max']:.0f}\n\n")
                
                f.write("=" * 80 + "\n")
            
            print(f"\n{'='*80}")
            print("TUEBINGEN AUTO-AGGREGATE")
            print(f"{'='*80}")
            print(f"Collected {len(pair_results)} pairs")
            print(f"[CSV] {csv_path}")
            print(f"[SUMMARY] {summary_path}")
            print(f"\nEdge F1:         {summary['edge_f1_mean']:.1%} ± {summary['edge_f1_std']:.1%}")
            print(f"Orient Acc:      {summary['orientation_accuracy_mean']:.1%} ± {summary['orientation_accuracy_std']:.1%}")
            print(f"Full SHD:        {summary['full_shd_mean']:.2f} ± {summary['full_shd_std']:.2f}")
            print(f"{'='*80}\n")
        
        except Exception as e:
            # Silently fail if aggregation doesn't work
            # (e.g., first pair, missing files, etc.)
            pass


class TuebingenBatchEvaluator:
    """
    Batch evaluator for Tuebingen cause-effect pairs
    
    Aggregates results from multiple pairs and computes average metrics
    """
    
    def __init__(self):
        """Initialize batch evaluator"""
        self.results = []
        self.pair_metrics = []
        self.pair_timings = []  # Track timing for each pair
    
    def add_pair_result(self, pair_id: str, metrics: Dict, learned_edges: Set[Tuple[str, str]], 
                       ground_truth_edges: Set[Tuple[str, str]], config: Dict = None, timing: float = None):
        """
        Add evaluation result for a single pair
        
        Args:
            pair_id: Pair identifier (e.g., 'pair0001')
            metrics: Evaluation metrics dictionary
            learned_edges: Set of learned edges
            ground_truth_edges: Set of ground truth edges
            config: Configuration dictionary (optional)
            timing: Time taken for this pair in seconds (optional)
        """
        result = {
            'pair_id': pair_id,
            'metrics': metrics,
            'learned_edges': learned_edges,
            'ground_truth_edges': ground_truth_edges,
            'config': config or {},
            'timing': timing
        }
        
        self.results.append(result)
        self.pair_metrics.append(metrics)
        if timing is not None:
            self.pair_timings.append(timing)
    
    def compute_aggregate_metrics(self) -> Dict:
        """
        Compute aggregate metrics across all pairs
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self.pair_metrics:
            return {}
        
        # Metrics to aggregate
        metric_keys = [
            'edge_precision', 'edge_recall', 'edge_f1',
            'directed_precision', 'directed_recall', 'directed_f1',
            'orientation_accuracy',
            'skeleton_shd', 'full_shd', 'reversals'
        ]
        
        aggregate = {}
        
        for key in metric_keys:
            values = [m[key] for m in self.pair_metrics if key in m]
            
            if values:
                aggregate[f'{key}_mean'] = sum(values) / len(values)
                aggregate[f'{key}_std'] = (sum((x - aggregate[f'{key}_mean'])**2 for x in values) / len(values))**0.5
                aggregate[f'{key}_min'] = min(values)
                aggregate[f'{key}_max'] = max(values)
        
        # Add counts
        aggregate['total_pairs'] = len(self.pair_metrics)
        aggregate['total_learned_edges'] = sum(m['learned_edges'] for m in self.pair_metrics)
        aggregate['total_ground_truth_edges'] = sum(m['ground_truth_edges'] for m in self.pair_metrics)
        
        return aggregate
    
    def save_csv_results(self, output_path: str, config: Dict = None):
        """
        Save results as CSV file (Tuebingen benchmark format)
        
        Args:
            output_path: Path to save CSV file
            config: Configuration dictionary with metadata
        """
        if not self.results:
            print("[WARN] No results to save")
            return
        
        # Prepare data for CSV
        rows = []
        
        for result in self.results:
            pair_id = result['pair_id']
            metrics = result['metrics']
            cfg = result.get('config', {})
            
            row = {
                'pair_id': pair_id,
                'pair_number': int(pair_id.replace('pair', '')) if 'pair' in pair_id else 0,
                
                # Edge metrics (undirected)
                'edge_precision': metrics.get('edge_precision', 0),
                'edge_recall': metrics.get('edge_recall', 0),
                'edge_f1': metrics.get('edge_f1', 0),
                
                # Directed metrics
                'directed_precision': metrics.get('directed_precision', 0),
                'directed_recall': metrics.get('directed_recall', 0),
                'directed_f1': metrics.get('directed_f1', 0),
                
                # Orientation
                'orientation_accuracy': metrics.get('orientation_accuracy', 0),
                'correctly_oriented': metrics.get('correctly_oriented', 0),
                'incorrectly_oriented': metrics.get('incorrectly_oriented', 0),
                
                # SHD
                'skeleton_shd': metrics.get('skeleton_shd', 0),
                'full_shd': metrics.get('full_shd', 0),
                'reversals': metrics.get('reversals', 0),
                
                # Counts
                'learned_edges': metrics.get('learned_edges', 0),
                'ground_truth_edges': metrics.get('ground_truth_edges', 0),
                
                # TP/FP/FN
                'undirected_tp': metrics.get('undirected_tp', 0),
                'undirected_fp': metrics.get('undirected_fp', 0),
                'undirected_fn': metrics.get('undirected_fn', 0),
                'directed_tp': metrics.get('directed_tp', 0),
                'directed_fp': metrics.get('directed_fp', 0),
                'directed_fn': metrics.get('directed_fn', 0),
                
                # Configuration
                'llm_model': cfg.get('llm_model', 'None'),
                'use_llm_prior': cfg.get('use_llm_prior', False),
                'learning_rate': cfg.get('learning_rate', 0),
                'n_epochs': cfg.get('n_epochs', 0),
                'threshold': cfg.get('threshold', 0.3),
                
                # Timing
                'time_seconds': result.get('timing', 0),
            }
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Sort by pair number
        df = df.sort_values('pair_number')
        
        # Save CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"\n[CSV] Saved to: {output_path}")
        print(f"  Pairs: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
    
    def save_summary_report(self, output_path: str, config: Dict = None):
        """
        Save summary report with aggregate statistics
        
        Args:
            output_path: Path to save summary report
            config: Configuration dictionary with metadata
        """
        if not self.results:
            print("[WARN] No results to save")
            return
        
        aggregate = self.compute_aggregate_metrics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TUEBINGEN BATCH EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            if config:
                f.write("CONFIGURATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"LLM Model:      {config.get('llm_model', 'None')}\n")
                f.write(f"Use LLM Prior:  {config.get('use_llm_prior', False)}\n")
                f.write(f"Learning Rate:  {config.get('learning_rate', 0)}\n")
                f.write(f"Epochs:         {config.get('n_epochs', 0)}\n")
                f.write(f"Threshold:      {config.get('threshold', 0.3)}\n")
                f.write(f"Timestamp:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Pairs:           {aggregate['total_pairs']}\n")
            f.write(f"Total Learned Edges:   {aggregate['total_learned_edges']}\n")
            f.write(f"Total GT Edges:        {aggregate['total_ground_truth_edges']}\n")
            
            # Timing statistics (if available)
            if self.pair_timings:
                import numpy as np
                total_time = sum(self.pair_timings)
                mean_time = np.mean(self.pair_timings)
                std_time = np.std(self.pair_timings)
                f.write(f"\nTiming Statistics:\n")
                f.write(f"  Total Time:        {total_time:.1f}s ({total_time/60:.1f} min)\n")
                f.write(f"  Average per Pair:  {mean_time:.1f}s ± {std_time:.1f}s\n")
                f.write(f"  Min Time:          {min(self.pair_timings):.1f}s\n")
                f.write(f"  Max Time:          {max(self.pair_timings):.1f}s\n")
            f.write("\n")
            
            # Aggregate metrics
            f.write("AGGREGATE METRICS (Mean ± Std)\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Edge-Level (Undirected Skeleton)\n")
            f.write(f"  Precision:  {aggregate['edge_precision_mean']:.1%} ± {aggregate['edge_precision_std']:.1%}\n")
            f.write(f"  Recall:     {aggregate['edge_recall_mean']:.1%} ± {aggregate['edge_recall_std']:.1%}\n")
            f.write(f"  F1 Score:   {aggregate['edge_f1_mean']:.1%} ± {aggregate['edge_f1_std']:.1%}\n\n")
            
            f.write("Directed Edge-Level\n")
            f.write(f"  Precision:  {aggregate['directed_precision_mean']:.1%} ± {aggregate['directed_precision_std']:.1%}\n")
            f.write(f"  Recall:     {aggregate['directed_recall_mean']:.1%} ± {aggregate['directed_recall_std']:.1%}\n")
            f.write(f"  F1 Score:   {aggregate['directed_f1_mean']:.1%} ± {aggregate['directed_f1_std']:.1%}\n\n")
            
            f.write("Orientation Accuracy\n")
            f.write(f"  Accuracy:   {aggregate['orientation_accuracy_mean']:.1%} ± {aggregate['orientation_accuracy_std']:.1%}\n\n")
            
            f.write("Structural Hamming Distance (SHD)\n")
            f.write(f"  Skeleton:   {aggregate['skeleton_shd_mean']:.2f} ± {aggregate['skeleton_shd_std']:.2f}\n")
            f.write(f"  Full:       {aggregate['full_shd_mean']:.2f} ± {aggregate['full_shd_std']:.2f}\n")
            f.write(f"  Reversals:  {aggregate['reversals_mean']:.2f} ± {aggregate['reversals_std']:.2f}\n\n")
            
            # Range statistics
            f.write("RANGE STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Edge F1 Score\n")
            f.write(f"  Min:  {aggregate['edge_f1_min']:.1%}\n")
            f.write(f"  Max:  {aggregate['edge_f1_max']:.1%}\n\n")
            
            f.write("Orientation Accuracy\n")
            f.write(f"  Min:  {aggregate['orientation_accuracy_min']:.1%}\n")
            f.write(f"  Max:  {aggregate['orientation_accuracy_max']:.1%}\n\n")
            
            f.write("Full SHD\n")
            f.write(f"  Min:  {aggregate['full_shd_min']:.0f}\n")
            f.write(f"  Max:  {aggregate['full_shd_max']:.0f}\n\n")
            
            # Per-pair results
            f.write("=" * 80 + "\n")
            f.write("PER-PAIR RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Pair ID':<12} {'Edge F1':<10} {'Orient Acc':<12} {'Full SHD':<10} {'Learned':<10} {'GT':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in sorted(self.results, key=lambda x: x['pair_id']):
                pair_id = result['pair_id']
                m = result['metrics']
                
                f.write(f"{pair_id:<12} "
                       f"{m['edge_f1']:<10.1%} "
                       f"{m['orientation_accuracy']:<12.1%} "
                       f"{m['full_shd']:<10.0f} "
                       f"{m['learned_edges']:<10} "
                       f"{m['ground_truth_edges']:<10}\n")
        
        print(f"[SUMMARY] Saved to: {output_path}")
    
    def print_aggregate_summary(self):
        """Print aggregate summary to console"""
        if not self.results:
            print("[WARN] No results to summarize")
            return
        
        aggregate = self.compute_aggregate_metrics()
        
        print("\n" + "=" * 80)
        print("TUEBINGEN BATCH EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Pairs: {aggregate['total_pairs']}")
        print(f"Total Edges: {aggregate['total_learned_edges']} learned, {aggregate['total_ground_truth_edges']} ground truth")
        
        # Timing statistics (if available)
        if self.pair_timings:
            import numpy as np
            total_time = sum(self.pair_timings)
            mean_time = np.mean(self.pair_timings)
            std_time = np.std(self.pair_timings)
            print(f"\n--- TIMING ---")
            print(f"Total Time:         {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"Average per Pair:   {mean_time:.1f}s ± {std_time:.1f}s")
            print(f"Range:              [{min(self.pair_timings):.1f}s, {max(self.pair_timings):.1f}s]")
        
        print("\n--- AGGREGATE METRICS (Mean ± Std) ---")
        print(f"Edge F1:            {aggregate['edge_f1_mean']:.1%} ± {aggregate['edge_f1_std']:.1%}")
        print(f"Directed F1:        {aggregate['directed_f1_mean']:.1%} ± {aggregate['directed_f1_std']:.1%}")
        print(f"Orientation Acc:    {aggregate['orientation_accuracy_mean']:.1%} ± {aggregate['orientation_accuracy_std']:.1%}")
        print(f"Full SHD:           {aggregate['full_shd_mean']:.2f} ± {aggregate['full_shd_std']:.2f}")
        
        print("\n--- RANGE ---")
        print(f"Edge F1:            [{aggregate['edge_f1_min']:.1%}, {aggregate['edge_f1_max']:.1%}]")
        print(f"Orientation Acc:    [{aggregate['orientation_accuracy_min']:.1%}, {aggregate['orientation_accuracy_max']:.1%}]")
        print(f"Full SHD:           [{aggregate['full_shd_min']:.0f}, {aggregate['full_shd_max']:.0f}]")
        print("=" * 80)


def evaluate_llm_output(llm_csv_path: str, ground_truth_path: str, 
                        ground_truth_type: str = 'edge_list',
                        output_dir: str = None) -> Dict:
    """
    Evaluate LLM outputs against ground truth
    
    This function evaluates the FCI+LLM hybrid outputs to measure LLM's
    contribution to causal direction resolution.
    
    Args:
        llm_csv_path: Path to FCI+LLM outputs CSV (e.g., edges_FCI_LLM_GPT35_*.csv)
        ground_truth_path: Path to ground truth file
        ground_truth_type: Type of ground truth ('bif' or 'edge_list')
        output_dir: Optional directory to save evaluation results
    
    Returns:
        Dictionary with evaluation metrics
    """
    import pandas as pd
    
    print("\n" + "=" * 80)
    print("EVALUATING LLM OUTPUT")
    print("=" * 80)
    print(f"LLM CSV: {llm_csv_path}")
    print(f"Ground Truth: {ground_truth_path}")
    
    # Load LLM outputs
    df_llm = pd.read_csv(llm_csv_path)
    
    # Extract learned edges from LLM outputs
    learned_edges = set()
    for _, row in df_llm.iterrows():
        source = row['source']
        target = row['target']
        edge_type = row.get('edge_type', 'directed')
        status = row.get('status', 'accepted')
        
        # Only include accepted edges
        if status == 'accepted':
            learned_edges.add((source, target))
    
    print(f"LLM Edges: {len(learned_edges)}")
    
    # Load ground truth
    ground_truth_edges = set()
    all_variables = set()
    
    if ground_truth_type == 'edge_list':
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '->' in line:
                    parts = line.split('->')
                    if len(parts) == 2:
                        source = parts[0].strip()
                        target = parts[1].strip()
                        if source and target:
                            ground_truth_edges.add((source, target))
                            all_variables.add(source)
                            all_variables.add(target)
    elif ground_truth_type == 'bif':
        # Parse BIF file
        with open(ground_truth_path, 'r') as f:
            content = f.read()
        
        import re
        var_pattern = r'variable\s+(\w+)\s*\{'
        variables = re.findall(var_pattern, content)
        all_variables = set(variables)
        
        prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'
        for match in re.finditer(prob_pattern, content):
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(',')]
            for parent in parents:
                if parent:
                    ground_truth_edges.add((parent, child))
    
    print(f"Ground Truth Edges: {len(ground_truth_edges)}")
    
    # Compute metrics (same as CausalGraphEvaluator.evaluate)
    learned_undirected = {tuple(sorted([e[0], e[1]])) for e in learned_edges}
    gt_undirected = {tuple(sorted([e[0], e[1]])) for e in ground_truth_edges}
    
    # Undirected metrics
    undirected_tp = len(learned_undirected & gt_undirected)
    undirected_fp = len(learned_undirected - gt_undirected)
    undirected_fn = len(gt_undirected - learned_undirected)
    
    edge_precision = undirected_tp / (undirected_tp + undirected_fp) if (undirected_tp + undirected_fp) > 0 else 0
    edge_recall = undirected_tp / (undirected_tp + undirected_fn) if (undirected_tp + undirected_fn) > 0 else 0
    edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall) if (edge_precision + edge_recall) > 0 else 0
    
    # Directed metrics
    directed_tp = len(learned_edges & ground_truth_edges)
    directed_fp = len(learned_edges - ground_truth_edges)
    directed_fn = len(ground_truth_edges - learned_edges)
    
    directed_precision = directed_tp / (directed_tp + directed_fp) if (directed_tp + directed_fp) > 0 else 0
    directed_recall = directed_tp / (directed_tp + directed_fn) if (directed_tp + directed_fn) > 0 else 0
    directed_f1 = 2 * directed_precision * directed_recall / (directed_precision + directed_recall) if (directed_precision + directed_recall) > 0 else 0
    
    # Orientation accuracy
    correctly_oriented = 0
    incorrectly_oriented = 0
    
    for learned_edge in learned_edges:
        undirected_edge = tuple(sorted([learned_edge[0], learned_edge[1]]))
        if undirected_edge in gt_undirected:
            if learned_edge in ground_truth_edges:
                correctly_oriented += 1
            else:
                reversed_edge = (learned_edge[1], learned_edge[0])
                if reversed_edge in ground_truth_edges:
                    incorrectly_oriented += 1
    
    orientation_accuracy = correctly_oriented / (correctly_oriented + incorrectly_oriented) if (correctly_oriented + incorrectly_oriented) > 0 else 0
    
    # SHD
    reversals = 0
    for learned_edge in learned_edges:
        reversed_edge = (learned_edge[1], learned_edge[0])
        if reversed_edge in ground_truth_edges and learned_edge not in ground_truth_edges:
            reversals += 1
    
    skeleton_shd = undirected_fp + undirected_fn
    full_shd = undirected_fp + undirected_fn + reversals
    
    metrics = {
        'edge_precision': edge_precision,
        'edge_recall': edge_recall,
        'edge_f1': edge_f1,
        'undirected_tp': undirected_tp,
        'undirected_fp': undirected_fp,
        'undirected_fn': undirected_fn,
        'directed_precision': directed_precision,
        'directed_recall': directed_recall,
        'directed_f1': directed_f1,
        'directed_tp': directed_tp,
        'directed_fp': directed_fp,
        'directed_fn': directed_fn,
        'orientation_accuracy': orientation_accuracy,
        'correctly_oriented': correctly_oriented,
        'incorrectly_oriented': incorrectly_oriented,
        'skeleton_shd': skeleton_shd,
        'full_shd': full_shd,
        'reversals': reversals,
        'learned_edges': len(learned_edges),
        'ground_truth_edges': len(ground_truth_edges)
    }
    
    # Print results
    print("\n" + "=" * 80)
    print("LLM EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nEdge F1:            {metrics['edge_f1']:.1%}")
    print(f"Directed F1:        {metrics['directed_f1']:.1%}")
    print(f"Orientation Acc:    {metrics['orientation_accuracy']:.1%}")
    print(f"Full SHD:           {metrics['full_shd']}")
    print(f"\nLearned Edges:      {metrics['learned_edges']}")
    print(f"Ground Truth Edges: {metrics['ground_truth_edges']}")
    print("=" * 80)
    
    # Save results if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = output_path / 'llm_evaluation_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'llm_csv_path': str(llm_csv_path),
                'ground_truth_path': str(ground_truth_path),
                'metrics': metrics,
                'learned_edges': [{'source': e[0], 'target': e[1]} for e in sorted(learned_edges)],
                'ground_truth_edges': [{'source': e[0], 'target': e[1]} for e in sorted(ground_truth_edges)],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        # Save TXT
        txt_path = output_path / 'llm_evaluation_results.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM OUTPUT EVALUATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"LLM CSV: {llm_csv_path}\n")
            f.write(f"Ground Truth: {ground_truth_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("METRICS\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("Edge-Level (Undirected)\n")
            f.write(f"  Precision:  {metrics['edge_precision']:.1%}\n")
            f.write(f"  Recall:     {metrics['edge_recall']:.1%}\n")
            f.write(f"  F1 Score:   {metrics['edge_f1']:.1%}\n\n")
            
            f.write("Directed Edge-Level\n")
            f.write(f"  Precision:  {metrics['directed_precision']:.1%}\n")
            f.write(f"  Recall:     {metrics['directed_recall']:.1%}\n")
            f.write(f"  F1 Score:   {metrics['directed_f1']:.1%}\n\n")
            
            f.write("Orientation Accuracy\n")
            f.write(f"  Accuracy:   {metrics['orientation_accuracy']:.1%}\n")
            f.write(f"  Correct:    {metrics['correctly_oriented']}\n")
            f.write(f"  Incorrect:  {metrics['incorrectly_oriented']}\n\n")
            
            f.write("Structural Hamming Distance\n")
            f.write(f"  Skeleton:   {metrics['skeleton_shd']}\n")
            f.write(f"  Full:       {metrics['full_shd']}\n")
            f.write(f"  Reversals:  {metrics['reversals']}\n\n")
            
            f.write("Summary\n")
            f.write(f"  Learned:    {metrics['learned_edges']}\n")
            f.write(f"  GT:         {metrics['ground_truth_edges']}\n")
        
        print(f"\n[SAVED] LLM evaluation results:")
        print(f"  JSON: {json_path}")
        print(f"  TXT:  {txt_path}")
    
    return metrics


if __name__ == "__main__":
    # Test the evaluator
    import sys
    sys.path.append('..')
    from modules.data_loader import CausalDataLoader
    from modules.prior_builder import PriorBuilder
    from modules.model import CausalDiscoveryModel
    
    # Load data
    loader = CausalDataLoader(
        data_path='data/alarm_data_10000.csv',
        metadata_path='outputs/knowledge_graph_metadata.json'
    )
    var_structure = loader.get_variable_structure()
    
    # Build priors
    prior_builder = PriorBuilder(var_structure)
    priors = prior_builder.get_all_priors(
        fci_csv_path='data/edges_Hybrid_FCI_LLM_20251207_230956.csv',
        llm_rules_path='llm_prior_rules'
    )
    
    # Initialize model
    model = CausalDiscoveryModel(
        n_states=var_structure['n_states'],
        skeleton_mask=priors['skeleton_mask'],
        direction_prior=priors['direction_prior']
    )
    
    # Initialize evaluator
    evaluator = CausalGraphEvaluator(
        ground_truth_path='../alarm.bif',
        var_structure=var_structure
    )
    
    # Extract learned edges (before training, should match LLM prior)
    adjacency = model.get_adjacency()
    learned_edges = evaluator.extract_learned_edges(adjacency, threshold=0.3)
    
    # Evaluate
    metrics = evaluator.evaluate(learned_edges)
    evaluator.print_metrics(metrics)

