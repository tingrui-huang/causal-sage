"""
Ground Truth Loader - Extensible for Multiple Formats

Supports:
- BIF format (ALARM, Insurance, etc.)
- JSON format (Tuebingen pairs)
- Edge list format (Sachs, etc.)
- Future formats can be easily added
"""

import json
import re
from pathlib import Path
from typing import Set, Tuple, Dict, Optional


class GroundTruthLoader:
    """
    Base class for loading ground truth in different formats
    
    This design allows easy extension for new datasets without
    modifying existing code.
    """
    
    def __init__(self, ground_truth_path: str, ground_truth_type: str = 'bif'):
        """
        Args:
            ground_truth_path: Path to ground truth file
            ground_truth_type: Type of ground truth ('bif', 'json', 'edge_list')
        """
        self.path = Path(ground_truth_path) if ground_truth_path else None
        self.type = ground_truth_type
        self.edges = None
        
        if self.path and self.path.exists():
            self.edges = self._load()
    
    def _load(self) -> Set[Tuple[str, str]]:
        """Load ground truth based on type"""
        if self.type == 'bif':
            return self._load_bif()
        elif self.type == 'json':
            return self._load_json()
        elif self.type == 'edge_list':
            return self._load_edge_list()
        else:
            raise ValueError(f"Unsupported ground truth type: {self.type}")
    
    def _load_bif(self) -> Set[Tuple[str, str]]:
        """
        Load BIF format (Bayesian Network)
        
        Format:
            probability ( CHILD | PARENT1, PARENT2, ... )
        
        Returns:
            Set of directed edges (parent, child)
        """
        edges = set()
        
        with open(self.path, 'r') as f:
            content = f.read()
        
        # Extract probability declarations
        prob_pattern = r'probability\s*\(\s*(\w+)\s*\|\s*([^)]+)\s*\)'
        
        for match in re.finditer(prob_pattern, content):
            child = match.group(1)
            parents_str = match.group(2)
            parents = [p.strip() for p in parents_str.split(',')]
            
            for parent in parents:
                if parent:
                    edges.add((parent, child))
        
        return edges
    
    def _load_json(self) -> Dict[str, str]:
        """
        Load JSON format (Tuebingen pairs)
        
        Format:
            {
                "pair0001": "-->",  // column 1 -> column 2
                "pair0002": "<--",  // column 2 -> column 1
                ...
            }
        
        Returns:
            Dictionary mapping pair_id to direction
        """
        with open(self.path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _load_edge_list(self) -> Set[Tuple[str, str]]:
        """
        Load edge list format (Sachs, etc.)
        
        Format:
            # Comments start with #
            source1 -> target1
            source2 -> target2
            ...
        
        Returns:
            Set of directed edges (source, target)
        """
        edges = set()
        
        with open(self.path, 'r', encoding='utf-8') as f:
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
                            edges.add((source, target))
        
        return edges
    
    def get_edges(self) -> Optional[Set[Tuple[str, str]]]:
        """Get ground truth edges (for graph-based evaluation)"""
        if self.type in ['bif', 'edge_list']:
            return self.edges
        else:
            # For non-graph formats (e.g., JSON for pairwise), return None
            return None
    
    def get_direction(self, pair_id: str) -> Optional[str]:
        """
        Get ground truth direction for a specific pair (for pairwise evaluation)
        
        Args:
            pair_id: Identifier for the pair (e.g., 'pair0001')
        
        Returns:
            Direction string ('-->', '<--', or None)
        """
        if self.type == 'json' and self.edges:
            return self.edges.get(pair_id)
        return None
    
    def is_available(self) -> bool:
        """Check if ground truth is available"""
        return self.edges is not None
    
    def get_type(self) -> str:
        """Get ground truth type"""
        return self.type


class TuebingenEvaluator:
    """
    Specialized evaluator for Tuebingen pairs
    
    Evaluates pairwise causal direction (A->B or B->A)
    instead of full graph metrics.
    """
    
    def __init__(self, ground_truth: GroundTruthLoader, var_structure: Dict):
        """
        Args:
            ground_truth: GroundTruthLoader instance
            var_structure: Variable structure from DataLoader
        """
        self.ground_truth = ground_truth
        self.var_structure = var_structure
    
    def evaluate_pairwise(self, adjacency, pair_id: str = None) -> Dict:
        """
        Evaluate pairwise causal direction
        
        Args:
            adjacency: (n_states, n_states) adjacency matrix
            pair_id: Identifier for the pair (optional, for ground truth lookup)
        
        Returns:
            Dictionary with evaluation results
        """
        # Get variable names (assume 2 variables for Tuebingen)
        var_names = list(self.var_structure['var_to_states'].keys())
        
        if len(var_names) != 2:
            raise ValueError(f"Tuebingen evaluation expects 2 variables, got {len(var_names)}")
        
        var_a, var_b = var_names
        
        # Get state indices for each variable
        indices_a = self.var_structure['var_to_states'][var_a]
        indices_b = self.var_structure['var_to_states'][var_b]
        
        # Extract block strengths
        # A->B: rows from A, columns from B
        block_a_to_b = adjacency[indices_a][:, indices_b]
        score_forward = block_a_to_b.mean().item()
        
        # B->A: rows from B, columns from A
        block_b_to_a = adjacency[indices_b][:, indices_a]
        score_backward = block_b_to_a.mean().item()
        
        # Determine predicted direction
        if score_forward > score_backward:
            predicted_direction = '-->'
            confidence = score_forward - score_backward
        else:
            predicted_direction = '<--'
            confidence = score_backward - score_forward
        
        # Get ground truth if available
        gt_direction = None
        correct = None
        
        if pair_id and self.ground_truth.is_available():
            gt_direction = self.ground_truth.get_direction(pair_id)
            if gt_direction:
                correct = (predicted_direction == gt_direction)
        
        return {
            'var_a': var_a,
            'var_b': var_b,
            'score_forward': score_forward,
            'score_backward': score_backward,
            'predicted_direction': predicted_direction,
            'confidence': confidence,
            'ground_truth_direction': gt_direction,
            'correct': correct
        }
    
    def print_results(self, results: Dict):
        """Print evaluation results in a readable format"""
        print("\n" + "=" * 80)
        print("TUEBINGEN PAIRWISE EVALUATION")
        print("=" * 80)
        
        print(f"\nVariables: {results['var_a']} and {results['var_b']}")
        print(f"\nScores:")
        print(f"  {results['var_a']} -> {results['var_b']}: {results['score_forward']:.4f}")
        print(f"  {results['var_b']} -> {results['var_a']}: {results['score_backward']:.4f}")
        
        print(f"\nPredicted Direction: {results['var_a']} {results['predicted_direction']} {results['var_b']}")
        print(f"Confidence: {results['confidence']:.4f}")
        
        if results['ground_truth_direction']:
            print(f"\nGround Truth: {results['var_a']} {results['ground_truth_direction']} {results['var_b']}")
            status = "✓ CORRECT" if results['correct'] else "✗ INCORRECT"
            print(f"Result: {status}")
        else:
            print("\n[INFO] No ground truth available for comparison")
        
        print("=" * 80)


if __name__ == "__main__":
    """Unit tests"""
    print("=" * 80)
    print("GROUND TRUTH LOADER TESTS")
    print("=" * 80)
    
    # Test 1: BIF format (if alarm.bif exists)
    print("\nTest 1: BIF Format")
    print("-" * 80)
    
    bif_path = Path("../data/alarm/alarm.bif")
    if bif_path.exists():
        loader = GroundTruthLoader(str(bif_path), 'bif')
        edges = loader.get_edges()
        print(f"[OK] Loaded {len(edges)} edges from BIF")
        print(f"  Sample edges: {list(edges)[:3]}")
    else:
        print(f"[SKIP] BIF file not found: {bif_path}")
    
    # Test 2: JSON format (example)
    print("\nTest 2: JSON Format")
    print("-" * 80)
    
    # Create example JSON
    example_json = {
        "pair0001": "-->",
        "pair0002": "<--",
        "pair0003": "-->"
    }
    
    json_path = Path("test_ground_truth.json")
    with open(json_path, 'w') as f:
        json.dump(example_json, f)
    
    loader = GroundTruthLoader(str(json_path), 'json')
    print(f"[OK] Loaded JSON ground truth")
    print(f"  pair0001 direction: {loader.get_direction('pair0001')}")
    print(f"  pair0002 direction: {loader.get_direction('pair0002')}")
    
    # Cleanup
    json_path.unlink()
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)

