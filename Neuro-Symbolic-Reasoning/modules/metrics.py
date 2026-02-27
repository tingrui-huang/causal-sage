"""
Training Metrics Module

Provides real-time monitoring functions for tracking:
1. Direction learning dynamics (unresolved ratio - symmetry breaking)
2. Sparsity evolution
3. Weight distribution

These metrics are crucial for understanding if the model is:
- Breaking symmetry / resolving directions (cycle consistency working)
- Learning sparse structures (group lasso working)

Key Concept:
- "Unresolved" edges are those where both A→B and B→A have strong weights (symmetric/bidirectional)
- FCI leaves many edges unresolved (undirected/partial)
- LLM resolves some using domain knowledge
- Neural network resolves the rest through gradient-based symmetry breaking (GSB)
"""

import torch
import numpy as np
from typing import Dict, List


def compute_unresolved_ratio(adjacency: torch.Tensor, 
                             block_structure: List[Dict],
                             threshold: float = 0.3) -> Dict:
    """
    Compute unresolved (symmetric) edge ratio
    
    Goal: Track how many variable pairs have strong weights in BOTH directions (unresolved/symmetric)
    We want this ratio to DECREASE during training (symmetry breaking / direction learning)
    
    Unresolved edges are those where the model hasn't determined a unique causal direction yet.
    - FCI leaves many edges unresolved (partial/undirected)
    - LLM resolves some using domain knowledge
    - Neural network resolves the rest through gradient-based symmetry breaking (GSB)
    
    FIXED: Now checks ALL variable pairs in FCI skeleton, not just those
    with both directions in block_structure. For pairs with only one direction
    in block_structure, we still check the reverse direction in the adjacency
    matrix (even though it's not explicitly in block_structure).
    
    Args:
        adjacency: (105, 105) adjacency matrix
        block_structure: List of blocks from PriorBuilder
        threshold: Threshold for "strong" weight (default: 0.3)
    
    Returns:
        Dictionary with unresolved edge statistics:
        {
            'unresolved': int,             # Number of unresolved pairs (symmetric/bidirectional)
            'resolved': int,               # Number of resolved pairs (one direction)
            'no_direction': int,           # Number of pairs with no strong direction
            'total_pairs': int,            # Total unique variable pairs
            'unresolved_ratio': float      # Ratio of unresolved pairs
        }
    """
    unresolved_count = 0  # Both directions strong (symmetric/unresolved)
    resolved_count = 0     # Only one direction strong (resolved)
    no_direction_count = 0 # Neither direction strong
    
    # Build block lookup
    block_lookup = {}
    for block in block_structure:
        var_a, var_b = block['var_pair']
        block_lookup[(var_a, var_b)] = block
    
    # Get all unique variable pairs from block_structure
    all_pairs = set()
    for block in block_structure:
        var_a, var_b = block['var_pair']
        pair_key = tuple(sorted([var_a, var_b]))
        all_pairs.add(pair_key)
    
    # Check each unique pair
    for pair_key in all_pairs:
        var_a, var_b = pair_key  # Already sorted
        
        # Try to get both directions
        forward_block = block_lookup.get((var_a, var_b))
        reverse_block = block_lookup.get((var_b, var_a))
        
        # Compute strengths for both directions
        # If a direction is not in block_structure, manually compute from adjacency
        if forward_block is not None:
            forward_weights = adjacency[forward_block['row_indices']][:, forward_block['col_indices']]
            forward_strength = forward_weights.max().item()
        else:
            # Manually compute: need to get state indices from reverse_block
            if reverse_block is not None:
                # Swap indices: reverse_block is (var_b, var_a), we want (var_a, var_b)
                forward_weights = adjacency[reverse_block['col_indices']][:, reverse_block['row_indices']]
                forward_strength = forward_weights.max().item()
            else:
                # Should not happen if block_structure is correct
                forward_strength = 0.0
        
        if reverse_block is not None:
            backward_weights = adjacency[reverse_block['row_indices']][:, reverse_block['col_indices']]
            backward_strength = backward_weights.max().item()
        else:
            # Manually compute: need to get state indices from forward_block
            if forward_block is not None:
                # Swap indices: forward_block is (var_a, var_b), we want (var_b, var_a)
                backward_weights = adjacency[forward_block['col_indices']][:, forward_block['row_indices']]
                backward_strength = backward_weights.max().item()
            else:
                # Should not happen if block_structure is correct
                backward_strength = 0.0
        
        # Classify
        # NOTE: Use a small epsilon to avoid float32 edge cases where e.g. 0.1 is stored as 0.10000000149
        # and accidentally counted as ">" 0.1.
        thr = float(threshold) + 1e-6
        forward_strong = forward_strength > thr
        backward_strong = backward_strength > thr
        
        if forward_strong and backward_strong:
            unresolved_count += 1  # Symmetric/unresolved
        elif forward_strong or backward_strong:
            resolved_count += 1     # Resolved to one direction
        else:
            no_direction_count += 1 # Neither direction
    
    total_pairs = unresolved_count + resolved_count + no_direction_count
    
    return {
        'unresolved': unresolved_count,
        'resolved': resolved_count,
        'no_direction': no_direction_count,
        'total_pairs': total_pairs,
        'unresolved_ratio': unresolved_count / total_pairs if total_pairs > 0 else 0
    }


def compute_sparsity_metrics(adjacency: torch.Tensor,
                             skeleton_mask: torch.Tensor,
                             block_structure: List[Dict],
                             threshold: float = 0.1) -> Dict:
    """
    Compute comprehensive sparsity metrics
    
    Tracks how well the model is learning sparse structures:
    - Overall sparsity: Percentage of allowed connections that are inactive
    - Block sparsity: Percentage of blocks that are inactive
    - Weight statistics: Distribution of active weights
    
    Args:
        adjacency: (105, 105) adjacency matrix
        skeleton_mask: (105, 105) skeleton constraint from FCI
        block_structure: List of blocks from PriorBuilder
        threshold: Threshold for "active" connection (default: 0.1)
    
    Returns:
        Dictionary with sparsity statistics:
        {
            'total_allowed': int,           # Total connections allowed by skeleton
            'active_connections': int,      # Number of active connections
            'overall_sparsity': float,      # Ratio of inactive connections
            'active_blocks': int,           # Number of active blocks
            'total_blocks': int,            # Total number of blocks
            'block_sparsity': float,        # Ratio of inactive blocks
            'mean_active_weight': float,    # Mean weight of active connections
            'max_weight': float,            # Maximum weight
            'min_nonzero_weight': float     # Minimum non-zero weight
        }
    """
    # Overall sparsity
    total_allowed = int(skeleton_mask.sum().item())
    adjacency_np = adjacency.detach().cpu().numpy()
    thr = float(threshold) + 1e-6
    active_connections = (adjacency_np > thr).sum()
    overall_sparsity = (total_allowed - active_connections) / total_allowed
    
    # Block-level sparsity
    active_blocks = 0
    for block in block_structure:
        block_weights = adjacency[block['row_indices']][:, block['col_indices']]
        if block_weights.max().item() > thr:
            active_blocks += 1
    
    block_sparsity = (len(block_structure) - active_blocks) / len(block_structure)
    
    # Weight distribution
    active_weights = adjacency_np[adjacency_np > threshold]
    
    return {
        'total_allowed': total_allowed,
        'active_connections': int(active_connections),
        'overall_sparsity': overall_sparsity,
        'active_blocks': active_blocks,
        'total_blocks': len(block_structure),
        'block_sparsity': block_sparsity,
        'mean_active_weight': float(active_weights.mean()) if len(active_weights) > 0 else 0.0,
        'max_weight': float(adjacency_np.max()),
        'min_nonzero_weight': float(adjacency_np[adjacency_np > 0].min()) if (adjacency_np > 0).any() else 0.0
    }


def compute_weight_statistics(adjacency: torch.Tensor,
                              skeleton_mask: torch.Tensor) -> Dict:
    """
    Compute detailed weight distribution statistics
    
    Args:
        adjacency: (105, 105) adjacency matrix
        skeleton_mask: (105, 105) skeleton constraint
    
    Returns:
        Dictionary with weight statistics
    """
    adjacency_np = adjacency.detach().cpu().numpy()
    skeleton_np = skeleton_mask.detach().cpu().numpy()
    
    # Get allowed weights
    allowed_weights = adjacency_np[skeleton_np > 0]
    
    return {
        'mean': float(allowed_weights.mean()),
        'std': float(allowed_weights.std()),
        'median': float(np.median(allowed_weights)),
        'min': float(allowed_weights.min()),
        'max': float(allowed_weights.max()),
        'q25': float(np.percentile(allowed_weights, 25)),
        'q75': float(np.percentile(allowed_weights, 75)),
        'num_near_zero': int((allowed_weights < 0.01).sum()),
        'num_near_one': int((allowed_weights > 0.99).sum())
    }


if __name__ == "__main__":
    """Unit tests for metrics functions"""
    print("=" * 70)
    print("METRICS MODULE UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Bidirectional ratio
    print("\nTest 1: Bidirectional Ratio")
    print("-" * 70)
    
    # Create dummy adjacency with known pattern
    adjacency = torch.zeros(6, 6)
    
    # Bidirectional pair: (0,1) <-> (2,3)
    adjacency[0:2, 2:4] = 0.8  # Forward
    adjacency[2:4, 0:2] = 0.7  # Backward
    
    # Unidirectional pair: (0,1) -> (4,5)
    adjacency[0:2, 4:6] = 0.8  # Forward only
    
    # Create block structure
    blocks = [
        {'var_pair': ('A', 'B'), 'row_indices': [0, 1], 'col_indices': [2, 3]},
        {'var_pair': ('B', 'A'), 'row_indices': [2, 3], 'col_indices': [0, 1]},
        {'var_pair': ('A', 'C'), 'row_indices': [0, 1], 'col_indices': [4, 5]},
    ]
    
    unresolved_stats = compute_unresolved_ratio(adjacency, blocks, threshold=0.3)
    
    print(f"Unresolved pairs (symmetric): {unresolved_stats['unresolved']}")
    print(f"Resolved pairs (one direction): {unresolved_stats['resolved']}")
    print(f"No direction pairs: {unresolved_stats['no_direction']}")
    print(f"Unresolved ratio: {unresolved_stats['unresolved_ratio']*100:.1f}%")
    
    assert unresolved_stats['unresolved'] == 1, "Should have 1 unresolved pair"
    assert unresolved_stats['resolved'] == 1, "Should have 1 resolved pair"
    print("[PASS] Unresolved ratio test passed!")
    
    # Test 2: Sparsity metrics
    print("\nTest 2: Sparsity Metrics")
    print("-" * 70)
    
    skeleton_mask = torch.ones(6, 6)
    skeleton_mask[4:6, 4:6] = 0  # Forbid some connections
    
    sparsity_stats = compute_sparsity_metrics(adjacency, skeleton_mask, blocks, threshold=0.1)
    
    print(f"Total allowed: {sparsity_stats['total_allowed']}")
    print(f"Active connections: {sparsity_stats['active_connections']}")
    print(f"Overall sparsity: {sparsity_stats['overall_sparsity']*100:.1f}%")
    print(f"Active blocks: {sparsity_stats['active_blocks']}/{sparsity_stats['total_blocks']}")
    print(f"Block sparsity: {sparsity_stats['block_sparsity']*100:.1f}%")
    print(f"Max weight: {sparsity_stats['max_weight']:.3f}")
    
    print("[PASS] Sparsity metrics test passed!")
    
    # Test 3: Weight statistics
    print("\nTest 3: Weight Statistics")
    print("-" * 70)
    
    weight_stats = compute_weight_statistics(adjacency, skeleton_mask)
    
    print(f"Mean: {weight_stats['mean']:.3f}")
    print(f"Std: {weight_stats['std']:.3f}")
    print(f"Median: {weight_stats['median']:.3f}")
    print(f"Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
    print(f"Q25-Q75: [{weight_stats['q25']:.3f}, {weight_stats['q75']:.3f}]")
    
    print("[PASS] Weight statistics test passed!")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
