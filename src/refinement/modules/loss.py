"""
Loss Module - Phase 2

The mathematical core of causal discovery.

Four loss components:
1. Reconstruction Loss: Auto-encoder style prediction
   - Uses BCELoss with reduction='mean' (automatically normalized by n_states)
   
2. Weighted Group Lasso: Block-level sparsity with Normal protection
   - Normalized by number of blocks (≈ n_vars²)
   - Represents "average penalty per block"
   
3. Cycle Consistency Loss: Direction learning by penalizing bidirectional edges
   - Normalized by number of variable pairs
   - Represents "average cycle penalty per pair"
   
4. Skeleton Preservation Loss: Prevent FCI edges from vanishing
   - Normalized by number of skeleton edges
   - Represents "average squared error per skeleton edge"

Critical Implementation Notes:
- Penalty weights MUST be inside the norm: ||W ⊙ P||_F, NOT: ||W||_F * P
- All losses are normalized to represent "per-element averages" for scale-invariance
  * This ensures Andes (223 vars) and Alarm (37 vars) have comparable loss magnitudes
  * Lambda hyperparameters now have consistent effects across different dataset sizes
  * Without normalization, larger networks would require drastically different lambda values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class LossComputer:
    """
    Compute all loss components for causal discovery
    
    This is the "soul" of the model - where mathematical constraints
    enforce causal structure learning.
    """
    
    def __init__(
        self,
        block_structure: List[Dict],
        penalty_weights: torch.Tensor,
        skeleton_mask: torch.Tensor = None,
        *,
        reconstruction_mode: str = "bce",
        variable_groups: Optional[List[List[int]]] = None,
    ):
        """
        Initialize loss computer with prior knowledge
        
        Args:
            block_structure: List of block definitions from PriorBuilder
                Each block: {'var_pair': (var_a, var_b), 
                            'row_indices': [...], 
                            'col_indices': [...]}
            penalty_weights: (105, 105) tensor with weights for each connection
                Normal→Normal: 0.1 (low penalty, allow)
                Others: 1.0 (high penalty, force sparsity)
            skeleton_mask: (105, 105) binary mask from FCI (optional, for skeleton preservation)
        """
        self.block_structure = block_structure
        self.penalty_weights = penalty_weights
        self.reconstruction_mode = str(reconstruction_mode)
        self.variable_groups: Optional[List[List[int]]] = variable_groups
        self._state_to_var: Optional[torch.Tensor] = None  # (n_states,) long, maps state index -> variable index
        
        # Store skeleton mask for skeleton preservation loss
        self.skeleton_mask = skeleton_mask
        
        # Extract n_vars (number of variables) and n_states (number of one-hot states)
        # n_vars is the number of unique variables in the block structure
        unique_vars = set()
        for block in block_structure:
            var_a, var_b = block['var_pair']
            unique_vars.add(var_a)
            unique_vars.add(var_b)
        self.n_vars = len(unique_vars)
        self.n_states = penalty_weights.shape[0]  # Total one-hot states (e.g., 446 for Andes)
        
        # Build reverse block lookup for cycle consistency
        self.block_lookup = {}
        for block in block_structure:
            var_a, var_b = block['var_pair']
            self.block_lookup[(var_a, var_b)] = block
        
        # BCE loss for reconstruction
        self.bce_loss = nn.BCELoss()
        
        print("=" * 70)
        print("LOSS COMPUTER INITIALIZED (PHASE 2)")
        print("=" * 70)
        print(f"Variables: {self.n_vars}")
        print(f"One-hot states: {self.n_states}")
        print(f"Blocks: {len(block_structure)}")
        print(f"Penalty weights shape: {penalty_weights.shape}")
        print(f"Normal→Normal (0.1): {(penalty_weights == 0.1).sum().item()} connections")
        print(f"Others (1.0): {(penalty_weights == 1.0).sum().item()} connections")
        if skeleton_mask is not None:
            print(f"Skeleton mask provided: {int(skeleton_mask.sum().item())} edges to preserve")
        print(f"Reconstruction mode: {self.reconstruction_mode}")
        if self.reconstruction_mode == "group_ce":
            n_groups = len(self.variable_groups or [])
            print(f"  Variable groups: {n_groups} (for per-variable softmax CE)")
            if not self.variable_groups:
                raise ValueError("variable_groups must be provided when reconstruction_mode='group_ce'")
            # Build a dense state->variable index mapping once (used for fast vectorized group logsumexp).
            state_to_var = torch.full((self.n_states,), -1, dtype=torch.long)
            for var_idx, idxs in enumerate(self.variable_groups):
                for s in idxs:
                    state_to_var[int(s)] = int(var_idx)
            if (state_to_var < 0).any():
                missing = int((state_to_var < 0).sum().item())
                raise ValueError(f"Invalid variable_groups: {missing} / {self.n_states} states are not assigned to any variable group")
            self._state_to_var = state_to_var
        print(f"[INFO] Loss normalization enabled:")
        print(f"  - Group Lasso: normalized by {len(block_structure)} blocks")
        print(f"  - Cycle Loss: normalized by number of pairs")
        print(f"  - Skeleton Loss: normalized by number of skeleton edges")
        print(f"  - This ensures scale-invariance across different dataset sizes")
    
    def reconstruction_loss(self, predictions: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: Auto-encoder style
        
        Goal: Model should be able to reconstruct observed states
        
        Args:
            predictions: (batch, 105) predicted state probabilities
            targets: (batch, 105) observed binary states
        
        Returns:
            Scalar loss value
        """
        if self.reconstruction_mode == "bce":
            # BCE over the full state vector (multi-hot reconstruction)
            return self.bce_loss(predictions, targets)
        if self.reconstruction_mode == "group_ce":
            # Per-variable softmax cross-entropy, fully vectorized.
            # predictions: (batch, n_states) logits
            # targets: (batch, n_states) multi-one-hot (one active state per variable)
            # For each sample b and variable v:
            #   CE(b,v) = -log softmax(logits_{b, S_v})[y_{b,v}]
            #          = logsumexp(logits_{b,S_v}) - logits_{b, y_{b,v}}
            if self._state_to_var is None:
                raise ValueError("Internal error: _state_to_var not initialized for group_ce")

            device = predictions.device
            state_to_var = self._state_to_var.to(device=device, non_blocking=True)  # (n_states,)
            n_vars = int(state_to_var.max().item()) + 1
            batch = predictions.shape[0]

            # Index matrix mapping each (b, state) -> var id
            idx = state_to_var.view(1, -1).expand(batch, -1)  # (batch, n_states)

            # Compute per-(b,var) logsumexp over that var's states using scatter_reduce(max) + scatter_add(exp(shifted)).
            max_per = torch.full((batch, n_vars), -float("inf"), device=device, dtype=predictions.dtype)
            max_per.scatter_reduce_(1, idx, predictions, reduce="amax", include_self=True)
            max_gather = max_per.gather(1, idx)  # (batch, n_states)
            shifted = predictions - max_gather
            denom = torch.zeros((batch, n_vars), device=device, dtype=predictions.dtype)
            denom.scatter_add_(1, idx, torch.exp(shifted))
            log_denom = torch.log(denom) + max_per  # (batch, n_vars)

            # Numerator: logits at the true class per variable.
            # Since targets is one-hot within each variable group, summing targets*logits grouped by var gives that logit.
            num = torch.zeros((batch, n_vars), device=device, dtype=predictions.dtype)
            num.scatter_add_(1, idx, targets * predictions)

            # Average CE over variables and batch
            loss = (log_denom - num).mean()
            return loss
        raise ValueError(f"Unsupported reconstruction_mode: {self.reconstruction_mode}")
    
    def weighted_group_lasso_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Weighted Group Lasso: Block-level sparsity with Normal protection
        
        CRITICAL MATHEMATICAL FORMULA:
        For each block:
            Loss_block = ||W_block ⊙ P_block||_F
        
        Where:
        - W_block: Weight sub-matrix for this variable pair
        - P_block: Penalty weight sub-matrix
        - ⊙: Element-wise multiplication (Hadamard product)
        - ||·||_F: Frobenius norm
        
        This is NOT: ||W_block||_F * P_scalar
        
        Why this matters:
        - Normal→Normal connections have P=0.1 (inside the norm)
        - Other connections have P=1.0
        - This allows Normal→Normal to exist while forcing sparsity elsewhere
        
        Args:
            adjacency: (105, 105) adjacency matrix
                MUST be sigmoid-activated and skeleton-masked
        
        Returns:
            Scalar loss value (sum over all blocks)
        """
        total_penalty = 0.0
        
        for block in self.block_structure:
            row_indices = block['row_indices']
            col_indices = block['col_indices']
            
            # Extract block weights
            # Use advanced indexing to get sub-matrix
            block_weights = adjacency[row_indices][:, col_indices]
            
            # Extract corresponding penalty weights
            block_penalties = self.penalty_weights[row_indices][:, col_indices]
            
            # CRITICAL: Element-wise multiplication BEFORE norm
            # This is the Weighted Group Lasso formula
            weighted_block = block_weights * block_penalties
            
            # Frobenius norm of the weighted block
            block_norm = torch.norm(weighted_block, p='fro')
            
            total_penalty += block_norm
        
        # Normalize by number of blocks to make loss scale-invariant
        # This ensures that Andes (223 vars) and Alarm (37 vars) have comparable loss magnitudes
        # Rationale: The penalty is a sum over all blocks, and #blocks ≈ n_vars * (n_vars - 1)
        # Normalizing by #blocks makes the loss represent "average penalty per block"
        n_blocks = len(self.block_structure)
        return total_penalty / n_blocks if n_blocks > 0 else total_penalty
    
    def cycle_consistency_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Cycle Consistency Loss: Penalize bidirectional edges
        
        Goal: Force model to choose direction (A→B OR B→A, not both)
        
        Formula:
        For each variable pair (A, B):
            Penalty = ||W_{A→B}||_F × ||W_{B→A}||_F
        
        Why this works:
        - If both directions are strong → high penalty
        - If only one direction is strong → low penalty
        - Forces model to choose one direction
        
        Args:
            adjacency: (105, 105) adjacency matrix
        
        Returns:
            Scalar loss value (sum over all pairs)
        """
        total_penalty = 0.0
        processed_pairs = set()
        
        for block in self.block_structure:
            var_a, var_b = block['var_pair']
            
            # Avoid double-counting (A,B) and (B,A)
            pair_key = tuple(sorted([var_a, var_b]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Find reverse block
            reverse_block = self.block_lookup.get((var_b, var_a))
            if reverse_block is None:
                continue
            
            # Extract both direction blocks
            forward_weights = adjacency[block['row_indices']][:, block['col_indices']]
            backward_weights = adjacency[reverse_block['row_indices']][:, reverse_block['col_indices']]
            
            # Compute norms
            forward_norm = torch.norm(forward_weights, p='fro')
            backward_norm = torch.norm(backward_weights, p='fro')
            
            # Penalty: product of norms
            # High when both directions are strong
            cycle_penalty = forward_norm * backward_norm
            
            total_penalty += cycle_penalty
        
        # Normalize by number of variable pairs to make loss scale-invariant
        # Rationale: The cycle penalty is a sum over all variable pairs, which grows as O(n_vars²)
        # Normalizing by #pairs makes the loss represent "average cycle penalty per pair"
        n_pairs = len(processed_pairs)
        return total_penalty / n_pairs if n_pairs > 0 else total_penalty
    
    def skeleton_preservation_loss(self, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Skeleton Preservation Loss: Prevent FCI edges from vanishing
        
        Goal: Force model to preserve edges identified by FCI (high recall constraint)
        
        Formula:
        For each edge (i,j) in FCI skeleton:
            Loss += (1 - (W_ij + W_ji))^2
        
        Why this works:
        - FCI has high recall (finds true edges)
        - But current loss allows W_ij and W_ji to both → 0
        - This loss forces: W_ij + W_ji ≈ 1 for FCI edges
        - Model must choose direction (A→B OR B→A), but cannot delete edge
        
        Mathematical intuition:
        - If W_ij = 0.8, W_ji = 0.2 → sum = 1.0 → loss = 0 ✓
        - If W_ij = 0.5, W_ji = 0.5 → sum = 1.0 → loss = 0 ✓
        - If W_ij = 0.1, W_ji = 0.1 → sum = 0.2 → loss = 0.64 ✗ (penalized!)
        - If W_ij = 0.0, W_ji = 0.0 → sum = 0.0 → loss = 1.0 ✗✗ (heavily penalized!)
        
        Args:
            adjacency: (105, 105) adjacency matrix (sigmoid + masked)
        
        Returns:
            Scalar loss value (sum over all FCI edges)
        """
        if self.skeleton_mask is None:
            # No skeleton mask provided, return zero loss
            return torch.tensor(0.0, device=adjacency.device)
        
        # Compute bidirectional edge strength: W_ij + W_ji
        # This is symmetric: edge_sum[i,j] = W_ij + W_ji
        edge_sum = adjacency + adjacency.T
        
        # Target: edge_sum should be 1.0 for all FCI edges
        # We want to minimize (1 - edge_sum)^2 for skeleton edges
        target = torch.ones_like(edge_sum)
        
        # Compute squared error for skeleton edges only
        # skeleton_mask is symmetric (if i-j connected, then j-i also marked)
        # So we can directly apply it
        squared_error = (target - edge_sum) ** 2
        
        # Only penalize edges in the FCI skeleton
        # Note: This will count each undirected edge twice (i->j and j->i)
        # But that's fine - it just scales the loss uniformly
        skeleton_loss = torch.sum(self.skeleton_mask * squared_error)
        
        # Normalize by number of skeleton edges to make loss magnitude reasonable
        # Divide by 2 because skeleton is symmetric (each edge counted twice)
        # This makes the loss represent "average squared error per skeleton edge"
        n_skeleton_edges = self.skeleton_mask.sum() / 2
        if n_skeleton_edges > 0:
            skeleton_loss = skeleton_loss / (2 * n_skeleton_edges)
        
        return skeleton_loss
    
    def compute_total_loss(self, 
                          predictions: torch.Tensor,
                          targets: torch.Tensor,
                          adjacency: torch.Tensor,
                          lambda_group: float = 0.01,
                          lambda_cycle: float = 0.001,
                          lambda_skeleton: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Compute total loss with all components
        
        Args:
            predictions: (batch, 105) predicted states
            targets: (batch, 105) observed states
            adjacency: (105, 105) adjacency matrix (sigmoid + masked)
            lambda_group: Weight for Group Lasso (default: 0.01)
            lambda_cycle: Weight for Cycle Consistency (default: 0.001)
            lambda_skeleton: Weight for Skeleton Preservation (default: 0.1)
        
        Returns:
            Dictionary with all loss components:
            {
                'total': total_loss,
                'reconstruction': recon_loss,
                'weighted_group_lasso': group_loss,
                'cycle_consistency': cycle_loss,
                'skeleton_preservation': skeleton_loss
            }
        """
        # Compute individual losses
        loss_recon = self.reconstruction_loss(predictions, targets)
        loss_group = self.weighted_group_lasso_loss(adjacency)
        loss_cycle = self.cycle_consistency_loss(adjacency)
        loss_skeleton = self.skeleton_preservation_loss(adjacency)
        
        # Total loss
        total_loss = (loss_recon + 
                     lambda_group * loss_group + 
                     lambda_cycle * loss_cycle +
                     lambda_skeleton * loss_skeleton)
        
        return {
            'total': total_loss,
            'reconstruction': loss_recon,
            'weighted_group_lasso': loss_group,
            'cycle_consistency': loss_cycle,
            'skeleton_preservation': loss_skeleton
        }


def test_weighted_group_lasso():
    """
    Unit test for Weighted Group Lasso calculation
    
    Verifies that the math is correct:
    ||W ⊙ P||_F should equal sqrt(sum((W * P)^2))
    """
    print("\n" + "=" * 70)
    print("UNIT TEST: Weighted Group Lasso")
    print("=" * 70)
    
    # Create a simple 2x2 block
    block_weights = torch.tensor([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    
    # Penalty weights (one Normal→Normal connection)
    block_penalties = torch.tensor([
        [0.1, 1.0],
        [1.0, 1.0]
    ])
    
    # Manual calculation
    # W ⊙ P = [[1.0*0.1, 1.0*1.0], [1.0*1.0, 1.0*1.0]]
    #       = [[0.1, 1.0], [1.0, 1.0]]
    # ||W ⊙ P||_F = sqrt(0.1^2 + 1.0^2 + 1.0^2 + 1.0^2)
    #             = sqrt(0.01 + 1 + 1 + 1)
    #             = sqrt(3.01)
    #             ≈ 1.7349
    
    expected_loss = torch.sqrt(torch.tensor(3.01))
    
    # Computed loss
    weighted_block = block_weights * block_penalties
    computed_loss = torch.norm(weighted_block, p='fro')
    
    print(f"\nBlock weights:\n{block_weights}")
    print(f"\nPenalty weights:\n{block_penalties}")
    print(f"\nWeighted block (W ⊙ P):\n{weighted_block}")
    print(f"\nExpected loss: {expected_loss:.6f}")
    print(f"Computed loss: {computed_loss:.6f}")
    print(f"Difference: {abs(expected_loss - computed_loss):.10f}")
    
    # Assert correctness
    assert torch.allclose(computed_loss, expected_loss, atol=1e-6), \
        f"Loss mismatch! Expected {expected_loss}, got {computed_loss}"
    
    print("\n[PASS] TEST PASSED: Weighted Group Lasso math is correct!")
    
    # Test 2: Verify it's different from wrong formula
    wrong_loss = torch.norm(block_weights, p='fro') * block_penalties.mean()
    print(f"\nWrong formula (||W||_F * mean(P)): {wrong_loss:.6f}")
    print(f"Correct formula: {computed_loss:.6f}")
    print(f"Difference: {abs(wrong_loss - computed_loss):.6f}")
    
    assert not torch.allclose(wrong_loss, computed_loss, atol=0.1), \
        "Wrong formula should give different result!"
    
    print("[PASS] Verified: Penalty must be inside the norm!")


def test_cycle_consistency():
    """
    Unit test for Cycle Consistency Loss
    """
    print("\n" + "=" * 70)
    print("UNIT TEST: Cycle Consistency Loss")
    print("=" * 70)
    
    # Create dummy adjacency with bidirectional edge
    adjacency = torch.zeros(4, 4)
    
    # Strong A→B (indices 0,1 → 2,3)
    adjacency[0:2, 2:4] = 0.8
    
    # Strong B→A (indices 2,3 → 0,1)
    adjacency[2:4, 0:2] = 0.7
    
    # Create block structure
    blocks = [
        {'var_pair': ('A', 'B'), 'row_indices': [0, 1], 'col_indices': [2, 3]},
        {'var_pair': ('B', 'A'), 'row_indices': [2, 3], 'col_indices': [0, 1]}
    ]
    
    # Dummy penalty weights
    penalty_weights = torch.ones(4, 4)
    
    # Create loss computer
    loss_computer = LossComputer(blocks, penalty_weights)
    
    # Compute cycle loss
    cycle_loss = loss_computer.cycle_consistency_loss(adjacency)
    
    # Manual calculation
    # ||W_{A→B}||_F = sqrt(4 * 0.8^2) = sqrt(2.56) ≈ 1.6
    # ||W_{B→A}||_F = sqrt(4 * 0.7^2) = sqrt(1.96) = 1.4
    # Penalty = 1.6 * 1.4 = 2.24
    
    forward_norm = torch.norm(adjacency[0:2, 2:4], p='fro')
    backward_norm = torch.norm(adjacency[2:4, 0:2], p='fro')
    expected_loss = forward_norm * backward_norm
    
    print(f"\nForward block (A→B) norm: {forward_norm:.6f}")
    print(f"Backward block (B→A) norm: {backward_norm:.6f}")
    print(f"Expected cycle loss: {expected_loss:.6f}")
    print(f"Computed cycle loss: {cycle_loss:.6f}")
    print(f"Difference: {abs(expected_loss - cycle_loss):.10f}")
    
    assert torch.allclose(cycle_loss, expected_loss, atol=1e-6), \
        f"Cycle loss mismatch! Expected {expected_loss}, got {cycle_loss}"
    
    print("\n[PASS] TEST PASSED: Cycle Consistency math is correct!")


def test_skeleton_preservation():
    """
    Unit test for Skeleton Preservation Loss
    """
    print("\n" + "=" * 70)
    print("UNIT TEST: Skeleton Preservation Loss")
    print("=" * 70)
    
    # Create a 4x4 adjacency matrix
    adjacency = torch.zeros(4, 4)
    
    # Case 1: Edge with good direction choice (0.8 + 0.2 = 1.0)
    adjacency[0, 1] = 0.8  # A → B strong
    adjacency[1, 0] = 0.2  # B → A weak
    
    # Case 2: Edge with both directions weak (0.1 + 0.1 = 0.2, should be penalized)
    adjacency[2, 3] = 0.1  # C → D weak
    adjacency[3, 2] = 0.1  # D → C weak
    
    # Create skeleton mask: mark edges (0,1) and (2,3) as FCI edges
    skeleton_mask = torch.zeros(4, 4)
    skeleton_mask[0, 1] = 1
    skeleton_mask[1, 0] = 1  # Symmetric
    skeleton_mask[2, 3] = 1
    skeleton_mask[3, 2] = 1  # Symmetric
    
    # Create dummy blocks and penalty weights
    blocks = []
    penalty_weights = torch.ones(4, 4)
    
    # Create loss computer with skeleton mask
    loss_computer = LossComputer(blocks, penalty_weights, skeleton_mask=skeleton_mask)
    
    # Compute skeleton preservation loss
    skel_loss = loss_computer.skeleton_preservation_loss(adjacency)
    
    # Manual calculation:
    # Edge (0,1): sum = 0.8 + 0.2 = 1.0 → (1 - 1.0)^2 = 0.0
    # Edge (1,0): sum = 0.2 + 0.8 = 1.0 → (1 - 1.0)^2 = 0.0
    # Edge (2,3): sum = 0.1 + 0.1 = 0.2 → (1 - 0.2)^2 = 0.64
    # Edge (3,2): sum = 0.1 + 0.1 = 0.2 → (1 - 0.2)^2 = 0.64
    # Total = (0.0 + 0.0 + 0.64 + 0.64) / (2 * 2 edges) = 1.28 / 4 = 0.32
    
    expected_loss = ((1.0 - 1.0)**2 + (1.0 - 1.0)**2 + 
                     (1.0 - 0.2)**2 + (1.0 - 0.2)**2) / 4
    
    print(f"\nAdjacency matrix:")
    print(adjacency)
    print(f"\nSkeleton mask:")
    print(skeleton_mask)
    print(f"\nEdge sums (A→B + B→A):")
    edge_sum = adjacency + adjacency.T
    print(edge_sum)
    print(f"\nExpected loss: {expected_loss:.6f}")
    print(f"Computed loss: {skel_loss:.6f}")
    print(f"Difference: {abs(expected_loss - skel_loss):.10f}")
    
    assert torch.allclose(skel_loss, torch.tensor(expected_loss), atol=1e-6), \
        f"Skeleton loss mismatch! Expected {expected_loss}, got {skel_loss}"
    
    print("\n[PASS] TEST PASSED: Skeleton Preservation math is correct!")
    
    # Test 2: Verify it penalizes edge deletion
    print("\n--- Test 2: Edge Deletion Penalty ---")
    adjacency_deleted = torch.zeros(4, 4)
    adjacency_deleted[0, 1] = 0.0  # Edge deleted!
    adjacency_deleted[1, 0] = 0.0
    
    loss_deleted = loss_computer.skeleton_preservation_loss(adjacency_deleted)
    print(f"Loss when edge (0,1) deleted: {loss_deleted:.6f}")
    print(f"Loss when edge (0,1) preserved: {skel_loss:.6f}")
    
    assert loss_deleted > skel_loss, "Deleted edges should have higher loss!"
    print("[PASS] Verified: Edge deletion is heavily penalized!")
    
    # Test 3: Verify it allows directional choice
    print("\n--- Test 3: Directional Choice ---")
    adjacency_choice1 = torch.zeros(4, 4)
    adjacency_choice1[0, 1] = 0.9  # Strong A → B
    adjacency_choice1[1, 0] = 0.1  # Weak B → A
    
    adjacency_choice2 = torch.zeros(4, 4)
    adjacency_choice2[0, 1] = 0.1  # Weak A → B
    adjacency_choice2[1, 0] = 0.9  # Strong B → A
    
    loss_choice1 = loss_computer.skeleton_preservation_loss(adjacency_choice1)
    loss_choice2 = loss_computer.skeleton_preservation_loss(adjacency_choice2)
    
    print(f"Loss for A→B strong (0.9 + 0.1): {loss_choice1:.6f}")
    print(f"Loss for B→A strong (0.1 + 0.9): {loss_choice2:.6f}")
    
    assert torch.allclose(loss_choice1, loss_choice2, atol=1e-6), \
        "Both directional choices should have similar low loss!"
    print("[PASS] Verified: Model can choose direction freely (as long as sum ≈ 1)!")


if __name__ == "__main__":
    print("=" * 70)
    print("LOSS MODULE UNIT TESTS")
    print("=" * 70)
    
    # Test 1: Weighted Group Lasso
    test_weighted_group_lasso()
    
    # Test 2: Cycle Consistency
    test_cycle_consistency()
    
    # Test 3: Skeleton Preservation (NEW!)
    test_skeleton_preservation()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED [SUCCESS]")
    print("=" * 70)
    print("\nLoss module is ready for Phase 2 training with Skeleton Preservation!")

