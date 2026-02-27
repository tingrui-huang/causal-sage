"""
Prior Builder Module

Dataset-agnostic prior builder for causal discovery pipeline.

Integrates prior knowledge from:
1. FCI: Skeleton mask (which variable pairs are connected)
2. LLM: Direction prior (initial weights for specific rules)
3. Domain knowledge: Normal state handling (dataset-specific)
"""

import torch
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path

from modules.vstructure_postprocess import enforce_vstructure_on_skeleton_mask


class PriorBuilder:
    """
    Dataset-agnostic prior builder for causal discovery
    
    Key responsibilities:
    1. FCI skeleton mask: (n_states, n_states) binary mask
    2. LLM direction prior: (n_states, n_states) initial weights
    3. Domain-specific penalty weights: (n_states, n_states) for weighted Group Lasso
    4. Block structure: List of blocks for Group Lasso
    
    Supports any dataset with proper variable structure metadata.
    """
    
    def __init__(self, var_structure: Dict, dataset_name: Optional[str] = None):
        """
        Args:
            var_structure: Variable structure from DataLoader
            dataset_name: Optional dataset name for dataset-specific logic
        """
        self.var_structure = var_structure
        self.n_states = var_structure['n_states']
        self.dataset_name = dataset_name or "Unknown"
        
        print("=" * 70)
        print("PRIOR BUILDER INITIALIZED")
        print("=" * 70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Variables: {var_structure['n_variables']}")
        print(f"States: {self.n_states}")
    
    def build_skeleton_mask_from_fci(self, fci_csv_path: str) -> torch.Tensor:
        """
        Build skeleton mask from FCI results
        
        Args:
            fci_csv_path: Path to FCI edges CSV (variable-level)
        
        Returns:
            Binary mask (105, 105) where 1 = allowed, 0 = forbidden
            
        Logic:
            - If FCI says A-B connected (any direction) -> entire block A->B is 1
            - Otherwise -> block is 0
        """
        print("\n" + "=" * 70)
        print("BUILDING SKELETON MASK FROM FCI")
        print("=" * 70)
        
        # Initialize mask to zeros (nothing allowed)
        skeleton_mask = torch.zeros(self.n_states, self.n_states)
        
        # Load FCI edges
        df_fci = pd.read_csv(fci_csv_path)
        print(f"Loaded {len(df_fci)} edges from FCI")
        
        # Determine column names
        if 'Source' in df_fci.columns and 'Target' in df_fci.columns:
            source_col, target_col = 'Source', 'Target'
        elif 'source' in df_fci.columns and 'target' in df_fci.columns:
            source_col, target_col = 'source', 'target'
        else:
            # Use first two columns
            source_col, target_col = df_fci.columns[0], df_fci.columns[1]
        
        print(f"Using columns: {source_col} -> {target_col}")
        
        # Count edges
        edge_count = 0
        directed_count = 0
        bidirectional_count = 0
        
        # Check if edge_type column exists
        has_edge_type = 'edge_type' in df_fci.columns
        
        # For each FCI edge, enable the entire block
        for _, row in df_fci.iterrows():
            var_a = row[source_col]
            var_b = row[target_col]
            
            # Check if variables exist
            if var_a not in self.var_structure['var_to_states']:
                print(f"Warning: Variable {var_a} not found in structure")
                continue
            if var_b not in self.var_structure['var_to_states']:
                print(f"Warning: Variable {var_b} not found in structure")
                continue
            
            # Get state indices for both variables
            states_a = self.var_structure['var_to_states'][var_a]
            states_b = self.var_structure['var_to_states'][var_b]
            
            # Get edge type (if available)
            edge_type = row.get('edge_type', 'directed') if has_edge_type else 'directed'
            
            # Enable block A -> B
            for i in states_a:
                for j in states_b:
                    skeleton_mask[i, j] = 1
            
            # For undirected/partial/tail-tail edges, also enable B -> A
            if edge_type in ['undirected', 'partial', 'tail-tail']:
                for i in states_b:
                    for j in states_a:
                        skeleton_mask[i, j] = 1
                bidirectional_count += 1
            else:
                directed_count += 1
            
            edge_count += 1
        
        # Calculate statistics
        total_possible = self.n_states * self.n_states
        allowed = int(skeleton_mask.sum().item())
        
        print(f"\nSkeleton mask statistics:")
        print(f"  Edges processed: {edge_count}")
        if has_edge_type:
            print(f"    - Directed (one direction): {directed_count}")
            print(f"    - Undirected/Partial/Tail-tail (both directions): {bidirectional_count}")
            print(f"  FCI Bidirectional Ratio: {bidirectional_count/edge_count*100:.1f}% ({bidirectional_count}/{edge_count} edges)")
        print(f"  Allowed connections: {allowed} / {total_possible} ({allowed/total_possible*100:.2f}%)")
        print(f"  Forbidden connections: {total_possible - allowed}")
        
        return skeleton_mask
    
    def build_direction_prior_from_llm(self, llm_csv_path: str, 
                                       high_confidence: float = 0.7,
                                       low_confidence: float = 0.3) -> torch.Tensor:
        """
        Build direction prior from FCI+LLM hybrid CSV
        
        This reads the edges_Hybrid_FCI_LLM_*.csv file where LLM has resolved
        directions for partial/undirected edges from FCI.
        
        Args:
            llm_csv_path: Path to FCI+LLM hybrid CSV (e.g., edges_Hybrid_FCI_LLM_20251207_230956.csv)
            high_confidence: Weight for LLM-resolved or directed edges (0.7)
            low_confidence: Weight for undirected/partial edges (0.3)
        
        Returns:
            Direction prior matrix (105, 105)
        """
        print("\n" + "=" * 70)
        print("BUILDING DIRECTION PRIOR FROM FCI+LLM HYBRID")
        print("=" * 70)
        
        # Initialize with zeros (no prior)
        direction_prior = torch.zeros(self.n_states, self.n_states)
        
        # Load CSV
        df = pd.read_csv(llm_csv_path)
        
        print(f"Loaded {len(df)} edges from FCI+LLM hybrid")
        
        # Determine columns
        if 'source' in df.columns and 'target' in df.columns:
            source_col, target_col = 'source', 'target'
        elif 'Source' in df.columns and 'Target' in df.columns:
            source_col, target_col = 'Source', 'Target'
        else:
            raise ValueError(f"Cannot find source/target columns in {llm_csv_path}")
        
        print(f"Using columns: {source_col} -> {target_col}")
        
        # Process edges
        directed_count = 0
        llm_resolved_count = 0
        undirected_count = 0
        
        for _, row in df.iterrows():
            var_source = row[source_col]
            var_target = row[target_col]
            edge_type = row.get('edge_type', 'directed')
            
            # Get state indices for these variables
            source_states = self.var_structure['var_to_states'][var_source]
            target_states = self.var_structure['var_to_states'][var_target]
            
            # Determine confidence based on edge type
            if edge_type == 'llm_resolved':
                confidence = high_confidence
                llm_resolved_count += 1
            elif edge_type == 'directed':
                confidence = high_confidence
                directed_count += 1
            else:  # undirected, partial, tail-tail
                confidence = low_confidence
                undirected_count += 1
            
            # Set all state-to-state connections for this variable pair
            for i in source_states:
                for j in target_states:
                    direction_prior[i, j] = confidence
        
        print(f"\nDirection prior statistics:")
        print(f"  Directed edges: {directed_count} (confidence={high_confidence})")
        print(f"  LLM-resolved edges: {llm_resolved_count} (confidence={high_confidence})")
        print(f"  Undirected/partial edges: {undirected_count} (confidence={low_confidence})")
        print(f"  Total edges: {len(df)}")
        print(f"  Non-zero weights: {(direction_prior > 0).sum().item()}")
        print(f"  Mean non-zero weight: {direction_prior[direction_prior > 0].mean().item():.4f}")
        
        return direction_prior
    
    def build_random_direction_prior(self, fci_csv_path: str, 
                                      high_confidence: float = 0.7,
                                      low_confidence: float = 0.3,
                                      seed: Optional[int] = None) -> torch.Tensor:
        """
        Build RANDOM direction prior for control experiment
        
        CRITICAL CHANGE: Now uses SAME magnitude as LLM (0.7/0.3), only direction is random.
        This ensures fair comparison - same "energy", only direction differs.
        
        For each edge pair (A, B):
            - Randomly choose: A->B strong OR B->A strong (50/50 chance)
            - Strong direction: high_confidence (0.7, same as LLM)
            - Weak direction: low_confidence (0.3, same as LLM)
        
        This tests if LLM is a "blind perturbation" (just breaking symmetry) or 
        an "intelligent guide" (pointing in the right direction).
        
        Hypothesis:
            - If random direction works as well as LLM -> LLM is just "blind perturbation"
            - If random direction has low orientation accuracy (~50%) -> LLM is "intelligent guide"
        
        Args:
            fci_csv_path: Path to FCI edges CSV (to know which pairs to randomize)
            high_confidence: Weight for strong direction (default: 0.7, same as LLM)
            low_confidence: Weight for weak direction (default: 0.3, same as LLM)
            seed: Random seed for reproducibility
        
        Returns:
            Random direction prior matrix (n_states, n_states)
        """
        print("\n" + "=" * 70)
        print("BUILDING RANDOM DIRECTION PRIOR (CONTROL EXPERIMENT)")
        print("=" * 70)
        print(f"High confidence (strong direction): {high_confidence}")
        print(f"Low confidence (weak direction): {low_confidence}")
        print(f"Random seed: {seed}")
        print("\n[FAIR COMPARISON] Same magnitude as LLM, only direction is random")
        
        # IMPORTANT: Use a local RNG so we don't accidentally perturb the
        # global torch RNG state used later in training.
        rng = None
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(int(seed))
        
        # Initialize with zeros
        direction_prior = torch.zeros(self.n_states, self.n_states)
        
        # Load FCI edges
        df_fci = pd.read_csv(fci_csv_path)
        print(f"Loaded {len(df_fci)} edges from FCI")
        
        # Determine column names
        if 'Source' in df_fci.columns and 'Target' in df_fci.columns:
            source_col, target_col = 'Source', 'Target'
        elif 'source' in df_fci.columns and 'target' in df_fci.columns:
            source_col, target_col = 'source', 'target'
        else:
            source_col, target_col = df_fci.columns[0], df_fci.columns[1]
        
        edge_count = 0
        forward_strong_count = 0
        backward_strong_count = 0

        processed_pairs = set()

        # For each FCI edge, randomly assign strong/weak directions
        for _, row in df_fci.iterrows():
            var_a = row[source_col]
            var_b = row[target_col]

            # Check if variables exist
            if var_a not in self.var_structure['var_to_states']:
                continue
            if var_b not in self.var_structure['var_to_states']:
                continue

            pair_key = tuple(sorted([var_a, var_b]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # Get state indices
            states_a = self.var_structure['var_to_states'][var_a]
            states_b = self.var_structure['var_to_states'][var_b]

            # Randomly decide which direction is strong (50/50 chance)
            coin = torch.rand(1, generator=rng).item() if rng is not None else torch.rand(1).item()
            if coin > 0.5:
                # A -> B is strong, B -> A is weak
                forward_weight = high_confidence
                backward_weight = low_confidence
                forward_strong_count += 1
            else:
                # B -> A is strong, A -> B is weak
                forward_weight = low_confidence
                backward_weight = high_confidence
                backward_strong_count += 1

            # Assign weights for A -> B
            for i in states_a:
                for j in states_b:
                    direction_prior[i, j] = forward_weight

            # Assign weights for B -> A
            for i in states_b:
                for j in states_a:
                    direction_prior[i, j] = backward_weight

            edge_count += 1

        print(f"\nRandom direction prior statistics:")
        print(f"  Total edge pairs: {edge_count}")
        print(f"  Forward strong (A->B): {forward_strong_count} ({forward_strong_count/edge_count*100:.1f}%)")
        print(f"  Backward strong (B->A): {backward_strong_count} ({backward_strong_count/edge_count*100:.1f}%)")
        print(f"  Non-zero weights: {(direction_prior > 0).sum().item()}")
        print(f"  Mean weight: {direction_prior[direction_prior > 0].mean().item():.4f}")
        print(f"  High confidence weights: {(direction_prior == high_confidence).sum().item()}")
        print(f"  Low confidence weights: {(direction_prior == low_confidence).sum().item()}")
        print("\n[CONTROL EXPERIMENT] Random direction with same magnitude as LLM")
        print("   Expected result: Unresolved ratio -> 0%, but Orientation accuracy ~50%")
        print("   (Because directions are random, not guided by domain knowledge)")

        return direction_prior

    def build_normal_penalty_weights(self, normal_weight: float = 0.1,
                                     abnormal_weight: float = 1.0,
                                     normal_keyword: str = 'Normal') -> torch.Tensor:
        """
        Build penalty weight matrix for Weighted Group Lasso (dataset-agnostic)

        This implements domain-specific penalty weighting. For datasets with
        "normal" states (like ALARM), we can give lower penalty to normal->normal
        connections. For other datasets, this returns uniform weights.

        Args:
            normal_weight: Weight for Normal -> Normal (low, e.g., 0.1)
            abnormal_weight: Weight for other connections (high, e.g., 1.0)
            normal_keyword: Keyword to identify normal states (default: 'Normal')

        Returns:
            Penalty weight matrix (n_states, n_states)

        Logic:
            - If dataset has "normal" states: Normal -> Normal gets low weight
            - Otherwise: Uniform weights for all connections
        """
        print("\n" + "=" * 70)
        print("BUILDING PENALTY WEIGHTS")
        print("=" * 70)

        penalty_weights = torch.ones(self.n_states, self.n_states) * abnormal_weight

        # Check if dataset has "normal" states
        has_normal_states = any(normal_keyword in self.var_structure['idx_to_state'][i]
                               for i in range(self.n_states))

        if not has_normal_states:
            print(f"Dataset does not have '{normal_keyword}' states - using uniform weights")
            print(f"  All connections: weight={abnormal_weight}")
            return penalty_weights

        # Dataset has normal states - apply differential weighting
        print(f"Dataset has '{normal_keyword}' states - applying differential weighting")

        normal_to_normal_count = 0

        for i in range(self.n_states):
            for j in range(self.n_states):
                state_i_name = self.var_structure['idx_to_state'][i]
                state_j_name = self.var_structure['idx_to_state'][j]

                is_i_normal = normal_keyword in state_i_name
                is_j_normal = normal_keyword in state_j_name

                # Only Normal -> Normal gets low weight
                if is_i_normal and is_j_normal:
                    penalty_weights[i, j] = normal_weight
                    normal_to_normal_count += 1

        print(f"\nPenalty weight statistics:")
        print(f"  Normal -> Normal connections: {normal_to_normal_count} (weight={normal_weight})")
        print(f"  Other connections: {self.n_states**2 - normal_to_normal_count} (weight={abnormal_weight})")
        print(f"  Ratio: {normal_to_normal_count / (self.n_states**2) * 100:.2f}% protected")

        return penalty_weights

    def build_block_structure(self, skeleton_mask: torch.Tensor) -> List[Dict]:
        """
        Build block structure for Group Lasso

        CRITICAL: Only create blocks for edges allowed by FCI skeleton!
        This prevents the model from wasting computation on 1332 variable pairs
        when only ~45 are actually allowed by FCI.

        Args:
            skeleton_mask: (105, 105) binary mask from FCI

        Returns:
            List of block definitions for FCI-allowed variable pairs only
        """
        print("\n" + "=" * 70)
        print("BUILDING BLOCK STRUCTURE (FCI-CONSTRAINED)")
        print("=" * 70)

        blocks = []
        var_names = self.var_structure['variable_names']

        for var_a in var_names:
            for var_b in var_names:
                if var_a == var_b:
                    continue  # Skip self-loops

                # Get state indices for this variable pair
                row_indices = self.var_structure['var_to_states'][var_a]
                col_indices = self.var_structure['var_to_states'][var_b]

                # Check if this block has ANY allowed connections in skeleton
                # Extract the sub-matrix for this block
                block_mask = skeleton_mask[row_indices][:, col_indices]

                # Only create block if at least one connection is allowed
                if block_mask.sum().item() > 0:
                    blocks.append({
                        'var_pair': (var_a, var_b),
                        'row_indices': row_indices,
                        'col_indices': col_indices
                    })

        print(f"Total blocks (FCI-allowed): {len(blocks)}")
        print(f"Total possible blocks: {len(var_names) * (len(var_names) - 1)} = {len(var_names) * (len(var_names) - 1)}")
        print(f"Reduction: {(1 - len(blocks) / (len(var_names) * (len(var_names) - 1))) * 100:.1f}%")

        # Show sample blocks
        print(f"\nSample FCI-allowed blocks:")
        for i, block in enumerate(blocks[:5]):
            var_a, var_b = block['var_pair']
            n_rows = len(block['row_indices'])
            n_cols = len(block['col_indices'])
            print(f"  {i+1}. {var_a} -> {var_b}: {n_rows} x {n_cols} = {n_rows * n_cols} connections")

        return blocks

    def get_all_priors(self, fci_skeleton_path: str, llm_direction_path: str = None,
                      use_llm_prior: bool = True, use_random_prior: bool = False,
                      random_seed: Optional[int] = None,
                      high_confidence: float = 0.7,
                      low_confidence: float = 0.3,
                      enforce_vstructure: bool = False,
                      vstructure_pag_csv_path: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Convenience method to build all priors at once

        IMPORTANT: Uses TWO separate CSV files:
        - fci_skeleton_path: Pure FCI results for HARD skeleton mask
        - llm_direction_path: FCI+LLM hybrid results for SOFT direction prior (optional)

        Args:
            fci_skeleton_path: Path to pure FCI edges (e.g., edges_FCI_20251207_230824.csv)
            llm_direction_path: Path to FCI+LLM edges (e.g., edges_Hybrid_FCI_LLM_20251207_230956.csv)
                               Can be None if use_llm_prior=False
            use_llm_prior: Whether to use LLM direction prior (default: True)
                          If False, uses uniform initialization (0.5 for all allowed edges)
            use_random_prior: Whether to use RANDOM direction prior (control experiment)
            random_seed: Random seed for reproducibility (required if use_random_prior=True)
        Returns:
            Dictionary with all prior structures
        """
        # Build skeleton from PURE FCI (hard constraint)
        skeleton_mask = self.build_skeleton_mask_from_fci(fci_skeleton_path)

        # Optional: enforce collider (v-structure) constraints directly on skeleton mask (training-time hard mask).
        if enforce_vstructure:
            pag_path = vstructure_pag_csv_path or fci_skeleton_path
            try:
                skeleton_mask, vstats = enforce_vstructure_on_skeleton_mask(
                    skeleton_mask=skeleton_mask,
                    var_structure=self.var_structure,
                    pag_csv_path=str(pag_path),
                )
                print("\n" + "=" * 70)
                print("APPLIED V-STRUCTURE HARD MASK (TRAINING-TIME)")
                print("=" * 70)
                print(f"PAG CSV: {pag_path}")
                print(f"Forced edges inferred: {vstats.forced_edges_total} (total) / {vstats.forced_edges_in_structure} (in-structure)")
                print(f"Skeleton reverse-blocks forbidden: {vstats.skeleton_blocks_forbidden}")
            except Exception as e:
                print(f"[WARN] Failed to enforce v-structure mask on skeleton: {e}")

        # Build direction prior
        if use_random_prior:
            if random_seed is None:
                raise ValueError("random_seed must be provided when use_random_prior=True (recommend: config.RANDOM_SEED)")
            # CONTROL EXPERIMENT: Random direction prior
            print("\n[CONTROL EXPERIMENT] Using RANDOM direction prior")
            direction_prior = self.build_random_direction_prior(
                fci_skeleton_path,
                high_confidence=high_confidence,  # Use custom values
                low_confidence=low_confidence,    # Use custom values
                seed=int(random_seed)
            )
        elif use_llm_prior and llm_direction_path:
            # Build direction prior from FCI+LLM hybrid (soft initialization)
            direction_prior = self.build_direction_prior_from_llm(
                llm_direction_path,
                high_confidence=high_confidence,  # Use custom values
                low_confidence=low_confidence     # Use custom values
            )
            print("\n[USING LLM DIRECTION PRIOR]")
        else:
            # Uniform initialization: 0.5 for all allowed edges, 0.0 for forbidden
            direction_prior = skeleton_mask * 0.5
            print("\n[NO LLM PRIOR - UNIFORM INITIALIZATION]")
            print("All FCI-allowed edges initialized with weight 0.5")
        
        # Build penalty weights for Normal state handling
        penalty_weights = self.build_normal_penalty_weights()
        
        # Build blocks ONLY for FCI-allowed edges
        blocks = self.build_block_structure(skeleton_mask)
        
        return {
            'skeleton_mask': skeleton_mask,
            'direction_prior': direction_prior,
            'penalty_weights': penalty_weights,
            'blocks': blocks
        }


if __name__ == "__main__":
    # Test the prior builder with ALARM dataset
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from modules.data_loader import CausalDataLoader
    
    print("Testing PriorBuilder with ALARM dataset\n")
    
    # Use absolute paths for testing
    base_dir = Path(__file__).parent.parent
    
    # Load data first to get variable structure
    loader = CausalDataLoader(
        data_path=str(base_dir / 'data' / 'alarm' / 'alarm_data_10000.csv'),
        metadata_path=str(base_dir / 'data' / 'alarm' / 'metadata.json')
    )
    var_structure = loader.get_variable_structure()
    
    # Build priors
    prior_builder = PriorBuilder(var_structure, dataset_name='ALARM')
    priors = prior_builder.get_all_priors(
        fci_skeleton_path=str(base_dir / 'data' / 'alarm' / 'edges_FCI_20251207_230824.csv'),
        llm_direction_path=str(base_dir / 'data' / 'alarm' / 'edges_Hybrid_FCI_LLM_20251207_230956.csv'),
        use_llm_prior=True
    )
    
    print("\n" + "=" * 70)
    print("ALL PRIORS BUILT SUCCESSFULLY")
    print("=" * 70)
    for key, value in priors.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {len(value)} items")

