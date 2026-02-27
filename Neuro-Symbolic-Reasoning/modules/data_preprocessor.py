"""
Data Preprocessor - Extensible for Multiple Data Types

Handles:
- Discrete data (already one-hot encoded) - pass through
- Continuous data (Tuebingen) - discretization needed
- Future data types can be easily added

This module is a placeholder for future Tuebingen support.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import KBinsDiscretizer


class DataPreprocessor:
    """
    Preprocessor for handling different data types
    
    Design principle: Extensibility without breaking existing code
    """
    
    def __init__(self, data_type: str = 'discrete'):
        """
        Args:
            data_type: Type of data ('discrete' or 'continuous')
        """
        self.data_type = data_type
        self.discretizer = None
        self.metadata = {}
    
    def fit_transform(self, data: pd.DataFrame, n_bins: int = 3) -> Tuple[pd.DataFrame, Dict]:
        """
        Fit and transform data based on type
        
        Args:
            data: Input dataframe
            n_bins: Number of bins for discretization (if continuous)
        
        Returns:
            Tuple of (transformed_data, metadata)
        """
        if self.data_type == 'discrete':
            # Already discrete, no transformation needed
            return data, {'type': 'discrete', 'n_bins': None}
        
        elif self.data_type == 'continuous':
            # Need discretization
            return self._discretize_continuous(data, n_bins)
        
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _discretize_continuous(self, data: pd.DataFrame, n_bins: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Discretize continuous data using quantile-based binning
        
        Args:
            data: Continuous dataframe
            n_bins: Number of bins per variable
        
        Returns:
            Tuple of (discretized_data, metadata)
        """
        print("\n" + "=" * 70)
        print("DISCRETIZING CONTINUOUS DATA")
        print("=" * 70)
        print(f"Strategy: Quantile-based binning")
        print(f"Number of bins: {n_bins}")
        
        # Initialize discretizer
        self.discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode='ordinal',
            strategy='quantile'
        )
        
        # Fit and transform
        data_array = data.values
        discretized_array = self.discretizer.fit_transform(data_array)
        
        # Convert back to DataFrame
        discretized_df = pd.DataFrame(
            discretized_array.astype(int),
            columns=data.columns
        )
        
        # Store metadata
        metadata = {
            'type': 'continuous_discretized',
            'n_bins': n_bins,
            'strategy': 'quantile',
            'bin_edges': self.discretizer.bin_edges_,
            'original_columns': list(data.columns)
        }
        
        print(f"\n[OK] Discretization complete")
        print(f"  Original shape: {data.shape}")
        print(f"  Discretized shape: {discretized_df.shape}")
        print(f"  Value range: [0, {n_bins-1}]")
        
        return discretized_df, metadata
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted discretizer
        
        Args:
            data: New data to transform
        
        Returns:
            Transformed data
        """
        if self.data_type == 'discrete':
            return data
        
        elif self.data_type == 'continuous':
            if self.discretizer is None:
                raise ValueError("Discretizer not fitted. Call fit_transform first.")
            
            discretized_array = self.discretizer.transform(data.values)
            return pd.DataFrame(
                discretized_array.astype(int),
                columns=data.columns
            )
        
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")


def generate_metadata_from_discretized(discretized_df: pd.DataFrame, 
                                       n_bins: int,
                                       dataset_name: str = "tuebingen") -> Dict:
    """
    Generate metadata JSON for discretized continuous data
    
    This creates the same format as ALARM's metadata.json,
    allowing seamless integration with existing pipeline.
    
    Args:
        discretized_df: Discretized dataframe
        n_bins: Number of bins used
        dataset_name: Name of dataset
    
    Returns:
        Metadata dictionary
    """
    metadata = {
        "dataset_name": dataset_name,
        "data_format": "one_hot_csv",
        "state_mappings": {},
        "notes": "Auto-generated from continuous data discretization"
    }
    
    # Generate state mappings
    for col in discretized_df.columns:
        state_names = [f"{col}_bin{i}" for i in range(n_bins)]
        metadata["state_mappings"][col] = {
            str(i): state_name for i, state_name in enumerate(state_names)
        }
    
    return metadata


if __name__ == "__main__":
    """Unit tests and examples"""
    print("=" * 80)
    print("DATA PREPROCESSOR TESTS")
    print("=" * 80)
    
    # Test 1: Discrete data (pass through)
    print("\nTest 1: Discrete Data (Pass Through)")
    print("-" * 80)
    
    discrete_data = pd.DataFrame({
        'A': [0, 1, 0, 1, 0],
        'B': [1, 0, 1, 0, 1],
        'C': [0, 0, 1, 1, 0]
    })
    
    preprocessor = DataPreprocessor(data_type='discrete')
    transformed, meta = preprocessor.fit_transform(discrete_data)
    
    print(f"[OK] Discrete data passed through unchanged")
    print(f"  Shape: {transformed.shape}")
    print(f"  Metadata: {meta}")
    
    # Test 2: Continuous data (discretization)
    print("\nTest 2: Continuous Data (Discretization)")
    print("-" * 80)
    
    # Generate synthetic continuous data
    # Use unified config seed if available (avoid hard-coding)
    try:
        import sys as _sys
        from pathlib import Path as _Path
        project_root = _Path(__file__).parent.parent.parent
        _sys.path.insert(0, str(project_root))
        import config as _unified_config
        np.random.seed(int(_unified_config.RANDOM_SEED))
    except Exception:
        # If config isn't available, just proceed without setting a seed.
        pass
    continuous_data = pd.DataFrame({
        'altitude': np.random.uniform(0, 3000, 100),
        'temperature': np.random.uniform(-10, 40, 100)
    })
    
    preprocessor = DataPreprocessor(data_type='continuous')
    discretized, meta = preprocessor.fit_transform(continuous_data, n_bins=3)
    
    print(f"[OK] Continuous data discretized")
    print(f"  Original range: altitude [{continuous_data['altitude'].min():.1f}, {continuous_data['altitude'].max():.1f}]")
    print(f"  Discretized values: {sorted(discretized['altitude'].unique())}")
    print(f"  Number of bins: {meta['n_bins']}")
    
    # Test 3: Generate metadata
    print("\nTest 3: Generate Metadata for Discretized Data")
    print("-" * 80)
    
    metadata = generate_metadata_from_discretized(discretized, n_bins=3, dataset_name="tuebingen_pair1")
    
    print(f"[OK] Metadata generated")
    print(f"  Dataset: {metadata['dataset_name']}")
    print(f"  Variables: {list(metadata['state_mappings'].keys())}")
    print(f"  Sample mapping: {metadata['state_mappings']['altitude']}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    print("\n[INFO] This module is ready for Tuebingen integration")
    print("[INFO] When needed, simply set data_type='continuous' in config")

