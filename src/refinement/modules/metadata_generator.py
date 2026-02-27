"""
Metadata Generator Module

This module provides a dataset-agnostic interface for generating metadata
required by the causal discovery pipeline. It handles the conversion of
various dataset formats into a standardized metadata format.

Key responsibilities:
1. Define standard metadata schema
2. Provide base class for dataset-specific generators
3. Validate metadata consistency
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod


class MetadataSchema:
    """
    Standard metadata schema for causal discovery pipeline
    
    Required fields:
    - dataset_name: Name of the dataset (e.g., "ALARM", "Tuebingen_pair1")
    - n_variables: Number of variables
    - n_states: Total number of states across all variables
    - variable_names: List of variable names
    - state_mappings: Dict mapping variable names to their state codes and names
    - variable_info: Additional information about each variable (optional)
    """
    
    REQUIRED_FIELDS = [
        'dataset_name',
        'n_variables',
        'n_states',
        'variable_names',
        'state_mappings'
    ]
    
    @staticmethod
    def validate(metadata: Dict) -> Tuple[bool, str]:
        """
        Validate metadata against schema
        
        Args:
            metadata: Metadata dictionary to validate
        
        Returns:
            (is_valid, error_message)
        """
        # Check required fields
        for field in MetadataSchema.REQUIRED_FIELDS:
            if field not in metadata:
                return False, f"Missing required field: {field}"
        
        # Validate state_mappings structure
        state_mappings = metadata['state_mappings']
        if not isinstance(state_mappings, dict):
            return False, "state_mappings must be a dictionary"
        
        # Count states and variables
        total_states = 0
        for var_name, state_mapping in state_mappings.items():
            if not isinstance(state_mapping, dict):
                return False, f"state_mapping for {var_name} must be a dictionary"
            total_states += len(state_mapping)
        
        # Verify counts
        if len(state_mappings) != metadata['n_variables']:
            return False, f"n_variables mismatch: expected {metadata['n_variables']}, got {len(state_mappings)}"
        
        if total_states != metadata['n_states']:
            return False, f"n_states mismatch: expected {metadata['n_states']}, got {total_states}"
        
        if len(metadata['variable_names']) != metadata['n_variables']:
            return False, f"variable_names length mismatch"
        
        return True, "Valid"


class BaseMetadataGenerator(ABC):
    """
    Abstract base class for dataset-specific metadata generators
    
    Subclasses must implement:
    - generate_metadata(): Generate metadata from dataset
    """
    
    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
    
    @abstractmethod
    def generate_metadata(self) -> Dict:
        """
        Generate metadata for the dataset
        
        Returns:
            Dictionary conforming to MetadataSchema
        """
        pass
    
    def save_metadata(self, metadata: Dict, output_path: Union[str, Path]) -> None:
        """
        Save metadata to JSON file
        
        Args:
            metadata: Metadata dictionary
            output_path: Path to save JSON file
        """
        # Validate metadata
        is_valid, message = MetadataSchema.validate(metadata)
        if not is_valid:
            raise ValueError(f"Invalid metadata: {message}")
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {output_path}")
        print(f"  Dataset: {metadata['dataset_name']}")
        print(f"  Variables: {metadata['n_variables']}")
        print(f"  States: {metadata['n_states']}")
    
    def load_metadata(self, metadata_path: Union[str, Path]) -> Dict:
        """
        Load and validate metadata from JSON file
        
        Args:
            metadata_path: Path to metadata JSON file
        
        Returns:
            Validated metadata dictionary
        """
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate
        is_valid, message = MetadataSchema.validate(metadata)
        if not is_valid:
            raise ValueError(f"Invalid metadata in {metadata_path}: {message}")
        
        return metadata


class DiscreteDataMetadataGenerator(BaseMetadataGenerator):
    """
    Metadata generator for discrete/categorical data
    
    This handles datasets where:
    - Each variable has discrete states
    - Data is in CSV format with one-hot encoded states
    """
    
    def __init__(self, dataset_name: str, data_path: Union[str, Path]):
        """
        Args:
            dataset_name: Name of the dataset
            data_path: Path to CSV file with one-hot encoded data
        """
        super().__init__(dataset_name)
        self.data_path = Path(data_path)
    
    def generate_metadata(self) -> Dict:
        """
        Generate metadata by analyzing CSV column names
        
        Assumes column naming convention: VARIABLE_STATE
        (e.g., "LVFAILURE_False", "LVFAILURE_True")
        
        Returns:
            Metadata dictionary
        """
        print("=" * 70)
        print(f"GENERATING METADATA FOR: {self.dataset_name}")
        print("=" * 70)
        
        # Load CSV to get column names
        df = pd.read_csv(self.data_path)
        
        # Get state columns (exclude ID columns)
        state_columns = [col for col in df.columns 
                        if col not in ['sample_id', 'patient_id', 'subject_id', 'id']]
        
        print(f"Total columns: {len(df.columns)}")
        print(f"State columns: {len(state_columns)}")
        
        # Parse variable-state structure
        state_mappings = {}
        variable_names_ordered = []  # Use list to preserve order
        variable_names_seen = set()  # Track seen variables
        
        for col in state_columns:
            # Split on last underscore to separate variable from state
            parts = col.rsplit('_', 1)
            if len(parts) != 2:
                raise ValueError(f"Column name {col} doesn't follow VARIABLE_STATE convention")
            
            var_name, state_name = parts
            
            # Add to ordered list only on first occurrence (preserves CSV column order)
            if var_name not in variable_names_seen:
                variable_names_ordered.append(var_name)
                variable_names_seen.add(var_name)
            
            if var_name not in state_mappings:
                state_mappings[var_name] = {}
            
            # Use sequential state codes
            state_code = str(len(state_mappings[var_name]))
            state_mappings[var_name][state_code] = col  # Store full column name as state name
        
        # FIXED: Use CSV column order, NOT alphabetical sorting
        variable_names = variable_names_ordered
        
        # Count total states
        n_states = sum(len(states) for states in state_mappings.values())
        
        metadata = {
            'dataset_name': self.dataset_name,
            'n_variables': len(variable_names),
            'n_states': n_states,
            'variable_names': variable_names,
            'state_mappings': state_mappings,
            'data_format': 'one_hot_csv',
            'source_file': str(self.data_path.name)
        }
        
        print(f"\nGenerated metadata:")
        print(f"  Variables: {metadata['n_variables']}")
        print(f"  States: {metadata['n_states']}")
        print(f"  Avg states per variable: {n_states / len(variable_names):.2f}")
        
        # Show sample variables
        print(f"\nSample variables:")
        for i, var in enumerate(variable_names[:5]):
            states = state_mappings[var]
            print(f"  {i+1}. {var}: {len(states)} states")
        
        return metadata


class ContinuousDataMetadataGenerator(BaseMetadataGenerator):
    """
    Metadata generator for continuous data (e.g., Tuebingen pairs)
    
    This handles datasets where:
    - Variables are continuous
    - Data is in CSV format with raw continuous values
    - No pre-defined states (each variable is treated as having 1 "continuous" state)
    """
    
    def __init__(self, dataset_name: str, data_path: Union[str, Path], 
                 variable_names: Optional[List[str]] = None):
        """
        Args:
            dataset_name: Name of the dataset
            data_path: Path to CSV file with continuous data
            variable_names: Optional list of variable names (if None, use column names)
        """
        super().__init__(dataset_name)
        self.data_path = Path(data_path)
        self.variable_names = variable_names
    
    def generate_metadata(self) -> Dict:
        """
        Generate metadata for continuous data
        
        For continuous data, each variable is treated as having a single "continuous" state
        
        Returns:
            Metadata dictionary
        """
        print("=" * 70)
        print(f"GENERATING METADATA FOR: {self.dataset_name}")
        print("=" * 70)
        
        # Load CSV to get column names
        df = pd.read_csv(self.data_path)
        
        # Get variable columns
        if self.variable_names is None:
            # Use all columns except common ID columns
            variable_names = [col for col in df.columns 
                            if col not in ['sample_id', 'patient_id', 'subject_id', 'id', 'index']]
        else:
            variable_names = self.variable_names
        
        print(f"Total columns: {len(df.columns)}")
        print(f"Variable columns: {len(variable_names)}")
        
        # For continuous data, each variable has one "continuous" state
        state_mappings = {}
        for var_name in variable_names:
            state_mappings[var_name] = {
                '0': f"{var_name}_continuous"
            }
        
        metadata = {
            'dataset_name': self.dataset_name,
            'n_variables': len(variable_names),
            'n_states': len(variable_names),  # One state per variable for continuous
            'variable_names': variable_names,
            'state_mappings': state_mappings,
            'data_format': 'continuous_csv',
            'source_file': str(self.data_path.name)
        }
        
        print(f"\nGenerated metadata:")
        print(f"  Variables: {metadata['n_variables']}")
        print(f"  States: {metadata['n_states']} (continuous)")
        print(f"  Variables: {', '.join(variable_names)}")
        
        return metadata


def create_metadata_for_dataset(dataset_type: str, dataset_name: str, 
                                data_path: Union[str, Path],
                                output_path: Union[str, Path],
                                **kwargs) -> Dict:
    """
    Convenience function to create metadata for a dataset
    
    Args:
        dataset_type: Type of dataset ('discrete' or 'continuous')
        dataset_name: Name of the dataset
        data_path: Path to data CSV file
        output_path: Path to save metadata JSON
        **kwargs: Additional arguments for specific generators
    
    Returns:
        Generated metadata dictionary
    """
    if dataset_type == 'discrete':
        generator = DiscreteDataMetadataGenerator(dataset_name, data_path)
    elif dataset_type == 'continuous':
        variable_names = kwargs.get('variable_names', None)
        generator = ContinuousDataMetadataGenerator(dataset_name, data_path, variable_names)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Generate metadata
    metadata = generator.generate_metadata()
    
    # Save metadata
    generator.save_metadata(metadata, output_path)
    
    return metadata


if __name__ == "__main__":
    # Example: Generate metadata for ALARM dataset
    print("Example: Generating metadata for ALARM dataset\n")
    
    metadata = create_metadata_for_dataset(
        dataset_type='discrete',
        dataset_name='ALARM',
        data_path='data/alarm/alarm_data_10000.csv',
        output_path='data/alarm/metadata.json'
    )
    
    print("\n" + "=" * 70)
    print("METADATA GENERATION COMPLETE")
    print("=" * 70)

