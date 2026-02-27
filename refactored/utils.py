"""
Utility functions for causal discovery experiments
"""

from config import DATASET, DATASET_CONFIGS, get_current_dataset_config
from modules.data_loader import DataLoader, LUCASDataLoader, ALARMDataLoader


def get_active_data_loader():
    """
    Get data loader for the active dataset specified in config.py
    
    Returns:
    --------
    data_loader : DataLoader instance
        Configured data loader for the active dataset
    """
    dataset_config = get_current_dataset_config()
    
    # IMPORTANT: FCI needs variable-level data, NOT one-hot encoded data
    # Use 'fci_data_path' if available, otherwise fall back to 'data_path'
    data_path = dataset_config.get('fci_data_path', dataset_config['data_path'])
    
    # Determine loader type based on dataset name
    if DATASET.lower() == 'lucas':
        return LUCASDataLoader(data_path)
    elif DATASET.lower() == 'alarm':
        return ALARMDataLoader(data_path)
    else:
        return DataLoader(data_path, dataset_name=DATASET.upper())


def print_dataset_info():
    """Print information about the active dataset"""
    dataset_config = get_current_dataset_config()
    
    print("\n" + "=" * 60)
    print(f"Active Dataset: {DATASET.upper()}")
    print(f"Data Path: {dataset_config['data_path']}")
    print(f"Data Type: {dataset_config.get('data_type', 'unknown')}")
    print(f"Ground Truth: {dataset_config.get('ground_truth_path', 'N/A')}")
    print(f"To change dataset, edit config.py at project root")
    print("=" * 60 + "\n")

