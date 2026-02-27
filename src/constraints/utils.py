"""
Utility helpers for constraint-discovery scripts.
"""

from src.constraints.modules.data_loader import DataLoader, LUCASDataLoader, ALARMDataLoader
import config


def get_active_data_loader():
    """Return a dataset-aware loader for the currently selected dataset."""
    dataset_config = config.get_current_dataset_config()

    # FCI/RFCI expects variable-level input if provided.
    data_path = dataset_config.get("fci_data_path", dataset_config["data_path"])
    dataset_name = str(config.DATASET).lower()

    if dataset_name == "lucas":
        return LUCASDataLoader(data_path)
    if dataset_name == "alarm":
        return ALARMDataLoader(data_path)
    return DataLoader(data_path, dataset_name=config.DATASET.upper())


def print_dataset_info():
    """Print active dataset and key paths."""
    dataset_config = config.get_current_dataset_config()
    print("\n" + "=" * 60)
    print(f"Active Dataset: {str(config.DATASET).upper()}")
    print(f"Data Path (training): {dataset_config['data_path']}")
    print(f"Data Path (constraint): {dataset_config.get('fci_data_path', dataset_config['data_path'])}")
    print(f"Ground Truth: {dataset_config.get('ground_truth_path', 'N/A')}")
    print("=" * 60 + "\n")
