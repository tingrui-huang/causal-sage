"""
FCI/LLM Configuration (refactored/)

This file now imports from the unified config at project root.
All settings are managed in ../config.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from unified config (use absolute import to avoid circular import)
import importlib.util
spec = importlib.util.spec_from_file_location("unified_config", PROJECT_ROOT / "config.py")
unified_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified_config)

# Extract needed variables
DATASET = unified_config.DATASET
DATASET_CONFIGS = unified_config.DATASET_CONFIGS
get_current_dataset_config = unified_config.get_current_dataset_config
FCI_INDEPENDENCE_TEST = unified_config.FCI_INDEPENDENCE_TEST
FCI_ALPHA = unified_config.FCI_ALPHA
RFCI_ALPHA = getattr(unified_config, "RFCI_ALPHA", FCI_ALPHA)
RFCI_DEPTH = getattr(unified_config, "RFCI_DEPTH", -1)
RFCI_MAX_DISC_PATH_LEN = getattr(unified_config, "RFCI_MAX_DISC_PATH_LEN", -1)
RFCI_MAX_ROWS = getattr(unified_config, "RFCI_MAX_ROWS", None)
VALIDATION_ALPHA = unified_config.VALIDATION_ALPHA
LLM_MODEL = unified_config.LLM_MODEL
LLM_TEMPERATURE = getattr(unified_config, "LLM_TEMPERATURE", 0.0)
LLM_MAX_TOKENS = getattr(unified_config, "LLM_MAX_TOKENS", 500)
FCI_OUTPUT_DIR = unified_config.FCI_OUTPUT_DIR
get_fci_config = unified_config.get_fci_config
print_unified_config = unified_config.print_config
VERBOSE = getattr(unified_config, "VERBOSE", False)
RANDOM_SEED = getattr(unified_config, "RANDOM_SEED", 0)

# Per-dataset overrides (if provided in DATASET_CONFIGS[DATASET])
_ds_cfg = get_current_dataset_config()
RFCI_DEPTH = int(_ds_cfg.get("rfci_depth", RFCI_DEPTH)) if RFCI_DEPTH is not None else -1
RFCI_MAX_DISC_PATH_LEN = int(_ds_cfg.get("rfci_max_disc_path_len", RFCI_MAX_DISC_PATH_LEN)) if RFCI_MAX_DISC_PATH_LEN is not None else -1
RFCI_MAX_ROWS = _ds_cfg.get("rfci_max_rows", RFCI_MAX_ROWS)

# Legacy imports (kept for backward compatibility)
from_config = (
)

# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================
# These are kept for backward compatibility with existing scripts

# Active dataset
ACTIVE_DATASET = DATASET

# Output directory
OUTPUT_DIR = str(unified_config.get_constraint_output_dir(DATASET))

# Ground truth path
GROUND_TRUTH_PATH = str(get_current_dataset_config()['ground_truth_path'])

# Neuro-Symbolic data directory
NEURO_SYMBOLIC_DATA_DIR = str(get_current_dataset_config()['data_path'].parent)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_output_dir():
    """Get outputs directory for current dataset"""
    # Compute dynamically to respect DATASET/SAMPLE_SIZE environment overrides.
    return str(unified_config.get_constraint_output_dir(DATASET))


def print_config():
    """Print FCI configuration"""
    print("\n" + "=" * 80)
    print("FCI/LLM CONFIGURATION")
    print("=" * 80)
    print(f"\nDataset: {DATASET}")
    print(f"Independence Test: {FCI_INDEPENDENCE_TEST}")
    print(f"Alpha: {FCI_ALPHA}")
    print(f"Validation Alpha: {VALIDATION_ALPHA}")
    print(f"LLM Model: {LLM_MODEL if LLM_MODEL else 'None (FCI only)'}")
    print(f"Output Dir: {get_output_dir()}")
    print(f"Ground Truth: {GROUND_TRUTH_PATH}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print_config()
    print("\n[INFO] All settings are managed in ../config.py")
    print("[INFO] Edit ../config.py to change any settings")
