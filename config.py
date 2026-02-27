"""
Unified Configuration for Causal Discovery Pipeline

This single config file controls BOTH:
1. FCI/LLM algorithms (refactored/)
2. Neuro-Symbolic training (Neuro-Symbolic-Reasoning/)

Usage:
    - Edit this file to change any settings
    - Both modules will automatically use these settings
"""

from pathlib import Path
import os
import re
from typing import Optional, Dict

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
REFACTORED_DIR = PROJECT_ROOT / 'refactored'
NEURO_SYMBOLIC_DIR = PROJECT_ROOT / 'Neuro-Symbolic-Reasoning'

# ============================================================================
# DATASET SELECTION
# ============================================================================
# Options: 'alarm', 'insurance', 'sachs', 'child', 'hailfinder', 'win95pts', 'tuebingen_pair0001', etc.
# Allow environment override so batch runners can switch datasets without editing this file.
DATASET = os.environ.get('DATASET', 'alarm')

# ============================================================================
# SAMPLE SIZE SETTINGS (for scalable N-sweep experiments)
# ============================================================================
# When enabled, non-pigs/link datasets will try to resolve dataset_data_<N>.csv
# style files first, then fall back to legacy fixed paths.
ENABLE_SAMPLE_SIZE_SWEEP = True
SAMPLE_SIZE = 10000

# ============================================================================
# STEP 1: FCI ALGORITHM SETTINGS
# ============================================================================
# Independence test for FCI
# Options: 'chisq' (discrete data), 'fisherz' (continuous Gaussian), 'gsq'
FCI_INDEPENDENCE_TEST = 'chisq'

# Significance level for FCI
FCI_ALPHA = 0.05

# ============================================================================
# STEP 1b: RFCI ALGORITHM SETTINGS (for large graphs)
# ============================================================================
# NOTE: Your installed causal-learn package does not include RFCI, so RFCI is
# run via Java Tetrad (see refactored/main_rfci.py and refactored/third_party/tetrad/).
#
# We keep RFCI config separate so you can tune it for pigs/link without affecting FCI.
RFCI_ALPHA = FCI_ALPHA

# Depth for RFCI adjacency search (-1 = unlimited). Smaller values speed up.
RFCI_DEPTH = 2

# Max discriminating path length (-1 = unlimited). Smaller values speed up.
RFCI_MAX_DISC_PATH_LEN = 2

# Max rows used for RFCI only (None = use full dataset).
# Rationale: CI tests cost grows ~linearly with N, while downstream refinement can still use full N.
RFCI_MAX_ROWS = 20000

# Validation alpha (for LLM-based direction resolution)
VALIDATION_ALPHA = 0.01

# ============================================================================
# STEP 2: LLM SETTINGS (Optional)
# ============================================================================
# LLM Model Selection
# Options:
#   - None (no LLM, use FCI skeleton only)
#   - 'gpt-3.5-turbo' (GPT-3.5)
#   - 'gpt-4' (GPT-4)
#   - 'zephyr-7b' (Zephyr)
LLM_MODEL = 'gpt-3.5-turbo'  # Set to None for FCI-only pipeline (testing GSB framework)

# Use LLM prior for direction initialization in neural training?
USE_LLM_PRIOR = True  # LLM prior enabled - using semantic variable names

# LLM API settings
LLM_TEMPERATURE = 0.0  # 0.0 for deterministic results
LLM_MAX_TOKENS = 500  # Prevent overly long responses

# ============================================================================
# STEP 3: NEURAL TRAINING SETTINGS
# ============================================================================
# Training hyperparameters
LEARNING_RATE = 0.01
N_EPOCHS = 300  # Number of training epochs, andes 1500
N_HOPS = 1  # Number of reasoning hops (1 = single-hop)
BATCH_SIZE = None  # None = full batch

# Regularization
LAMBDA_GROUP_LASSO = 0.01  # Group lasso penalty weight
LAMBDA_CYCLE = 5    # Cycle consistency penalty weight
LAMBDA_SKELETON = 0.1    # Skeleton preservation penalty weight (NEW!)
# alarm is 0.01, 0.005, insurance is 0.001,0.05, different datasets have different configurations.
# Sachs for these two should be as much lower as possible, like 0 and 0.001
# child is 0.005, 0,.005
# hailfinder is 0.01 0.005
# andes is 0.0001 0.01


# Threshold for edge detection
THRESHOLD = 0.8
# Lower = more edges, Higher = fewer edges Sachs dataset should be lower, like 0.008
# child is 0.05
# hailfinder is 0.05
# andes is 0.008

# Logging
LOG_INTERVAL = 20  # Print training stats every N epochs
VERBOSE = True

# ============================================================================
# DATASET-SPECIFIC PATHS
# ============================================================================
DATASET_CONFIGS = {
    'alarm': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (37 columns), neural training needs one-hot data (105 columns)
        'fci_data_path': PROJECT_ROOT / 'alarm_data.csv',  # Variable-level data for FCI (37 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'alarm' / 'alarm_data_10000.csv',  # One-hot data for neural training (105 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'alarm' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': PROJECT_ROOT / 'alarm.bif',
        'ground_truth_type': 'bif',  # Type: 'bif', 'json', or None
        
        # Data type
        'data_type': 'discrete',  # 'discrete' or 'continuous'
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'insurance': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (27 columns), neural training needs one-hot data (88 columns)
        'fci_data_path': PROJECT_ROOT / 'insurance_data.csv',  # Variable-level data for FCI (27 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'insurance' / 'insurance_data_10000.csv',  # One-hot data for neural training (88 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'insurance' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'insurance' / 'insurance.bif',
        'ground_truth_type': 'bif',  # Type: 'bif', 'json', or None
        
        # Data type
        'data_type': 'discrete',  # 'discrete' or 'continuous'
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'tuebingen_pair0001': {
        # Data files (generated by generate_tuebingen_data.py)
        'fci_data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'pair0001_data_variable.csv',  # For FCI (2 columns)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'pair0001_data.csv',  # For neural network (10 columns one-hot)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'metadata.json',
        
        # Ground truth
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'pair0001_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        
        # Data type (now discretized via quantile binning)
        'data_type': 'discrete',  # Discretized from continuous
        
        # FCI/LLM outputs (generated with semantic variable names)
        'fci_skeleton_path': PROJECT_ROOT / 'refactored' / 'outputs' / 'tuebingen_pair0001' / 'edges_FCI_20251226_012031.csv',
        'llm_direction_path': PROJECT_ROOT / 'refactored' / 'outputs' / 'tuebingen_pair0001' / 'edges_FCI_LLM_GPT35_20251226_012044.csv',
    },
    
    # Legacy alias for backward compatibility
    'tuebingen_pair1': {
        # Data files (generated by generate_tuebingen_data.py)
        'fci_data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'pair0001_data_variable.csv',  # For FCI (2 columns)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'pair0001_data.csv',  # For neural network (10 columns one-hot)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'metadata.json',
        
        # Ground truth
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0001' / 'pair0001_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        
        # Data type (now discretized via quantile binning)
        'data_type': 'discrete',  # Discretized from continuous
        
        # FCI/LLM outputs (generated with semantic variable names)
        'fci_skeleton_path': PROJECT_ROOT / 'refactored' / 'outputs' / 'tuebingen_pair0001' / 'edges_FCI_20251226_012031.csv',
        'llm_direction_path': PROJECT_ROOT / 'refactored' / 'outputs' / 'tuebingen_pair0001' / 'edges_FCI_LLM_GPT35_20251226_012044.csv',
    },
    
    # Additional Tuebingen pairs (add more as needed)
    'tuebingen_pair0002': {
        'fci_data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0002' / 'pair0002_data_variable.csv',
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0002' / 'pair0002_data.csv',
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0002' / 'metadata.json',
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0002' / 'pair0002_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        'data_type': 'discrete',
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    
    'tuebingen_pair0003': {
        'fci_data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0003' / 'pair0003_data_variable.csv',
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0003' / 'pair0003_data.csv',
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0003' / 'metadata.json',
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0003' / 'pair0003_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        'data_type': 'discrete',
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    
    'tuebingen_pair0004': {
        'fci_data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0004' / 'pair0004_data_variable.csv',
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0004' / 'pair0004_data.csv',
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0004' / 'metadata.json',
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0004' / 'pair0004_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        'data_type': 'discrete',
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    
    'tuebingen_pair0005': {
        'fci_data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0005' / 'pair0005_data_variable.csv',
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0005' / 'pair0005_data.csv',
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0005' / 'metadata.json',
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / 'pair0005' / 'pair0005_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        'data_type': 'discrete',
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    
    'sachs': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (11 columns), neural training needs one-hot data (~33 columns)
        'fci_data_path': PROJECT_ROOT / 'sachs_data_variable.csv',  # Variable-level data for FCI (11 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'sachs' / 'sachs_data.csv',  # One-hot data for neural training (~33 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'sachs' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'sachs' / 'sachs_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Pre-discretized from bnlearn (3 states per variable)
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'child': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (20 columns), neural training needs one-hot data
        'fci_data_path': PROJECT_ROOT / 'child_data_variable.csv',  # Variable-level data for FCI (20 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'child' / 'child_data.csv',  # One-hot data for neural training
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'child' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'child' / 'child_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Discrete medical diagnosis variables
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'hailfinder': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (56 columns), neural training needs one-hot data (223 columns)
        'fci_data_path': PROJECT_ROOT / 'hailfinder_data_variable.csv',  # Variable-level data for FCI (56 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'hailfinder' / 'hailfinder_data.csv',  # One-hot data for neural training (223 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'hailfinder' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'hailfinder' / 'hailfinder_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Discrete meteorological variables
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'win95pts': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (76 columns), neural training needs one-hot data (151 columns)
        'fci_data_path': PROJECT_ROOT / 'win95pts_data_variable.csv',  # Variable-level data for FCI (76 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'win95pts' / 'win95pts_data.csv',  # One-hot data for neural training (151 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'win95pts' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'win95pts' / 'win95pts_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Discrete troubleshooting variables
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },
    
    'andes': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (223 columns), neural training needs one-hot data (446 columns)
        'fci_data_path': PROJECT_ROOT / 'andes_data_variable.csv',  # Variable-level data for FCI (223 vars)
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'andes' / 'andes_data.csv',  # One-hot data for neural training (446 states)
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'andes' / 'metadata.json',
        
        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'andes' / 'andes_ground_truth.txt',
        'ground_truth_type': 'edge_list',  # Type: 'bif', 'json', 'edge_list', or None
        
        # Data type
        'data_type': 'discrete',  # Discrete tutoring variables
        
        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,  # Will be auto-detected
        'llm_direction_path': None,  # Will be auto-detected
    },

    'pigs': {
        # Data files
        # IMPORTANT: FCI needs variable-level data (N vars columns), neural training needs one-hot data (sum(states) columns)
        'fci_data_path': PROJECT_ROOT / 'pigs_data_variable.csv',  # Variable-level data for FCI
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'pigs' / 'pigs_data_50000.csv',  # One-hot data for neural training
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'pigs' / 'metadata.json',

        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'pigs' / 'pigs.bif',
        'ground_truth_type': 'bif',

        # Data type
        'data_type': 'discrete',

        # Constraint-based discovery default (FCI is too slow for pigs-sized graphs)
        # Options: 'fci' | 'rfci'
        'constraint_algo': 'rfci',

        # RFCI tuning (optional per-dataset overrides; falls back to global RFCI_* settings)
        'rfci_depth': 2,
        'rfci_max_disc_path_len': 2,
        'rfci_max_rows': 20000,

        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },

    'link': {
        # Data files
        # IMPORTANT: RFCI/FCI needs variable-level data, neural training needs one-hot data
        'fci_data_path': PROJECT_ROOT / 'link_data_variable.csv',  # Variable-level data for RFCI/FCI
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'link' / 'link_data_50000.csv',  # One-hot for neural training
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'link' / 'metadata.json',

        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'link' / 'link.bif',
        'ground_truth_type': 'bif',

        # Data type
        'data_type': 'discrete',

        # Constraint-based discovery default (link is large; prefer RFCI)
        'constraint_algo': 'rfci',

        # RFCI tuning (optional per-dataset overrides; falls back to global RFCI_* settings)
        'rfci_depth': 2,
        'rfci_max_disc_path_len': 2,
        'rfci_max_rows': 20000,

        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },

    'hepar2': {
        # Data files
        'fci_data_path': PROJECT_ROOT / 'hepar2_data_variable.csv',  # Variable-level data for FCI/RFCI
        'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'HEPAR2' / 'hepar2_data_10000.csv',  # One-hot for training
        'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'HEPAR2' / 'metadata.json',

        # Ground truth (for evaluation)
        'ground_truth_path': NEURO_SYMBOLIC_DIR / 'data' / 'HEPAR2' / 'hepar2.bif',
        'ground_truth_type': 'bif',

        # Data type
        'data_type': 'discrete',

        # Default constraint algo (HEPAR2 is moderate-size; can switch to 'rfci' if needed)
        'constraint_algo': 'fci',

        # FCI/LLM outputs (auto-detected, leave as None)
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    },
    
    # Add more datasets here...
}


def _effective_sample_size() -> Optional[int]:
    """Resolve sample size from env override first, then global config."""
    env_size = os.environ.get("SAMPLE_SIZE")
    if env_size is not None:
        try:
            return int(env_size)
        except ValueError:
            print(f"[WARN] Ignoring invalid SAMPLE_SIZE env var: {env_size}")
    return int(SAMPLE_SIZE) if SAMPLE_SIZE is not None else None


def _dataset_uses_sample_sweep(dataset_name: str) -> bool:
    """Whether sample-size-based path resolution should be applied."""
    ds = str(dataset_name).lower()
    if ds in {"pigs", "link"}:
        return False
    if ds == "tuebingen" or ds.startswith("tuebingen_pair"):
        return False
    return bool(ENABLE_SAMPLE_SIZE_SWEEP)


def _first_existing(paths):
    """Return the first existing Path in paths, otherwise None."""
    for p in paths:
        if p.exists():
            return p
    return None


def _replace_trailing_numeric_suffix(path: Path, sample_size: int) -> Path:
    """
    Replace a trailing _<digits> in filename stem with _<sample_size>.
    If no trailing numeric suffix exists, append one.
    """
    stem = path.stem
    if re.search(r'_\d+$', stem):
        stem = re.sub(r'_\d+$', f'_{sample_size}', stem)
    else:
        stem = f"{stem}_{sample_size}"
    return path.with_name(f"{stem}{path.suffix}")


def resolve_dataset_paths(dataset_name: str, dataset_cfg: Dict, sample_size: Optional[int] = None) -> Dict:
    """
    Resolve data/fci/metadata paths for a dataset, with optional sample-size awareness.

    Behavior:
      - pigs/link/tuebingen*: keep legacy paths unchanged.
      - other datasets: prefer sample-size-specific files if present; fallback to legacy paths.
    """
    resolved = dict(dataset_cfg)

    if not _dataset_uses_sample_sweep(dataset_name):
        return resolved

    n = _effective_sample_size() if sample_size is None else int(sample_size)
    if n is None:
        return resolved

    ds = str(dataset_name)
    data_path = Path(dataset_cfg['data_path'])
    fci_data_path = Path(dataset_cfg['fci_data_path'])
    metadata_path = Path(dataset_cfg['metadata_path'])

    # One-hot training data
    data_candidates = [
        data_path.parent / f"{ds}_data_{n}{data_path.suffix}",
        _replace_trailing_numeric_suffix(data_path, n),
    ]
    data_hit = _first_existing(data_candidates)
    resolved['data_path'] = data_hit if data_hit is not None else data_path

    # Variable-level data for FCI/RFCI
    fci_candidates = [
        fci_data_path.parent / f"{ds}_data_variable_{n}{fci_data_path.suffix}",
        _replace_trailing_numeric_suffix(fci_data_path, n),
    ]
    fci_hit = _first_existing(fci_candidates)
    resolved['fci_data_path'] = fci_hit if fci_hit is not None else fci_data_path

    # Optional per-size metadata support
    meta_candidates = [
        metadata_path.parent / f"metadata_{n}{metadata_path.suffix}",
        metadata_path,
    ]
    meta_hit = _first_existing(meta_candidates)
    resolved['metadata_path'] = meta_hit if meta_hit is not None else metadata_path
    resolved['sample_size'] = n

    return resolved

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
# FCI/LLM outputs
FCI_OUTPUT_DIR = REFACTORED_DIR / 'outputs'

# Neural training results
TRAINING_RESULTS_DIR = NEURO_SYMBOLIC_DIR / 'results'

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
# Random seed
RANDOM_SEED = 42

# Device
DEVICE = 'cpu'  # 'cuda' or 'cpu'

# Early stopping
EARLY_STOPPING = False
PATIENCE = 50

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_current_dataset_config():
    """Get configuration for current dataset"""
    # Special handling for batch mode
    if DATASET == 'tuebingen':
        # In batch mode, return a dummy config (actual configs will be generated per-pair)
        return {
            'data_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen',
            'metadata_path': NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen',
            'ground_truth_path': None,
            'ground_truth_type': 'edge_list',
            'data_type': 'discrete',
            'fci_skeleton_path': None,
            'llm_direction_path': None,
        }
    
    # For specific Tuebingen pairs, generate config dynamically
    if DATASET.startswith('tuebingen_pair'):
        pair_id = DATASET.replace('tuebingen_', '')
        return get_tuebingen_pair_config(pair_id)
    
    base_cfg = DATASET_CONFIGS.get(DATASET, DATASET_CONFIGS['alarm'])
    return resolve_dataset_paths(DATASET, base_cfg)


def get_fci_config():
    """Get FCI-specific configuration"""
    dataset_cfg = get_current_dataset_config()
    
    # Auto-select independence test based on data type
    default_test = 'fisherz' if dataset_cfg['data_type'] == 'continuous' else 'chisq'
    
    return {
        'dataset': DATASET,
        'sample_size': dataset_cfg.get('sample_size'),
        'independence_test': default_test,  # Auto-selected based on data_type
        'alpha': FCI_ALPHA,
        'validation_alpha': VALIDATION_ALPHA,
        'output_dir': str(FCI_OUTPUT_DIR),
        'ground_truth_path': str(dataset_cfg['ground_truth_path']),
        'ground_truth_type': dataset_cfg.get('ground_truth_type', 'bif'),
        'llm_temperature': LLM_TEMPERATURE,
        'llm_max_tokens': LLM_MAX_TOKENS,
    }


def get_training_config():
    """Get training-specific configuration"""
    dataset_cfg = get_current_dataset_config()
    
    # Allow environment variable override for batch processing
    fci_skeleton_override = os.environ.get('FCI_SKELETON_PATH')
    llm_direction_override = os.environ.get('LLM_DIRECTION_PATH')

    # Which constraint-based algorithm produced the skeleton?
    # Default remains FCI, but large graphs (e.g., pigs/link) can use RFCI.
    constraint_algo = dataset_cfg.get("constraint_algo", "fci").lower()
    if constraint_algo not in {"fci", "rfci"}:
        constraint_algo = "fci"

    if constraint_algo == "rfci":
        skeleton_patterns = ["edges_RFCI_*.csv", "edges_FCI_*.csv"]
        llm_patterns = ["edges_RFCI_LLM_*.csv", "edges_FCI_LLM_*.csv"]
    else:
        skeleton_patterns = ["edges_FCI_*.csv", "edges_RFCI_*.csv"]
        llm_patterns = ["edges_FCI_LLM_*.csv", "edges_RFCI_LLM_*.csv"]
    
    return {
        # Dataset
        'dataset_name': DATASET,
        'sample_size': dataset_cfg.get('sample_size'),
        'data_path': str(dataset_cfg['data_path']),
        'metadata_path': str(dataset_cfg['metadata_path']),
        'ground_truth_path': str(dataset_cfg['ground_truth_path']) if dataset_cfg['ground_truth_path'] else None,
        'ground_truth_type': dataset_cfg.get('ground_truth_type', 'bif'),
        'data_type': dataset_cfg.get('data_type', 'discrete'),
        
        # Constraint skeleton + optional LLM directions.
        # Can be overridden by environment variables for batch processing.
        'fci_skeleton_path': (
            fci_skeleton_override
            if fci_skeleton_override
            else _auto_detect_latest_file_any(skeleton_patterns, FCI_OUTPUT_DIR / DATASET)
        ),
        'llm_direction_path': (
            llm_direction_override
            if llm_direction_override
            else (_auto_detect_latest_file_any(llm_patterns, FCI_OUTPUT_DIR / DATASET) if USE_LLM_PRIOR else None)
        ),
        
        # LLM settings
        'llm_model': LLM_MODEL,
        'use_llm_prior': USE_LLM_PRIOR,
        'llm_temperature': LLM_TEMPERATURE,
        'llm_max_tokens': LLM_MAX_TOKENS,
        
        # Hyperparameters
        'learning_rate': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'n_hops': N_HOPS,
        'batch_size': BATCH_SIZE,
        'lambda_group_lasso': LAMBDA_GROUP_LASSO,
        'lambda_cycle': LAMBDA_CYCLE,
        'lambda_skeleton': LAMBDA_SKELETON,  # NEW: Skeleton preservation
        'threshold': THRESHOLD,
        
        # Output
        'results_dir': str(TRAINING_RESULTS_DIR),
        'verbose': VERBOSE,
        'log_interval': LOG_INTERVAL,
        
        # Advanced
        'random_seed': RANDOM_SEED,
        'device': DEVICE,
        'early_stopping': EARLY_STOPPING,
        'patience': PATIENCE,
    }


def _auto_detect_latest_file(pattern, directory):
    """Auto-detect the latest file matching pattern"""
    from pathlib import Path
    
    data_dir = Path(directory)
    if not data_dir.exists():
        return None
    
    files = list(data_dir.glob(pattern))
    if not files:
        return None
    
    # Return the most recent file
    latest_file = max(files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)


def _auto_detect_latest_file_any(patterns, directory):
    """
    Auto-detect the latest file matching ANY of the provided glob patterns.

    Patterns are tried in order; the first pattern with >=1 match wins (and returns its latest file).
    This is used to allow RFCI/FCI compatibility without changing downstream code.
    """
    for pat in patterns:
        hit = _auto_detect_latest_file(pat, directory)
        if hit:
            return hit
    return None


def get_constraint_output_dir(dataset_name: Optional[str] = None, sample_size: Optional[int] = None) -> Path:
    """
    Get constraint-discovery outputs directory (FCI/RFCI/LLM).

    For sample-sweep datasets, use:
      refactored/outputs/<dataset>/n_<sample_size>/
    Otherwise keep legacy:
      refactored/outputs/<dataset>/
    """
    ds = str(dataset_name or DATASET)
    base = FCI_OUTPUT_DIR / ds
    if not _dataset_uses_sample_sweep(ds):
        return base
    n = _effective_sample_size() if sample_size is None else int(sample_size)
    if n is None:
        return base
    return base / f"n_{int(n)}"


def print_config():
    """Print current configuration"""
    print("\n" + "=" * 80)
    print("UNIFIED CONFIGURATION")
    print("=" * 80)
    
    print(f"\nDataset: {DATASET}")
    if _dataset_uses_sample_sweep(DATASET):
        print(f"Sample Size: {_effective_sample_size()} (ENABLE_SAMPLE_SIZE_SWEEP={ENABLE_SAMPLE_SIZE_SWEEP})")
    else:
        print("Sample Size: fixed by dataset defaults (sample sweep disabled for this dataset)")
    print(f"Ground Truth: {get_current_dataset_config()['ground_truth_path']}")
    
    print(f"\n--- STEP 1: FCI Algorithm ---")
    print(f"  Independence Test: {FCI_INDEPENDENCE_TEST}")
    print(f"  Alpha: {FCI_ALPHA}")
    print(f"  Output: {FCI_OUTPUT_DIR}")
    
    print(f"\n--- STEP 2: LLM (Optional) ---")
    print(f"  Model: {LLM_MODEL if LLM_MODEL else 'None (FCI only)'}")
    print(f"  Use LLM Prior: {USE_LLM_PRIOR}")
    
    print(f"\n--- STEP 3: Neural Training ---")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Lambda Lasso: {LAMBDA_GROUP_LASSO}")
    print(f"  Lambda Cycle: {LAMBDA_CYCLE}")
    print(f"  Lambda Skeleton: {LAMBDA_SKELETON}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Results: {TRAINING_RESULTS_DIR}")
    
    print("\n" + "=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration"""
    dataset_cfg = get_current_dataset_config()
    
    # Check if data files exist
    if not dataset_cfg['data_path'].exists():
        raise FileNotFoundError(f"Data file not found: {dataset_cfg['data_path']}")
    
    if not dataset_cfg['metadata_path'].exists():
        raise FileNotFoundError(f"Metadata file not found: {dataset_cfg['metadata_path']}")
    
    # Check ground truth (optional)
    if dataset_cfg['ground_truth_path'] and not Path(dataset_cfg['ground_truth_path']).exists():
        print(f"[WARN] Ground truth file not found: {dataset_cfg['ground_truth_path']}")
    
    # Check LLM prior requirement
    if USE_LLM_PRIOR and not LLM_MODEL:
        raise ValueError("USE_LLM_PRIOR is True but LLM_MODEL is None")
    
    print("[OK] Configuration validated")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_tuebingen_pair_config(pair_number):
    """
    Get configuration for a Tuebingen pair dynamically
    
    Args:
        pair_number: Pair number (1-108) or formatted string ('pair0001')
    
    Returns:
        Configuration dict for the pair
    
    Example:
        >>> config = get_tuebingen_pair_config(1)
        >>> config = get_tuebingen_pair_config('pair0001')
    """
    # Parse pair number
    if isinstance(pair_number, str):
        if pair_number.startswith('pair'):
            pair_id = pair_number
        else:
            pair_id = f"pair{int(pair_number):04d}"
    else:
        pair_id = f"pair{int(pair_number):04d}"
    
    # Check if already in DATASET_CONFIGS
    dataset_name = f"tuebingen_{pair_id}"
    if dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]
    
    # Generate config dynamically
    pair_dir = NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / pair_id
    
    config = {
        'fci_data_path': pair_dir / f'{pair_id}_data_variable.csv',  # For FCI (2 columns)
        'data_path': pair_dir / f'{pair_id}_data.csv',  # For neural network (10 columns one-hot)
        'metadata_path': pair_dir / 'metadata.json',
        'ground_truth_path': pair_dir / f'{pair_id}_ground_truth.txt',
        'ground_truth_type': 'edge_list',
        'data_type': 'discrete',
        'fci_skeleton_path': None,
        'llm_direction_path': None,
    }
    
    return config


def use_tuebingen_pair(pair_number):
    """
    Switch to a Tuebingen pair dataset
    
    Args:
        pair_number: Pair number (1-108) or formatted string ('pair0001')
    
    Example:
        >>> use_tuebingen_pair(1)  # Use pair0001
        >>> use_tuebingen_pair('pair0050')  # Use pair0050
    """
    global DATASET
    
    # Parse pair number
    if isinstance(pair_number, str):
        if pair_number.startswith('pair'):
            pair_id = pair_number
        else:
            pair_id = f"pair{int(pair_number):04d}"
    else:
        pair_id = f"pair{int(pair_number):04d}"
    
    dataset_name = f"tuebingen_{pair_id}"
    
    # Add to DATASET_CONFIGS if not present
    if dataset_name not in DATASET_CONFIGS:
        DATASET_CONFIGS[dataset_name] = get_tuebingen_pair_config(pair_number)
    
    DATASET = dataset_name
    print(f"[CONFIG] Switched to {dataset_name}")
    
    return DATASET


def get_all_tuebingen_pairs(start=1, end=108):
    """
    Get list of all Tuebingen pair numbers
    
    Args:
        start: Starting pair number (default: 1)
        end: Ending pair number (default: 108)
    
    Returns:
        List of pair numbers that have data files
    
    Example:
        >>> pairs = get_all_tuebingen_pairs()
        >>> pairs = get_all_tuebingen_pairs(start=1, end=20)
    """
    available_pairs = []
    
    for i in range(start, end + 1):
        pair_id = f"pair{i:04d}"
        pair_dir = NEURO_SYMBOLIC_DIR / 'data' / 'tuebingen' / pair_id
        
        # Check if pair data exists
        data_file = pair_dir / f'{pair_id}_data.csv'
        if data_file.exists():
            available_pairs.append(i)
    
    return available_pairs


def is_batch_mode():
    """Check if we're in batch mode (DATASET == 'tuebingen')"""
    return DATASET == 'tuebingen'


# ============================================================================
# QUICK PRESETS
# ============================================================================
def use_fci_only():
    """Preset: FCI only (no LLM)"""
    global LLM_MODEL, USE_LLM_PRIOR
    LLM_MODEL = None
    USE_LLM_PRIOR = False
    print("[CONFIG] Using FCI only (no LLM)")


def use_fci_gpt35():
    """Preset: FCI + GPT-3.5"""
    global LLM_MODEL, USE_LLM_PRIOR
    LLM_MODEL = 'gpt-3.5-turbo'
    USE_LLM_PRIOR = True
    print("[CONFIG] Using FCI + GPT-3.5")


def use_fci_zephyr():
    """Preset: FCI + Zephyr"""
    global LLM_MODEL, USE_LLM_PRIOR
    LLM_MODEL = 'zephyr-7b'
    USE_LLM_PRIOR = True
    print("[CONFIG] Using FCI + Zephyr")


# ============================================================================
# MAIN (for testing)
# ============================================================================
if __name__ == "__main__":
    print_config()
    
    print("\n" + "=" * 80)
    print("TESTING PRESETS")
    print("=" * 80)
    
    print("\n1. FCI Only:")
    use_fci_only()
    print(f"   LLM Model: {LLM_MODEL}, Use LLM Prior: {USE_LLM_PRIOR}")
    
    print("\n2. FCI + GPT-3.5:")
    use_fci_gpt35()
    print(f"   LLM Model: {LLM_MODEL}, Use LLM Prior: {USE_LLM_PRIOR}")
    
    print("\n3. FCI + Zephyr:")
    use_fci_zephyr()
    print(f"   LLM Model: {LLM_MODEL}, Use LLM Prior: {USE_LLM_PRIOR}")
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    try:
        validate_config()
    except Exception as e:
        print(f"[ERROR] {e}")

