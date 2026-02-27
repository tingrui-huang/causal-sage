"""
Export FCI/LLM Results to Neuro-Symbolic-Reasoning

Automatically copies the latest FCI and LLM results to the
Neuro-Symbolic-Reasoning data folder with version control.
"""

import shutil
from pathlib import Path
from datetime import datetime
from config import OUTPUT_DIR, NEURO_SYMBOLIC_DATA_DIR


def find_latest_file(pattern, directory=OUTPUT_DIR):
    """Find the most recent file matching pattern"""
    from config import DATASET
    
    output_path = Path(directory)
    
    if not output_path.exists():
        return None
    
    def _pick(files):
        # Avoid accidentally exporting debug/smoke artifacts
        files = [p for p in files if "smoke" not in p.name.lower() and "tmp" not in p.name.lower()]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    # Try dataset-specific directory first (e.g., outputs/alarm/)
    dataset_dir = output_path / DATASET
    if dataset_dir.exists():
        files = list(dataset_dir.glob(pattern))
        if files:
            picked = _pick(files)
            if picked:
                return picked
    
    # Fall back to root outputs directory
    files = list(output_path.glob(pattern))
    
    if not files:
        return None
    
    picked = _pick(files)
    return picked


def export_fci():
    """Export constraint skeleton (FCI or RFCI)"""
    print("\n" + "=" * 80)
    print("EXPORTING FCI SKELETON")
    print("=" * 80)
    
    # Find latest constraint CSV (prefer RFCI if present)
    latest_fci = find_latest_file('edges_RFCI_*.csv') or find_latest_file('edges_FCI_*.csv')
    
    if not latest_fci:
        print("[WARN] No FCI/RFCI CSV found. Skipping skeleton export.")
        return None
    
    print(f"Found: {latest_fci.name}")
    
    # Destination
    neuro_data_dir = Path(NEURO_SYMBOLIC_DATA_DIR)
    neuro_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy with timestamp preservation
    dest_path = neuro_data_dir / latest_fci.name
    shutil.copy2(latest_fci, dest_path)
    
    print(f"[OK] Exported to: {dest_path}")
    return dest_path


def export_llm(llm_name='GPT35'):
    """
    Export FCI+LLM results
    
    Args:
        llm_name: Name of LLM (e.g., 'GPT35', 'Zephyr')
    """
    print("\n" + "=" * 80)
    print(f"EXPORTING FCI + {llm_name}")
    print("=" * 80)
    
    # Find latest LLM CSV
    # Prefer RFCI_LLM if present (for large graphs like link/pigs), otherwise FCI_LLM.
    patterns = [
        f'edges_RFCI_LLM_{llm_name}_*.csv',
        f'edges_FCI_LLM_{llm_name}_*.csv',
    ]
    latest_llm = None
    used_pattern = None
    for pat in patterns:
        hit = find_latest_file(pat)
        if hit:
            latest_llm = hit
            used_pattern = pat
            break
    
    if not latest_llm:
        print(f"[WARN] No {llm_name} CSV found (patterns: {patterns}). Skipping {llm_name} export.")
        return None
    
    print(f"Found: {latest_llm.name}")
    if used_pattern:
        print(f"  (matched: {used_pattern})")
    
    # Destination
    neuro_data_dir = Path(NEURO_SYMBOLIC_DATA_DIR)
    neuro_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy with timestamp preservation
    dest_path = neuro_data_dir / latest_llm.name
    shutil.copy2(latest_llm, dest_path)
    
    print(f"[OK] Exported to: {dest_path}")
    return dest_path


def export_all():
    """Export all available results"""
    print("\n" + "=" * 80)
    print("EXPORTING ALL RESULTS TO NEURO-SYMBOLIC-REASONING")
    print("=" * 80)
    
    exported = {}
    
    # Export FCI
    fci_path = export_fci()
    if fci_path:
        exported['fci'] = str(fci_path)
    
    # Export GPT-3.5 (if available)
    gpt35_path = export_llm('GPT35')
    if gpt35_path:
        exported['gpt35'] = str(gpt35_path)
    
    # Export Zephyr (if available)
    zephyr_path = export_llm('Zephyr')
    if zephyr_path:
        exported['zephyr'] = str(zephyr_path)
    
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    
    if exported:
        print(f"Exported {len(exported)} file(s):")
        for key, path in exported.items():
            print(f"  - {key.upper()}: {Path(path).name}")
    else:
        print("No files exported. Run FCI/LLM algorithms first.")
    
    print("=" * 80)
    
    return exported


if __name__ == "__main__":
    export_all()
