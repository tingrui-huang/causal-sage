# Anonymous Reproducibility Package

This repository is an anonymous reproducibility package for a UAI submission.
It provides the ALARM pipeline used in the paper and reproduces the reported results with frozen inputs.

## Scope

- Dataset: `ALARM`,`insurance`
- Main entrypoint: `Neuro-Symbolic-Reasoning/experiment_llm_vs_random.py`
- Goal: reproduce the paper metrics (not re-tune for better new performance)

## Environment

- OS: Linux/macOS/Windows
- Python: `>=3.10` (validated with `3.13.1` in this package)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Repository Modes

### 1) Paper-Repro Mode (default, used for review)

Use the frozen files already shipped in this repository:
- one-hot training CSV
- metadata JSON
- FCI skeleton CSV
- pre-generated LLM skeleton CSV (`edges_FCI_LLM_*.csv`)

This mode is expected to match paper numbers.
No online LLM call is required for LLM-prior runs in this package.

### 2) Regenerate Mode (optional)

You may regenerate data/constraints via provided scripts.
Regenerated artifacts can lead to different metrics and are not used for strict paper-number reproduction.

## Quick Start (Paper-Repro)

### Step 1: Check config

Open `config.py` and ensure:
- `DATASET = 'alarm'`
- paper-aligned settings are kept unchanged

### Step 2: Run experiment (paper-aligned command)

```bash
cd Neuro-Symbolic-Reasoning
python experiment_llm_vs_random.py --datasets alarm --run_mode random --seeds 5 --sample_size 10000 --epochs 140 --high_conf 0.9 --low_conf 0.1 --reconstruction_mode group_ce --vstructure_in_mask --dag_check --dag_project_on_cycle
```

### Step 2b: Run with pre-generated LLM prior (offline)

LLM-prior mode in this repository uses the shipped LLM skeleton file and does not query external APIs.

```bash
cd Neuro-Symbolic-Reasoning
python experiment_llm_vs_random.py --datasets alarm --run_mode llm --seeds 5 --sample_size 10000 --epochs 140 --high_conf 0.9 --low_conf 0.1 --reconstruction_mode group_ce --vstructure_in_mask --dag_check --dag_project_on_cycle
```

To run both prior configurations in one command:

```bash
cd Neuro-Symbolic-Reasoning
python experiment_llm_vs_random.py --datasets alarm --run_mode both --seeds 5 --sample_size 10000 --epochs 140 --high_conf 0.9 --low_conf 0.1 --reconstruction_mode group_ce --vstructure_in_mask --dag_check --dag_project_on_cycle
```

### Step 3: Inspect outputs

Outputs are saved under:

- `Neuro-Symbolic-Reasoning/results/experiment_llm_vs_random/alarm/n_10000/seed_5/<run_id>/run_report.txt`
- `Neuro-Symbolic-Reasoning/results/experiment_llm_vs_random/alarm/n_10000/seed_5/<run_id>/random_prior/complete_metrics.json`
- `Neuro-Symbolic-Reasoning/results/experiment_llm_vs_random/alarm/n_10000/seed_5/<run_id>/llm_prior/complete_metrics.json` (when `run_mode=llm` or `both`)
- `Neuro-Symbolic-Reasoning/results/experiment_llm_vs_random/alarm/n_10000/seed_5/<run_id>/comparison_report.txt` (when `run_mode=both`)

## Expected Result Pattern

For paper-repro runs (ALARM, `run_mode=random`, `seed=5`, `n=10000`, `epochs=140`), key metrics should match:

- Edge F1: `0.88`
- Full SHD: `8`

Small floating-point formatting differences are acceptable.

## Optional: Regenerate Data (Non-paper mode)

```bash
python Neuro-Symbolic-Reasoning/scripts/generate_multi_sample_size_data.py --datasets alarm --sizes 10000 --seed 42
```

Then rerun the experiment.
Note: regenerated data/FCI may produce different metrics.

## Minimal File Structure

- `config.py` (root unified config)
- `reproducibility.py`
- `refactored/` (constraint discovery and related modules)
- `Neuro-Symbolic-Reasoning/experiment_llm_vs_random.py`
- `Neuro-Symbolic-Reasoning/train_complete.py`
- `Neuro-Symbolic-Reasoning/modules/`
- `Neuro-Symbolic-Reasoning/data/alarm/` (frozen artifacts for paper-repro)

## Reproducibility Notes

- Random seed is controlled in code/config.
- Results are deterministic given the same:
  - input data + metadata
  - constraint skeleton files
  - config and dependency versions

## Anonymous Submission Notice

This repository is anonymized for double-blind review.
All identifying information has been removed from documentation and metadata where possible.
