# CausalSAGE

Codebase for PAG-to-DAG refinement experiments in CausalSAGE.

## Scope

- Main datasets in this repository: `alarm`, `insurance`
- Main experiment entrypoint: `scripts/run_llm_vs_random.py`
- Goal: reproduce paper-aligned settings and provide a clean, maintainable layout

## Repository Structure

- `config.py`: single source of truth for dataset paths and runtime settings
- `scripts/`: runnable entrypoints
- `src/refinement/`: PAG-to-DAG differentiable refinement implementation
- `src/constraints/`: constraint discovery modules
- `data/<dataset>/`: dataset artifacts (`*_data_*.csv`, `metadata*.json`, `*.bif`)
- `outputs/constraints/`: FCI/RFCI/LLM constraint outputs
- `outputs/experiments/`: refinement experiment outputs
- `legacy/` (optional): archived pre-refactor code kept only for historical reference

## Environment

- Python `>=3.10`

```bash
pip install -r requirements.txt
```

## Quick Start

### 1) Check config

Open `config.py` and confirm:

- `DATASET = 'alarm'`
- sample-size and training settings match your run target

### 2) Run random-prior refinement (paper-aligned style)

```bash
python scripts/run_llm_vs_random.py --datasets alarm --run_mode random --seeds 5 --sample_size 10000 --epochs 140 --high_conf 0.9 --low_conf 0.1 --reconstruction_mode group_ce --vstructure_in_mask --dag_check --dag_project_on_cycle
```

### 3) Run llm-prior refinement

```bash
python scripts/run_llm_vs_random.py --datasets alarm --run_mode llm --seeds 5 --sample_size 10000 --epochs 140 --high_conf 0.9 --low_conf 0.1 --reconstruction_mode group_ce --vstructure_in_mask --dag_check --dag_project_on_cycle
```

### 4) Run both priors in one command

```bash
python scripts/run_llm_vs_random.py --datasets alarm --run_mode both --seeds 5 --sample_size 10000 --epochs 140 --high_conf 0.9 --low_conf 0.1 --reconstruction_mode group_ce --vstructure_in_mask --dag_check --dag_project_on_cycle
```

## Output Locations

Example outputs are written under:

- `outputs/experiments/llm_vs_random/alarm/n_10000/seed_5/<run_id>/run_report.txt`
- `outputs/experiments/llm_vs_random/alarm/n_10000/seed_5/<run_id>/random_prior/complete_metrics.json`
- `outputs/experiments/llm_vs_random/alarm/n_10000/seed_5/<run_id>/llm_prior/complete_metrics.json`
- `outputs/experiments/llm_vs_random/alarm/n_10000/seed_5/<run_id>/comparison_report.txt`

Constraint outputs are written under:

- `outputs/constraints/<dataset>/n_<sample_size>/`

## Expected Pattern (alarm)

For paper-aligned random-prior runs (`seed=5`, `n=10000`, `epochs=140`), results should be close to:

- Edge F1: `0.88`
- Full SHD: `8`

Small floating-point differences are acceptable.

## Reproducibility Notes

- Random seed is controlled in config and scripts.
- Results are deterministic given:
  - same input data and metadata
  - same constraint skeleton files
  - same config and dependency versions
