"""
Result Manager Module

Manages experiment results with proper organization by dataset and configuration.
"""

import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ResultManager:
    """
    Manages saving and loading of experiment results
    
    Directory structure:
    results/
    ├── alarm/
    │   ├── no_llm_20251219_140530/
    │   │   ├── model.pt
    │   │   ├── adjacency.pt
    │   │   ├── evaluation_results.json
    │   │   ├── evaluation_results.txt
    │   │   └── config.json
    │   ├── gpt35_20251219_141020/
    │   └── ...
    ├── tuebingen/
    └── ...
    """
    
    def __init__(self, base_dir: str = 'results'):
        """
        Args:
            base_dir: Base directory for all results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_run_directory(self, 
                            dataset_name: str,
                            llm_model: Optional[str] = None,
                            config: Optional[Dict] = None) -> Path:
        """
        Create a directory for this run
        
        Args:
            dataset_name: Name of dataset (e.g., 'alarm', 'tuebingen_pair1')
            llm_model: LLM model name (e.g., 'gpt35', 'zephyr', None for no LLM)
            config: Configuration dictionary (optional, for custom naming)
        
        Returns:
            Path to run directory
        """
        # Create dataset directory
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Generate run name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if llm_model:
            run_name = f"{llm_model}_{timestamp}"
        else:
            run_name = f"no_llm_{timestamp}"
        
        # Add config-specific suffix if provided
        if config:
            if 'lambda_cycle' in config:
                run_name += f"_cycle{config['lambda_cycle']}"
            if 'lambda_group_lasso' in config:
                run_name += f"_lasso{config['lambda_group_lasso']}"
        
        # Create run directory
        run_dir = dataset_dir / run_name
        run_dir.mkdir(exist_ok=True)
        
        return run_dir
    
    def save_model(self, 
                   model: torch.nn.Module,
                   run_dir: Path,
                   name: str = 'model.pt'):
        """Save model state dict"""
        model_path = run_dir / name
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
    
    def save_adjacency(self,
                      adjacency: torch.Tensor,
                      run_dir: Path,
                      name: str = 'adjacency.pt'):
        """Save adjacency matrix"""
        adj_path = run_dir / name
        torch.save(adjacency, adj_path)
        print(f"Adjacency saved to: {adj_path}")
    
    def save_config(self,
                   config: Dict,
                   run_dir: Path,
                   name: str = 'config.json'):
        """Save configuration"""
        config_path = run_dir / name
        
        # Add metadata
        config_with_meta = {
            'metadata': {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': config.get('dataset_name', 'Unknown'),
                'llm_model': config.get('llm_model', 'None')
            },
            'config': config
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_with_meta, f, indent=2)
        
        print(f"Config saved to: {config_path}")
    
    def save_history(self,
                    history: Dict,
                    run_dir: Path,
                    name: str = 'training_history.json'):
        """
        Save training history (loss curves, metrics over time)
        
        Args:
            history: Dictionary with training metrics over epochs
            run_dir: Run directory
            name: Filename for history
        """
        history_path = run_dir / name
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to: {history_path}")
    
    def load_model(self,
                  model: torch.nn.Module,
                  run_dir: Path,
                  name: str = 'model.pt'):
        """Load model state dict"""
        model_path = run_dir / name
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from: {model_path}")
        return model
    
    def load_adjacency(self,
                      run_dir: Path,
                      name: str = 'adjacency.pt') -> torch.Tensor:
        """Load adjacency matrix"""
        adj_path = run_dir / name
        adjacency = torch.load(adj_path)
        print(f"Adjacency loaded from: {adj_path}")
        return adjacency
    
    def load_config(self,
                   run_dir: Path,
                   name: str = 'config.json') -> Dict:
        """Load configuration"""
        config_path = run_dir / name
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        print(f"Config loaded from: {config_path}")
        return config_data.get('config', config_data)
    
    def list_runs(self, dataset_name: Optional[str] = None) -> Dict:
        """
        List all runs, optionally filtered by dataset
        
        Args:
            dataset_name: Filter by dataset (None = all datasets)
        
        Returns:
            Dictionary mapping dataset -> list of run directories
        """
        runs = {}
        
        if dataset_name:
            datasets = [dataset_name]
        else:
            datasets = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
        for dataset in datasets:
            dataset_dir = self.base_dir / dataset
            if dataset_dir.exists():
                runs[dataset] = [
                    run_dir.name 
                    for run_dir in dataset_dir.iterdir() 
                    if run_dir.is_dir()
                ]
        
        return runs
    
    def get_run_info(self, dataset_name: str, run_name: str) -> Dict:
        """
        Get information about a specific run
        
        Args:
            dataset_name: Dataset name
            run_name: Run directory name
        
        Returns:
            Dictionary with run information
        """
        run_dir = self.base_dir / dataset_name / run_name
        
        info = {
            'dataset': dataset_name,
            'run_name': run_name,
            'path': str(run_dir),
            'exists': run_dir.exists()
        }
        
        if run_dir.exists():
            # Check for files
            info['has_model'] = (run_dir / 'model.pt').exists()
            info['has_adjacency'] = (run_dir / 'adjacency.pt').exists()
            info['has_config'] = (run_dir / 'config.json').exists()
            info['has_results'] = (run_dir / 'evaluation_results.json').exists()
            
            # Load config if available
            if info['has_config']:
                info['config'] = self.load_config(run_dir)
            
            # Load results if available
            if info['has_results']:
                with open(run_dir / 'evaluation_results.json', 'r') as f:
                    results = json.load(f)
                info['metrics'] = results.get('metrics', {})
        
        return info


if __name__ == "__main__":
    # Test the result manager
    print("Testing ResultManager\n")
    
    manager = ResultManager()
    
    # Create run directories
    print("Creating run directories:")
    run_dir1 = manager.create_run_directory('alarm', 'gpt35')
    print(f"  Created: {run_dir1}")
    
    run_dir2 = manager.create_run_directory('alarm', None)
    print(f"  Created: {run_dir2}")
    
    run_dir3 = manager.create_run_directory(
        'alarm', 
        'gpt35', 
        config={'lambda_cycle': 0.01, 'lambda_group_lasso': 0.1}
    )
    print(f"  Created: {run_dir3}")
    
    # List runs
    print("\nListing runs:")
    runs = manager.list_runs()
    for dataset, run_list in runs.items():
        print(f"  {dataset}:")
        for run in run_list:
            print(f"    - {run}")
    
    # Save dummy config
    print("\nSaving config:")
    config = {
        'dataset_name': 'ALARM',
        'llm_model': 'gpt-3.5-turbo',
        'learning_rate': 0.01,
        'n_epochs': 1000
    }
    manager.save_config(config, run_dir1)
    
    print("\nResultManager test complete!")
