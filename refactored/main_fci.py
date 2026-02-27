"""
Main Program: FCI
"""

import sys
import os
import time
import json
from datetime import datetime

# Import config and utils
from config import get_output_dir
from utils import get_active_data_loader, print_dataset_info

# Import modules from the modules package
from modules.algorithms import FCIAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class FCIPipeline:
    def __init__(self, data_loader, output_dir=None):
        print("=" * 60)
        print("Initializing FCI Pipeline")
        print("=" * 60)

        self.output_dir = output_dir or get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = data_loader
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up FCI algorithm...")
        self.algorithm = FCIAlgorithm(self.df, self.nodes)

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(self.output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(self.output_dir)

        self.model_name = "fci"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.algorithm_runtime_seconds = None
        
        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)
    
    def run(self, independence_test='chisq', alpha=0.05):
        print(f"\n{'='*60}")
        print(f"Running FCI Algorithm")
        print(f"Independence test: {independence_test}")
        print(f"Significance level: {alpha}")
        print(f"{'='*60}\n")

        algo_start = time.perf_counter()
        self.graph = self.algorithm.run(
            independence_test=independence_test,
            alpha=alpha
        )
        self.algorithm_runtime_seconds = time.perf_counter() - algo_start
        
        print(f"\n{'='*60}")
        print("FCI Algorithm Completed")
        print(f"{'='*60}")
        print(f"[TIME] FCI algorithm runtime: {self.algorithm_runtime_seconds:.2f}s")
        self._print_statistics()
        self._save_results()
    
    def _print_statistics(self):
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS (PAG - Partial Ancestral Graph)")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")
        
        # Count all edge types
        directed = sum(1 for u, v, d in self.graph.edges(data=True) 
                      if d.get('type') == 'directed')
        bidirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'bidirected')
        partial = sum(1 for u, v, d in self.graph.edges(data=True) 
                     if d.get('type') == 'partial')
        undirected = sum(1 for u, v, d in self.graph.edges(data=True) 
                        if d.get('type') == 'undirected')
        tail_tail = sum(1 for u, v, d in self.graph.edges(data=True) 
                       if d.get('type') == 'tail-tail')
        
        print(f"\nEdge Type Breakdown:")
        print(f"  Directed (->):      {directed:3d}  [certain causal direction]")
        print(f"  Bidirected (<->):   {bidirected:3d}  [latent confounder]")
        print(f"  Partial (o->/-o):   {partial:3d}  [ambiguous direction]")
        print(f"  Undirected (o-o):   {undirected:3d}  [completely ambiguous]")
        print(f"  Tail-tail (--):     {tail_tail:3d}  [no clear direction]")
        print(f"{'='*60}")
    
    def _save_results(self):
        print(f"\n{'='*60}")
        print("Saving Results")
        print(f"{'='*60}")

        self.reporter.save_text_report(self.graph, model_name="FCI")

        self.reporter.save_edge_list(self.graph, model_name="FCI")
        
        # Skip visualization for Tuebingen dataset (only 2 nodes, not informative)
        from config import DATASET
        if not DATASET.lower().startswith('tuebingen'):
            filename = f"causal_graph_{self.model_name}_{self.timestamp}"
            self.visualizer.visualize(self.graph, 
                                     title="Causal Graph (FCI Algorithm - PAG)",
                                     filename=filename,
                                     save_only=True,
                                     node_color='lightyellow',
                                     edge_color='red')
        
        print(f"{'='*60}")


def main():
    """Main function - runs FCI with parameters from config.py"""
    from config import FCI_INDEPENDENCE_TEST, FCI_ALPHA, GROUND_TRUTH_PATH
    
    total_start = time.perf_counter()
    evaluation_runtime_seconds = None

    print_dataset_info()
    
    # Use parameters from config.py (non-interactive mode)
    independence_test = FCI_INDEPENDENCE_TEST
    alpha = FCI_ALPHA
    
    print(f"\nUsing parameters from config.py:")
    print(f"  Independence test: {independence_test}")
    print(f"  Significance level: {alpha}")
    
    # Initialize pipeline
    data_loader = get_active_data_loader()
    pipeline = FCIPipeline(data_loader)
    
    # Run pipeline
    print(f"\nStarting FCI algorithm...")
    pipeline.run(independence_test=independence_test, alpha=alpha)
    
    print("\n" + "=" * 60)
    print(f"FCI completed! Results saved to {get_output_dir()}/")
    print("=" * 60)
    
    # === AUTO-EVALUATION ===
    print("\n" + "=" * 60)
    print("Running automatic evaluation...")
    print("=" * 60)
    
    try:
        eval_start = time.perf_counter()
        from evaluate_fci import evaluate_fci, find_latest_fci_csv
        from pathlib import Path
        
        latest_fci = find_latest_fci_csv(get_output_dir())
        gt_path = Path(GROUND_TRUTH_PATH)
        
        if latest_fci and gt_path.exists():
            print(f"\n[INFO] Evaluating: {latest_fci.name}")
            print(f"[INFO] Ground truth: {gt_path.name}\n")
            
            metrics = evaluate_fci(latest_fci, gt_path, output_dir=get_output_dir())
            
            # Print key metrics for easy reference
            print("\n" + "=" * 60)
            print("KEY METRICS (FCI Only)")
            print("=" * 60)
            print(f"SHD:                  {metrics['shd']}")
            print(f"Unresolved Ratio:     {metrics['unresolved_ratio']*100:.1f}%")
            print(f"Edge F1:              {metrics['edge_f1']*100:.1f}%")
            print(f"Orientation Accuracy: {metrics['orientation_accuracy']*100:.1f}%")
            print("=" * 60)
            
        elif not latest_fci:
            print("[WARN] Could not find FCI outputs for evaluation")
        elif not gt_path.exists():
            print(f"[WARN] Ground truth file not found: {gt_path}")
            print("Update GROUND_TRUTH_PATH in config.py to enable evaluation")
        evaluation_runtime_seconds = time.perf_counter() - eval_start
    except Exception as e:
        import traceback
        print(f"[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        print("\nYou can run 'python evaluate_fci.py' manually later.")
        evaluation_runtime_seconds = time.perf_counter() - eval_start
    
    total_runtime_seconds = time.perf_counter() - total_start
    
    # Persist timing information for reproducible reporting
    timing_payload = {
        "algorithm": "FCI",
        "timestamp": pipeline.timestamp,
        "output_dir": str(get_output_dir()),
        "independence_test": independence_test,
        "alpha": float(alpha),
        "algorithm_runtime_seconds": float(pipeline.algorithm_runtime_seconds) if pipeline.algorithm_runtime_seconds is not None else None,
        "evaluation_runtime_seconds": float(evaluation_runtime_seconds) if evaluation_runtime_seconds is not None else None,
        "total_runtime_seconds": float(total_runtime_seconds),
    }
    timing_file = os.path.join(get_output_dir(), f"timing_FCI_{pipeline.timestamp}.json")
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2, ensure_ascii=True)
    
    print("\n" + "=" * 60)
    print("All done!")
    print(f"[TIME] Algorithm runtime:  {pipeline.algorithm_runtime_seconds:.2f}s")
    if evaluation_runtime_seconds is not None:
        print(f"[TIME] Evaluation runtime: {evaluation_runtime_seconds:.2f}s")
    print(f"[TIME] Total runtime:      {total_runtime_seconds:.2f}s")
    print(f"[TIME] Timing JSON:       {timing_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

