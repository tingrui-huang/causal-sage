"""
Main Program: RFCI (Tetrad)

Runs Java Tetrad RFCI via modules.algorithms.RFCIAlgorithm and writes edges_RFCI_*.csv
to refactored/output/<dataset>/, compatible with the rest of the pipeline.
"""

import os
import time
import json
from datetime import datetime

from config import get_output_dir
from utils import get_active_data_loader, print_dataset_info

from modules.algorithms import RFCIAlgorithm
from modules.visualizers import GraphVisualizer
from modules.reporters import ReportGenerator


class RFCIPipeline:
    def __init__(self, data_loader, output_dir=None):
        print("=" * 60)
        print("Initializing RFCI Pipeline")
        print("=" * 60)

        self.output_dir = output_dir or get_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {self.output_dir}/")

        print("\n[1/4] Loading data...")
        self.data_loader = data_loader
        self.df, self.nodes = self.data_loader.load_csv()

        print("\n[2/4] Setting up RFCI algorithm...")
        self.algorithm = RFCIAlgorithm(self.df, self.nodes, data_path=str(self.data_loader.data_path))

        print("\n[3/4] Setting up visualizer...")
        self.visualizer = GraphVisualizer(self.output_dir)

        print("\n[4/4] Setting up reporter...")
        self.reporter = ReportGenerator(self.output_dir)

        self.model_name = "rfci"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.algorithm_runtime_seconds = None

        print("\n" + "=" * 60)
        print("Pipeline initialized successfully!")
        print("=" * 60)

    def run(self, alpha=0.05, depth=-1, max_disc_path_len=-1, max_rows=None, verbose=False):
        print(f"\n{'='*60}")
        print("Running RFCI Algorithm (Tetrad)")
        print(f"Significance level (alpha): {alpha}")
        print(f"Depth: {depth}")
        print(f"Max discriminating path length: {max_disc_path_len}")
        print(f"{'='*60}\n")

        edges_out = os.path.join(self.output_dir, f"edges_RFCI_{self.timestamp}.csv")
        algo_start = time.perf_counter()
        self.graph = self.algorithm.run(
            alpha=alpha,
            depth=depth,
            max_disc_path_len=max_disc_path_len,
            max_rows=max_rows,
            verbose=verbose,
            output_edges_path=edges_out,
        )
        self.algorithm_runtime_seconds = time.perf_counter() - algo_start

        print(f"\n{'='*60}")
        print("RFCI Algorithm Completed")
        print(f"{'='*60}")
        print(f"[TIME] RFCI algorithm runtime: {self.algorithm_runtime_seconds:.2f}s")
        self._print_statistics()
        self._save_results()

    def _print_statistics(self):
        print(f"\n{'='*60}")
        print("GRAPH STATISTICS (PAG - Partial Ancestral Graph)")
        print(f"{'='*60}")
        print(f"Total nodes:          {self.graph.number_of_nodes()}")
        print(f"Total edges:          {self.graph.number_of_edges()}")

        directed = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "directed")
        bidirected = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "bidirected")
        partial = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "partial")
        undirected = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "undirected")
        tail_tail = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("type") == "tail-tail")

        print("\nEdge Type Breakdown:")
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

        # Text report
        self.reporter.save_text_report(self.graph, model_name="RFCI")

        # The edges CSV is already written by the Java runner as edges_RFCI_<timestamp>.csv
        # Avoid writing a second edges file with a different naming scheme.

        from config import DATASET
        if not DATASET.lower().startswith("tuebingen"):
            filename = f"causal_graph_{self.model_name}_{self.timestamp}"
            self.visualizer.visualize(
                self.graph,
                title="Causal Graph (RFCI Algorithm - PAG)",
                filename=filename,
                save_only=True,
                node_color="lightyellow",
                edge_color="red",
            )

        print(f"{'='*60}")


def main():
    """
    Main function - runs RFCI with parameters from config.py (if present),
    otherwise defaults are used.
    """
    total_start = time.perf_counter()
    print_dataset_info()

    from config import RFCI_ALPHA, RFCI_DEPTH, RFCI_MAX_DISC_PATH_LEN, RFCI_MAX_ROWS, VERBOSE as CFG_VERBOSE

    alpha = RFCI_ALPHA
    depth = RFCI_DEPTH
    max_disc_path_len = RFCI_MAX_DISC_PATH_LEN
    max_rows = RFCI_MAX_ROWS
    verbose = bool(CFG_VERBOSE)

    print("\nUsing parameters from config.py:")
    print(f"  Alpha: {alpha}")
    print(f"  Depth: {depth}")
    print(f"  Max discriminating path length: {max_disc_path_len}")
    print(f"  Max rows (RFCI only): {max_rows}")

    data_loader = get_active_data_loader()
    pipeline = RFCIPipeline(data_loader)

    print("\nStarting RFCI algorithm...")
    # Note: RFCI_MAX_ROWS only affects RFCI. Downstream training can still use full N.
    pipeline.run(alpha=alpha, depth=depth, max_disc_path_len=max_disc_path_len, max_rows=max_rows, verbose=verbose)

    total_runtime_seconds = time.perf_counter() - total_start
    
    # Persist timing information for reproducible reporting
    timing_payload = {
        "algorithm": "RFCI",
        "timestamp": pipeline.timestamp,
        "output_dir": str(get_output_dir()),
        "alpha": float(alpha),
        "depth": int(depth),
        "max_disc_path_len": int(max_disc_path_len),
        "max_rows": int(max_rows) if max_rows is not None else None,
        "algorithm_runtime_seconds": float(pipeline.algorithm_runtime_seconds) if pipeline.algorithm_runtime_seconds is not None else None,
        "total_runtime_seconds": float(total_runtime_seconds),
    }
    timing_file = os.path.join(get_output_dir(), f"timing_RFCI_{pipeline.timestamp}.json")
    with open(timing_file, "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, indent=2, ensure_ascii=True)
    
    print("\n" + "=" * 60)
    print(f"RFCI completed! Results saved to {get_output_dir()}/")
    print(f"[TIME] Algorithm runtime:  {pipeline.algorithm_runtime_seconds:.2f}s")
    print(f"[TIME] Total runtime:      {total_runtime_seconds:.2f}s")
    print(f"[TIME] Timing JSON:       {timing_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

