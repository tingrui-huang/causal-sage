"""
Algorithm Module

Implements various causal discovery algorithms (GES, FCI, PC, etc.)
These are baseline methods that don't use LLMs.
"""

import numpy as np
import networkx as nx
import os
import csv
import subprocess
import warnings
from pathlib import Path
import re


class BaseAlgorithm:
    def __init__(self, dataframe, nodes):
        self.df = dataframe
        self.nodes = nodes
        self.data_matrix = dataframe.values
        self.graph = None
    
    def run(self):
        raise NotImplementedError("Subclasses must implement run()")


class GESAlgorithm(BaseAlgorithm):
    def __init__(self, dataframe, nodes):
        super().__init__(dataframe, nodes)
        print("[ALGORITHM] GES (Greedy Equivalence Search)")
        print("[ALGORITHM] Assumes: No latent confounders")
        print("[ALGORITHM] NOTE: Choose score function by data type")
        print("[ALGORITHM]       - discrete:   local_score_BDeu")
        print("[ALGORITHM]       - continuous: local_score_BIC")
    
    def run(self, score_func='local_score_BIC'):
        try:
            from causallearn.search.ScoreBased.GES import ges

            # Accept a few common aliases to reduce configuration mistakes.
            score_aliases = {
                "bic": "local_score_BIC",
                "bdeu": "local_score_BDeu",
            }
            canonical = score_aliases.get(str(score_func).strip().lower(), score_func)

            print(f"[GES] Running with score function: {score_func}")
            print(f"[GES] Data shape: {self.data_matrix.shape}")

            # Suppress noisy pandas FutureWarning emitted inside causal-learn's
            # LocalScoreFunction for groupby.apply behavior changes.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*DataFrameGroupBy\.apply operated on the grouping columns.*",
                    category=FutureWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r".*When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group.*",
                    category=FutureWarning,
                )
                # Run GES
                record = ges(self.data_matrix, score_func=canonical)
            
            # Extract graph
            learned_graph = record['G']
            
            print(f"[GES] Algorithm completed")
            print(f"[GES] Found {learned_graph.get_num_edges()} edges")
            
            # Convert to NetworkX DiGraph
            self.graph = self._convert_to_networkx(learned_graph)
            
            return self.graph
            
        except ImportError:
            print("[ERROR] causal-learn not installed!")
            print("[ERROR] Install with: pip install causal-learn")
            raise
        except Exception as e:
            print(f"[ERROR] GES failed: {e}")
            print("[HINT] For discrete datasets, prefer score_func='local_score_BDeu'.")
            print("[HINT] For continuous datasets, prefer score_func='local_score_BIC'.")
            raise
    
    def _convert_to_networkx(self, causallearn_graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        
        # Get adjacency matrix
        adj_matrix = causallearn_graph.graph
        
        # Add edges
        edge_count = 0
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if i != j and adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    # Directed edge: i -> j
                    graph.add_edge(self.nodes[i], self.nodes[j])
                    edge_count += 1
                elif i < j and adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1:
                    # Undirected edge (in CPDAG): add both directions
                    graph.add_edge(self.nodes[i], self.nodes[j], type='undirected')
                    graph.add_edge(self.nodes[j], self.nodes[i], type='undirected')
                    edge_count += 1
        
        print(f"[GES] Converted to NetworkX: {edge_count} directed edges")
        
        return graph


class FCIAlgorithm(BaseAlgorithm):
    def __init__(self, dataframe, nodes):
        super().__init__(dataframe, nodes)
        print("[ALGORITHM] FCI (Fast Causal Inference)")
        print("[ALGORITHM] Allows: Latent confounders")
        print("[ALGORITHM] CRITICAL: Use 'chisq' for discrete data (LUCAS, ALARM)")
        print("[ALGORITHM]           Use 'fisherz' only for continuous Gaussian data")
        self.causallearn_graph = None
    
    def run(self, independence_test='chisq', alpha=0.05):
        try:
            from causallearn.search.ConstraintBased.FCI import fci
            
            print(f"[FCI] Running with test: {independence_test}, alpha: {alpha}")
            print(f"[FCI] Data shape: {self.data_matrix.shape}")
            
            if independence_test == 'fisherz':
                print("[FCI] WARNING: fisherz assumes continuous Gaussian data!")
                print("[FCI]          For discrete data, use 'chisq' or 'gsq' instead.")
            
            # Run FCI
            graph_result, edges = fci(
                self.data_matrix,
                independence_test_method=independence_test,
                alpha=alpha
            )
            
            print(f"[FCI] Algorithm completed")
            print(f"[FCI] Found {len(edges)} edges")
            
            # Store the causal-learn graph for later analysis
            self.causallearn_graph = graph_result
            
            # Convert to NetworkX DiGraph
            self.graph = self._convert_to_networkx(graph_result)
            
            return self.graph
            
        except ImportError:
            print("[ERROR] causal-learn not installed!")
            print("[ERROR] Install with: pip install causal-learn")
            raise
        except Exception as e:
            print(f"[ERROR] FCI failed: {e}")
            raise
    
    def get_ambiguous_edges(self):
        """
        Extract edges that FCI cannot determine direction for
        
        These are edges with circles (o-o, o->, o-, -o) or tail-tail (--)
        that could benefit from LLM's domain knowledge.
        
        NOTE: Bidirected edges (<->) are NOT included because they represent
        a definite structure (latent confounder), not directional ambiguity.
        
        Returns:
        --------
        list of tuples: [(node_a, node_b, edge_type), ...] where direction is ambiguous
        """
        if self.causallearn_graph is None:
            print("[WARNING] FCI has not been run yet!")
            return []
        
        ambiguous_pairs = []
        adj_matrix = self.causallearn_graph.graph
        
        # causal-learn encoding:
        # -1: arrowhead (>), 1: tail (-), 2: circle (o)
        
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):  # Only upper triangle
                edge_i = adj_matrix[i, j]
                edge_j = adj_matrix[j, i]
                
                if edge_i == 0 and edge_j == 0:
                    continue
                
                # Skip bidirected edges - they represent latent confounders, not directional ambiguity
                if edge_i == -1 and edge_j == -1:
                    continue
                
                # Any edge with at least one circle is ambiguous
                # o-o (both circles - completely ambiguous)
                if edge_i == 2 and edge_j == 2:
                    ambiguous_pairs.append((self.nodes[i], self.nodes[j], 'o-o'))
                
                # o-> (circle at i, arrow at j - partially ambiguous)
                elif edge_i == 2 and edge_j == -1:
                    ambiguous_pairs.append((self.nodes[i], self.nodes[j], 'o->'))
                
                # <-o (arrow at i, circle at j - partially ambiguous)
                elif edge_i == -1 and edge_j == 2:
                    ambiguous_pairs.append((self.nodes[j], self.nodes[i], '<-o'))
                
                # o- (circle at i, tail at j - ambiguous)
                elif edge_i == 2 and edge_j == 1:
                    ambiguous_pairs.append((self.nodes[i], self.nodes[j], 'o-'))
                
                # -o (tail at i, circle at j - ambiguous)
                elif edge_i == 1 and edge_j == 2:
                    ambiguous_pairs.append((self.nodes[i], self.nodes[j], '-o'))
                
                # -- (tail-tail - also ambiguous, no clear direction)
                elif edge_i == 1 and edge_j == 1:
                    ambiguous_pairs.append((self.nodes[i], self.nodes[j], '--'))
        
        print(f"[FCI] Found {len(ambiguous_pairs)} ambiguous edges for LLM arbitration")
        print(f"[FCI] (Bidirected edges excluded - they represent latent confounders)")
        return ambiguous_pairs
    
    def _convert_to_networkx(self, causallearn_graph):
        """
        Convert causal-learn PAG to NetworkX DiGraph
        
        PAG edge encoding in causal-learn:
        - -1: arrowhead (>)
        - 1: tail (-)
        - 2: circle (o)
        
        PAG edge types:
        - i -> j:   (-1, 1) directed, i causes j
        - i <-> j:  (-1, -1) bidirected, latent confounder
        - i o-> j:  (2, -1) partially directed, circle at i
        - i o- j:   (2, 1) circle-tail
        - i o-o j:  (2, 2) undirected with circles
        - i -- j:   (1, 1) tail-tail
        """
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        
        # Get adjacency matrix
        adj_matrix = causallearn_graph.graph
        
        # Add edges
        edge_pair_count = 0  # Number of variable pairs with edges
        directed_count = 0
        bidirected_count = 0
        partial_count = 0
        undirected_count = 0
        tail_tail_count = 0
        
        for i in range(len(self.nodes)):
            for j in range(i+1, len(self.nodes)):  # Only check upper triangle to avoid duplicates
                edge_i = adj_matrix[i, j]
                edge_j = adj_matrix[j, i]
                
                if edge_i == 0 and edge_j == 0:
                    continue
                
                # Process edge types
                if edge_i == -1 and edge_j == 1:
                    # i -> j (directed)
                    graph.add_edge(self.nodes[i], self.nodes[j], type='directed')
                    directed_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == 1 and edge_j == -1:
                    # i <- j (directed, reverse)
                    graph.add_edge(self.nodes[j], self.nodes[i], type='directed')
                    directed_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == -1 and edge_j == -1:
                    # i <-> j (bidirected, latent confounder)
                    # This is CRITICAL for FCI - represents hidden common cause!
                    graph.add_edge(self.nodes[i], self.nodes[j], type='bidirected')
                    graph.add_edge(self.nodes[j], self.nodes[i], type='bidirected')
                    bidirected_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == 2 and edge_j == -1:
                    # i o-> j (partially directed, circle at i, arrow at j)
                    # Direction: likely j is effect, but not certain
                    graph.add_edge(self.nodes[i], self.nodes[j], type='partial', subtype='o->', pag_encoding='(2,-1)')
                    partial_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == -1 and edge_j == 2:
                    # i <-o j (partially directed, arrow at i, circle at j)
                    # Direction: likely i is effect, but not certain
                    # Stored as j->i with subtype to indicate original was <-o
                    graph.add_edge(self.nodes[j], self.nodes[i], type='partial', subtype='<-o', pag_encoding='(-1,2)')
                    partial_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == 2 and edge_j == 1:
                    # i o- j (circle at i, tail at j)
                    # j is definitely not a descendant of i, but direction unclear
                    graph.add_edge(self.nodes[i], self.nodes[j], type='partial', subtype='o-', pag_encoding='(2,1)')
                    partial_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == 1 and edge_j == 2:
                    # i -o j (tail at i, circle at j)
                    # i is definitely not a descendant of j, but direction unclear
                    graph.add_edge(self.nodes[i], self.nodes[j], type='partial', subtype='-o', pag_encoding='(1,2)')
                    partial_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == 2 and edge_j == 2:
                    # i o-o j (undirected with circles - completely ambiguous)
                    graph.add_edge(self.nodes[i], self.nodes[j], type='undirected')
                    graph.add_edge(self.nodes[j], self.nodes[i], type='undirected')
                    undirected_count += 1
                    edge_pair_count += 1
                    
                elif edge_i == 1 and edge_j == 1:
                    # i -- j (tail-tail, undirected)
                    graph.add_edge(self.nodes[i], self.nodes[j], type='tail-tail')
                    graph.add_edge(self.nodes[j], self.nodes[i], type='tail-tail')
                    tail_tail_count += 1
                    edge_pair_count += 1
        
        print(f"[FCI] Converted to NetworkX: {edge_pair_count} edge pairs")
        print(f"[FCI] (Total directed edges in graph: {graph.number_of_edges()})")
        print(f"[FCI]   Directed (->):      {directed_count}")
        print(f"[FCI]   Bidirected (<->):   {bidirected_count}  [latent confounders]")
        print(f"[FCI]   Partial (o->/-o):   {partial_count}  [ambiguous direction]")
        print(f"[FCI]   Undirected (o-o):   {undirected_count}  [completely ambiguous]")
        print(f"[FCI]   Tail-tail (--):     {tail_tail_count}")
        
        return graph


class RFCIAlgorithm(BaseAlgorithm):
    """
    RFCI (Tetrad) wrapper.

    Why: your installed causal-learn does not ship RFCI, and pigs/link graphs are too large for FCI.
    This wrapper runs Java Tetrad RFCI via a tiny bundled runner (refactored/third_party/tetrad/RunRfci.java)
    and returns a NetworkX graph with edge_type attributes compatible with existing reporters/prior_builder.
    """
    def __init__(self, dataframe, nodes, data_path: str | None = None):
        super().__init__(dataframe, nodes)
        self.data_path = data_path
        print("[ALGORITHM] RFCI (Recursive FCI) via Tetrad (Java)")
        print("[ALGORITHM] Allows: Latent confounders (like FCI), typically faster on large graphs")

    def run(
        self,
        alpha: float = 0.05,
        depth: int = -1,
        max_disc_path_len: int = -1,
        max_rows: int | None = None,
        verbose: bool = False,
        output_edges_path: str | None = None,
    ):
        """
        Run RFCI on a discrete, integer-coded CSV (header required).

        Parameters
        ----------
        alpha : float
            Significance level for CI tests.
        depth : int
            Search depth (-1 = unlimited). Smaller speeds up.
        max_disc_path_len : int
            Max discriminating path length (-1 = unlimited). Smaller speeds up.
        output_edges_path : str | None
            Where to write edges CSV. If None, a temp file under refactored/third_party/tetrad/ is used.
        """
        # Resolve paths
        repo_root = Path(__file__).resolve().parents[1]  # refactored/
        tetrad_dir = repo_root / "third_party" / "tetrad"
        jar_path = tetrad_dir / "tetrad-lib-7.6.8-shaded.jar"
        java_src = tetrad_dir / "RunRfci.java"
        bin_dir = tetrad_dir / "bin"
        class_name = "RunRfci"

        if not jar_path.exists():
            raise FileNotFoundError(f"Missing Tetrad jar: {jar_path}")
        if not java_src.exists():
            raise FileNotFoundError(f"Missing Java runner source: {java_src}")

        if not self.data_path:
            raise ValueError(
                "RFCIAlgorithm needs the CSV path used to load the dataframe. "
                "Pass data_path=... when constructing RFCIAlgorithm."
            )

        in_csv = Path(self.data_path)
        if not in_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {in_csv}")

        out_csv = Path(output_edges_path) if output_edges_path else (tetrad_dir / "rfci_edges_tmp.csv")

        # Compile runner if needed
        bin_dir.mkdir(parents=True, exist_ok=True)
        class_file = bin_dir / f"{class_name}.class"
        if not class_file.exists() or class_file.stat().st_mtime < java_src.stat().st_mtime:
            print("[RFCI] Compiling Java runner...")
            compile_cmd = [
                "javac",
                "-cp",
                str(jar_path),
                "-d",
                str(bin_dir),
                str(java_src),
            ]
            proc = subprocess.run(compile_cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    "javac failed.\n"
                    f"cmd={' '.join(compile_cmd)}\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}\n"
                )

        # Run RFCI
        print("[RFCI] Running Tetrad RFCI...")
        cp_sep = ";" if os.name == "nt" else ":"
        classpath = f"{bin_dir}{cp_sep}{jar_path}"
        run_cmd = [
            "java",
            "-cp",
            classpath,
            class_name,
            str(in_csv),
            str(out_csv),
            str(alpha),
            str(depth),
            str(max_disc_path_len),
            str(max_rows) if max_rows is not None else "-1",
            "true" if verbose else "false",
        ]
        proc = subprocess.run(run_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "java RFCI runner failed.\n"
                f"cmd={' '.join(run_cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

        if not out_csv.exists():
            raise RuntimeError(f"RFCI did not produce outputs edges file: {out_csv}")

        # Convert edges CSV to NetworkX
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)

        with out_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = row.get("source")
                v = row.get("target")
                edge_type = row.get("edge_type", "directed")
                status = row.get("status", "accepted")
                if status == "rejected":
                    graph.add_edge(u, v, type="rejected")
                else:
                    graph.add_edge(u, v, type=edge_type)

        # Basic stats
        directed = sum(1 for _, _, d in graph.edges(data=True) if d.get("type") == "directed")
        bidirected = sum(1 for _, _, d in graph.edges(data=True) if d.get("type") == "bidirected")
        partial = sum(1 for _, _, d in graph.edges(data=True) if d.get("type") == "partial")
        undirected = sum(1 for _, _, d in graph.edges(data=True) if d.get("type") == "undirected")
        tail_tail = sum(1 for _, _, d in graph.edges(data=True) if d.get("type") == "tail-tail")

        print(f"[RFCI] Output edges: {graph.number_of_edges()}")
        print(f"[RFCI]   Directed (->):      {directed}")
        print(f"[RFCI]   Bidirected (<->):   {bidirected}  [latent confounders]")
        print(f"[RFCI]   Partial (o->/-o):   {partial}  [ambiguous direction]")
        print(f"[RFCI]   Undirected (o-o):   {undirected}  [completely ambiguous]")
        print(f"[RFCI]   Tail-tail (--):     {tail_tail}")

        self.graph = graph
        return graph

    def get_ambiguous_edges(self):
        """
        Extract edges with ambiguous direction from the RFCI PAG outputs.

        Returns:
          list of (node_a, node_b, edge_type) tuples, similar to FCIAlgorithm.get_ambiguous_edges.

        Notes:
          - We skip bidirected edges (<->) because they represent latent confounding, not ambiguity.
          - Tetrad CSV edge types are mapped into a small set compatible with the hybrid LLM scripts:
              partial    -> 'o->'  (direction ambiguous at the tail/circle end)
              undirected -> 'o-o'
              tail-tail  -> '--'
          - If the Java runner ever emits other ambiguous types, we conservatively pass them through.
        """
        if self.graph is None:
            print("[WARNING] RFCI has not been run yet!")
            return []

        ambiguous = []
        seen_pairs = set()

        for u, v, d in self.graph.edges(data=True):
            et = d.get("type", "directed")
            if et == "bidirected":
                continue

            if et == "partial":
                out_type = "o->"
            elif et == "undirected":
                out_type = "o-o"
            elif et == "tail-tail":
                out_type = "--"
            else:
                # Not ambiguous (directed, rejected, etc.)
                continue

            key = tuple(sorted((u, v)))
            if key in seen_pairs and out_type == "o-o":
                continue
            seen_pairs.add(key)
            ambiguous.append((u, v, out_type))

        print(f"[RFCI] Found {len(ambiguous)} ambiguous edges for LLM arbitration")
        print(f"[RFCI] (Bidirected edges excluded - they represent latent confounders)")
        return ambiguous

    @staticmethod
    def load_graph_from_edges_csv(edges_csv_path: str, nodes: list[str] | None = None):
        """
        Load a NetworkX graph from an edges_RFCI_*.csv produced by the Tetrad runner.

        Expected columns: source,target,edge_type,status (status optional).
        """
        import networkx as nx

        p = Path(edges_csv_path)
        if not p.exists():
            raise FileNotFoundError(p)

        graph = nx.DiGraph()
        if nodes is not None:
            graph.add_nodes_from(nodes)

        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                u = row.get("source")
                v = row.get("target")
                edge_type = row.get("edge_type", "directed")
                status = row.get("status", "accepted")
                if status == "rejected":
                    graph.add_edge(u, v, type="rejected")
                else:
                    graph.add_edge(u, v, type=edge_type)

        return graph

    @staticmethod
    def load_graph_from_report_txt(report_path: str, nodes: list[str] | None = None):
        """
        Best-effort loader from refactored/report_RFCI_*.txt (human-readable report).

        Parses lines like:
          [YES] A -> B
          [YES] A o-> B
          [YES] A o-o B
          [YES] A <-> B
          [YES] A -- B

        Returns a DiGraph with edge attribute 'type' matching:
          directed | partial | undirected | bidirected | tail-tail
        """
        import networkx as nx

        p = Path(report_path)
        if not p.exists():
            raise FileNotFoundError(p)

        graph = nx.DiGraph()
        if nodes is not None:
            graph.add_nodes_from(nodes)

        # Match: "[YES] <u> <etype> <v>"
        pat = re.compile(r"\[YES\]\s+(\S+)\s+(->|o->|o-o|<->|--)\s+(\S+)")

        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pat.search(line)
            if not m:
                continue
            u, et_sym, v = m.group(1), m.group(2), m.group(3)

            if et_sym == "->":
                graph.add_edge(u, v, type="directed")
            elif et_sym == "o->":
                graph.add_edge(u, v, type="partial")
            elif et_sym == "o-o":
                graph.add_edge(u, v, type="undirected")
                graph.add_edge(v, u, type="undirected")
            elif et_sym == "<->":
                graph.add_edge(u, v, type="bidirected")
                graph.add_edge(v, u, type="bidirected")
            elif et_sym == "--":
                graph.add_edge(u, v, type="tail-tail")
                graph.add_edge(v, u, type="tail-tail")

        return graph


class PCAlgorithm(BaseAlgorithm):
    """
    PC Algorithm (Constraint-based, assumes causal sufficiency / no latent confounders).
    
    Output is typically a CPDAG: directed + undirected edges.
    We convert it into a NetworkX DiGraph with edge attributes:
      - type='directed' for oriented edges
      - type='undirected' for unoriented edges (stored in both directions)
    """
    def __init__(self, dataframe, nodes):
        super().__init__(dataframe, nodes)
        print("[ALGORITHM] PC (Peter-Clark)")
        print("[ALGORITHM] Assumes: No latent confounders (causal sufficiency)")
        print("[ALGORITHM] Output: CPDAG (directed + undirected edges)")

    def run(self, independence_test='chisq', alpha=0.05, stable=True, uc_rule=0, uc_priority=2):
        """
        Run PC algorithm from causal-learn.
        
        Parameters
        ----------
        independence_test : str
            'chisq' / 'gsq' for discrete; 'fisherz' for continuous Gaussian.
        alpha : float
            Significance level.
        stable : bool
            Use stable-PC variant.
        uc_rule : int
            Unshielded collider orientation rule in causal-learn PC implementation.
        uc_priority : int
            Priority for collider orientation in causal-learn.
        """
        try:
            from causallearn.search.ConstraintBased.PC import pc

            print(f"[PC] Running with test: {independence_test}, alpha: {alpha}, stable: {stable}")
            print(f"[PC] Data shape: {self.data_matrix.shape}")

            # causal-learn PC signature differs across versions; be defensive.
            try:
                cg = pc(
                    self.data_matrix,
                    alpha=alpha,
                    indep_test=independence_test,
                    stable=stable,
                    uc_rule=uc_rule,
                    uc_priority=uc_priority,
                )
            except TypeError:
                # Fallback for older versions
                cg = pc(
                    self.data_matrix,
                    alpha=alpha,
                    indep_test=independence_test,
                    stable=stable,
                )

            learned_graph = getattr(cg, "G", cg)
            print("[PC] Algorithm completed")

            self.graph = self._convert_to_networkx(learned_graph)
            return self.graph

        except ImportError:
            print("[ERROR] causal-learn not installed!")
            print("[ERROR] Install with: pip install causal-learn")
            raise
        except Exception as e:
            print(f"[ERROR] PC failed: {e}")
            raise

    def _convert_to_networkx(self, causallearn_graph):
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)

        adj_matrix = causallearn_graph.graph

        pair_count = 0
        directed_count = 0
        undirected_count = 0

        # causal-learn encoding (for CPDAG):
        # -1: arrowhead (>), 1: tail (-), 0: no edge
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                a = adj_matrix[i, j]
                b = adj_matrix[j, i]

                if a == 0 and b == 0:
                    continue

                pair_count += 1

                if a == -1 and b == 1:
                    # i -> j
                    graph.add_edge(self.nodes[i], self.nodes[j], type='directed')
                    directed_count += 1
                elif a == 1 and b == -1:
                    # j -> i
                    graph.add_edge(self.nodes[j], self.nodes[i], type='directed')
                    directed_count += 1
                elif a == -1 and b == -1:
                    # undirected edge i - j in CPDAG (store both directions)
                    graph.add_edge(self.nodes[i], self.nodes[j], type='undirected')
                    graph.add_edge(self.nodes[j], self.nodes[i], type='undirected')
                    undirected_count += 1
                else:
                    # Unexpected encoding across versions; treat as undirected to be safe.
                    graph.add_edge(self.nodes[i], self.nodes[j], type='undirected', pc_encoding=f"({a},{b})")
                    graph.add_edge(self.nodes[j], self.nodes[i], type='undirected', pc_encoding=f"({b},{a})")
                    undirected_count += 1

        print(f"[PC] Converted to NetworkX: {pair_count} edge pairs")
        print(f"[PC]   Directed (->):   {directed_count}")
        print(f"[PC]   Undirected (-):  {undirected_count}")
        print(f"[PC]   Total edges in DiGraph: {graph.number_of_edges()}")

        return graph
