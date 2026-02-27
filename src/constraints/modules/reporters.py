"""
Reporter Module

Generates reports and logs from causal discovery results.
"""

import json
import os
from datetime import datetime


class ReportGenerator:

    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_text_report(self, graph, model_name="model", cot_log=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, 
                               f"report_{model_name.replace(' ', '_')}_{timestamp}.txt")
        
        # Classify edges with proper type notation
        accepted_edges = []
        rejected_edges = []
        seen_pairs = set()  # To avoid duplicating bidirected/undirected edges
        
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('type', 'directed')
            
            # Format edge based on type
            if edge_type == 'bidirected':
                # Bidirected edge: A <-> B (latent confounder)
                # Only print once for each pair
                pair = tuple(sorted([u, v]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge_str = f"{u} <-> {v}"
            elif edge_type == 'undirected':
                # Undirected edge: A o-o B (completely ambiguous)
                # Only print once for each pair
                pair = tuple(sorted([u, v]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge_str = f"{u} o-o {v}"
            elif edge_type == 'tail-tail':
                # Tail-tail edge: A -- B
                # Only print once for each pair
                pair = tuple(sorted([u, v]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge_str = f"{u} -- {v}"
            elif edge_type == 'partial':
                # Partial edge: show subtype (o->, <-o, o-, -o)
                subtype = data.get('subtype', 'o->')
                edge_str = f"{u} {subtype} {v}"
            elif edge_type == 'llm_resolved':
                # LLM resolved edge: A -> B [LLM]
                edge_str = f"{u} -> {v} [LLM]"
            else:
                # Directed edge: A -> B (from FCI or other methods)
                edge_str = f"{u} -> {v}"
            
            if data.get('type') == 'rejected':
                rejected_edges.append(edge_str)
            else:
                accepted_edges.append(edge_str)

        with open(filename, "w", encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"Causal Discovery Report: {model_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Nodes: {graph.number_of_nodes()}\n")
            f.write(f"Total Iterations: {len(cot_log) if cot_log else 'N/A'}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"ACCEPTED CAUSAL EDGES: {len(accepted_edges)}\n")
            f.write("-" * 60 + "\n")
            if accepted_edges:
                for i, edge in enumerate(sorted(accepted_edges), 1):
                    f.write(f"  {i:2d}. [YES] {edge}\n")
            else:
                f.write("  (No edges accepted)\n")
            
            f.write("\n" + "=" * 60 + "\n\n")
            
            # Rejected edges
            f.write(f"REJECTED EDGES: {len(rejected_edges)}\n")
            f.write("-" * 60 + "\n")
            if rejected_edges:
                for i, edge in enumerate(sorted(rejected_edges), 1):
                    f.write(f"  {i:2d}. [NO]  {edge}\n")
            else:
                f.write("  (No edges rejected)\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"[REPORT] Text report saved to: {filename}")
        return filename
    
    def save_edge_list(self, graph, model_name="model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir,
                               f"edges_{model_name.replace(' ', '_')}_{timestamp}.csv")
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("source,target,edge_type,status\n")
            for u, v, data in graph.edges(data=True):
                edge_type = data.get('type', 'directed')
                status = "rejected" if edge_type == 'rejected' else "accepted"
                # Include edge type in CSV for full transparency
                f.write(f"{u},{v},{edge_type},{status}\n")
        
        print(f"[REPORT] Edge list saved to: {filename}")
        return filename
    
    def save_cot_log(self, cot_log, model_name="model"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir,
                               f"cot_reasoning_{model_name}_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(cot_log, f, indent=2, ensure_ascii=True)
        
        print(f"[LOG] CoT reasoning saved to: {filename}")
        print(f"[LOG] Total reasoning steps: {len(cot_log)}")
        return filename

