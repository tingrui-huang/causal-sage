"""
Visualizer Module

Handles graph visualization and saving to files.
"""

import networkx as nx
import matplotlib.pyplot as plt
import os


class GraphVisualizer:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize(self, graph, title="Causal Graph", 
                  filename=None, save_only=False,
                  node_color='lightblue', edge_color='green'):
        """
        Visualize causal graph with edge type differentiation
        
        Edge styling:
        - Directed (->): Black solid, confident
        - LLM Resolved: Green thick, LLM contribution
        - Partial (o->, -o, o-): Orange dashed, uncertain
        - Undirected (o-o): Gray dotted, uncertain
        - Tail-tail (--): Gray dotted, uncertain
        - Bidirected (<->): Red solid, latent confounder
        """
        from matplotlib.lines import Line2D
        
        # Use unified seed for deterministic layouts (when available)
        layout_seed = None
        try:
            from config import RANDOM_SEED
            layout_seed = int(RANDOM_SEED)
        except Exception:
            layout_seed = None
        
        # Categorize edges by type
        edges_directed = []
        edges_llm_resolved = []
        edges_partial = []
        edges_undirected = []
        edges_bidirected = []
        
        for u, v, d in graph.edges(data=True):
            edge_type = d.get('type', 'directed')
            
            # Skip rejected edges
            if edge_type == 'rejected':
                continue
            
            if edge_type == 'directed':
                edges_directed.append((u, v))
            elif edge_type == 'llm_resolved':
                edges_llm_resolved.append((u, v))
            elif edge_type == 'partial':
                edges_partial.append((u, v))
            elif edge_type in ['undirected', 'tail-tail']:
                edges_undirected.append((u, v))
            elif edge_type == 'bidirected':
                edges_bidirected.append((u, v))
        
        # Layout - use spring layout for better spacing
        pos = nx.spring_layout(graph, seed=layout_seed, k=1.5, iterations=50)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, 
                              node_size=3000, 
                              node_color=node_color,
                              edgecolors='black',
                              linewidths=2)
        
        # Draw labels
        nx.draw_networkx_labels(graph, pos, font_size=11, font_weight='bold')
        
        # Draw edges with different styles
        # 1. Directed edges: Black solid
        if edges_directed:
            nx.draw_networkx_edges(graph, pos, 
                                  edgelist=edges_directed,
                                  edge_color='black', 
                                  width=3, 
                                  arrowsize=35,
                                  arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=20,
                                  min_target_margin=20)
        
        # 2. LLM Resolved: Green thick (highlighting LLM contribution)
        if edges_llm_resolved:
            nx.draw_networkx_edges(graph, pos, 
                                  edgelist=edges_llm_resolved,
                                  edge_color='#00AA00', 
                                  width=5, 
                                  arrowsize=45,
                                  arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=20,
                                  min_target_margin=20)
        
        # 3. Partial edges: Orange dashed (uncertain)
        if edges_partial:
            nx.draw_networkx_edges(graph, pos, 
                                  edgelist=edges_partial,
                                  edge_color='#FF8C00', 
                                  width=3.5, 
                                  style='dashed',
                                  arrowsize=35,
                                  arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=20,
                                  min_target_margin=20)
        
        # 4. Undirected edges: Gray dotted (uncertain, no direction)
        if edges_undirected:
            nx.draw_networkx_edges(graph, pos, 
                                  edgelist=edges_undirected,
                                  edge_color='gray', 
                                  width=2.5, 
                                  style='dotted',
                                  arrows=False)
        
        # 5. Bidirected edges: Red solid (latent confounder)
        if edges_bidirected:
            nx.draw_networkx_edges(graph, pos, 
                                  edgelist=edges_bidirected,
                                  edge_color='red', 
                                  width=3, 
                                  arrowsize=35,
                                  arrowstyle='<->',
                                  connectionstyle='arc3,rad=0.1',
                                  min_source_margin=20,
                                  min_target_margin=20)
        
        # Add legend
        legend_elements = []
        if edges_directed:
            legend_elements.append(Line2D([0], [0], color='black', lw=3, 
                                         label='Directed (Confident)'))
        if edges_llm_resolved:
            legend_elements.append(Line2D([0], [0], color='#00AA00', lw=5, 
                                         label='LLM Resolved'))
        if edges_partial:
            legend_elements.append(Line2D([0], [0], color='#FF8C00', lw=3, 
                                         linestyle='--', label='Partial (Uncertain)'))
        if edges_undirected:
            legend_elements.append(Line2D([0], [0], color='gray', lw=2.5, 
                                         linestyle=':', label='Undirected'))
        if edges_bidirected:
            legend_elements.append(Line2D([0], [0], color='red', lw=3, 
                                         label='Bidirected (Confounder)'))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper left', 
                      fontsize=11, framealpha=0.9)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        if filename:
            fig_path = os.path.join(self.output_dir, f"{filename}.png")
        else:
            fig_path = os.path.join(self.output_dir, "causal_graph.png")
        
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[OUTPUT] Graph saved to: {fig_path}")

        if not save_only:
            plt.show()
        else:
            plt.close()
        
        return fig_path
    
    def visualize_comparison(self, graph_fci, graph_hybrid, 
                            filename="comparison", save_only=False):
        """
        Create side-by-side comparison of FCI and Hybrid FCI+LLM results
        
        Highlights:
        - Orange dashed lines for uncertain edges (partial, undirected)
        - Green thick arrows for LLM-resolved edges
        - Large, prominent arrows for clarity
        """
        from matplotlib.lines import Line2D
        
        # Use unified seed for deterministic layouts (when available)
        layout_seed = None
        try:
            from config import RANDOM_SEED
            layout_seed = int(RANDOM_SEED)
        except Exception:
            layout_seed = None
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Get all nodes for consistent layout
        all_nodes = list(set(graph_fci.nodes()) | set(graph_hybrid.nodes()))
        
        # Use spring layout with fixed seed for consistency
        # Create a complete graph for better spacing
        temp_graph = nx.Graph()
        temp_graph.add_nodes_from(all_nodes)
        pos = nx.spring_layout(temp_graph, seed=layout_seed, k=1.5, iterations=50)
        
        # --- LEFT: Pure FCI ---
        ax = axes[0]
        ax.set_title("Before: Pure FCI Output\n(Uncertainty in Orange/Dashed)", 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Categorize FCI edges
        edges_directed = [(u, v) for u, v, d in graph_fci.edges(data=True) 
                         if d.get('type') == 'directed']
        edges_partial = [(u, v) for u, v, d in graph_fci.edges(data=True) 
                        if d.get('type') == 'partial']
        edges_undirected = [(u, v) for u, v, d in graph_fci.edges(data=True) 
                           if d.get('type') in ['undirected', 'tail-tail']]
        
        # Draw nodes
        nx.draw_networkx_nodes(graph_fci, pos, ax=ax, 
                              node_color='lightgray', 
                              node_size=3500, 
                              edgecolors='black', 
                              linewidths=2)
        nx.draw_networkx_labels(graph_fci, pos, ax=ax, 
                               font_size=11, 
                               font_weight='bold')
        
        # Draw edges with PROMINENT arrows
        # Directed: Black solid with large arrows
        nx.draw_networkx_edges(graph_fci, pos, ax=ax, 
                              edgelist=edges_directed,
                              edge_color='black', 
                              width=3, 
                              arrowsize=35,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=20,
                              min_target_margin=20)
        
        # Partial: Orange dashed with large arrows
        nx.draw_networkx_edges(graph_fci, pos, ax=ax, 
                              edgelist=edges_partial,
                              edge_color='#FF8C00', 
                              width=3.5, 
                              style='dashed',
                              arrowsize=35,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=20,
                              min_target_margin=20)
        
        # Undirected: Gray dotted, no arrows
        if edges_undirected:
            nx.draw_networkx_edges(graph_fci, pos, ax=ax, 
                                  edgelist=edges_undirected,
                                  edge_color='gray', 
                                  width=2.5, 
                                  style='dotted',
                                  arrows=False)
        
        ax.axis('off')
        
        # --- RIGHT: Hybrid FCI+LLM ---
        ax = axes[1]
        ax.set_title("After: Hybrid (FCI + LLM)\n(LLM Resolved in Green)", 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Categorize Hybrid edges
        edges_directed_hy = [(u, v) for u, v, d in graph_hybrid.edges(data=True) 
                            if d.get('type') == 'directed']
        edges_resolved = [(u, v) for u, v, d in graph_hybrid.edges(data=True) 
                         if d.get('type') == 'llm_resolved']
        edges_undirected_hy = [(u, v) for u, v, d in graph_hybrid.edges(data=True) 
                              if d.get('type') in ['undirected', 'tail-tail']]
        
        # Draw nodes
        nx.draw_networkx_nodes(graph_hybrid, pos, ax=ax, 
                              node_color='lightgray', 
                              node_size=3500, 
                              edgecolors='black', 
                              linewidths=2)
        nx.draw_networkx_labels(graph_hybrid, pos, ax=ax, 
                               font_size=11, 
                               font_weight='bold')
        
        # Draw edges with PROMINENT arrows
        # Directed: Black solid (unchanged from FCI)
        nx.draw_networkx_edges(graph_hybrid, pos, ax=ax, 
                              edgelist=edges_directed_hy,
                              edge_color='black', 
                              width=3, 
                              arrowsize=35,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=20,
                              min_target_margin=20)
        
        # LLM Resolved: GREEN THICK with EXTRA LARGE arrows
        nx.draw_networkx_edges(graph_hybrid, pos, ax=ax, 
                              edgelist=edges_resolved,
                              edge_color='#00AA00', 
                              width=5, 
                              arrowsize=45,
                              arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1',
                              min_source_margin=20,
                              min_target_margin=20)
        
        # Undirected: Gray dotted (if any remained)
        if edges_undirected_hy:
            nx.draw_networkx_edges(graph_hybrid, pos, ax=ax, 
                                  edgelist=edges_undirected_hy,
                                  edge_color='gray', 
                                  width=2.5, 
                                  style='dotted',
                                  arrows=False)
        
        ax.axis('off')
        
        # Add legends with larger fonts
        legend_elements_1 = [
            Line2D([0], [0], color='black', lw=3, label='Directed (Confident)'),
            Line2D([0], [0], color='#FF8C00', lw=3, linestyle='--', 
                   label='Partial/Uncertain (o->)'),
            Line2D([0], [0], color='gray', lw=2.5, linestyle=':', 
                   label='Undirected (o-o)')
        ]
        axes[0].legend(handles=legend_elements_1, loc='upper left', 
                      fontsize=12, framealpha=0.9)
        
        legend_elements_2 = [
            Line2D([0], [0], color='black', lw=3, label='Originally Confident'),
            Line2D([0], [0], color='#00AA00', lw=5, label='LLM Resolved (Fixed)')
        ]
        axes[1].legend(handles=legend_elements_2, loc='upper left', 
                      fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f"{filename}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"[OUTPUT] Comparison graph saved to: {fig_path}")
        
        if not save_only:
            plt.show()
        else:
            plt.close()
        
        return fig_path

