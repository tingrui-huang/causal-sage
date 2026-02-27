"""
Scanner Module

Implements various methods to scan for potential causal relationships
using statistical measures (Mutual Information, Correlation, etc.)
"""

from sklearn.metrics import mutual_info_score
import pandas as pd


class BaseScanner:
    def __init__(self, dataframe, graph, nodes):
        self.df = dataframe
        self.graph = graph
        self.nodes = nodes
    
    def scan(self, threshold):
        raise NotImplementedError("Subclasses must implement scan()")


class MutualInformationScanner(BaseScanner):

    def scan(self, threshold=0.05):
        candidates = []
        
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                # Skip diagonal and lower triangle
                if i >= j:
                    continue
                
                # Skip if edge already exists in either direction
                if self.graph.has_edge(node_a, node_b) or self.graph.has_edge(node_b, node_a):
                    continue
                
                # Calculate Mutual Information
                # MI >= 0, higher means stronger dependency
                mi_score = mutual_info_score(self.df[node_a], self.df[node_b])
                
                if mi_score > threshold:
                    candidates.append((node_a, node_b, mi_score))
        
        # RankedExpand: prioritize highest MI scores
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates


class PearsonCorrelationScanner(BaseScanner):
    
    def scan(self, threshold=0.3):
        candidates = []
        corr_matrix = self.df.corr().abs()
        
        for i, node_a in enumerate(self.nodes):
            for j, node_b in enumerate(self.nodes):
                if i >= j:
                    continue
                
                if self.graph.has_edge(node_a, node_b) or self.graph.has_edge(node_b, node_a):
                    continue
                
                corr_score = corr_matrix.loc[node_a, node_b]
                
                if corr_score > threshold:
                    candidates.append((node_a, node_b, corr_score))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return candidates

