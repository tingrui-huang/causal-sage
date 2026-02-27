"""
Data Loading Module

Handles loading and preprocessing of datasets for causal discovery.
"""

import pandas as pd
import networkx as nx


class DataLoader:
    def __init__(self, data_path, dataset_name="Custom"):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.df = None
        self.nodes = None
        self.descriptions = {}
        
    def load_csv(self):
        self.df = pd.read_csv(self.data_path)
        self.nodes = self.df.columns.tolist()
        
        # Auto-generate descriptions if not provided
        if not self.descriptions:
            self._generate_descriptions()
        
        print(f"[DATA] Dataset: {self.dataset_name}")
        print(f"[DATA] Loaded: {self.df.shape}")
        print(f"[DATA] Variables: {self.nodes}")
        
        return self.df, self.nodes
    
    def _generate_descriptions(self):
        """Auto-generate variable descriptions based on data"""
        for node in self.nodes:
            unique_vals = self.df[node].nunique()
            
            if unique_vals == 2:
                desc = f"Binary variable (0/1)"
            elif unique_vals <= 10:
                desc = f"Categorical variable ({unique_vals} categories)"
            else:
                desc = f"Continuous or high-cardinality variable"
            
            self.descriptions[node] = desc
    
    def get_description(self, node_name):
        return self.descriptions.get(node_name, "A variable in the dataset.")
    
    def create_empty_graph(self):
        if self.nodes is None:
            raise ValueError("Data not loaded. Call load_csv() first.")
        
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        
        return graph


class LUCASDataLoader(DataLoader):
    def __init__(self, data_path):
        super().__init__(data_path, dataset_name="LUCAS")
        
        # Domain-specific variable descriptions
        self.descriptions = {
            "Smoking": "A binary variable indicating if the patient smokes.",
            "Lung_cancer": "A binary variable indicating if the patient has lung cancer.",
            "Genetics": "Genetic predisposition to cancer.",
            "Yellow_Fingers": "Discoloration of fingers, often associated with tar.",
            "Anxiety": "Mental health status.",
            "Peer_Pressure": "Social influence to smoke.",
        }


class ALARMDataLoader(DataLoader):
    """
    Data loader for ALARM dataset (medical monitoring system)
    ALARM: A Large ARtificial Medical monitoring system
    37 variables related to anesthesia monitoring
    """
    def __init__(self, data_path):
        super().__init__(data_path, dataset_name="ALARM")
        
        # ALARM-specific descriptions will be auto-generated
        # You can add domain-specific ones here if needed

