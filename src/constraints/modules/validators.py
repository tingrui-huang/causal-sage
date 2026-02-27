"""
Validator Module

Validates LLM-proposed edges against statistical tests on data.
Implements data-driven pruning to prevent hallucinations.
"""

import pandas as pd
from scipy.stats import chi2_contingency


class BaseValidator:
    def __init__(self, dataframe):
        self.df = dataframe
    
    def validate(self, node_a, node_b, significance_level=0.05):
        raise NotImplementedError("Subclasses must implement validate()")


class ChiSquareValidator(BaseValidator):
    def validate(self, node_a, node_b, significance_level=0.05):
        # Create contingency table
        contingency_table = pd.crosstab(self.df[node_a], self.df[node_b])

        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # If p < significance_level, reject H0 -> variables are dependent
        is_valid = p_value < significance_level
        
        return is_valid, p_value


class CorrelationValidator(BaseValidator):
    """
    Correlation-based validator for continuous data
    
    Note: Less appropriate for binary/categorical data.
    Use ChiSquareValidator for discrete data instead.
    """
    
    def validate(self, node_a, node_b, threshold=0.3):
        corr_matrix = self.df.corr()
        correlation = abs(corr_matrix.loc[node_a, node_b])
        
        is_valid = correlation > threshold
        
        return is_valid, correlation

