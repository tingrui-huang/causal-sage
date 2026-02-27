"""
Parser Module

Parses LLM responses to extract causal direction decisions.
Handles various outputs formats and edge cases.
"""


class BaseParser:
    def parse(self, response_text, node_a, node_b):
        raise NotImplementedError("Subclasses must implement parse()")


class RobustDirectionParser(BaseParser):
    """
    Robust parser that handles multiple outputs formats

    Supports:
    - Variable names: "Smoking->Lung_cancer"
    - Generic notation: "A->B" or "B->A"
    - Various arrow types: ->, →, etc.
    - Case insensitive matching
    """
    
    def parse(self, response_text, node_a, node_b):
        """
        Parse LLM response to extract causal direction
        
        Parameters:
        -----------
        response_text : str
            LLM response text
        node_a : str
            First variable name (corresponds to "A")
        node_b : str
            Second variable name (corresponds to "B")
            
        Returns:
        --------
        tuple or None: 
            - (node_a, node_b) if A->B
            - (node_b, node_a) if B->A
            - None if no edge or bidirectional
        """
        # Normalize text: lowercase, remove spaces and newlines
        text = response_text.lower().replace(" ", "").replace("\n", "")
        
        # Normalize arrows (handle Unicode and ASCII variants)
        text = text.replace("→", "->").replace("—>", "->").replace("=>", "->")
        
        # Strategy 1: Match specific variable names
        pattern_ab = f"{node_a.lower()}->{node_b.lower()}"
        pattern_ba = f"{node_b.lower()}->{node_a.lower()}"
        
        if pattern_ab in text:
            return (node_a, node_b)
        elif pattern_ba in text:
            return (node_b, node_a)
        
        # Strategy 2: Match generic A->B or B->A notation
        if "direction:a->b" in text or ("a->b" in text and "b->a" not in text):
            return (node_a, node_b)
        elif "direction:b->a" in text or ("b->a" in text and "a->b" not in text):
            return (node_b, node_a)
        
        # Strategy 3: Check for bidirectional or no edge
        if "direction:a<->b" in text or "a<->b" in text:
            return None  # Bidirectional or confounded
        
        if "direction:none" in text or "noedge" in text:
            return None
        
        # If no clear pattern found, return None (conservative approach)
        return None


class SimpleDirectionParser(BaseParser):
    def parse(self, response_text, node_a, node_b):
        text = response_text.lower()
        
        if "direction: a->b" in text or "direction:a->b" in text:
            return (node_a, node_b)
        elif "direction: b->a" in text or "direction:b->a" in text:
            return (node_b, node_a)
        else:
            return None

