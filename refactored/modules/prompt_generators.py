"""
Prompt Generator Module

Generates prompts for LLMs to perform causal reasoning.
Different generators for different prompting strategies (CoT, Direct, etc.)
"""


class BasePromptGenerator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def generate(self, node_a, node_b):
        raise NotImplementedError("Subclasses must implement generate()")


class CoTPromptGenerator(BasePromptGenerator):
    def generate(self, node_a, node_b):
        # Get variable descriptions
        if hasattr(self.data_loader, 'get_description'):
            desc_a = self.data_loader.get_description(node_a)
            desc_b = self.data_loader.get_description(node_b)
        else:
            desc_a = "A variable."
            desc_b = "A variable."
        
        prompt = f"""Role: You are an expert in Medical Science performing Causal Discovery.

Task: Analyze the relationship between two variables from a dataset:
Variable A: {node_a} ({desc_a})
Variable B: {node_b} ({desc_b})

Context: The data shows a strong statistical correlation between them.

Instructions:
1. Think step-by-step about the biological or logical mechanism.
2. Determine if A causes B, B causes A, or if they are likely confounded (no direct link).
3. Provide your reasoning briefly.
4. Final Answer format: "Direction: A->B", "Direction: B->A", or "Direction: None".

Reasoning:"""
        
        return prompt


class DirectPromptGenerator(BasePromptGenerator):
    def generate(self, node_a, node_b):
        if hasattr(self.data_loader, 'get_description'):
            desc_a = self.data_loader.get_description(node_a)
            desc_b = self.data_loader.get_description(node_b)
        else:
            desc_a = "A variable."
            desc_b = "A variable."
        
        prompt = f"""You are an expert in medical causal inference.

Given two correlated variables:
- Variable A: {node_a} ({desc_a})
- Variable B: {node_b} ({desc_b})

Determine the causal direction:
- If A causes B, respond: "Direction: A->B"
- If B causes A, respond: "Direction: B->A"
- If no direct causal link, respond: "Direction: None"

Your answer:"""
        
        return prompt


class ZephyrCoTPromptGenerator(BasePromptGenerator):
    # Uses [INST] tags required by Zephyr's instruction format.
    def generate(self, node_a, node_b):
        if hasattr(self.data_loader, 'get_description'):
            desc_a = self.data_loader.get_description(node_a)
            desc_b = self.data_loader.get_description(node_b)
        else:
            desc_a = "A variable."
            desc_b = "A variable."
        
        prompt = f"""[INST] You are an expert in Medical Science performing Causal Discovery.

Task: Analyze the relationship between two variables from a dataset:
Variable A: {node_a} ({desc_a})
Variable B: {node_b} ({desc_b})

Context: The data shows a strong statistical correlation between them.

Instructions:
1. Think step-by-step about the biological or logical mechanism.
2. Determine if A causes B, B causes A, or if they are likely confounded (no direct link).
3. Provide your reasoning briefly.
4. Final Answer format: "Direction: A->B", "Direction: B->A", or "Direction: None".

Reasoning: [/INST]"""
        
        return prompt

