'''
Thoughts-Augmented FunSearch - LLM Interface Module
====================================================
Milestone 2: LLM Interface Refactoring & Thoughts-Augmented Enhancement

CRITICAL: All LLM outputs must have:
1. Mathematical reasoning in triple quotes before Python code
2. Python code in standard Markdown python blocks
'''

import json
import re
import ast
import time
import http.client
import urllib.request
import urllib.error
from typing import Collection, Tuple

from implementation.sampler import LLM


# ============================================================================
# OPENAI COMPATIBLE API CONFIGURATION
# ============================================================================
import os

# API Configuration - DO NOT HARDCODE IN PRODUCTION
# Use environment variable API_KEY or set directly (for this session only)
_API_KEY = os.environ.get("API_KEY", "")

# Host URL for the API
_API_HOST = "https://api.bltcy.ai"
_API_MODEL = "gpt-5-nano"


# ============================================================================
# THOUGHTS-AUGMENTED PROMPT (CRITICAL INSTRUCTION)
# ============================================================================
THOUGHTS_AUGMENTED_SYSTEM_PROMPT = '''You are an expert algorithm designer specializing in bin packing optimization.

CRITICAL INSTRUCTIONS - You MUST follow this format exactly:

1. First, provide your mathematical and logical reasoning inside triple quotes (""")
2. Then, output your Python code inside a standard Markdown python block

Example format:
    """My reasoning: I will use logarithmic scaling because it helps balance utilization
    across bins with different remaining capacities. This prevents premature bin exhaustion."""

    ```python
    def priority(item: float, bins: np.ndarray) -> np.ndarray:
        ratios = item / bins
        priorities = -np.log(ratios + 1e-10)
        return priorities
    ```

DO NOT output code without first providing reasoning in triple quotes.
'''


class ThoughtsAugmentedLLM(LLM):
    """
    Thoughts-Augmented LLM interface for FunSearch.
    
    Features:
    - Forces LLM to output reasoning before code
    - Robust regex extraction of thoughts and code
    - Built-in safety validation
    """

    def __init__(self, samples_per_prompt: int, api_key: str = ""):
        super().__init__(samples_per_prompt)
        self._api_key = api_key or _API_KEY
        self._system_prompt = THOUGHTS_AUGMENTED_SYSTEM_PROMPT
        self._host = _API_HOST
        self._model = _API_MODEL

    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        results = []
        for _ in range(self._samples_per_prompt):
            sample = self._draw_sample(prompt)
            results.append(sample)
            time.sleep(0.5)  # Rate limiting
        return results

    def _draw_sample(self, content: str) -> str:
        """Call OpenAI-compatible API with thoughts-augmented prompt."""
        
        if not self._api_key:
            raise ValueError(
                "API key is not set. Cannot make LLM API calls."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            )
        
        client = OpenAI(
            base_url=f"{self._host}/v1",
            api_key=self._api_key,
            timeout=120,
        )
        
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=1.0,
                max_tokens=4000,  # Need high value for reasoning models
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API call failed: {str(e)}")


# ============================================================================
# ROBUST REGEX EXTRACTOR (High-robustness implementation)
# ============================================================================

def extract_thoughts_and_code(response: str) -> Tuple[str, str]:
    '''
    Extract thoughts and code from LLM response.
    
    Supports multiple formats:
    - Standard: triple-quoted thoughts followed by python code block
    - Inline: Single code block without explicit thoughts
    - Fallback: Attempts to find any function definition
    
    Returns:
        (thoughts: str, code: str)
    '''
    thoughts = ""
    code = ""
    
    # Pattern 1: Thoughts in triple quotes, then code in python block
    thoughts_match = re.search(r'"""(.*?)"""', response, re.DOTALL)
    code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    
    if thoughts_match and code_match:
        thoughts = thoughts_match.group(1).strip()
        code = code_match.group(1).strip()
        # Fix indentation if code has wrong indentation
        import textwrap
        code = textwrap.dedent(code)
        return thoughts, code
    
    # Pattern 2: Only code block (no explicit thoughts)
    code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        import textwrap
        code = textwrap.dedent(code)
        thoughts = "(No explicit reasoning provided)"
        return thoughts, code
    
    # Pattern 3: Any code block with python hint
    code_match = re.search(r'```\s*python\n?(.*?)\n?```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        return "", code
    
    # Pattern 4: Look for def priority function directly (simplified regex)
    try:
        func_match = re.search(
            r'(def priority\s*\([^)]*\).*?:)',
            response,
            re.DOTALL
        )
        if func_match:
            # Get from function start to next def or end
            start_pos = func_match.start()
            remaining = response[start_pos:]
            next_def = re.search(r'\n\s*def ', remaining[50:])  # Skip first 50 chars
            if next_def:
                code = remaining[:50 + next_def.start()].strip()
            else:
                code = remaining[:500].strip()  # Limit length
            return "(Extracted function directly)", code
    except re.error:
        pass
    
    # Pattern 5: Last resort - return entire response as code
    return thoughts, ""


# ============================================================================
# SAFETY VALIDATOR (AST + Keyword checking)
# ============================================================================

DANGEROUS_PATTERNS = [
    # Dangerous imports
    (r'\bimport\s+os\b', "Forbidden import: 'os'"),
    (r'\bimport\s+sys\b', "Forbidden import: 'sys'"),
    (r'\bimport\s+subprocess\b', "Forbidden import: 'subprocess'"),
    (r'\bimport\s+requests\b', "Forbidden import: 'requests'"),
    (r'\bimport\s+urllib\b', "Forbidden import: 'urllib'"),
    (r'\bimport\s+socket\b', "Forbidden import: 'socket'"),
    (r'\bfrom\s+os\s+import', "Forbidden: 'from os import'"),
    (r'\bfrom\s+sys\s+import', "Forbidden: 'from sys import'"),
    
    # Dangerous function calls
    (r'\bos\.system\s*\(', "Forbidden function: os.system()"),
    (r'\bos\.popen\s*\(', "Forbidden function: os.popen()"),
    (r'\bsubprocess\.', "Forbidden module: subprocess"),
    (r'\beval\s*\(', "Forbidden function: eval()"),
    (r'\bexec\s*\(', "Forbidden function: exec()"),
    (r'\b__import__\s*\(', "Forbidden function: __import__()"),
    (r'\bopen\s*\(', "Forbidden function: open()"),
    (r'\bfile\s*\(', "Forbidden function: file()"),
    
    # Network access
    (r'\.connect\s*\(', "Potential network access"),
    (r'\bsocket\s*\.', "Socket usage"),
    
    # File operations
    (r'\bwrite\s*\(', "Potential file write"),
    (r'\bread\s*\(', "Potential file read"),
]


def validate_code_safety(code: str) -> Tuple[bool, str]:
    """
    Validate code for safety using both regex and AST analysis.
    
    Returns:
        (is_safe: bool, error_message: str)
    """
    # Step 1: Regex pattern matching
    for pattern, message in DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            return False, message
    
    # Step 2: AST-based validation
    try:
        tree = ast.parse(code)
        
        # Check for dangerous AST nodes
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'sys', 'subprocess', 'requests', 'urllib', 'socket']:
                        return False, f"Forbidden import: '{alias.name}'"
            elif isinstance(node, ast.ImportFrom):
                if node.module in ['os', 'sys', 'subprocess', 'requests', 'urllib', 'socket']:
                    return False, f"Forbidden import: 'from {node.module} import ...'"
            
            # Check for calls to dangerous functions
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'eval', 'exec', '__import__']:
                        return False, f"Forbidden function call: '{node.func.id}()'"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'spawn', 'Popen']:
                        return False, f"Forbidden attribute: '{node.func.attr}'"
        
        return True, ""
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


def extract_function_signature(code: str) -> str:
    """Extract function signature from code."""
    match = re.search(r'def\s+priority\s*\([^)]*\)', code)
    if match:
        return match.group(0)
    return ""


# ============================================================================
# DEBUG LOG FORMATTER (for Milestone 3)
# ============================================================================

def format_debug_log(attempt_num: int, thoughts: str, code: str, 
                     extraction_success: bool, safety_passed: bool,
                     score: float = None, error: str = None,
                     optimal: int = 167) -> str:
    """
    Format debug log entry in the required format.
    
    Args:
        attempt_num: Attempt number
        thoughts: LLM's reasoning text
        code: Extracted Python code
        extraction_success: Whether regex extraction succeeded
        safety_passed: Whether safety validation passed
        score: Evaluation score (negative bins used)
        error: Error message if any
        optimal: Optimal number of bins
    
    Returns:
        Formatted debug log string
    """
    log = f"""
================================================================================
[Attempt {attempt_num}]
================================================================================

[Thoughts]
\"\"\"{thoughts}\"\"\"

[Extracted Code]
```python
{code}
```

[Execution]
- Regex Extraction: {'SUCCESS' if extraction_success else 'FAILED'}
- Safety Validation: {'PASSED' if safety_passed else f'REJECTED - {error}'}

[Result]
"""
    if error:
        log += f"- Error: {error}\n"
    elif score is not None:
        bins_used = -score  # Score is negative of bins used
        gap = bins_used - optimal
        log += f"- Bins Used: {bins_used}\n"
        log += f"- Optimal: {optimal}\n"
        log += f"- Gap: {gap}\n"
    else:
        log += "- No result\n"
    
    return log
