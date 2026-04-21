#!/usr/bin/env python3
"""
Thoughts-Augmented FunSearch - Run with Real API
================================================
Run full FunSearch experiment with real GPT-5-nano API

Usage:
    python3 run_real_api.py
"""

import re
import time
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

# Project modules
from implementation.llm_interface import (
    ThoughtsAugmentedLLM,
    extract_thoughts_and_code,
    validate_code_safety,
)
from implementation.evaluator import Sandbox
from bin_packing_utils import parse_binpack_file


# ============================================================================
# SPECIFICATION (Baseline priority function)
# ============================================================================
SPECIFICATION = r'''
import numpy as np


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
        items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    packing = [[] for _ in bins]
    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        priorities = priority(item, bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    num_bins = []
    for name in instances:
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        bins = np.array([capacity for _ in range(instance['num_items'])])
        _, bins_packed = online_binpack(items, bins)
        num_bins.append((bins_packed != capacity).sum())
    return -np.mean(num_bins)


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.

    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.

    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Baseline: logarithmic priority based on utilization
    remaining = bins - item
    priorities = -remaining
    return priorities
'''


# ============================================================================
# BASELINE PRIORITY FUNCTION (for initial prompt)
# ============================================================================
BASELINE_PRIORITY = '''def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    priorities = -remaining
    return priorities
'''


# ============================================================================
# DEBUG LOG MANAGER
# ============================================================================

class DebugLogManager:
    """Manages debug log creation and formatting."""

    def __init__(self, output_path: str = "debug_log.md",
                 instance_names: List[str] = None,
                 optimal: float = 0):
        self.output_path = output_path
        self.instance_names = instance_names or []
        self.optimal = optimal
        self.entries: List[Dict[str, Any]] = []

    def add_entry(self, entry: Dict[str, Any]):
        """Add a debug log entry."""
        self.entries.append(entry)

    def write_log(self):
        """Write complete debug log to markdown file."""

        instance_str = ", ".join(self.instance_names) if self.instance_names else "N/A"

        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("# Thoughts-Augmented FunSearch - Debug Log (Real API)\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- Model: GPT-5-nano\n")
            f.write(f"- API Host: api.bltcy.ai\n")
            f.write(f"- Instances: {instance_str}\n")
            f.write(f"- Optimal (avg): {self.optimal:.1f}\n")
            f.write(f"- Timeout: 30 seconds\n\n")

            f.write("---\n\n")

            f.write("## Trial Run Results\n\n")

            for entry in self.entries:
                f.write(self._format_entry(entry))
                f.write("\n\n---\n\n")

        print(f"\n[Debug Log] Written to: {self.output_path}")
        return self.output_path

    def _format_entry(self, entry: Dict[str, Any]) -> str:
        """Format a single debug log entry."""

        lines = []
        lines.append(f"## [Attempt {entry['attempt']}]\n")
        lines.append(f"**Timestamp:** {entry['timestamp']}\n")

        if 'raw_response' in entry:
            lines.append(f"\n### Raw Response\n")
            lines.append(f"```text\n{entry['raw_response'][:2000]}\n```\n")

        lines.append(f"\n### Thoughts (LLM Reasoning)\n")
        lines.append(f'"""{entry["thoughts"]}"""\n')

        lines.append(f"\n### Extracted Code\n")
        lines.append(f"```python\n{entry['code']}\n```\n")

        lines.append(f"\n### Execution Details\n")
        lines.append(f"- Regex Extraction: {'SUCCESS' if entry['extraction_success'] else 'FAILED'}\n")

        if entry['safety_passed']:
            lines.append(f"- Safety Validation: PASSED\n")
        else:
            lines.append(f"- Safety Validation: REJECTED - {entry.get('safety_error', 'Unknown')}\n")

        # Timing information
        if 'api_time' in entry and entry['api_time'] is not None:
            lines.append(f"- **API Time: {entry['api_time']:.1f}s**\n")
        if 'eval_time' in entry and entry['eval_time'] is not None:
            lines.append(f"- Evaluation Time: {entry['eval_time']*1000:.1f}ms\n")

        if 'score' in entry and entry['score'] is not None:
            bins_used = -entry['score']
            optimal = entry.get('optimal', self.optimal)
            gap = bins_used - optimal
            lines.append(f"\n### Result\n")
            lines.append(f"- Bins Used: **{bins_used:.1f}**\n")
            lines.append(f"- Optimal: {optimal:.1f}\n")
            lines.append(f"- Gap: {gap:+.1f}\n")

            if entry.get('is_best'):
                lines.append(f"- **NEW BEST!** ★\n")
        elif 'error' in entry:
            lines.append(f"\n### Result\n")
            lines.append(f"- Error: {entry['error']}\n")

        return ''.join(lines)


# ============================================================================
# PROMPT BUILDER
# ============================================================================

def extract_priority_function(spec: str) -> str:
    """Extract the priority function from a specification."""
    pattern = r'(def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:.*?)(?=\n(?:def |\'\'\'|\"\"\"|\Z))'
    match = re.search(pattern, spec, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern2 = r'(def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:.*?)(?=\n(?:def |\Z))'
    match2 = re.search(pattern2, spec, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    return BASELINE_PRIORITY


def build_feedback_prompt(current_priority_func: str) -> str:
    """
    Build prompt that includes the current best priority function.
    
    Each iteration sees the current best solution and tries to improve it.
    
    Args:
        current_priority_func: The current best priority function code
        
    Returns:
        Prompt string that includes the current best solution
    """
    return f'''Improve the 'priority' function for online bin packing.

Current best implementation:
```python
{current_priority_func}
```

TASK:
1. First, describe your mathematical and logical reasoning in triple quotes (""")
2. Then write an improved Python 'priority' function

REQUIRED FORMAT:
    """Your reasoning"""

    ```python
    def priority(item: float, bins: np.ndarray) -> np.ndarray:
        # Your implementation
    ```

Important:
- The program selects the bin with np.argmax(priorities)
- Better bins must receive larger scores
- Keep the code simple, valid, and NumPy-only

Novelty requirement:
- Do not return a plain Best-Fit scoring rule
- Do NOT use any baseline backbone such as:
  - score = -r
  - score = -remaining
  - score = -abs(remaining)
  - score = -r**2
  - base = -r plus some extra bonus/penalty
- The final score must NOT be centered on a standard tight-fit backbone

Good directions:
- piecewise scoring rules
- threshold bonuses for very small leftover space
- explicit penalties for specific leftover ranges
- non-monotonic residual preference
- fragmentation-aware scoring without using -r as the main term

Do not add helper functions or imports.
'''

def fallback_extract_priority_code(response: str) -> str:
    """
    Fallback extractor: directly grab the first def priority(...) block.
    """
    match = re.search(
        r'(def\s+priority\s*\(.*?\)\s*(?:->.*?)?:.*?)(?=\n(?:def\s+|\Z))',
        response,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""

# ============================================================================
# MAIN FUNSEARCH RUNNER
# ============================================================================

def replace_priority_function(spec: str, new_code: str) -> str:
    """Replace priority function body in specification."""

    priority_def = 'def priority(item: float, bins: np.ndarray) -> np.ndarray:'
    func_start = spec.find(priority_def)
    if func_start == -1:
        return spec

    search_start = func_start + len(priority_def)
    next_def = re.search(r'\ndef ', spec[search_start:])
    if next_def:
        func_end = search_start + next_def.start()
    else:
        func_end = len(spec)

    new_spec = spec[:func_start] + new_code + spec[func_end:]
    return new_spec


def run_real_api_experiment(api_key: str, num_iterations: int = 10, 
                            mock_mode: bool = False,
                            mock_responses: Optional[List[str]] = None):
    """
    Run FunSearch with real API.
    
    Args:
        api_key: API key for LLM
        num_iterations: Number of iterations to run
        mock_mode: If True, use mock LLM responses instead of real API
        mock_responses: List of mock responses to use (for testing)
    """

    print("="*70)
    print("THOUGHTS-AUGMENTED FUNSEARCH - REAL API RUN")
    print("="*70)

    # Load ALL instances (removed [:2] limit for rigor)
    instances = parse_binpack_file('binpack8.txt')
    instance_names = list(instances.keys())
    
    # Compute average optimal from data (removed hardcoded 167)
    optimal = sum(inst['optimal'] for inst in instances.values()) / len(instances)

    print(f"\n[Config]")
    print(f"  Model: GPT-5-nano")
    print(f"  API Host: api.bltcy.ai")
    print(f"  Instances: {len(instance_names)} ({', '.join(instance_names[:5])}{'...' if len(instance_names) > 5 else ''})")
    print(f"  Optimal (avg): {optimal:.1f}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Mock Mode: {mock_mode}")

    # Initialize
    llm = ThoughtsAugmentedLLM(samples_per_prompt=1, api_key=api_key)
    sandbox = Sandbox(verbose=False)
    debug_log = DebugLogManager(
        output_path="debug_log.md",
        instance_names=instance_names,
        optimal=optimal
    )

    # Track best
    best_score = float('-inf')
    best_program = SPECIFICATION
    current_spec = SPECIFICATION

    iteration = 0
    attempt = 0
    start_time = time.time()
    mock_index = 0

    while iteration < num_iterations:
        iteration += 1

        print(f"\n{'='*60}")
        print(f"[Iteration {iteration}/{num_iterations}]")
        print(f"{'='*60}")
        
        # Build prompt with current best (feedback mechanism)
        current_priority_func = extract_priority_function(current_spec)
        prompt = build_feedback_prompt(current_priority_func)
        
        if iteration > 1:
            print(f"[Feedback] Improving upon best solution with {-best_score:.1f} bins")

        # Get sample from LLM
        print(f"[LLM] Requesting sample...")

        try:
            api_start = time.time()
            
            if mock_mode:
                if mock_responses and mock_index < len(mock_responses):
                    sample = mock_responses[mock_index]
                    mock_index += 1
                else:
                    sample = '''"""Using baseline approach."""\n\n```python\ndef priority(item: float, bins: np.ndarray) -> np.ndarray:\n    ratios = item / bins\n    log_ratios = np.log(ratios)\n    priorities = -log_ratios\n    return priorities\n```'''
                api_time = 0.1
            else:
                sample = ""
                for retry_idx in range(3):
                    candidate = llm.draw_samples(prompt)[0]
                    if candidate and candidate.strip():
                        sample = candidate
                        break
                    print(f"[LLM] Empty response on retry {retry_idx + 1}/3")
                api_time = time.time() - api_start
                
            print(f"[LLM] Response received ({api_time:.1f}s)")
        except Exception as e:
            print(f"[LLM Error] {e}")
            continue

        attempt += 1

        # Extract thoughts and code
        thoughts, code = extract_thoughts_and_code(sample)

        if not code:
            code = fallback_extract_priority_code(sample)

        print(f"  Raw response preview: {sample[:300]!r}")
        print(f"  Thoughts: {thoughts[:80]}..." if len(thoughts) > 80 else f"  Thoughts: {thoughts}")
        print(f"  Extracted code preview: {code[:200]!r}" if code else "  Extracted code preview: <EMPTY>")

        # Validate safety
        is_safe, safety_error = validate_code_safety(code)

        entry = {
            'attempt': attempt,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'raw_response': sample,
            'thoughts': thoughts,
            'code': code,
            'extraction_success': bool(code),
            'safety_passed': is_safe,
            'safety_error': safety_error if not is_safe else None,
            'api_time': api_time,
            'eval_time': None,
        }

        if not is_safe:
            print(f"  [Safety] REJECTED - {safety_error}")
            debug_log.add_entry(entry)
            continue

        if not code:
            print(f"  [Extract] No code found")
            entry['error'] = "No code extracted"
            debug_log.add_entry(entry)
            continue

        # Build program
        try:
            program = replace_priority_function(current_spec, code)
        except Exception as e:
            print(f"  [Program Build] Error: {e}")
            entry['error'] = str(e)
            debug_log.add_entry(entry)
            continue

        # Evaluate (removed unused test_input parameter)
        eval_start = time.time()
        result, success = sandbox.run(
            program=program,
            function_to_run='evaluate',
            function_to_evolve='priority',
            inputs=instances,
            timeout_seconds=30
        )
        eval_time = time.time() - eval_start
        entry['eval_time'] = eval_time

        if success and result is not None:
            score = result
            bins_used = -score
            gap = bins_used - optimal

            entry['score'] = score
            entry['optimal'] = optimal
            entry['is_best'] = False

            print(f"  [Result] Bins: {bins_used:.1f} (Optimal: {optimal:.1f}, Gap: {gap:+.1f})")

            if score > best_score:
                best_score = score
                best_program = program
                entry['is_best'] = True
                print(f"  [★ NEW BEST!]")
                
                # Update current_spec for next iteration's prompt
                current_spec = program
        else:
            entry['error'] = "Execution failed or timeout"
            print(f"  [Execution] Failed or timeout")

        debug_log.add_entry(entry)

        # Progress report every 5 iterations
        elapsed = time.time() - start_time
        if iteration % 5 == 0 or elapsed > 180:
            bins_used = -best_score if best_score > float('-inf') else 0
            print(f"\n[Progress] Iteration {iteration}/{num_iterations}. Current best: {bins_used:.1f} bins (optimal: {optimal:.1f})")
            print(f"[Time] Elapsed: {elapsed/60:.1f} minutes")

    # Write debug log
    print(f"\n[Debug Log] Writing to debug_log.md...")
    log_path = debug_log.write_log()

    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest Score: {best_score}")
    print(f"Best Bins: {-best_score if best_score > float('-inf') else 'N/A'}")
    print(f"Optimal (avg): {optimal:.1f}")
    print(f"Total Attempts: {attempt}")
    print(f"Total Time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"\nDebug Log: {log_path}")

    return best_score, best_program, log_path, debug_log


# ============================================================================
# SELF-TEST (No API required)
# ============================================================================

def run_self_test():
    """Self-test that verifies the feedback mechanism works correctly."""
    print("\n" + "="*70)
    print("SELF-TEST MODE (No API Required)")
    print("="*70)
    
    # Mock responses simulating improvement
    mock_responses = [
        '''"""First iteration: Using squared log for better fit."""\n\n```python\ndef priority(item: float, bins: np.ndarray) -> np.ndarray:\n    ratios = item / bins\n    log_ratios = np.log(ratios)\n    priorities = -log_ratios * 1.5\n    return priorities\n```''',
        '''"""Second iteration: Adding exponential decay."""\n\n```python\ndef priority(item: float, bins: np.ndarray) -> np.ndarray:\n    ratios = item / bins\n    log_ratios = np.log(ratios)\n    priorities = -log_ratios * 2.0 + np.exp(-bins)\n    return priorities\n```''',
    ]
    
    print("\n[1] Testing prompt building...")
    test_priority = '''def priority(item: float, bins: np.ndarray) -> np.ndarray:
    ratios = item / bins
    log_ratios = np.log(ratios)
    priorities = -log_ratios * 1.5
    return priorities'''
    
    prompt = build_feedback_prompt(test_priority)
    assert "Current best implementation:" in prompt
    assert test_priority in prompt
    print("   ✓ Prompt building works correctly")
    
    print("\n[2] Testing priority function extraction...")
    extracted = extract_priority_function(SPECIFICATION)
    assert "def priority" in extracted
    assert "remaining = bins - item" in extracted
    print("   ✓ Priority function extraction works correctly")
    
    print("\n[3] Testing replace_priority_function...")
    new_code = '''def priority(item: float, bins: np.ndarray) -> np.ndarray:
    ratios = item / bins
    priorities = -np.square(ratios)
    return priorities'''
    
    new_spec = replace_priority_function(SPECIFICATION, new_code)
    assert new_spec != SPECIFICATION
    assert "np.square(ratios)" in new_spec
    print("   ✓ Function replacement works correctly")
    
    print("\n[4] Testing sandbox evaluation (all instances)...")
    sandbox = Sandbox(verbose=False)
    instances = parse_binpack_file('binpack8.txt')
    result, success = sandbox.run(
        program=SPECIFICATION,
        function_to_run='evaluate',
        function_to_evolve='priority',
        inputs=instances,
        timeout_seconds=30
    )
    assert success, "Sandbox evaluation failed"
    assert result is not None, "No result returned"
    bins_used = -result
    optimal_avg = sum(inst['optimal'] for inst in instances.values()) / len(instances)
    print(f"   ✓ Sandbox works (baseline: {bins_used:.1f} bins, avg optimal: {optimal_avg:.1f})")
    
    print("\n[5] Testing dynamic optimal calculation...")
    assert optimal_avg > 0, "Optimal should be computed from data"
    assert isinstance(optimal_avg, float), "Optimal should be float"
    print(f"   ✓ Dynamic optimal computed: {optimal_avg:.1f}")
    
    print("\n[6] Testing DebugLogManager with dynamic config...")
    dl = DebugLogManager(
        output_path="/tmp/test_debug.md",
        instance_names=['t501_00', 't501_01', 't501_02'],
        optimal=175.5
    )
    assert dl.instance_names == ['t501_00', 't501_01', 't501_02']
    assert dl.optimal == 175.5
    print("   ✓ DebugLogManager accepts dynamic config")
    
    print("\n[7] Testing full iteration loop (mock mode, all instances)...")
    best_score, best_program, log_path, debug_log = run_real_api_experiment(
        api_key="MOCK_KEY",
        num_iterations=3,
        mock_mode=True,
        mock_responses=mock_responses
    )
    
    assert best_score != float('-inf'), "Should have a best score"
    assert best_program != SPECIFICATION, "Best program should be improved"
    assert len(debug_log.instance_names) == len(instances), "Should track all instances"
    print(f"   ✓ Full iteration loop works (best: {-best_score:.1f} bins)")
    
    print("\n[8] Verifying all instances are loaded...")
    all_inst = parse_binpack_file('binpack8.txt')
    print(f"   ✓ Total instances loaded: {len(all_inst)}")
    assert len(all_inst) > 2, "Should load more than 2 instances"
    
    print("\n" + "="*70)
    print("ALL SELF-TESTS PASSED ✓")
    print("="*70)
    
    return True


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        print("Running self-tests to verify code correctness...")
        success = run_self_test()
        if success:
            print("\n✅ Self-tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Self-tests failed!")
            sys.exit(1)
    
    import getpass

    print("Please enter the API Key to run the experiment.")
    API_KEY = getpass.getpass("API_KEY: ")
    if not API_KEY.strip():
        print("ERROR: API Key cannot be empty.")
        exit(1)

    print("Starting Thoughts-Augmented FunSearch with Real API...")

    best_score, best_program, log_path, debug_log = run_real_api_experiment(
        api_key=API_KEY,
        num_iterations=10
    )
