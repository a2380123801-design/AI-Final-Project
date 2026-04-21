"""Sandbox for executing generated code with timeout and process isolation."""

import multiprocessing
import traceback
from typing import Any


class Sandbox:
    """Sandbox for executing generated code with timeout protection."""

    def __init__(self, verbose: bool = False):
        self._verbose = verbose

    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,
            timeout_seconds: int,
            **kwargs
    ) -> tuple[Any, bool]:
        """
        Execute generated code with timeout.
        
        Returns:
            (result, success) - execution result and whether it succeeded
        """
        result_queue = multiprocessing.Queue()
        
        process = multiprocessing.Process(
            target=self._compile_and_run,
            args=(program, function_to_run, function_to_evolve, inputs, result_queue)
        )
        
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            process.terminate()
            process.join()
            if self._verbose:
                print(f"[Sandbox] Timeout after {timeout_seconds}s")
            return None, False
        
        if result_queue.qsize() != 0:
            result = result_queue.get_nowait()
            return result
        else:
            return None, False

    def _compile_and_run(self, program: str, function_to_run: str, 
                         function_to_evolve: str, inputs: Any, 
                         result_queue: multiprocessing.Queue):
        """Internal method to compile and run the generated code."""
        try:
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            
            if function_to_run not in all_globals_namespace:
                result_queue.put((None, False))
                return
            
            function = all_globals_namespace[function_to_run]
            result = function(inputs)  # evaluate() takes full inputs dict
            
            if not isinstance(result, (int, float)):
                result_queue.put((None, False))
                return
            
            result_queue.put((result, True))
        except Exception:
            error_msg = traceback.format_exc(limit=10)
            result_queue.put((f"RUNTIME_ERROR:\n{error_msg}", False))
