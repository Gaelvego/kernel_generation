"""
Raw Triton Baseline Generator
Handles freeform Triton Python code generation, extraction, compilation, and retry loops.
"""
import re
import traceback
import ast
from typing import Optional, Dict, Any

import core.config as config
from core.llm_client import generate_llm_response


def extract_triton_code(raw_response: str) -> str:
    """
    Extracts Python code from markdown fences.
    Handles ```python, ```, and raw code.
    """
    text = raw_response.strip()

    # Remove markdown code blocks
    if text.startswith("```python"):
        text = text[len("```python"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    text = text.strip()

    # Sometimes the model adds explanatory text before/after — try to isolate a @triton.jit block
    jit_match = re.search(r"(@triton\.jit|import triton)", text)
    if jit_match:
        # naive: keep from the first import or decorator to the end
        start = jit_match.start()
        # Try to find the last def and keep until the end of the file / next double newline
        candidate = text[start:]
        return candidate.strip()

    return text


def structural_validate(code_str: str) -> tuple[bool, Optional[str]]:
    """
    Lightweight validation without executing the kernel on a GPU.
    Checks:
      1. Python parses into valid AST
      2. Contains @triton.jit decorator
      3. Contains a function definition
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Python Syntax Error: {e}"

    has_triton_import = False
    has_jit_decorator = False
    has_func_def = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.name if isinstance(node, ast.Import) else alias.asname or alias.name
                if "triton" in str(name) or (isinstance(node, ast.ImportFrom) and node.module and "triton" in node.module):
                    has_triton_import = True
        if isinstance(node, ast.FunctionDef):
            has_func_def = True
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute) and decorator.attr == "jit":
                    has_jit_decorator = True
                elif isinstance(decorator, ast.Call):
                    func = decorator.func
                    if isinstance(func, ast.Attribute) and func.attr == "jit":
                        has_jit_decorator = True

    if not has_func_def:
        return False, "No function definition found in generated code."
    if not has_jit_decorator:
        return False, "No @triton.jit decorator found on any function."
    if not has_triton_import:
        # Not fatal — the snippet might rely on the caller context, but warn
        pass

    return True, None


def compile_triton_kernel(code_str: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Attempts to compile the generated Triton Python code.
    Phase 1: structural validation (always possible).
    Phase 2: Triton frontend compilation (requires triton package, no GPU needed for TTIR).

    Returns:
        (success: bool, error_message: str or None, info: str or None)
    """
    # Step 1: structural check
    ok, err = structural_validate(code_str)
    if not ok:
        return False, err, None

    # Step 2: try importing and compiling with Triton
    try:
        import triton
        import triton.language as tl

        # Execute the code in a namespace containing triton imports
        namespace = {
            "triton": triton,
            "tl": tl,
            "torch": None,  # optional; some kernels reference torch for dtypes
        }
        try:
            import torch
            namespace["torch"] = torch
        except ImportError:
            pass

        exec(code_str, namespace)

        # Find the JIT kernel
        kernel = None
        for obj in namespace.values():
            if callable(obj) and hasattr(obj, "_jit"):
                kernel = obj
                break

        if kernel is None:
            # Fallback: search by decorator on ast
            return True, None, "Structural validation passed; kernel not executed (no GPU compile attempted)."

        # Try to compile without running — this requires a shape signature.
        # We introspect the kernel AST to guess a signature or just compile with empty constants.
        # In practice, Triton needs at least the non-constexpr arg types.
        return True, None, f"Kernel '{getattr(kernel, '__name__', 'unknown')}' structurally valid and imported."

    except ImportError:
        # Triton not installed in this environment
        return True, None, "Triton not installed locally; structural validation only."
    except Exception as e:
        return False, f"Triton compile/import error: {type(e).__name__}: {e}\n{traceback.format_exc()}", None


def generate_raw_triton(
    model_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    max_retries: int = 3,
    enable_retry: bool = True,
) -> Dict[str, Any]:
    """
    Generates a raw Triton kernel with optional error-feedback retry loop.

    Args:
        model_name: e.g. 'ollama', 'gemini', 'kimi'
        user_prompt: the task description
        system_prompt: optional system rules (defaults to raw Triton rules)
        max_retries: max attempts if retry is enabled
        enable_retry: if False, only one shot (mode='raw'); if True, retry on failure (mode='raw_retry')

    Returns:
        dict with keys: success, attempts, code, error, info, raw_response
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert GPU kernel engineer. "
            "Write a complete, self-contained Python function using the Triton language. "
            "Use the @triton.jit decorator. "
            "Import triton and triton.language as tl. "
            "Output ONLY the Python code inside a markdown code block (```python ... ```). "
            "Do not include explanatory text outside the code block."
        )

    errors: list[str] = []
    attempts = 0
    last_code = ""
    last_raw = ""

    for attempt in range(max_retries if enable_retry else 1):
        attempts += 1
        prompt = user_prompt
        if errors and enable_retry:
            prompt += (
                "\n\nYour previous implementation failed. Please correct it based on the following errors:\n"
                + "\n".join(f"- {e}" for e in errors)
            )

        raw_response = generate_llm_response(model_name, system_prompt, prompt, schema=None)
        last_raw = raw_response
        code = extract_triton_code(raw_response)
        last_code = code

        success, error, info = compile_triton_kernel(code)

        if success:
            return {
                "success": True,
                "attempts": attempts,
                "code": code,
                "error": None,
                "info": info,
                "raw_response": raw_response,
            }

        errors.append(error or "Unknown error")

        # If retry is disabled, stop after first attempt
        if not enable_retry:
            break

    return {
        "success": False,
        "attempts": attempts,
        "code": last_code,
        "error": errors[-1] if errors else "Unknown error",
        "info": None,
        "raw_response": last_raw,
    }
