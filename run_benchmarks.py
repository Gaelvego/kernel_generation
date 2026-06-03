"""
Run benchmarks across pipeline modes and models.
Produces JSON results for comparison.
"""
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List

import core.config as config
from core.benchmarks import BENCHMARKS, Benchmark
from core.triton_generator import generate_raw_triton
from core.llm_client import generate_llm_response
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator


def run_json_pipeline(benchmark: Benchmark, model_name: str) -> Dict[str, Any]:
    """
    Run the constrained JSON -> MLIR pipeline for a single benchmark.
    """
    import json as json_mod
    from pydantic import ValidationError

    results = {
        "benchmark": benchmark.name,
        "mode": "json",
        "model": model_name,
        "attempts": 0,
        "success": False,
        "error": None,
        "mlir_code": None,
        "latency_ms": 0,
    }

    system_prompt = (
        "You are an expert compiler engineer targeting Triton MLIR. "
        "You must output ONLY a valid JSON object that strictly follows the provided JSON Schema. "
        "DO NOT include markdown code blocks or explanatory text outside the JSON. "
        "Use only the opcodes listed in the schema. "
        "For arith.constant, you MUST provide the 'value' field. "
        "For tt.load and other memory ops, specify 'out_type' explicitly. "
        "For scf.if, provide 'result_types' if the if-block produces values."
    )

    schema_str = json_mod.dumps(MlirResponse.model_json_schema(), indent=2)
    system_prompt += f"\n\nJSON Schema:\n{schema_str}"

    max_retries = config.MAX_RETRIES
    errors = []
    start = time.time()

    try:
        translator = MLIRTranslator()
    except ImportError as e:
        results["error"] = f"MLIR not available: {e}"
        results["latency_ms"] = (time.time() - start) * 1000
        return results

    for attempt in range(max_retries):
        results["attempts"] = attempt + 1
        prompt = benchmark.user_prompt
        if errors:
            prompt += (
                "\n\nYour previous JSON attempt failed with these errors. Please fix the JSON:\n"
                + "\n".join(f"- {e}" for e in errors)
            )

        try:
            raw_response = generate_llm_response(model_name, system_prompt, prompt, schema=MlirResponse)
        except Exception as e:
            results["error"] = f"API error: {e}"
            break

        # Clean markdown
        clean = raw_response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            parsed = json_mod.loads(clean)
            response_obj = MlirResponse(**parsed)
            mlir_code = translator.translate_to_module(response_obj.code)
            results["success"] = True
            results["mlir_code"] = mlir_code
            break
        except json_mod.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
        except ValidationError as e:
            errors.append(f"Schema validation error: {e}")
        except RuntimeError as e:
            errors.append(f"MLIR verification error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")

    results["latency_ms"] = (time.time() - start) * 1000
    if not results["success"] and not results["error"]:
        results["error"] = errors[-1] if errors else "Unknown failure"
    return results


def run_raw_pipeline(benchmark: Benchmark, model_name: str, enable_retry: bool) -> Dict[str, Any]:
    """
    Run the raw Triton generation pipeline for a single benchmark.
    """
    start = time.time()
    result = generate_raw_triton(
        model_name=model_name,
        user_prompt=benchmark.user_prompt,
        enable_retry=enable_retry,
        max_retries=config.MAX_RETRIES,
    )
    latency_ms = (time.time() - start) * 1000

    return {
        "benchmark": benchmark.name,
        "mode": "raw_retry" if enable_retry else "raw",
        "model": model_name,
        "attempts": result["attempts"],
        "success": result["success"],
        "error": result["error"],
        "code": result["code"],
        "info": result["info"],
        "latency_ms": latency_ms,
    }


def run_evaluation(
    models: List[str],
    modes: List[str],
    benchmarks: List[Benchmark] = None,
) -> List[Dict[str, Any]]:
    """
    Run the full evaluation matrix: models × modes × benchmarks.

    Args:
        models: list of model names, e.g. ["ollama", "gemini"]
        modes: list of modes, e.g. ["json", "raw", "raw_retry"]
        benchmarks: list of Benchmark objects (defaults to core.benchmarks.BENCHMARKS)

    Returns:
        List of result dictionaries.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS

    all_results: List[Dict[str, Any]] = []

    for benchmark in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {benchmark.name}")
        print(f"{'='*60}")

        for model in models:
            for mode in modes:
                print(f"  [{model}] mode={mode} ... ", end="", flush=True)
                try:
                    if mode == "json":
                        res = run_json_pipeline(benchmark, model)
                    elif mode in ("raw", "raw_retry"):
                        res = run_raw_pipeline(benchmark, model, enable_retry=(mode == "raw_retry"))
                    else:
                        res = {
                            "benchmark": benchmark.name,
                            "mode": mode,
                            "model": model,
                            "success": False,
                            "error": f"Unknown mode: {mode}",
                        }
                    all_results.append(res)
                    status = "SUCCESS" if res["success"] else "FAIL"
                    print(f"{status} (attempts={res.get('attempts', 1)}, latency={res.get('latency_ms', 0):.0f}ms)")
                except Exception as e:
                    print(f"CRASH: {e}")
                    all_results.append({
                        "benchmark": benchmark.name,
                        "mode": mode,
                        "model": model,
                        "success": False,
                        "error": str(e),
                    })

    return all_results


def save_results(results: List[Dict[str, Any]], filename: str = None):
    """Save evaluation results to a JSON file in the eval_results directory."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{timestamp}.json"
    filepath = os.path.join(config.EVAL_OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    return filepath


def print_summary(results: List[Dict[str, Any]]):
    """Print a formatted summary table of results."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    # Group by benchmark, then model, then mode
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(dict))
    for r in results:
        grouped[r["benchmark"]][r["model"]][r["mode"]] = r

    for bench_name in grouped:
        print(f"\nBenchmark: {bench_name}")
        print("-" * 70)
        for model in grouped[bench_name]:
            for mode in grouped[bench_name][model]:
                r = grouped[bench_name][model][mode]
                status = "PASS" if r["success"] else "FAIL"
                attempts = r.get("attempts", "-")
                lat = r.get("latency_ms", 0)
                print(f"  {model:12s} | {mode:12s} | {status:4s} | attempts={attempts} | {lat:.0f}ms")

    # Overall stats
    total = len(results)
    passed = sum(1 for r in results if r["success"])
    print(f"\nOverall: {passed}/{total} successes ({100*passed/total:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Triton kernel generation benchmarks")
    parser.add_argument("--models", nargs="+", default=["ollama"], help="Models to test")
    parser.add_argument("--modes", nargs="+", default=["json", "raw", "raw_retry"], help="Modes to test")
    parser.add_argument("--benchmarks", nargs="+", default=None, help="Specific benchmark names to run")
    parser.add_argument("--output", default=None, help="Output JSON filename")
    args = parser.parse_args()

    # Filter benchmarks if requested
    benchmark_list = BENCHMARKS
    if args.benchmarks:
        bench_map = {b.name: b for b in BENCHMARKS}
        benchmark_list = [bench_map[b] for b in args.benchmarks if b in bench_map]

    results = run_evaluation(args.models, args.modes, benchmark_list)
    filepath = save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
