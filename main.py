import json
import argparse
import traceback
from pydantic import ValidationError

from core.llm_client import generate_llm_response
from core.mlops_tracker import MLOpsTracker
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator
from core.triton_generator import generate_raw_triton
import core.config as config


def run_json_pipeline(system_prompt: str, user_prompt: str, model_name: str, tracker=None):
    """
    Original constrained JSON -> MLIR pipeline.
    """
    schema_str = json.dumps(MlirResponse.model_json_schema(), indent=2)
    full_system = (
        system_prompt
        + f"\n\nJSON Schema:\n{schema_str}\n\n"
        + "STRICT RULES:\n"
        + "1. DO NOT include markdown code blocks or text outside the JSON.\n"
        + "2. In 'arguments' and 'returns', use only strings (e.g., '%arg0'), NOT dictionaries.\n"
        + "3. You MUST ONLY use the opcodes provided in the JSON Schema Enum.\n"
        + "4. In iter_args and operands, you MUST ONLY use valid registers (e.g., '%0').\n"
        + "5. For arith.constant, ALWAYS provide the 'value' field.\n"
        + "6. For tt.load / tt.store, specify 'out_type' explicitly.\n"
        + "7. For scf.if producing values, provide 'result_types'.\n"
    )

    max_retries = config.MAX_RETRIES
    success = False
    mlir_code = ""
    error_msg = None

    try:
        translator = MLIRTranslator()
    except ImportError as e:
        print(f"[!] Environment error: {e}")
        if tracker:
            tracker.finish()
        return

    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        if tracker:
            tracker.start_timer()

        print("Waiting for LLM generation (Constrained Decoding enabled)...")
        try:
            raw_response = generate_llm_response(model_name, full_system, user_prompt, schema=MlirResponse)
        except Exception as e:
            print(f"[!] Error calling the API: {e}")
            break

        print("LLM responded. Parsing JSON contract...")
        error_msg = None

        try:
            clean_json = raw_response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
            if clean_json.startswith("```"):
                clean_json = clean_json[3:]
            if clean_json.endswith("```"):
                clean_json = clean_json[:-3]
            clean_json = clean_json.strip()

            print(f"[DEBUG] Raw response length: {len(raw_response)}")
            print(f"[DEBUG] First 100 chars of clean_json: {clean_json[:100]}")

            parsed_json = json.loads(clean_json)
            response_obj = MlirResponse(**parsed_json)

            print("Translating to MLIR Dialects...")
            mlir_code = translator.translate_to_module(response_obj.code)

            success = True
            print("[✓] Successful MLIR Compilation!")
            print(mlir_code)

            if tracker:
                tracker.log_iteration(attempt, user_prompt, raw_response, mlir_code, success, None)
            break

        except json.JSONDecodeError as e:
            error_msg = f"Malformed JSON: {e}"
        except ValidationError as e:
            error_msg = f"JSON Schema not respected: {e}"
        except RuntimeError as e:
            error_msg = str(e)
        except Exception as e:
            error_msg = f"General error: {traceback.format_exc()}"

        success = False
        print(f"[X] Semantic/Syntactic failure intercepted:\n{error_msg}")
        if tracker:
            tracker.log_iteration(attempt, user_prompt, raw_response, mlir_code, success, error_msg)

        user_prompt += f"\n\nYour previous attempt failed with this error:\n{error_msg}\nCorrect the JSON to fix it."

    if tracker:
        tracker.finish()


def run_raw_pipeline(user_prompt: str, model_name: str, enable_retry: bool, tracker=None):
    """
    Raw Triton Python generation pipeline.
    """
    print(f"\nRunning RAW pipeline (retry={enable_retry})...")
    if tracker:
        tracker.start_timer()

    result = generate_raw_triton(
        model_name=model_name,
        user_prompt=user_prompt,
        enable_retry=enable_retry,
        max_retries=config.MAX_RETRIES,
    )

    print(f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}")
    print(f"Attempts: {result['attempts']}")
    if result['success']:
        print("\n--- Generated Code ---")
        print(result['code'])
        if result.get('info'):
            print(f"\nInfo: {result['info']}")
    else:
        print(f"\nError: {result['error']}")

    if tracker:
        latency = tracker.get_elapsed_time_ms()
        tracker.log_iteration(
            iteration_idx=0,
            prompt=user_prompt,
            raw_json=result['raw_response'],
            mlir_code=result['code'] if result['success'] else "",
            success=result['success'],
            error_msg=result['error'],
        )
        tracker.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Semantic GPU Kernel Generator — LLM-MLIR Compiler"
    )
    parser.add_argument(
        "--mode",
        choices=["json", "raw", "raw_retry"],
        default=config.DEFAULT_MODE,
        help="Pipeline mode: json (constrained JSON->MLIR), raw (single-shot Triton), raw_retry (Triton with error feedback)"
    )
    parser.add_argument(
        "--model",
        default="ollama",
        help="Model backend to use (ollama, gemini, kimi)"
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Custom user prompt. If omitted, uses the default softmax prompt."
    )
    parser.add_argument(
        "--no-mlops",
        action="store_true",
        help="Disable Weights & Biases logging even if key is present"
    )
    args = parser.parse_args()

    print(f"Starting pipeline: mode={args.mode}, model={args.model}")

    # Setup MLOps Tracker (Optional)
    tracker = None
    if not args.no_mlops:
        try:
            if config.WANDB_API_KEY:
                tracker = MLOpsTracker(job_type=f"{args.mode}-experiment")
            else:
                print("[Info] WANDB_API_KEY not found. Running without MLOps.")
        except Exception as e:
            print(f"[Info] Running without MLOps due to error: {e}")

    # Default prompt if none provided
    user_prompt = args.prompt or (
        "Generate a kernel that performs one-pass online softmax. "
        "Goal: compute the softmax operation across the rows of an input matrix in a single global memory pass, "
        "using an online reduction algorithm to dynamically update the local maximum and the sum of exponentials."
    )

    if args.mode == "json":
        system_prompt = (
            "You are an expert LLM compiler in Triton MLIR. You must output ONLY a valid JSON object "
            "that EXACTLY complies with the provided JSON Schema."
        )
        run_json_pipeline(system_prompt, user_prompt, args.model, tracker)
    elif args.mode in ("raw", "raw_retry"):
        run_raw_pipeline(user_prompt, args.model, enable_retry=(args.mode == "raw_retry"), tracker=tracker)
    else:
        print(f"[!] Unknown mode: {args.mode}")

    print("\nProcess finished.")


if __name__ == "__main__":
    main()
