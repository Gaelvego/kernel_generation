import json
import traceback
from pydantic import ValidationError

from core.llm_client import generate_llm_response
from core.mlops_tracker import MLOpsTracker
from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator

def main():
    """
    Main Orchestrator - Generation Loop and Strict Validation.
    Phase 1 and 2 of the LLM-Guided Compiler Project.
    """
    print("Starting MLOps and MLIR infrastructure...")
    
    # 1. Setup MLOps Tracker (Optional)
    try:
        # Try to initialize W&B only if the key exists in .env
        import core.config as config
        if config.WANDB_API_KEY:
            tracker = MLOpsTracker(job_type="vector-add-experiment")
        else:
            tracker = None
            print("[Info] WANDB_API_KEY not found. Running without MLOps.")
    except Exception as e:
        tracker = None
        print(f"[Info] Running without MLOps due to error: {e}")
    
    # 2. Instantiate the MLIR translator
    try:
        translator = MLIRTranslator()
    except ImportError as e:
        print(f"[!] Environment error: {e}")
        if tracker: tracker.finish()
        return

    # 3. Prompt Definition and Allowed Triton Operations
    schema_str = json.dumps(MlirResponse.model_json_schema(), indent=2)
    
    system_prompt = (
        "Role: You are a Block-Level GPU Architect and expert LLM compiler in Triton MLIR. "
        "Your job is to translate high-level mathematical descriptions to a strict intermediate language in JSON format (Static Single Assignment - SSA).\n\n"
        f"You must output ONLY a valid JSON object that EXACTLY complies with the following JSON Schema:\n{schema_str}\n\n"
        "CRITICAL RULES:\n"
        "1. Immutability & Scoping: Each operation must save its result in a new temporary register (e.g., '%0'). Registers defined inside a block (e.g., scf.for or scf.if) CANNOT be accessed outside. To use them outside, you MUST pass them through 'iter_args' and 'results' (for scf.for) or 'results' (for scf.if), and yield them using 'scf.yield'.\n"
        "2. Data Types: Define the exact MLIR type ONLY in the function 'arguments' block or for explicit casting. Do not invent types.\n"
        "3. Valid Operands: In 'iter_args' and 'operands', you may use valid registers (e.g., '%0') or raw numerical constants (e.g., 2.0). Raw numbers will be automatically injected as 'arith.constant'. If you explicitly use 'arith.constant', you MUST provide the value in the 'attributes' field (e.g., \"attributes\": {\"value\": 1.0}).\n"
        "4. Exact Opcodes: You MUST ONLY use the opcodes provided in the JSON Schema Enum.\n"
        "5. Output Format: DO NOT include markdown code blocks (e.g., ```json) or text outside the JSON. Your output must be parseable by json.loads().\n"
        "6. Nested Blocks: For control flow like scf.for, use the nested 'body' list property. Do not use flat 'end_for' markers. To yield values to the next iteration of an scf.for loop or out of an scf.if block, add an object with 'opcode': 'scf.yield' and its 'operands' at the end of the 'body' list.\n"
        "7. NO Array Indexing: DO NOT use array indexing like '%arg0[%i]' or '%tensor[idx]'. This does NOT exist in MLIR SSA. To access tensor elements, use 'tensor.extract' with indices as separate operands. For whole-tensor operations, use vectorized ops directly.\n"
        "8. Reductions: If you use 'tt.reduce', you MUST include the 'region_combiner' field (e.g., \"region_combiner\": \"arith.addf\" or \"arith.maximumf\").\n"
        "9. NO UNDECLARED VARIABLES: You CANNOT use any register (e.g., '%is_final_pass', '%final_output') as an operand, condition, or return value unless it was EXPLICITLY defined beforehand either in the function 'arguments' or as the 'result' of a previous operation. Do not invent variables.\n"
        "10. Chain of Thought: The 'reasoning' field MUST be used to plan the topological order. You MUST list every register you will create (e.g., %0, %1) and explain where the 'iter_args' values come from before outputting the JSON.\n"
        "11. Tensor Operations: 'tensor.extract' returns a SINGLE scalar value, NOT a row or column. If you extract from a 2D tensor (e.g., tensor<128x128xf32>), you MUST provide TWO indices (e.g., operands: ['%tensor', '%row_idx', '%col_idx']).\n"
        "12. scf.for Rules: If a loop has 'iter_args', the VERY LAST operation in its 'body' array MUST be an 'scf.yield' with the exact same number of operands as the 'iter_args'. Do not forget the final yield.\n\n"
        "EXPECTED EXAMPLE (Iterative Max with scf.for and scf.if):\n"
        "User: \"Find the maximum value in a tensor by iterating and keeping the running max.\"\n"
        "Response JSON:\n"
        "{\n"
        "  \"reasoning\": \"I will use an scf.for loop to iterate. I need a loop variable %i. The running max will be passed via iter_args %current_max. Inside the loop, I load the value into %val, compare it using arith.cmpf into %is_greater, and use scf.if to select the new max into %new_max. Finally, I yield %new_max to the next iteration.\",\n"
        "  \"code\": {\n"
        "    \"function_name\": \"find_max\",\n"
        "    \"arguments\": [\n"
        "      {\"name\": \"%tensor\", \"type\": \"tensor<128xf32>\"}\n"
        "    ],\n"
        "    \"operations\": [\n"
        "      {\n"
        "        \"opcode\": \"scf.for\",\n"
        "        \"lower_bound\": 0,\n"
        "        \"upper_bound\": 128,\n"
        "        \"step\": 1,\n"
        "        \"loop_var\": \"%i\",\n"
        "        \"iter_args\": {\"%current_max\": 0.0},\n"
        "        \"results\": [\"%final_max\"],\n"
        "        \"body\": [\n"
        "          {\"opcode\": \"tensor.extract\", \"operands\": [\"%tensor\", \"%i\"], \"result\": \"%val\"},\n"
        "          {\"opcode\": \"arith.cmpf\", \"operands\": [\"%val\", \"%current_max\"], \"result\": \"%is_greater\", \"attributes\": {\"predicate\": 2}},\n"
        "          {\n"
        "            \"opcode\": \"scf.if\",\n"
        "            \"condition\": \"%is_greater\",\n"
        "            \"results\": [\"%new_max\"],\n"
        "            \"then_body\": [\n"
        "              {\"opcode\": \"scf.yield\", \"operands\": [\"%val\"]}\n"
        "            ],\n"
        "            \"else_body\": [\n"
        "              {\"opcode\": \"scf.yield\", \"operands\": [\"%current_max\"]}\n"
        "            ]\n"
        "          },\n"
        "          {\"opcode\": \"scf.yield\", \"operands\": [\"%new_max\"]}\n"
        "        ]\n"
        "      }\n"
        "    ],\n"
        "    \"returns\": [\"%final_max\"]\n"
        "  }\n"
        "}\n"
    )
    
    user_prompt = (
        "Generate a kernel that performs one-pass online softmax. "
        "Goal: compute the softmax operation across the rows of an input matrix in a single global memory pass, "
        "using an online reduction algorithm to dynamically update the local maximum and the sum of exponentials."
    )
    
    # Experiment parameters
    max_retries = 3
    success = False
    
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        if tracker: tracker.start_timer()
        
        # Call the model using Constrained Decoding (XGrammar / Structured Outputs)
        print("Waiting for LLM generation (Constrained Decoding enabled)...")
        try:
            # Pass the actual Pydantic schema to the client (XGrammar at API level)
            raw_response = generate_llm_response("ollama", system_prompt, user_prompt, schema=MlirResponse)
        except Exception as e:
            print(f"[!] Error calling the API: {e}")
            break
            
        print("LLM responded. Parsing JSON contract...")
        mlir_code = ""
        error_msg = None
        
        try:
            # 1. Syntactic Validation (Almost always passes thanks to Constrained Decoding)
            # Limpieza básica por si el modelo remoto responde con bloques de markdown
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
            
            # 2. Semantic AST Validation
            from core.semantic_validator import SemanticValidator
            semantic_errors = SemanticValidator.validate(response_obj)
            if semantic_errors:
                raise RuntimeError("Multiple Semantic Errors Found:\n" + "\n".join(semantic_errors))
            
            # 3. Translation to MLIR
            print("Translating to MLIR Dialects...")
            mlir_code = translator.translate_to_module(response_obj.code)
            
            success = True
            print("[✓] Successful MLIR Compilation!")
            print(mlir_code)
            
            if tracker: tracker.log_iteration(attempt, user_prompt, raw_response, mlir_code, success, None)
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
        if tracker: tracker.log_iteration(attempt, user_prompt, raw_response, mlir_code, success, error_msg)

        feedback_context = (
            f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
            f"You generated this JSON:\n{raw_response}\n\n"
            f"But it failed semantic validation with the following error(s):\n"
            f"[COMPILER ERROR]: {error_msg}\n"
        )

        if "not found in environment" in error_msg:
            user_prompt += (
                f"\n\n[COMPILER ERROR]: {error_msg}\n"
                f"CRITICAL RULE VIOLATION: You used a register that DOES NOT EXIST. "
                f"In MLIR, you cannot invent variables like '%is_final_pass'. "
                f"If you need a boolean condition for scf.if, you MUST compute it first using 'arith.cmpf' and assign it to a result register. "
                f"If you need to return a value, it MUST be the 'result' of a previous operation."
            )
        elif "requires a single operand" in error_msg:
            user_prompt += feedback_context + "Correct the arity of the operation and output the fixed JSON."
        else:
            user_prompt += feedback_context + "Analyze the error, correct the logical flaw in the JSON, and output the fixed version."

    if tracker: tracker.finish()
    print("\nProcess finished.")

if __name__ == "__main__":
    main()
