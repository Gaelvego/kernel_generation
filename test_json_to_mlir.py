#!/usr/bin/env python3
"""
Test script for MLIR translation from pre-existing JSON.
No LLM generation required - just loads JSON and runs it through the pipeline.
Designed to run in Colab with the extracted GitHub Actions tarball.
"""
import json
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.schemas import MlirResponse
from core.mlir_translator import MLIRTranslator
from core.semantic_validator import SemanticValidator

def test_json_file(json_path: str):
    """
    Load a JSON file and test the full MLIR translation pipeline.
    """
    print(f"[✓] Loading JSON from: {json_path}")
    
    with open(json_path, 'r') as f:
        raw_json = f.read()
    
    print(f"[✓] JSON loaded ({len(raw_json)} chars)")
    
    # Clean up markdown fences if present
    clean_json = raw_json.strip()
    if clean_json.startswith("```json"):
        clean_json = clean_json[7:]
    if clean_json.startswith("```"):
        clean_json = clean_json[3:]
    if clean_json.endswith("```"):
        clean_json = clean_json[:-3]
    clean_json = clean_json.strip()
    
    # 1. Parse JSON
    print("[✓] Parsing JSON...")
    try:
        parsed = json.loads(clean_json)
        response = MlirResponse(**parsed)
        print("[✓] JSON parsed and validated against Pydantic schema")
    except json.JSONDecodeError as e:
        print(f"[✗] JSON parsing failed: {e}")
        return False
    except Exception as e:
        print(f"[✗] Pydantic validation failed: {e}")
        return False
    
    # 2. Semantic validation
    print("[✓] Running semantic validation...")
    try:
        errors = SemanticValidator.validate(response)
        if errors:
            print(f"[✗] Semantic errors found:")
            for err in errors:
                print(f"  - {err}")
            return False
        print("[✓] Semantic validation passed")
    except Exception as e:
        print(f"[✗] Semantic validation error: {e}")
        return False
    
    # 3. MLIR translation
    print("[✓] Translating to MLIR...")
    try:
        translator = MLIRTranslator()
        mlir_code = translator.translate_to_module(response.code)
        print("[✓] MLIR translation successful!")
        print("\n--- Generated MLIR ---")
        print(mlir_code)
        print("--- End MLIR ---\n")
        return True
    except ImportError as e:
        print(f"[✗] MLIR bindings not found: {e}")
        print("[ℹ] Make sure the tarball is extracted and in the PYTHONPATH")
        return False
    except Exception as e:
        print(f"[✗] MLIR translation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MLIR translation from JSON")
    parser.add_argument("--json", required=True, help="Path to JSON file")
    args = parser.parse_args()
    
    success = test_json_file(args.json)
    sys.exit(0 if success else 1)
