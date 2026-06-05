import os
import sys
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# --- NATIVE MLIR ENVIRONMENT SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Assumes llvm-project is inside the main llm-mlir-compiler directory
LLVM_PROJECT_DIR = os.path.join(PROJECT_ROOT, "llvm-project")
MLIR_BINDINGS_PATH = os.path.join(LLVM_PROJECT_DIR, "build", "tools", "mlir", "python_packages", "mlir_core")

# Colab fallback path (GitHub Actions tarball)
COLAB_MLIR_PATH = "/content/llvm-prebuilt/llvm-install/python_packages/mlir_core"

if os.path.exists(MLIR_BINDINGS_PATH):
    sys.path.append(MLIR_BINDINGS_PATH)
elif os.path.exists(COLAB_MLIR_PATH):
    sys.path.append(COLAB_MLIR_PATH)
else:
    print(f"[Warning] Native MLIR binding not found at {MLIR_BINDINGS_PATH} or {COLAB_MLIR_PATH}. Ensure LLVM is compiled.")

# --- API KEYS ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")

# --- MODEL CONFIGURATION ---
MODEL_KIMI = "meta/llama-3.1-405b-instruct" # Proxy model in Nvidia API
MODEL_GEMINI = "gemini-3.1-flash-lite"      # Gemini model
MODEL_OLLAMA = "gemma4:e2b"                 # Local Ollama model

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"

USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "0") == "1"
REMOTE_MODEL_URL = os.getenv("GEMMA_API_URL", "https://unsubtly-dash-economy.ngrok-free.dev")

# Strict parameters for code generation
GENERATION_PARAMS = {
    "max_tokens": 4096,
    "temperature": 0.1,  # Low temperature for determinism in code generation
    "top_p": 0.9,
}

# --- MLOPS CONFIGURATION ---
WANDB_PROJECT_NAME = "llm-mlir-compiler"
WANDB_ENTITY = None  # Automatic if configured in the environment

