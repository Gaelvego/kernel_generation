"""
Generate a self-contained Colab notebook that embeds the entire repo,
builds/restores LLVM/MLIR, sets up Ollama, and runs the evaluation.
"""
import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def read_file(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return f.read()

def write_cell(name, content, ext="py"):
    """Create a %%writefile cell."""
    lines = [f"%%writefile /content/kernel_generation/{name}\n"]
    lines.append(content)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text
    }

# Read all source files
files_to_embed = {
    "core/__init__.py": "",
    "core/config.py": read_file("core/config.py"),
    "core/schemas.py": read_file("core/schemas.py"),
    "core/llm_client.py": read_file("core/llm_client.py"),
    "core/mlir_translator.py": read_file("core/mlir_translator.py"),
    "core/triton_generator.py": read_file("core/triton_generator.py"),
    "core/benchmarks.py": read_file("core/benchmarks.py"),
    "core/mlops_tracker.py": read_file("core/mlops_tracker.py"),
    "run_benchmarks.py": read_file("run_benchmarks.py"),
    "main.py": read_file("main.py"),
}

cells = []

# --- Cell 1: Title ---
cells.append(markdown_cell([
    "# GPU Kernel Generation — Colab Evaluation Notebook\n",
    "\n",
    "This notebook is **self-contained**: it builds/restores LLVM/MLIR, sets up Ollama + Gemma, writes the entire codebase, and runs the evaluation matrix.\n",
    "\n",
    "**How to use:**\n",
    "1. Upload this file to Google Colab (Runtime → Change runtime type → GPU).\n",
    "2. Run cells sequentially top-to-bottom.\n",
    "3. The first run will build LLVM (~45 min) and download Gemma (~10 min).\n",
    "4. Both are cached to Google Drive so future sessions restore in ~2 minutes.\n",
    "\n",
    "**Drive space required:** ~12GB (LLVM build ~8GB + Gemma ~3GB + results).\n"
]))

# --- Cell 2: Mount Drive ---
cells.append(code_cell([
    "from google.colab import drive\n",
    "drive.mount('/content/drive')\n",
    "\n",
    "import os\n",
    "os.makedirs('/content/drive/MyDrive/llvm-build-cache', exist_ok=True)\n",
    "os.makedirs('/content/drive/MyDrive/ollama-models', exist_ok=True)\n",
    "os.makedirs('/content/drive/MyDrive/gpu-kernel-results', exist_ok=True)\n",
    "print('[✓] Drive mounted and cache directories ready.')\n"
]))

# --- Cell 3: System deps ---
cells.append(code_cell([
    "!apt-get update -qq\n",
    "!apt-get install -y -qq cmake ninja-build clang lld ccache\n",
    "print('[✓] System dependencies installed.')\n"
]))

# --- Cell 4: Install Ollama ---
cells.append(code_cell([
    "!curl -fsSL https://ollama.com/install.sh | sh\n",
    "print('[✓] Ollama installed.')\n"
]))

# --- Cell 5: Python deps ---
cells.append(code_cell([
    "!pip install -q torch wandb requests python-dotenv pydantic triton\n",
    "print('[✓] Python packages installed.')\n"
]))

# --- Cell 6: Environment variables ---
cells.append(code_cell([
    "import os\n",
    "os.environ['OLLAMA_MODELS'] = '/content/drive/MyDrive/ollama-models'\n",
    "os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'\n",
    "os.environ['COLAB_EVAL_DIR'] = '/content/drive/MyDrive/gpu-kernel-results'\n",
    "os.environ['PATH'] += ':/usr/local/bin'\n",
    "print('[✓] Environment configured for Colab.')\n"
]))

# --- Cell 7: LLVM Build or Restore ---
cells.append(code_cell([
    "import os\n",
    "import sys\n",
    "\n",
    "LLVM_BUILD_CACHE = '/content/drive/MyDrive/llvm-build-cache/build'\n",
    "LLVM_BUILD_LOCAL = '/content/llvm-project/build'\n",
    "\n",
    "def check_mlir_valid(path):\n",
    "    return os.path.exists(os.path.join(path, 'tools', 'mlir', 'python_packages', 'mlir_core'))\n",
    "\n",
    "if check_mlir_valid(LLVM_BUILD_CACHE):\n",
    "    print('[LLVM] Found cached build in Drive. Restoring...')\n",
    "    !mkdir -p /content/llvm-project\n",
    "    !cp -r {LLVM_BUILD_CACHE} {LLVM_BUILD_LOCAL}\n",
    "    print('[LLVM] Restored from Drive cache.')\n",
    "else:\n",
    "    print('[LLVM] No cache found. Building from source (~45 minutes)...')\n",
    "    !git clone --depth 1 --branch llvmorg-20.1.0 https://github.com/llvm/llvm-project.git /content/llvm-project\n",
    "    !pip install -q -r /content/llvm-project/mlir/python/requirements.txt\n",
    "    !mkdir -p {LLVM_BUILD_LOCAL}\n",
    "    !cd {LLVM_BUILD_LOCAL} && cmake -G Ninja ../llvm \\\n",
    "       -DLLVM_ENABLE_PROJECTS=\"mlir\" \\\n",
    "       -DLLVM_BUILD_EXAMPLES=OFF \\\n",
    "       -DLLVM_BUILD_TOOLS=OFF \\\n",
    "       -DLLVM_TARGETS_TO_BUILD=\"Native\" \\\n",
    "       -DMLIR_ENABLE_BINDINGS_PYTHON=ON \\\n",
    "       -DCMAKE_BUILD_TYPE=Release \\\n",
    "       -DPython3_EXECUTABLE=$(which python)\n",
    "    print('[LLVM] Starting build...')\n",
    "    !cd {LLVM_BUILD_LOCAL} && cmake --build . --target mlir-python-bindings\n",
    "    print('[LLVM] Build complete. Caching to Drive...')\n",
    "    !mkdir -p /content/drive/MyDrive/llvm-build-cache\n",
    "    !cp -r {LLVM_BUILD_LOCAL} {LLVM_BUILD_CACHE}\n",
    "    print('[LLVM] Cached to Drive.')\n",
    "\n",
    "# Add to PYTHONPATH\n",
    "sys.path.append(f'{LLVM_BUILD_LOCAL}/tools/mlir/python_packages/mlir_core')\n",
    "print('[✓] LLVM/MLIR path configured.')\n"
]))

# --- Cell 8: Verify MLIR ---
cells.append(code_cell([
    "from mlir.ir import Context\n",
    "print('[✓] MLIR Python bindings loaded successfully.')\n"
]))

# --- Cell 9: Start Ollama ---
cells.append(code_cell([
    "import subprocess\n",
    "import time\n",
    "\n",
    "# Start Ollama server in background\n",
    "subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "time.sleep(5)\n",
    "print('[✓] Ollama server started.')\n"
]))

# --- Cell 10: Pull Gemma ---
cells.append(code_cell([
    "import os\n",
    "model_tag = os.getenv('OLLAMA_MODEL', 'gemma4:e2b')\n",
    "print(f'[Ollama] Pulling model: {model_tag} ...')\n",
    "!ollama pull {model_tag}\n",
    "print('[✓] Model ready.')\n",
    "!ollama run {model_tag} \"Hello, are you working?\" --nowordwrap\n"
]))

# --- Cells 11-20: Write repo files ---
for rel_path, content in files_to_embed.items():
    if rel_path == "core/__init__.py":
        cells.append(write_cell(rel_path, "# GPU Kernel Generation\n"))
    else:
        cells.append(write_cell(rel_path, content))

# --- Cell 21: Sanity check ---
cells.append(code_cell([
    "import sys\n",
    "sys.path.insert(0, '/content/kernel_generation')\n",
    "from core.config import DEFAULT_MODE, EVAL_OUTPUT_DIR\n",
    "print(f'[✓] Config loaded. DEFAULT_MODE={DEFAULT_MODE}, EVAL_OUTPUT_DIR={EVAL_OUTPUT_DIR}')\n",
    "from core.benchmarks import BENCHMARKS\n",
    "print(f'[✓] Benchmarks loaded: {[b.name for b in BENCHMARKS]}')\n"
]))

# --- Cell 22: Quick JSON test ---
cells.append(code_cell([
    "!cd /content/kernel_generation && python main.py --mode json --model ollama --no-mlops --prompt \"Write a vector addition kernel\"\n"
]))

# --- Cell 23: Quick raw test ---
cells.append(code_cell([
    "!cd /content/kernel_generation && python main.py --mode raw --model ollama --no-mlops --prompt \"Write a vector addition kernel\"\n"
]))

# --- Cell 24: Full evaluation ---
cells.append(code_cell([
    "!cd /content/kernel_generation && python run_benchmarks.py --models ollama --modes json raw raw_retry\n"
]))

# --- Cell 25: Summarize results ---
cells.append(code_cell([
    "import json\n",
    "import glob\n",
    "import os\n",
    "\n",
    "result_dir = '/content/drive/MyDrive/gpu-kernel-results'\n",
    "files = sorted(glob.glob(os.path.join(result_dir, '*.json')))\n",
    "\n",
    "if not files:\n",
    "    print('No result files found in Drive.')\n",
    "else:\n",
    "    latest = files[-1]\n",
    "    print(f'Loading: {latest}')\n",
    "    with open(latest) as f:\n",
    "        data = json.load(f)\n",
    "\n",
    "    total = len(data)\n",
    "    passed = sum(1 for r in data if r['success'])\n",
    "    print(f\"\\nOverall: {passed}/{total} successes ({100*passed/total:.1f}%)\")\n",
    "\n",
    "    from collections import defaultdict\n",
    "    grouped = defaultdict(lambda: defaultdict(dict))\n",
    "    for r in data:\n",
    "        grouped[r['benchmark']][r['model']][r['mode']] = r\n",
    "\n",
    "    for bench in grouped:\n",
    "        print(f\"\\nBenchmark: {bench}\")\n",
    "        print('-'*50)\n",
    "        for model in grouped[bench]:\n",
    "            for mode in grouped[bench][model]:\n",
    "                r = grouped[bench][model][mode]\n",
    "                status = 'PASS' if r['success'] else 'FAIL'\n",
    "                attempts = r.get('attempts', '-')\n",
    "                lat = r.get('latency_ms', 0)\n",
    "                print(f\"  {model:12s} | {mode:12s} | {status:4s} | attempts={attempts} | {lat:.0f}ms\")\n"
]))

# Assemble notebook
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        }
    },
    "cells": cells
}

output_path = os.path.join(REPO_ROOT, "Colab_GPU_Kernel_Evaluation.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[✓] Notebook generated: {output_path}")
print(f"    Total cells: {len(cells)}")
print(f"    Upload this file to Google Colab to run the evaluation.")
