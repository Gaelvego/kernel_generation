# Semantic GPU Kernel Generator — LLM-MLIR Compiler

> A hybrid pipeline for generating semantically correct Triton/MLIR GPU kernels using Large Language Models and Constrained Decoding.

Traditional LLM pipelines for GPU kernel generation tend to fail silently — producing code that looks valid but contains syntax errors or hallucinated operations. This project takes a different approach through a **Semantically Constrained Generation** architecture. Rather than generating raw Triton or Python code directly, the pipeline drives an LLM (Gemma-4) to output a strict JSON Abstract Syntax Tree (AST). That AST is then translated locally into MLIR (Multi-Level Intermediate Representation) dialects, where static mathematical and semantic validation occurs before any compilation step.

---

## System Architecture

The pipeline runs across a decoupled local/cloud environment with three main components:

1. **Local Orchestrator (Python)** — Handles prompt formulation, JSON schema enforcement via Pydantic, AST-to-MLIR translation, and MLOps metric tracking.
2. **Remote Inference Engine (Kaggle/Colab)** — Hosts the `gemma-4-e4b-it` model behind a FastAPI endpoint exposed through an Ngrok tunnel.
3. **Constrained Decoding (XGrammar)** — The orchestrator sends a Pydantic JSON Schema to the remote endpoint, which compiles it with `xgrammar` to constrain the model's output logits, preventing malformed JSON or illegal MLIR operations at the generation level.

---

## 1. Initial Setup & Installation

### 1.1 — Clone the Repository and Create the Environment

Using a virtual environment is required to isolate project dependencies and avoid conflicts with system packages.

```bash
# Clone the repository
git clone https://github.com/Gaelvego/kernel_generation.git
cd llm-mlir-compiler

# Create a Python 3.10 virtual environment
python3.10 -m venv .venv

# Activate the virtual environment

# Linux / macOS:
source .venv/bin/activate

# Windows:
# .venv\Scripts\activate
```

### 1.2 — Install Python Dependencies

With the virtual environment active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 — Configure MLIR Python Bindings *(Critical)*

The local orchestrator depends on native LLVM/MLIR bindings for semantic validation.

👉 Refer to [INSTALL_MLIR.md](./INSTALL_MLIR.md) for the complete guide on building LLVM/MLIR from source or using the provided Docker containers. This step is required before running the pipeline.

---

## ☁️ 2. Remote Model Configuration (Kaggle)

The LLM runs remotely on Kaggle and connects back to the local orchestrator via API.

1. Open Kaggle and import the notebook included in the repository (`.ipynb`).
2. Turn on internet and enable GPU acceleration (T4 x2 or P100) in the session settings.
3. Run all cells to initialize FastAPI, download the model weights, and start the Ngrok tunnel.
4. Copy the generated **Ngrok Public URL** from the cell output — it will follow the pattern `https://<random-id>.ngrok-free.dev`.

---

## 🔐 3. Environment Variables

Create a `.env` file in the project root and fill in the values below:

```env
# Remote model endpoint (paste the Ngrok URL from Kaggle here)
USE_REMOTE_MODEL=1        # Set to 1 for remote inference, 0 for local Ollama
GEMMA_API_URL="https://your-ngrok-url.ngrok-free.dev"

# MLOps tracking (Weights & Biases)
WANDB_API_KEY=your_wandb_api_key_here

# External fallback APIs (optional)
NVIDIA_API_KEY=your_nvapi_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🚀 4. Running the Pipeline

Once MLIR is configured, the virtual environment is active, and the Kaggle server is running:

```bash
# Optional: authenticate with Weights & Biases for live MLOps tracking
wandb login

# Start the orchestrator
python main.py
```

Each run logs metrics to your W&B dashboard, including prompt latency, schema validation success rates, full JSON outputs, and the resulting MLIR AST representations.

---

## 📂 5. Project Structure

```
llm-mlir-compiler/
├── core/
│   ├── config.py           # Global environment variables and thresholds
│   ├── llm_client.py       # API router (Remote, Gemini, Kimi, Ollama)
│   ├── mlir_translator.py  # Converts JSON AST to MLIR dialects
│   ├── mlops_tracker.py    # Weights & Biases integration
│   ├── schemas.py          # Pydantic schemas that drive XGrammar constraints
│   └── triton_backend.py   # Lowering logic: TTIR → TTGIR → PTX (NVIDIA)
├── main.py                 # Entry point and orchestration loop
├── requirements.txt        # Python dependencies
├── INSTALL_MLIR.md         # MLIR/LLVM setup guide
└── .env                    # Environment variables (git-ignored)
```

---

## 🛠️ 6. Extending the Project

Currently in develop.

---

## 📚 7. References

- [MLIR Documentation](https://mlir.llvm.org/docs/) — Multi-Level Intermediate Representation
- [Triton MLIR Dialects](https://triton-lang.org/main/dialects/dialects.html)
- [XGrammar — HuggingFace](https://huggingface.co/docs/text-generation-inference/conceptual/guidance)
- [Weights & Biases MLOps](https://docs.wandb.ai/)