"""Constants and model configuration for CoT Legibility Phase 1."""

from pathlib import Path

# --- Working directories ---
BASE_DIR = Path(__file__).parent
LOG_DIR_GENERATION = str(BASE_DIR / "logs" / "step1_generation")
LOG_DIR_READERS = str(BASE_DIR / "logs" / "step2_readers")
RESULTS_DIR = BASE_DIR / "results"

# --- Generators ---
GENERATORS = {
    "G1": "openrouter/deepseek/deepseek-r1-distill-qwen-32b",
    "G2": "openrouter/deepseek/deepseek-r1-0528",
    "G3": "openrouter/qwen/qwq-32b",
}

# --- Readers ---
READERS = {
    "R1": "openrouter/qwen/qwen3-32b",
    "R2": "openrouter/meta-llama/llama-3.1-70b-instruct",
    "R3": "openrouter/deepseek/deepseek-v3",
    "R4": "openrouter/qwen/qwen3-4b",
}

FULL_READERS = ["R1", "R2", "R3"]  # R4 is C2-only

# --- Generation hyperparameters ---
K_SAMPLES = 6
COT_TEMPERATURE = 0.7
COT_MAX_TOKENS = 16384
ANSWER_MAX_TOKENS = 256

# --- HuggingFace dataset identifiers ---
GPQA_DATASET = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"
MATH_DATASET = "nlile/hendrycks-MATH-benchmark"
MATH_MIN_LEVEL = 3
MATH_MAX_LEVEL = 5
