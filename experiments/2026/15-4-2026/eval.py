"""CoT Legibility Phase 1 — @task definitions for inspect eval / eval-set.

Step 1 — Generate CoTs (run first):
    inspect eval-set eval.py@cot_gen_G1 eval.py@cot_gen_G2 eval.py@cot_gen_G3 \
        --log-dir logs/step1_generation

Step 2 — Reader evaluation (run after Step 1):
    # C1/C4 baselines
    inspect eval-set eval.py@reader_c1_R1 eval.py@reader_c1_R2 eval.py@reader_c1_R3 \
        eval.py@reader_c4_R1 eval.py@reader_c4_R2 eval.py@reader_c4_R3 \
        --log-dir logs/step2_readers

    # C2 crossfill (full readers)
    inspect eval-set eval.py@reader_c2_R1_G1 eval.py@reader_c2_R1_G2 eval.py@reader_c2_R1_G3 \
        eval.py@reader_c2_R2_G1 eval.py@reader_c2_R2_G2 eval.py@reader_c2_R2_G3 \
        eval.py@reader_c2_R3_G1 eval.py@reader_c2_R3_G2 eval.py@reader_c2_R3_G3 \
        --log-dir logs/step2_readers

    # C2 crossfill (R4 leak detector — canonical)
    inspect eval-set eval.py@reader_c2_R4_G1 eval.py@reader_c2_R4_G2 eval.py@reader_c2_R4_G3 \
        --log-dir logs/step2_readers

    # C2 crossfill (ALL readers — truncated last 64 tokens)
    inspect eval-set eval.py@reader_c2_R{1,2,3,4}_t64_G{1,2,3} --log-dir logs/step2_readers

    # C2 crossfill (ALL readers — truncated last 5% chars)
    inspect eval-set eval.py@reader_c2_R{1,2,3,4}_t5p_G{1,2,3} --log-dir logs/step2_readers

    # C2 crossfill (ALL readers — answer patterns masked with pad token)
    inspect eval-set eval.py@reader_c2_R{1,2,3,4}_mask_G{1,2,3} --log-dir logs/step2_readers

    # C2 crossfill (ALL readers — truncated at first answer-leak)
    inspect eval-set eval.py@reader_c2_R{1,2,3,4}_tleak_G{1,2,3} --log-dir logs/step2_readers

Individual tasks:
    inspect eval eval.py@cot_gen_G1 --max-samples 5 --epochs 1
    inspect eval eval.py@reader_c2_R1_G1 --max-samples 5

Full pipeline (Python):
    python eval.py
    python eval.py --step 1
    python eval.py --step 2
    python eval.py --test
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from inspect_ai import Task, eval, eval_set, task
from inspect_ai.model import GenerateConfig

from config import (
    GENERATORS,
    READERS,
    FULL_READERS,
    K_SAMPLES,
    COT_TEMPERATURE,
    COT_MAX_TOKENS,
    ANSWER_MAX_TOKENS,
    LOG_DIR_GENERATION,
    LOG_DIR_READERS,
)
from data import (
    build_combined_dataset,
    extract_cots_from_logs,
    build_c2_dataset,
    truncate_last_n_tokens,
    truncate_last_pct,
    make_mask_transform,
    truncate_at_first_leak,
)
from solvers import (
    cot_generation_solver,
    crossfill_solver,
    self_cot_solver,
    no_cot_solver,
)
from scorers import (
    generator_correctness_scorer,
    reader_correctness_scorer,
)


# ===================================================================
# Step 1: CoT Generation tasks
# ===================================================================

def _cot_gen(generator_id: str, epochs: int = K_SAMPLES) -> Task:
    return Task(
        dataset=build_combined_dataset(),
        solver=cot_generation_solver(),
        scorer=generator_correctness_scorer(),
        model=GENERATORS[generator_id],
        config=GenerateConfig(
            max_tokens=COT_MAX_TOKENS,
            temperature=COT_TEMPERATURE,
        ),
        epochs=epochs,
        name=f"cot_gen_{generator_id}",
        metadata={"generator_id": generator_id, "step": "generation"},
    )


@task
def cot_gen_G1(epochs: int = K_SAMPLES) -> Task:
    """Generate CoTs from G1 (DeepSeek-R1-Distill-Qwen-32B)."""
    return _cot_gen("G1", epochs=epochs)


@task
def cot_gen_G2(epochs: int = K_SAMPLES) -> Task:
    """Generate CoTs from G2 (DeepSeek-R1)."""
    return _cot_gen("G2", epochs=epochs)


@task
def cot_gen_G3(epochs: int = K_SAMPLES) -> Task:
    """Generate CoTs from G3 (QwQ-32B)."""
    return _cot_gen("G3", epochs=epochs)


# ===================================================================
# Step 2: Reader C1 tasks (self-CoT baseline)
# ===================================================================

def _reader_c1(reader_id: str) -> Task:
    return Task(
        dataset=build_combined_dataset(),
        solver=self_cot_solver(),
        scorer=reader_correctness_scorer(),
        model=READERS[reader_id],
        config=GenerateConfig(
            max_tokens=COT_MAX_TOKENS,
            temperature=0.0,
        ),
        name=f"reader_c1_{reader_id}",
        metadata={"reader_id": reader_id, "condition": "C1", "step": "reader"},
    )


@task
def reader_c1_R1() -> Task:
    """C1: R1 (Qwen3-32B) generates its own CoT."""
    return _reader_c1("R1")


@task
def reader_c1_R2() -> Task:
    """C1: R2 (Llama-3.1-70B) generates its own CoT."""
    return _reader_c1("R2")


@task
def reader_c1_R3() -> Task:
    """C1: R3 (DeepSeek-V3) generates its own CoT."""
    return _reader_c1("R3")


# ===================================================================
# Step 2: Reader C4 tasks (no-CoT baseline)
# ===================================================================

def _reader_c4(reader_id: str) -> Task:
    return Task(
        dataset=build_combined_dataset(),
        solver=no_cot_solver(reader_id=reader_id),
        scorer=reader_correctness_scorer(),
        model=READERS[reader_id],
        config=GenerateConfig(
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
        ),
        name=f"reader_c4_{reader_id}",
        metadata={"reader_id": reader_id, "condition": "C4", "step": "reader"},
    )


@task
def reader_c4_R1() -> Task:
    """C4: R1 (Qwen3-32B) answers without CoT."""
    return _reader_c4("R1")


@task
def reader_c4_R2() -> Task:
    """C4: R2 (Llama-3.1-70B) answers without CoT."""
    return _reader_c4("R2")


@task
def reader_c4_R3() -> Task:
    """C4: R3 (DeepSeek-V3) answers without CoT."""
    return _reader_c4("R3")


# ===================================================================
# Step 2: Reader C2 tasks (crossfill with generator CoT)
#
# These load CoTs from Step 1 logs at task creation time.
# Step 1 must be complete before running these.
# ===================================================================

def _load_cots() -> dict:
    """Load CoTs from Step 1 logs. Exits if none found."""
    cots = extract_cots_from_logs(LOG_DIR_GENERATION)
    if not cots:
        print(f"ERROR: No CoTs found in {LOG_DIR_GENERATION}. Run Step 1 first.")
        print("  inspect eval-set eval.py@cot_gen_G1 eval.py@cot_gen_G2 eval.py@cot_gen_G3 "
              f"--log-dir {LOG_DIR_GENERATION}")
        sys.exit(1)
    return cots


def _reader_c2(
    reader_id: str,
    generator_id: str,
    cot_transform=None,
    name_suffix: str = "",
) -> Task:
    cots = _load_cots()
    dataset = build_c2_dataset(cots, reader_id, generator_id, cot_transform=cot_transform)
    task_name = f"reader_c2_{reader_id}{name_suffix}_{generator_id}"
    return Task(
        dataset=dataset,
        solver=crossfill_solver(reader_id=reader_id),
        scorer=reader_correctness_scorer(),
        model=READERS[reader_id],
        config=GenerateConfig(
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
        ),
        name=task_name,
        metadata={
            "reader_id": reader_id,
            "generator_id": generator_id,
            "condition": "C2",
            "step": "reader",
            "cot_transform": name_suffix or "none",
        },
    )


# --- R1 x generators ---

@task
def reader_c2_R1_G1() -> Task:
    """C2: R1 reads G1's CoT."""
    return _reader_c2("R1", "G1")


@task
def reader_c2_R1_G2() -> Task:
    """C2: R1 reads G2's CoT."""
    return _reader_c2("R1", "G2")


@task
def reader_c2_R1_G3() -> Task:
    """C2: R1 reads G3's CoT."""
    return _reader_c2("R1", "G3")


# --- R2 x generators ---

@task
def reader_c2_R2_G1() -> Task:
    """C2: R2 reads G1's CoT."""
    return _reader_c2("R2", "G1")


@task
def reader_c2_R2_G2() -> Task:
    """C2: R2 reads G2's CoT."""
    return _reader_c2("R2", "G2")


@task
def reader_c2_R2_G3() -> Task:
    """C2: R2 reads G3's CoT."""
    return _reader_c2("R2", "G3")


# --- R3 x generators ---

@task
def reader_c2_R3_G1() -> Task:
    """C2: R3 reads G1's CoT."""
    return _reader_c2("R3", "G1")


@task
def reader_c2_R3_G2() -> Task:
    """C2: R3 reads G2's CoT."""
    return _reader_c2("R3", "G2")


@task
def reader_c2_R3_G3() -> Task:
    """C2: R3 reads G3's CoT."""
    return _reader_c2("R3", "G3")


# --- R4 x generators (leak detector, C2 only) ---

@task
def reader_c2_R4_G1() -> Task:
    """C2: R4 (tiny) reads G1's CoT — leak detection."""
    return _reader_c2("R4", "G1")


@task
def reader_c2_R4_G2() -> Task:
    """C2: R4 (tiny) reads G2's CoT — leak detection."""
    return _reader_c2("R4", "G2")


@task
def reader_c2_R4_G3() -> Task:
    """C2: R4 (tiny) reads G3's CoT — leak detection."""
    return _reader_c2("R4", "G3")


# --- R5 x generators (Google-lineage control, C2 only) ---

@task
def reader_c2_R5_G1() -> Task:
    """C2: R5 (Gemma-4-31B) reads G1's CoT — Google-lineage control."""
    return _reader_c2("R5", "G1")


@task
def reader_c2_R5_G2() -> Task:
    """C2: R5 (Gemma-4-31B) reads G2's CoT — Google-lineage control."""
    return _reader_c2("R5", "G2")


@task
def reader_c2_R5_G3() -> Task:
    """C2: R5 (Gemma-4-31B) reads G3's CoT — Google-lineage control."""
    return _reader_c2("R5", "G3")


# --- ALL readers: truncated last 64 tokens ---

@task
def reader_c2_R1_t64_G1() -> Task:
    """C2: R1 reads G1's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R1", "G1", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R1_t64_G2() -> Task:
    """C2: R1 reads G2's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R1", "G2", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R1_t64_G3() -> Task:
    """C2: R1 reads G3's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R1", "G3", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R2_t64_G1() -> Task:
    """C2: R2 reads G1's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R2", "G1", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R2_t64_G2() -> Task:
    """C2: R2 reads G2's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R2", "G2", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R2_t64_G3() -> Task:
    """C2: R2 reads G3's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R2", "G3", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R3_t64_G1() -> Task:
    """C2: R3 reads G1's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R3", "G1", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R3_t64_G2() -> Task:
    """C2: R3 reads G2's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R3", "G2", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R3_t64_G3() -> Task:
    """C2: R3 reads G3's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R3", "G3", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R4_t64_G1() -> Task:
    """C2: R4 reads G1's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R4", "G1", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R4_t64_G2() -> Task:
    """C2: R4 reads G2's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R4", "G2", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R4_t64_G3() -> Task:
    """C2: R4 reads G3's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R4", "G3", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R5_t64_G1() -> Task:
    """C2: R5 reads G1's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R5", "G1", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R5_t64_G2() -> Task:
    """C2: R5 reads G2's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R5", "G2", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


@task
def reader_c2_R5_t64_G3() -> Task:
    """C2: R5 reads G3's CoT truncated (last 64 tokens removed)."""
    return _reader_c2("R5", "G3", cot_transform=truncate_last_n_tokens, name_suffix="_t64")


# --- ALL readers: truncated last 5% of characters ---

@task
def reader_c2_R1_t5p_G1() -> Task:
    """C2: R1 reads G1's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R1", "G1", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R1_t5p_G2() -> Task:
    """C2: R1 reads G2's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R1", "G2", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R1_t5p_G3() -> Task:
    """C2: R1 reads G3's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R1", "G3", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R2_t5p_G1() -> Task:
    """C2: R2 reads G1's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R2", "G1", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R2_t5p_G2() -> Task:
    """C2: R2 reads G2's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R2", "G2", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R2_t5p_G3() -> Task:
    """C2: R2 reads G3's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R2", "G3", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R3_t5p_G1() -> Task:
    """C2: R3 reads G1's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R3", "G1", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R3_t5p_G2() -> Task:
    """C2: R3 reads G2's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R3", "G2", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R3_t5p_G3() -> Task:
    """C2: R3 reads G3's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R3", "G3", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R4_t5p_G1() -> Task:
    """C2: R4 reads G1's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R4", "G1", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R4_t5p_G2() -> Task:
    """C2: R4 reads G2's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R4", "G2", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R4_t5p_G3() -> Task:
    """C2: R4 reads G3's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R4", "G3", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R5_t5p_G1() -> Task:
    """C2: R5 reads G1's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R5", "G1", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R5_t5p_G2() -> Task:
    """C2: R5 reads G2's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R5", "G2", cot_transform=truncate_last_pct, name_suffix="_t5p")


@task
def reader_c2_R5_t5p_G3() -> Task:
    """C2: R5 reads G3's CoT truncated (last 5% of chars removed)."""
    return _reader_c2("R5", "G3", cot_transform=truncate_last_pct, name_suffix="_t5p")


# --- ALL readers: answer-leaking patterns masked with pad token ---

@task
def reader_c2_R1_mask_G1() -> Task:
    """C2: R1 reads G1's CoT with answer-leaking patterns masked."""
    return _reader_c2("R1", "G1", cot_transform=make_mask_transform("R1"), name_suffix="_mask")


@task
def reader_c2_R1_mask_G2() -> Task:
    """C2: R1 reads G2's CoT with answer-leaking patterns masked."""
    return _reader_c2("R1", "G2", cot_transform=make_mask_transform("R1"), name_suffix="_mask")


@task
def reader_c2_R1_mask_G3() -> Task:
    """C2: R1 reads G3's CoT with answer-leaking patterns masked."""
    return _reader_c2("R1", "G3", cot_transform=make_mask_transform("R1"), name_suffix="_mask")


@task
def reader_c2_R2_mask_G1() -> Task:
    """C2: R2 reads G1's CoT with answer-leaking patterns masked."""
    return _reader_c2("R2", "G1", cot_transform=make_mask_transform("R2"), name_suffix="_mask")


@task
def reader_c2_R2_mask_G2() -> Task:
    """C2: R2 reads G2's CoT with answer-leaking patterns masked."""
    return _reader_c2("R2", "G2", cot_transform=make_mask_transform("R2"), name_suffix="_mask")


@task
def reader_c2_R2_mask_G3() -> Task:
    """C2: R2 reads G3's CoT with answer-leaking patterns masked."""
    return _reader_c2("R2", "G3", cot_transform=make_mask_transform("R2"), name_suffix="_mask")


@task
def reader_c2_R3_mask_G1() -> Task:
    """C2: R3 reads G1's CoT with answer-leaking patterns masked."""
    return _reader_c2("R3", "G1", cot_transform=make_mask_transform("R3"), name_suffix="_mask")


@task
def reader_c2_R3_mask_G2() -> Task:
    """C2: R3 reads G2's CoT with answer-leaking patterns masked."""
    return _reader_c2("R3", "G2", cot_transform=make_mask_transform("R3"), name_suffix="_mask")


@task
def reader_c2_R3_mask_G3() -> Task:
    """C2: R3 reads G3's CoT with answer-leaking patterns masked."""
    return _reader_c2("R3", "G3", cot_transform=make_mask_transform("R3"), name_suffix="_mask")


@task
def reader_c2_R4_mask_G1() -> Task:
    """C2: R4 reads G1's CoT with answer-leaking patterns masked."""
    return _reader_c2("R4", "G1", cot_transform=make_mask_transform("R4"), name_suffix="_mask")


@task
def reader_c2_R4_mask_G2() -> Task:
    """C2: R4 reads G2's CoT with answer-leaking patterns masked."""
    return _reader_c2("R4", "G2", cot_transform=make_mask_transform("R4"), name_suffix="_mask")


@task
def reader_c2_R4_mask_G3() -> Task:
    """C2: R4 reads G3's CoT with answer-leaking patterns masked."""
    return _reader_c2("R4", "G3", cot_transform=make_mask_transform("R4"), name_suffix="_mask")


@task
def reader_c2_R5_mask_G1() -> Task:
    """C2: R5 reads G1's CoT with answer-leaking patterns masked."""
    return _reader_c2("R5", "G1", cot_transform=make_mask_transform("R5"), name_suffix="_mask")


@task
def reader_c2_R5_mask_G2() -> Task:
    """C2: R5 reads G2's CoT with answer-leaking patterns masked."""
    return _reader_c2("R5", "G2", cot_transform=make_mask_transform("R5"), name_suffix="_mask")


@task
def reader_c2_R5_mask_G3() -> Task:
    """C2: R5 reads G3's CoT with answer-leaking patterns masked."""
    return _reader_c2("R5", "G3", cot_transform=make_mask_transform("R5"), name_suffix="_mask")


# --- ALL readers x generators: truncated at first answer-leak ---

@task
def reader_c2_R1_tleak_G1() -> Task:
    """C2: R1 reads G1's CoT truncated at first answer-leak."""
    return _reader_c2("R1", "G1", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R1_tleak_G2() -> Task:
    """C2: R1 reads G2's CoT truncated at first answer-leak."""
    return _reader_c2("R1", "G2", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R1_tleak_G3() -> Task:
    """C2: R1 reads G3's CoT truncated at first answer-leak."""
    return _reader_c2("R1", "G3", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R2_tleak_G1() -> Task:
    """C2: R2 reads G1's CoT truncated at first answer-leak."""
    return _reader_c2("R2", "G1", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R2_tleak_G2() -> Task:
    """C2: R2 reads G2's CoT truncated at first answer-leak."""
    return _reader_c2("R2", "G2", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R2_tleak_G3() -> Task:
    """C2: R2 reads G3's CoT truncated at first answer-leak."""
    return _reader_c2("R2", "G3", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R3_tleak_G1() -> Task:
    """C2: R3 reads G1's CoT truncated at first answer-leak."""
    return _reader_c2("R3", "G1", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R3_tleak_G2() -> Task:
    """C2: R3 reads G2's CoT truncated at first answer-leak."""
    return _reader_c2("R3", "G2", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R3_tleak_G3() -> Task:
    """C2: R3 reads G3's CoT truncated at first answer-leak."""
    return _reader_c2("R3", "G3", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R4_tleak_G1() -> Task:
    """C2: R4 reads G1's CoT truncated at first answer-leak."""
    return _reader_c2("R4", "G1", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R4_tleak_G2() -> Task:
    """C2: R4 reads G2's CoT truncated at first answer-leak."""
    return _reader_c2("R4", "G2", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R4_tleak_G3() -> Task:
    """C2: R4 reads G3's CoT truncated at first answer-leak."""
    return _reader_c2("R4", "G3", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R5_tleak_G1() -> Task:
    """C2: R5 reads G1's CoT truncated at first answer-leak."""
    return _reader_c2("R5", "G1", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R5_tleak_G2() -> Task:
    """C2: R5 reads G2's CoT truncated at first answer-leak."""
    return _reader_c2("R5", "G2", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


@task
def reader_c2_R5_tleak_G3() -> Task:
    """C2: R5 reads G3's CoT truncated at first answer-leak."""
    return _reader_c2("R5", "G3", cot_transform=truncate_at_first_leak, name_suffix="_tleak")


# ===================================================================
# Python pipeline orchestration (for `python eval.py` usage)
# ===================================================================

def run_step1(max_samples: int | None = None, epochs: int = K_SAMPLES):
    """Step 1: Generate CoTs from all generators."""
    print(f"Step 1: Generating CoTs (epochs={epochs}, max_samples={max_samples})")
    tasks = [_cot_gen(gid, epochs=epochs) for gid in ["G1", "G2", "G3"]]
    success, logs = eval_set(
        tasks=tasks,
        log_dir=LOG_DIR_GENERATION,
        max_samples=max_samples,
        fail_on_error=0.5,
    )
    if not success:
        print("WARNING: Step 1 had failures. Check logs.")
        for log in logs:
            if hasattr(log, "status") and log.status != "success":
                task_name = log.eval.task if hasattr(log, "eval") else "unknown"
                print(f"  FAILED: {task_name}")
    print("Step 1 complete.")
    return success


def run_step2(max_samples: int | None = None):
    """Step 2: Run all reader evaluations (C1, C4, C2)."""
    cots = _load_cots()
    n_cots = len(cots)
    print(f"Step 2: Found {n_cots} CoTs from Step 1")

    tasks = []
    # C1 and C4 baselines
    for rid in FULL_READERS:
        tasks.append(_reader_c1(rid))
        tasks.append(_reader_c4(rid))
    # C2: full readers x generators
    for rid in FULL_READERS:
        for gid in ["G1", "G2", "G3"]:
            tasks.append(_reader_c2(rid, gid))
    # C2: R4, R5 x generators (canonical, un-truncated)
    for rid in ["R4", "R5"]:
        for gid in ["G1", "G2", "G3"]:
            tasks.append(_reader_c2(rid, gid))
    # C2: ALL readers x generators (truncated — last 64 tokens)
    for rid in FULL_READERS + ["R4", "R5"]:
        for gid in ["G1", "G2", "G3"]:
            tasks.append(_reader_c2(rid, gid, cot_transform=truncate_last_n_tokens, name_suffix="_t64"))
    # C2: ALL readers x generators (truncated — last 5% chars)
    for rid in FULL_READERS + ["R4", "R5"]:
        for gid in ["G1", "G2", "G3"]:
            tasks.append(_reader_c2(rid, gid, cot_transform=truncate_last_pct, name_suffix="_t5p"))
    # C2: ALL readers x generators (masked — answer-leaking patterns replaced with pad token)
    for rid in FULL_READERS + ["R4", "R5"]:
        for gid in ["G1", "G2", "G3"]:
            tasks.append(_reader_c2(rid, gid, cot_transform=make_mask_transform(rid), name_suffix="_mask"))
    # C2: ALL readers x generators (truncated at first answer-leak)
    for rid in FULL_READERS + ["R4", "R5"]:
        for gid in ["G1", "G2", "G3"]:
            tasks.append(_reader_c2(rid, gid,
                cot_transform=truncate_at_first_leak, name_suffix="_tleak"))

    print(f"Step 2: Running {len(tasks)} reader tasks")
    success, logs = eval_set(
        tasks=tasks,
        log_dir=LOG_DIR_READERS,
        max_samples=max_samples,
        fail_on_error=0.5,
    )
    if not success:
        print("WARNING: Step 2 had failures. Check logs.")
        for log in logs:
            if hasattr(log, "status") and log.status != "success":
                task_name = log.eval.task if hasattr(log, "eval") else "unknown"
                print(f"  FAILED: {task_name}")
    print("Step 2 complete.")
    return success


def run_pipeline(max_samples: int | None = None, epochs: int = K_SAMPLES):
    """Run full Phase 1 pipeline (Step 1 then Step 2)."""
    step1_ok = run_step1(max_samples=max_samples, epochs=epochs)
    if not step1_ok:
        print("Step 1 had failures. Proceeding to Step 2 with available data.")
    run_step2(max_samples=max_samples)


# ===================================================================
# CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="CoT Legibility Phase 1")
    parser.add_argument("--step", type=int, choices=[1, 2], default=None,
                        help="Run only step 1 (generation) or step 2 (readers)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 5 samples, 1 epoch")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per task")
    parser.add_argument("--epochs", type=int, default=K_SAMPLES,
                        help="Number of epochs for CoT generation")
    args = parser.parse_args()

    if args.test:
        args.max_samples = 5
        args.epochs = 1

    if args.step == 1:
        run_step1(max_samples=args.max_samples, epochs=args.epochs)
    elif args.step == 2:
        run_step2(max_samples=args.max_samples)
    else:
        run_pipeline(max_samples=args.max_samples, epochs=args.epochs)


if __name__ == "__main__":
    main()
