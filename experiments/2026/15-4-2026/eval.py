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

    # C2 crossfill (R4 leak detector)
    inspect eval-set eval.py@reader_c2_R4_G1 eval.py@reader_c2_R4_G2 eval.py@reader_c2_R4_G3 \
        --log-dir logs/step2_readers

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

from inspect_ai import Task, eval_set, task
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
        solver=no_cot_solver(),
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
        print(
            "  inspect eval-set eval.py@cot_gen_G1 eval.py@cot_gen_G2 eval.py@cot_gen_G3 "
            f"--log-dir {LOG_DIR_GENERATION}"
        )
        sys.exit(1)
    return cots


def _reader_c2(reader_id: str, generator_id: str) -> Task:
    cots = _load_cots()
    dataset = build_c2_dataset(cots, reader_id, generator_id)
    request_logprobs = reader_id in FULL_READERS
    return Task(
        dataset=dataset,
        solver=crossfill_solver(request_logprobs=request_logprobs),
        scorer=reader_correctness_scorer(),
        model=READERS[reader_id],
        config=GenerateConfig(
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
        ),
        name=f"reader_c2_{reader_id}_{generator_id}",
        metadata={
            "reader_id": reader_id,
            "generator_id": generator_id,
            "condition": "C2",
            "step": "reader",
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
    # C2: R4 x generators
    for gid in ["G1", "G2", "G3"]:
        tasks.append(_reader_c2("R4", gid))

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
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only step 1 (generation) or step 2 (readers)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Quick test: 5 samples, 1 epoch"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples per task"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=K_SAMPLES,
        help="Number of epochs for CoT generation",
    )
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
