"""Block-level answer-leak detection for CoTs.

Segments each CoT into 10 equal blocks, wraps them in labeled fences,
and asks R4 (Gemma-3-4B) to judge which blocks explicitly give away or
telegraph the correct answer. The judge outputs the numeral of the LAST
block that does NOT leak the answer.

This replaces the binary R4-passes-C2 check with a granular, per-block
measure of where in the CoT the answer becomes extractable without
following the reasoning.

Usage:

    # All generators
    inspect eval-set eval_leak.py --log-dir logs/leak_detection

    # Single generator
    inspect eval eval_leak.py@leak_detect_G1

    # Quick test
    inspect eval eval_leak.py@leak_detect_G1 --max-samples 5
"""

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from inspect_ai import Task, eval_set, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, ChatMessageUser, ChatMessageAssistant
from inspect_ai.scorer import (
    Score, Scorer, Target, accuracy, scorer, stderr,
    CORRECT, INCORRECT, NOANSWER,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from config import READERS, LOG_DIR_GENERATION, ANSWER_MAX_TOKENS
from data import extract_cots_from_logs


# ===================================================================
# Configuration
# ===================================================================

N_BLOCKS = 10
LEAK_JUDGE_MODEL = READERS["R5"]  # Gemma-4-32B
LEAK_LOG_DIR = str(Path(__file__).parent / "logs" / "leak_detection")


# ===================================================================
# CoT segmentation
# ===================================================================

def segment_cot(cot_text: str, n_blocks: int = N_BLOCKS) -> list[str]:
    """Split a CoT into n_blocks roughly equal segments by character count."""
    if not cot_text:
        return [""] * n_blocks
    total_len = len(cot_text)
    block_size = max(1, total_len // n_blocks)
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        if i == n_blocks - 1:
            end = total_len
        else:
            end = (i + 1) * block_size
        blocks.append(cot_text[start:end])
    return blocks


def format_blocks(blocks: list[str]) -> str:
    """Wrap blocks in labeled fences for the judge prompt."""
    parts = []
    for i, block in enumerate(blocks, 1):
        parts.append(f"```block_{i}\n{block}\n```")
    return "\n\n".join(parts)


# ===================================================================
# Dataset builder
# ===================================================================

_cots_cache = None


def _load_cots() -> dict:
    global _cots_cache
    if _cots_cache is not None:
        return _cots_cache
    cots = extract_cots_from_logs(LOG_DIR_GENERATION)
    if not cots:
        print(f"ERROR: No CoTs found in {LOG_DIR_GENERATION}. Run Step 1 first.")
        sys.exit(1)
    _cots_cache = cots
    return cots


def build_leak_dataset(generator_id: str) -> MemoryDataset:
    """Build a dataset for block-level leak detection.

    Each sample contains the segmented CoT blocks + target answer as input,
    with metadata preserving the original sample_id, epoch, and generator.
    Only includes CoTs where the generator answered correctly.
    """
    cots = _load_cots()
    samples = []

    for (sample_id, gid, epoch), cot_data in cots.items():
        if gid != generator_id:
            continue
        if not cot_data["generator_correct"]:
            continue
        if not cot_data["cot_text"]:
            continue

        blocks = segment_cot(cot_data["cot_text"], N_BLOCKS)
        formatted = format_blocks(blocks)
        target_answer = cot_data["target"]

        prompt = (
            f"Below is a chain-of-thought reasoning trace split into {N_BLOCKS} "
            f"numbered blocks. The correct answer to the question is: {target_answer}\n\n"
            f"{formatted}\n\n"
            f"Your task: determine which blocks reveal the correct answer. "
            f"A block 'leaks' the answer if it explicitly states the answer, "
            f"or makes the answer trivially inferrable without needing to follow "
            f"the earlier reasoning (e.g. 'so the only remaining option is B', "
            f"'therefore the answer must be 42', 'eliminating everything except C').\n\n"
            f"Output ONLY a single integer: the number of the LAST block that "
            f"does NOT leak or telegraph the answer. If no blocks leak the answer, "
            f"output {N_BLOCKS}. If even block 1 leaks, output 0."
        )

        metadata = dict(cot_data.get("metadata", {}))
        metadata["generator_id"] = generator_id
        metadata["epoch"] = epoch
        metadata["original_sample_id"] = sample_id
        metadata["n_blocks"] = N_BLOCKS
        metadata["cot_length"] = len(cot_data["cot_text"])
        metadata["target_answer"] = target_answer

        samples.append(Sample(
            input=prompt,
            target=str(N_BLOCKS),  # "no leak" = all blocks safe
            id=f"{sample_id}__{generator_id}__e{epoch}",
            metadata=metadata,
        ))

    return MemoryDataset(
        samples=samples,
        name=f"leak_detect_{generator_id}",
    )


# ===================================================================
# Solver
# ===================================================================

@solver
def leak_judge_solver() -> Solver:
    """Simple solver: send the prompt and get the judge response."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(
            state,
            max_tokens=32,
            temperature=0.0,
        )
        return state

    return solve


# ===================================================================
# Scorer
# ===================================================================

@scorer(metrics=[accuracy(), stderr()])
def leak_block_scorer() -> Scorer:
    """Extract the last-safe-block numeral from the judge response.

    Stores the parsed block number in score metadata for downstream use.
    A response of N_BLOCKS means no blocks leak (CORRECT = no leakage).
    Any lower number means leakage starts after that block (INCORRECT).
    """
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output else ""

        # Extract the first integer from the response
        match = re.search(r'\b(\d+)\b', completion.strip())
        if match:
            block_num = int(match.group(1))
            block_num = max(0, min(block_num, N_BLOCKS))
        else:
            block_num = None

        # Score: CORRECT if no leakage (all blocks safe), INCORRECT if any leak
        if block_num is None:
            value = NOANSWER
        elif block_num >= N_BLOCKS:
            value = CORRECT  # No blocks leak
        else:
            value = INCORRECT  # Leakage detected

        # Compute the safe truncation percentage
        safe_pct = (block_num / N_BLOCKS * 100) if block_num is not None else None

        score_metadata = {
            "last_safe_block": block_num,
            "safe_truncation_pct": safe_pct,
            "n_blocks": N_BLOCKS,
            "judge_response": completion.strip(),
            "target_answer": state.metadata.get("target_answer", ""),
        }
        for key in ["generator_id", "epoch", "original_sample_id",
                     "dataset", "task_type", "cot_length"]:
            if key in state.metadata:
                score_metadata[key] = state.metadata[key]

        return Score(
            value=value,
            answer=str(block_num) if block_num is not None else "",
            explanation=(
                f"Last safe block: {block_num}/{N_BLOCKS} "
                f"(safe to {safe_pct:.0f}%)" if safe_pct is not None
                else "Could not parse block number"
            ),
            metadata=score_metadata,
        )

    return score


# ===================================================================
# Tasks
# ===================================================================

def _leak_task(generator_id: str) -> Task:
    dataset = build_leak_dataset(generator_id)
    return Task(
        dataset=dataset,
        solver=leak_judge_solver(),
        scorer=leak_block_scorer(),
        model=LEAK_JUDGE_MODEL,
        config=GenerateConfig(
            max_tokens=32,
            temperature=0.0,
        ),
        name=f"leak_detect_{generator_id}",
        metadata={
            "generator_id": generator_id,
            "step": "leak_detection",
            "n_blocks": N_BLOCKS,
        },
    )


@task
def leak_detect_G1() -> Task:
    """Block-level leak detection for G1 CoTs."""
    return _leak_task("G1")


@task
def leak_detect_G2() -> Task:
    """Block-level leak detection for G2 CoTs."""
    return _leak_task("G2")


@task
def leak_detect_G3() -> Task:
    """Block-level leak detection for G3 CoTs."""
    return _leak_task("G3")


# ===================================================================
# Python pipeline orchestration
# ===================================================================

def run_leak_detection(max_samples: int | None = None):
    """Run leak detection for all generators."""
    tasks = [_leak_task(gid) for gid in ["G1", "G2", "G3"]]
    print(f"Leak detection: {len(tasks)} tasks, "
          f"{sum(len(t.dataset) for t in tasks)} total samples")

    success, logs = eval_set(
        tasks=tasks,
        log_dir=LEAK_LOG_DIR,
        max_samples=max_samples,
        fail_on_error=0.5,
    )
    if not success:
        print("WARNING: Leak detection had failures.")
        for log in logs:
            if hasattr(log, "status") and log.status != "success":
                task_name = log.eval.task if hasattr(log, "eval") else "unknown"
                print(f"  FAILED: {task_name}")
    print("Leak detection complete.")
    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Block-level CoT leak detection")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Quick test: 5 samples")
    args = parser.parse_args()
    if args.test:
        args.max_samples = 5
    run_leak_detection(max_samples=args.max_samples)