"""Self-judged CoT foreignness evaluation.

Each reader model rates how "foreign" a generator's CoT feels to itself,
providing a distributional-shift covariate for V1 regression when logprobs
are unavailable.

Background: SPEC 1.2 requires per-CoT surprisal to control for distribution
shift (confound #1), but our reader model providers (Qwen3, Llama-3.1,
DeepSeek-V3) do not return logprobs through OpenRouter. This self-judged
foreignness task is the fallback: each reader rates how foreign each CoT
feels to it on a 1-5 scale.

Design rationale: The reader judges itself rather than an external model
because foreignness is fundamentally about on/off-policy distance as
perceived by the reader. An external judge can only match surface-level
style markers; the reader itself captures whether the reasoning flow
feels natural or alien. This is the same model whose C2 performance
we're trying to explain, so its self-assessed foreignness is the most
ecologically valid covariate.

Run all 9 tasks:
    inspect eval-set \
      foreignness.py@foreignness_R1_G1 foreignness.py@foreignness_R1_G2 foreignness.py@foreignness_R1_G3 \
      foreignness.py@foreignness_R2_G1 foreignness.py@foreignness_R2_G2 foreignness.py@foreignness_R2_G3 \
      foreignness.py@foreignness_R3_G1 foreignness.py@foreignness_R3_G2 foreignness.py@foreignness_R3_G3 \
      --log-dir logs/foreignness --log-dir-allow-dirty \
      --limit 256 --max-connections 64 --max-tasks 3 \
      --retry-on-error 3 -M "provider={'sort': 'throughput'}"

Individual task:
    inspect eval foreignness.py@foreignness_R1_G1 --limit 5
"""

import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.scorer import (
    NOANSWER,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from config import (
    READERS,
    LOG_DIR_GENERATION,
)

FOREIGNNESS_MAX_TOKENS = 16384
from data import build_c2_dataset, extract_cots_from_logs


# ---------------------------------------------------------------------------
# Self-recognition prompt and rubric
# ---------------------------------------------------------------------------

FOREIGNNESS_RUBRIC = """\
1 - Completely familiar: This reasoning feels entirely like something you \
would produce. The vocabulary, structure, reasoning style, and formatting \
all feel natural to you.

2 - Mostly familiar: This reasoning is largely consistent with how you would \
approach this problem, but has minor differences -- slightly unusual phrasing \
or formatting that you would rarely but plausibly use.

3 - Mixed: Some parts of this reasoning feel natural to you, others feel \
off -- different vocabulary, unusual structure, or reasoning patterns you \
wouldn't typically use.

4 - Mostly foreign: This reasoning has a distinctly different style from \
yours. The structure, vocabulary, or formatting conventions feel noticeably \
different from how you would reason, though occasional elements may overlap.

5 - Completely foreign: This reasoning was almost certainly not produced by \
you. The style, structure, and reasoning patterns are fundamentally different \
from how you think and write."""

JUDGE_PROMPT_TEMPLATE = """\
Below is a chain-of-thought reasoning trace that was produced by an AI model \
while solving a problem. Your task is to rate how similar this reasoning is \
to your own style -- i.e., how likely it is that YOU produced this reasoning.

## Chain-of-Thought

<cot>
{cot_text}
</cot>

## Rating Scale

{rubric}

## Instructions

Read the reasoning carefully and consider:
- Does the overall structure feel like how you would approach this problem?
- Is the vocabulary and phrasing natural to you, or does it feel like \
someone else's writing?
- Do the formatting conventions (tags, notation, layout) match what you \
would use?
- Does the reasoning flow feel familiar, or does it follow a logic you \
wouldn't use?

Provide your rating as a single integer 1-5 on the last line of your response, \
formatted as: SCORE: N"""


# ---------------------------------------------------------------------------
# Foreignness solver
# ---------------------------------------------------------------------------

@solver
def foreignness_solver() -> Solver:
    """Ask the reader model to self-judge how foreign a CoT feels."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cot_text = state.metadata.get("cot_text", "")

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            cot_text=cot_text,
            rubric=FOREIGNNESS_RUBRIC,
        )

        # Replace the input message with the self-recognition prompt
        state.messages = [ChatMessageUser(content=prompt)]

        state = await generate(
            state,
            max_tokens=FOREIGNNESS_MAX_TOKENS,
            temperature=0.0,
        )
        return state

    return solve


# ---------------------------------------------------------------------------
# Foreignness scorer
# ---------------------------------------------------------------------------

_SCORE_PATTERN = re.compile(r"SCORE:\s*(\d)")


@scorer(metrics=[mean()])
def foreignness_scorer() -> Scorer:
    """Extract numeric 1-5 foreignness score from judge response."""

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output else ""

        # Extract SCORE: N from the response
        match = _SCORE_PATTERN.search(completion)
        foreignness_score = None
        if match:
            val = int(match.group(1))
            if 1 <= val <= 5:
                foreignness_score = val

        score_metadata = {
            "reader_id": state.metadata.get("reader_id", ""),
            "generator_id": state.metadata.get("generator_id", ""),
            "epoch": state.metadata.get("epoch", 0),
            "original_sample_id": state.metadata.get("original_sample_id", ""),
            "judge_completion": completion,
        }

        if foreignness_score is not None:
            score_metadata["foreignness_score"] = foreignness_score
            return Score(
                value=foreignness_score / 5.0,
                answer=str(foreignness_score),
                explanation=f"Foreignness: {foreignness_score}/5",
                metadata=score_metadata,
            )
        else:
            # Skip samples where we can't extract a score
            return Score(
                value=NOANSWER,
                answer="",
                explanation="Could not extract foreignness score from judge response",
                metadata=score_metadata,
            )

    return score


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

LOG_DIR_FOREIGNNESS = str(Path(__file__).parent / "logs" / "foreignness")


def _load_cots() -> dict:
    """Load CoTs from Step 1 logs."""
    import sys
    cots = extract_cots_from_logs(LOG_DIR_GENERATION)
    if not cots:
        print(f"ERROR: No CoTs found in {LOG_DIR_GENERATION}. Run Step 1 first.")
        sys.exit(1)
    return cots


def _build_foreignness_task(reader_id: str, generator_id: str) -> Task:
    """Build a foreignness evaluation task for a (reader, generator) pair.

    The reader model itself is the judge -- it rates how foreign each
    generator's CoT feels to it. This is more ecologically valid than
    an external judge because it captures on/off-policy distance as
    perceived by the same model whose C2 performance we're explaining.
    """
    cots = _load_cots()
    dataset = build_c2_dataset(cots, reader_id, generator_id)

    return Task(
        dataset=dataset,
        solver=foreignness_solver(),
        scorer=foreignness_scorer(),
        model=READERS[reader_id],
        config=GenerateConfig(
            max_tokens=FOREIGNNESS_MAX_TOKENS,
            temperature=0.0,
        ),
        name=f"foreignness_{reader_id}_{generator_id}",
        metadata={
            "reader_id": reader_id,
            "generator_id": generator_id,
            "condition": "foreignness",
        },
    )


# ---------------------------------------------------------------------------
# 9 task definitions: foreignness_R{1,2,3}_G{1,2,3}
# ---------------------------------------------------------------------------

@task
def foreignness_R1_G1() -> Task:
    """Foreignness: how foreign is G1's CoT to R1 (Qwen3-32B)?"""
    return _build_foreignness_task("R1", "G1")


@task
def foreignness_R1_G2() -> Task:
    """Foreignness: how foreign is G2's CoT to R1 (Qwen3-32B)?"""
    return _build_foreignness_task("R1", "G2")


@task
def foreignness_R1_G3() -> Task:
    """Foreignness: how foreign is G3's CoT to R1 (Qwen3-32B)?"""
    return _build_foreignness_task("R1", "G3")


@task
def foreignness_R2_G1() -> Task:
    """Foreignness: how foreign is G1's CoT to R2 (Llama-3.1-70B)?"""
    return _build_foreignness_task("R2", "G1")


@task
def foreignness_R2_G2() -> Task:
    """Foreignness: how foreign is G2's CoT to R2 (Llama-3.1-70B)?"""
    return _build_foreignness_task("R2", "G2")


@task
def foreignness_R2_G3() -> Task:
    """Foreignness: how foreign is G3's CoT to R2 (Llama-3.1-70B)?"""
    return _build_foreignness_task("R2", "G3")


@task
def foreignness_R3_G1() -> Task:
    """Foreignness: how foreign is G1's CoT to R3 (DeepSeek-V3)?"""
    return _build_foreignness_task("R3", "G1")


@task
def foreignness_R3_G2() -> Task:
    """Foreignness: how foreign is G2's CoT to R3 (DeepSeek-V3)?"""
    return _build_foreignness_task("R3", "G2")


@task
def foreignness_R3_G3() -> Task:
    """Foreignness: how foreign is G3's CoT to R3 (DeepSeek-V3)?"""
    return _build_foreignness_task("R3", "G3")
