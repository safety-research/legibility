"""Custom solvers for CoT Legibility Phase 1.

- cot_generation_solver: generate CoTs from generators (Step 1)
- crossfill_solver: prefill reader with generator's CoT (C2 condition)
- self_cot_solver: reader generates its own CoT (C1 condition)
- no_cot_solver: reader answers without CoT (C4 condition)
"""

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant

from config import COT_MAX_TOKENS, COT_TEMPERATURE, ANSWER_MAX_TOKENS


# ---------------------------------------------------------------------------
# Step 1: CoT generation
# ---------------------------------------------------------------------------

@solver
def cot_generation_solver() -> Solver:
    """Solver for generating CoTs from reasoning models.

    Adds instruction to think step by step, then generates
    at high temperature to get diverse reasoning traces.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_type = state.metadata.get("task_type", "")

        if task_type == "multiple_choice":
            suffix = (
                "\n\nAfter your reasoning, state your final answer as "
                "\"The answer is: X\" where X is the letter (A, B, C, or D)."
            )
        else:
            suffix = (
                "\n\nAfter your reasoning, state your final answer inside "
                "\\boxed{} notation."
            )

        # Append instruction to the last user message
        if state.messages and state.messages[-1].role == "user":
            content = state.messages[-1].content
            if isinstance(content, str):
                state.messages[-1].content = content + suffix
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        block.text += suffix
                        break

        state = await generate(
            state,
            max_tokens=COT_MAX_TOKENS,
            temperature=COT_TEMPERATURE,
        )
        return state

    return solve


# ---------------------------------------------------------------------------
# C2: Crossfill (prefill reader with generator's CoT)
# ---------------------------------------------------------------------------

@solver
def crossfill_solver() -> Solver:
    """C2 crossfill solver: prefill reader with generator's CoT.

    Reads CoT from state.metadata["cot_text"], inserts it as an assistant
    message (prefill), then asks the reader to answer based on that reasoning.

    The CoT text should contain only the reasoning trace (no final answer),
    having been stripped by extract_think_block() during CoT extraction.

    Logprobs / surprisal note:
        SPEC 1.2 requires per-CoT surprisal as a regression covariate,
        but our reader model providers (Qwen3, Llama-3.1, DeepSeek-V3)
        do not return logprobs through OpenRouter. Tested April 2026:
        R1 (404), R2 (404), R3 (unreliable all-zero), R4 (404).
        Use foreignness.py model-graded scores as the distributional-
        shift covariate instead.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cot_text = state.metadata.get("cot_text", "")
        if not cot_text:
            state = await generate(state, max_tokens=ANSWER_MAX_TOKENS, temperature=0.0)
            return state

        # Insert the CoT as an assistant prefill message
        state.messages.append(
            ChatMessageAssistant(content=cot_text)
        )

        task_type = state.metadata.get("task_type", "")
        if task_type == "multiple_choice":
            followup = (
                "Based on the reasoning above, what is the final answer? "
                "Reply with \"The answer is: X\" where X is the letter."
            )
        else:
            followup = (
                "Based on the reasoning above, what is the final answer? "
                "Reply with your answer inside \\boxed{} notation."
            )

        state.messages.append(
            ChatMessageUser(content=followup)
        )

        state = await generate(
            state,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
        )
        return state

    return solve


# ---------------------------------------------------------------------------
# C1: Self CoT (reader generates its own reasoning)
# ---------------------------------------------------------------------------

@solver
def self_cot_solver() -> Solver:
    """C1 solver: reader generates its own CoT before answering."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_type = state.metadata.get("task_type", "")

        if task_type == "multiple_choice":
            suffix = (
                "\n\nThink step by step before answering. "
                "After your reasoning, state your final answer as "
                "\"The answer is: X\" where X is the letter (A, B, C, or D)."
            )
        else:
            suffix = (
                "\n\nThink step by step before answering. "
                "After your reasoning, state your final answer inside "
                "\\boxed{} notation."
            )

        if state.messages and state.messages[-1].role == "user":
            content = state.messages[-1].content
            if isinstance(content, str):
                state.messages[-1].content = content + suffix
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        block.text += suffix
                        break

        state = await generate(
            state,
            max_tokens=COT_MAX_TOKENS,
            temperature=0.0,
        )
        return state

    return solve


# ---------------------------------------------------------------------------
# C4: No CoT (reader answers directly)
# ---------------------------------------------------------------------------

@solver
def no_cot_solver() -> Solver:
    """C4 solver: reader answers with no chain-of-thought."""
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_type = state.metadata.get("task_type", "")

        if task_type == "multiple_choice":
            suffix = (
                "\n\nAnswer with just the letter. No explanation. "
                "\"The answer is: X\" where X is the letter."
            )
        else:
            suffix = (
                "\n\nAnswer with just the answer inside \\boxed{} notation. "
                "No explanation."
            )

        if state.messages and state.messages[-1].role == "user":
            content = state.messages[-1].content
            if isinstance(content, str):
                state.messages[-1].content = content + suffix
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        block.text += suffix
                        break

        state = await generate(
            state,
            max_tokens=ANSWER_MAX_TOKENS,
            temperature=0.0,
        )
        return state

    return solve
