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
                '"The answer is: X" where X is the letter (A, B, C, or D).'
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


def _compute_mean_surprisal(output) -> float | None:
    """Compute mean surprisal from output token logprobs.

    Note: With the chat API, we can only get logprobs on the generated
    (answer) tokens, not on the prefilled CoT tokens. The SPEC (1.2)
    ideally wants surprisal over the CoT tokens themselves. True CoT
    token surprisal requires a completions API or local inference.

    This computes answer-token surprisal as an available proxy. The
    value is stored as "answer_surprisal" to distinguish from the
    ideal "cot_surprisal" metric.
    """
    if not hasattr(output, "choices") or not output.choices:
        return None
    choice = output.choices[0]
    if not hasattr(choice, "logprobs") or not choice.logprobs:
        return None
    logprobs = choice.logprobs
    if not hasattr(logprobs, "content") or not logprobs.content:
        return None
    token_logprobs = []
    for token_info in logprobs.content:
        if hasattr(token_info, "logprob") and token_info.logprob is not None:
            token_logprobs.append(token_info.logprob)
    if not token_logprobs:
        return None
    # surprisal = -mean(log_prob)
    return -sum(token_logprobs) / len(token_logprobs)


@solver
def crossfill_solver(request_logprobs: bool = False) -> Solver:
    """C2 crossfill solver: prefill reader with generator's CoT.

    Reads CoT from state.metadata["cot_text"], inserts it as an assistant
    message (prefill), then asks the reader to answer based on that reasoning.

    The CoT text should contain only the reasoning trace (no final answer),
    having been stripped by extract_think_block() during CoT extraction.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        cot_text = state.metadata.get("cot_text", "")
        if not cot_text:
            state = await generate(state, max_tokens=ANSWER_MAX_TOKENS, temperature=0.0)
            return state

        # Insert the CoT as an assistant prefill message
        state.messages.append(ChatMessageAssistant(content=cot_text))

        task_type = state.metadata.get("task_type", "")
        if task_type == "multiple_choice":
            followup = (
                "Based on the reasoning above, what is the final answer? "
                'Reply with "The answer is: X" where X is the letter.'
            )
        else:
            followup = (
                "Based on the reasoning above, what is the final answer? "
                "Reply with your answer inside \\boxed{} notation."
            )

        state.messages.append(ChatMessageUser(content=followup))

        # Generate answer; request logprobs for answer-token surprisal
        generate_kwargs = {
            "max_tokens": ANSWER_MAX_TOKENS,
            "temperature": 0.0,
        }
        if request_logprobs:
            generate_kwargs["logprobs"] = True

        state = await generate(state, **generate_kwargs)

        # Compute and store answer-token surprisal if logprobs available
        if request_logprobs and state.output:
            surprisal = _compute_mean_surprisal(state.output)
            if surprisal is not None:
                state.store.set("answer_surprisal", surprisal)
                # Note: true CoT token surprisal (SPEC 1.2) requires
                # completions API or local inference. This is an
                # answer-token proxy stored for downstream use.

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
                '"The answer is: X" where X is the letter (A, B, C, or D).'
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
                '"The answer is: X" where X is the letter.'
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
