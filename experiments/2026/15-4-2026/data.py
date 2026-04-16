"""Dataset loading (GPQA-Diamond, MATH-500) and CoT extraction from logs."""

import random
import re
from pathlib import Path

import datasets as hf_datasets

load_dataset = hf_datasets.load_dataset
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log, list_eval_logs
from inspect_ai.scorer import CORRECT

from config import (
    GPQA_DATASET,
    GPQA_CONFIG,
    MATH_DATASET,
    MATH_MIN_LEVEL,
    MATH_MAX_LEVEL,
)


# ---------------------------------------------------------------------------
# Answer extraction helpers (reused from 13-4-2026/lib/data.py patterns)
# ---------------------------------------------------------------------------


def extract_mc_answer(text: str) -> str | None:
    """Extract a multiple-choice answer letter (A-E) from model output."""
    if not text:
        return None
    # Pattern 1: "The answer is: X" or "The answer is (X)"
    match = re.search(r"[Tt]he answer is:?\s*\(?([A-E])\)?", text)
    if match:
        return match.group(1)
    # Pattern 2: \boxed{X} or \boxed{(X)}
    match = re.search(r"\\boxed\{?\(?([A-E])\)?\}?", text)
    if match:
        return match.group(1)
    # Pattern 3: Standalone letter at end of text (last 50 chars only to avoid
    # false positives from letters used as variables in scientific text)
    tail = text[-50:] if len(text) > 50 else text
    match = re.search(r"\b([A-E])\b", tail)
    if match:
        return match.group(1)
    return None


def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.find("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1].strip()
    return None


def extract_math_answer(text: str) -> str | None:
    """Extract a math answer from model output (boxed or text)."""
    if not text:
        return None
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    match = re.search(r"[Tt]he answer is:?\s*(.+?)\.?\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else None


def extract_gold_math_answer(solution: str) -> str | None:
    """Extract gold answer from MATH dataset solution field."""
    boxed = _extract_boxed(solution)
    if boxed is not None:
        return boxed
    return None


def normalize_math_expr(expr: str) -> str:
    """Normalize a math expression for comparison."""
    if not expr:
        return ""
    s = expr.strip()
    s = s.replace(" ", "")
    s = s.replace("\\$", "")
    s = s.replace(",", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\(", "(").replace("\\)", ")")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = s.replace("\\frac", "frac")
    s = s.replace("\\dfrac", "frac")
    s = s.replace("\\tfrac", "frac")
    try:
        return str(float(s))
    except ValueError:
        return s.lower()


def check_math_equivalence(predicted: str | None, target: str | None) -> bool:
    """Check if predicted and target math answers are equivalent."""
    if predicted is None or target is None:
        return False
    return normalize_math_expr(predicted) == normalize_math_expr(target)


# ---------------------------------------------------------------------------
# CoT text extraction and cleaning
# ---------------------------------------------------------------------------


def extract_think_block(text: str) -> str:
    """Extract content from <think>...</think> blocks.

    If no think tags found, strips known answer patterns from the end
    and returns the remaining text as the reasoning trace.
    """
    if not text:
        return ""

    # Try to extract <think> block content
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Some models use <thought> or similar tags
    match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No think tags found — strip final answer patterns from the end
    return strip_final_answer(text)


def strip_final_answer(text: str) -> str:
    """Remove final answer statements from the end of a CoT.

    Strips patterns like:
    - "The answer is: X"
    - "\\boxed{...}"
    - "Therefore, X." (final sentence)
    """
    if not text:
        return ""

    # Strip everything after the last \boxed{} (the final answer)
    boxed_match = re.search(r"\\boxed\{", text)
    if boxed_match:
        # Find the matching closing brace
        start = boxed_match.start()
        depth = 0
        i = boxed_match.end()
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        # Strip from \boxed to end
        result = text[:start].strip()
        if result:
            return result

    # Strip "The answer is: ..." at end
    result = re.sub(
        r"[Tt]he (?:final )?answer is:?\s*.+?\.?\s*$",
        "",
        text,
        flags=re.MULTILINE,
    ).strip()
    if result:
        return result

    # If stripping removed everything, return original (minus last line)
    lines = text.strip().split("\n")
    if len(lines) > 1:
        return "\n".join(lines[:-1]).strip()
    return text


# ---------------------------------------------------------------------------
# GPQA-Diamond loader
# ---------------------------------------------------------------------------


def _letter_from_index(idx: int) -> str:
    return chr(ord("A") + idx)


def load_gpqa_diamond() -> list[Sample]:
    """Load GPQA-Diamond as Inspect Samples (multiple-choice)."""
    ds = load_dataset(GPQA_DATASET, GPQA_CONFIG, split="train")
    samples = []
    for i, row in enumerate(ds):
        question = row["Question"]
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        # Shuffle deterministically using problem index as seed
        rng = random.Random(i)
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [choices[j] for j in order]
        correct_letter = _letter_from_index(order.index(0))

        choice_text = "\n".join(
            f"{_letter_from_index(k)}. {c}" for k, c in enumerate(shuffled)
        )
        full_input = f"{question}\n\n{choice_text}"

        samples.append(
            Sample(
                input=full_input,
                target=correct_letter,
                id=f"gpqa_{i}",
                metadata={
                    "dataset": "gpqa_diamond",
                    "task_type": "multiple_choice",
                    "question_idx": i,
                },
            )
        )
    return samples


# ---------------------------------------------------------------------------
# MATH-500 loader (Level 3-5)
# ---------------------------------------------------------------------------


def load_math500() -> list[Sample]:
    """Load MATH-500 test split, filtered to Level 3-5.

    Uses nlile/hendrycks-MATH-benchmark which has 500 test problems
    with columns: problem, solution, answer, subject, level, unique_id.
    """
    ds = load_dataset(MATH_DATASET, split="test")
    samples = []
    idx = 0
    for row in ds:
        # level is an integer in this dataset
        level = row.get("level", 0)
        if isinstance(level, str):
            level_match = re.search(r"(\d+)", level)
            level = int(level_match.group(1)) if level_match else 0
        if level < MATH_MIN_LEVEL or level > MATH_MAX_LEVEL:
            continue
        # Use the direct "answer" field if available, else extract from solution
        gold = row.get("answer")
        if not gold:
            gold = extract_gold_math_answer(row.get("solution", ""))
        if not gold:
            continue
        samples.append(
            Sample(
                input=row["problem"],
                target=gold,
                id=f"math_{idx}",
                metadata={
                    "dataset": "math500",
                    "task_type": "open_ended_math",
                    "level": level,
                    "math_subject": row.get("subject", ""),
                    "question_idx": idx,
                },
            )
        )
        idx += 1
    return samples


# ---------------------------------------------------------------------------
# Combined dataset builder
# ---------------------------------------------------------------------------

_combined_dataset_cache: MemoryDataset | None = None


def build_combined_dataset() -> MemoryDataset:
    """Build combined GPQA-Diamond + MATH-500 dataset. Cached after first call."""
    global _combined_dataset_cache
    if _combined_dataset_cache is not None:
        return _combined_dataset_cache
    samples = load_gpqa_diamond() + load_math500()
    _combined_dataset_cache = MemoryDataset(samples=samples, name="gpqa_math_combined")
    return _combined_dataset_cache


# ---------------------------------------------------------------------------
# CoT extraction from Step 1 logs (bridges Step 1 -> Step 2)
# ---------------------------------------------------------------------------


def _get_completion_text(sample) -> str:
    """Extract full completion text from a logged sample."""
    if hasattr(sample, "output") and sample.output:
        if hasattr(sample.output, "completion"):
            return sample.output.completion or ""
        if hasattr(sample.output, "choices") and sample.output.choices:
            return sample.output.choices[0].message.content or ""
    return ""


def _get_sample_score_value(sample) -> str | None:
    """Get the score value (C/I) from a logged sample."""
    if hasattr(sample, "scores") and sample.scores:
        for scorer_name, score in sample.scores.items():
            if hasattr(score, "value"):
                return score.value
    return None


def _detect_generator_id(log) -> str | None:
    """Detect generator_id from a log's metadata, task name, or model.

    Priority:
    1. Task metadata["generator_id"] (set explicitly)
    2. Task name pattern matching
    3. Model name matching
    """
    from config import GENERATORS

    # Priority 1: explicit metadata
    if hasattr(log, "eval") and hasattr(log.eval, "metadata") and log.eval.metadata:
        gid = log.eval.metadata.get("generator_id")
        if gid and gid in GENERATORS:
            return gid

    # Priority 2: task name
    if hasattr(log, "eval") and hasattr(log.eval, "task"):
        task_name = log.eval.task or ""
        for gid in ["G1", "G2", "G3"]:
            if f"_{gid}" in task_name or task_name.endswith(gid):
                return gid

    # Priority 3: model name
    if hasattr(log, "eval") and hasattr(log.eval, "model"):
        model_name = log.eval.model or ""
        for gid, model_path in GENERATORS.items():
            model_suffix = model_path.split("/")[-1]
            if model_suffix in model_name:
                return gid

    return None


def extract_cots_from_logs(log_dir: str) -> dict:
    """Read Step 1 eval logs and extract CoTs.

    Extracts only the reasoning trace (think block content), stripping
    the final answer to prevent answer leakage in crossfill.

    Returns dict keyed by (sample_id, generator_id, epoch) with values:
        {cot_text, generator_correct, input, target, metadata}
    """
    cots = {}
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        return cots

    log_files = list_eval_logs(log_dir)
    for log_info in log_files:
        log = read_eval_log(log_info.name)

        # Skip non-success logs
        if hasattr(log, "status") and log.status != "success":
            continue

        generator_id = _detect_generator_id(log)
        if generator_id is None:
            continue

        if not hasattr(log, "samples") or not log.samples:
            continue

        for sample in log.samples:
            sample_id = sample.id if hasattr(sample, "id") else None
            epoch = sample.epoch if hasattr(sample, "epoch") else 0
            if sample_id is None:
                continue

            full_completion = _get_completion_text(sample)
            # Extract only the reasoning trace, strip the final answer
            cot_text = extract_think_block(full_completion)

            score_val = _get_sample_score_value(sample)
            generator_correct = (score_val == CORRECT) if score_val else False

            input_text = ""
            if hasattr(sample, "input"):
                input_text = (
                    sample.input if isinstance(sample.input, str) else str(sample.input)
                )

            target_text = ""
            if hasattr(sample, "target"):
                target_text = (
                    sample.target
                    if isinstance(sample.target, str)
                    else str(sample.target)
                )

            metadata = {}
            if hasattr(sample, "metadata") and sample.metadata:
                metadata = dict(sample.metadata)

            key = (sample_id, generator_id, epoch)
            cots[key] = {
                "cot_text": cot_text,
                "generator_correct": generator_correct,
                "input": input_text,
                "target": target_text,
                "metadata": metadata,
            }

    return cots


def build_c2_dataset(
    cots: dict,
    reader_id: str,
    generator_id: str,
) -> MemoryDataset:
    """Build C2 crossfill dataset for a specific (reader, generator) pair.

    Each Sample contains the original question as input, correct answer as target,
    and the generator's CoT (reasoning only, no answer) in metadata for the
    crossfill solver.
    """
    samples = []
    for (sample_id, gid, epoch), cot_data in cots.items():
        if gid != generator_id:
            continue
        if not cot_data["generator_correct"]:
            continue  # only use CoTs where generator got it right
        if not cot_data["cot_text"]:
            continue

        metadata = dict(cot_data["metadata"])
        metadata["cot_text"] = cot_data["cot_text"]
        metadata["generator_id"] = generator_id
        metadata["reader_id"] = reader_id
        metadata["condition"] = "C2"
        metadata["generator_correct"] = cot_data["generator_correct"]
        metadata["epoch"] = epoch
        metadata["original_sample_id"] = sample_id

        samples.append(
            Sample(
                input=cot_data["input"],
                target=cot_data["target"],
                id=f"{sample_id}__{generator_id}__e{epoch}",
                metadata=metadata,
            )
        )

    return MemoryDataset(
        samples=samples,
        name=f"c2_{reader_id}_{generator_id}",
    )
