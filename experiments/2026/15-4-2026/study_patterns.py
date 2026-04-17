"""Side Inspect task: extract answer-leaking patterns from CoTs using a small model.

Usage:
    inspect eval study_patterns.py@extract_leak_patterns --limit 200 \
        --log-dir logs/pattern_study -M "provider={'sort': 'throughput'}"

Then run the pattern extraction:
    python study_patterns.py --extract
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.log import read_eval_log, list_eval_logs

from config import LOG_DIR_GENERATION
from data import extract_cots_from_logs


# ===================================================================
# Dataset: build from Step 1 CoTs
# ===================================================================


def build_pattern_study_dataset() -> MemoryDataset:
    """Build dataset of (CoT, target) pairs from Step 1 logs for pattern extraction."""
    cots = extract_cots_from_logs(LOG_DIR_GENERATION)
    if not cots:
        print(f"ERROR: No CoTs found in {LOG_DIR_GENERATION}. Run Step 1 first.")
        sys.exit(1)

    samples = []
    for (sample_id, gid, epoch), cot_data in cots.items():
        if not cot_data["generator_correct"]:
            continue
        if not cot_data["cot_text"]:
            continue

        target = cot_data["target"]
        task_type = cot_data["metadata"].get("task_type", "open_ended_math")

        samples.append(
            Sample(
                input=(
                    f"Here is a chain-of-thought reasoning trace produced by an AI model "
                    f"while solving a problem. The correct answer to the problem is: {target}\n\n"
                    f"The task type is: {task_type}\n\n"
                    f"<think>\n{cot_data['cot_text']}\n</think>\n\n"
                    f"Your job: identify every substring in the <think> block that DIRECTLY "
                    f'conveys or leaks the correct answer "{target}". A pattern \'directly '
                    f"conveys' the answer if someone reading ONLY that substring (without the "
                    f"surrounding reasoning) could determine the answer.\n\n"
                    f"Look for:\n"
                    f'- Explicit answer statements: "the answer is X", "therefore X", '
                    f'"so the answer must be X"\n'
                    f"- Boxed answers: \\boxed{{X}}\n"
                    f'- For multiple choice: direct letter mentions like "option A is correct", '
                    f'"the correct choice is B"\n'
                    f"- Conclusion/summary statements that state the answer\n"
                    f"- Intermediate results that exactly equal the final answer\n"
                    f"- Any other substring that gives away the answer\n\n"
                    f"For EACH such substring, output it EXACTLY as it appears in the CoT "
                    f"(verbatim copy, preserving whitespace and special characters) wrapped "
                    f"in <pattern> tags. Output NOTHING else besides the pattern tags.\n\n"
                    f"If no answer-leaking patterns are found, output: <pattern>NONE</pattern>\n\n"
                    f"Example output format:\n"
                    f"<pattern>the answer is 42</pattern>\n"
                    f"<pattern>\\boxed{{42}}</pattern>\n"
                    f"<pattern>Therefore, 42.</pattern>"
                ),
                target=target,
                id=f"{sample_id}__{gid}__e{epoch}",
                metadata={
                    "generator_id": gid,
                    "task_type": task_type,
                    "original_sample_id": sample_id,
                    "cot_length": len(cot_data["cot_text"]),
                },
            )
        )

    return MemoryDataset(samples=samples, name="pattern_study")


# ===================================================================
# Solver: just generate
# ===================================================================


@solver
def pattern_extract_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state, max_tokens=2048, temperature=0.0)
        return state

    return solve


# ===================================================================
# Scorer: check that model produced valid pattern tags
# ===================================================================


@scorer(metrics=[accuracy(), stderr()])
def pattern_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion if state.output else ""
        patterns = re.findall(r"<pattern>(.*?)</pattern>", completion, re.DOTALL)
        has_patterns = len(patterns) > 0 and not (
            len(patterns) == 1 and patterns[0].strip() == "NONE"
        )

        return Score(
            value=CORRECT if has_patterns else INCORRECT,
            answer=str(len(patterns)),
            explanation=f"Found {len(patterns)} patterns",
            metadata={
                "patterns": patterns,
                "n_patterns": len(patterns),
            },
        )

    return score


# ===================================================================
# Task
# ===================================================================


@task
def extract_leak_patterns() -> Task:
    """Extract answer-leaking patterns from CoTs using a small model."""
    return Task(
        dataset=build_pattern_study_dataset(),
        solver=pattern_extract_solver(),
        scorer=pattern_scorer(),
        model="openrouter/google/gemma-3-4b-it",
        config=GenerateConfig(max_tokens=2048, temperature=0.0),
        name="extract_leak_patterns",
        metadata={"step": "pattern_study"},
    )


# ===================================================================
# Post-hoc extraction: read logs and derive regex patterns
# ===================================================================


def extract_patterns_from_logs(log_dir: str):
    """Read pattern study logs and extract all discovered patterns."""
    log_files = list_eval_logs(log_dir)
    all_patterns = []
    samples_with_leaks = 0
    samples_without_leaks = 0

    for log_info in log_files:
        log = read_eval_log(log_info.name)
        if hasattr(log, "status") and log.status != "success":
            continue
        if not hasattr(log, "eval") or "extract_leak_patterns" not in (
            log.eval.task or ""
        ):
            continue

        for sample in log.samples or []:
            completion = ""
            if hasattr(sample, "output") and sample.output:
                if hasattr(sample.output, "completion"):
                    completion = sample.output.completion or ""

            patterns = re.findall(r"<pattern>(.*?)</pattern>", completion, re.DOTALL)
            patterns = [
                p.strip() for p in patterns if p.strip() and p.strip() != "NONE"
            ]

            if patterns:
                samples_with_leaks += 1
                task_type = ""
                gen_id = ""
                if hasattr(sample, "metadata") and sample.metadata:
                    task_type = sample.metadata.get("task_type", "")
                    gen_id = sample.metadata.get("generator_id", "")
                for p in patterns:
                    all_patterns.append(
                        {
                            "pattern": p,
                            "task_type": task_type,
                            "generator_id": gen_id,
                            "sample_id": sample.id,
                        }
                    )
            else:
                samples_without_leaks += 1

    return all_patterns, samples_with_leaks, samples_without_leaks


def categorize_and_derive_regexes(patterns: list[dict]):
    """Categorize extracted patterns and derive regex templates."""
    categories = {
        "boxed": [],
        "the_answer_is": [],
        "therefore_conclusion": [],
        "mc_letter_statement": [],
        "equals_statement": [],
        "option_correct": [],
        "other": [],
    }

    for entry in patterns:
        p = entry["pattern"]
        if re.search(r"\\boxed", p):
            categories["boxed"].append(p)
        elif re.search(
            r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s+(?:is|would be|should be|must be)",
            p,
            re.IGNORECASE,
        ):
            categories["the_answer_is"].append(p)
        elif re.search(
            r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer|result)",
            p,
            re.IGNORECASE,
        ):
            categories["therefore_conclusion"].append(p)
        elif re.search(r"(?:option|choice)\s+[A-E]\s+(?:is|seems)", p, re.IGNORECASE):
            categories["option_correct"].append(p)
        elif re.search(r"(?:=|equals)\s*\S+\s*$", p):
            categories["equals_statement"].append(p)
        elif re.search(r"\b[A-E]\b.*(?:correct|right|answer)", p, re.IGNORECASE):
            categories["mc_letter_statement"].append(p)
        else:
            categories["other"].append(p)

    # Report
    print("=" * 80)
    print("PATTERN CATEGORIES")
    print("=" * 80)
    for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"\n--- {cat} ({len(items)} occurrences) ---")
        # Show frequency of unique patterns
        counter = Counter(items)
        for pat, count in counter.most_common(10):
            print(f"  [{count:>4}x] {pat[:120]}")

    # Derive regexes
    print("\n" + "=" * 80)
    print("DERIVED REGEXES")
    print("=" * 80)
    regexes = [
        ("boxed_answer", r"\\boxed\{[^}]+\}"),
        (
            "the_answer_is",
            r"[Tt]he\s+(?:final\s+|correct\s+)?answer\s+(?:is|would be|should be|must be)\s*:?\s*\S+",
        ),
        (
            "therefore_answer",
            r"(?:[Ss]o|[Tt]herefore|[Tt]hus|[Hh]ence)\s*,?\s*(?:the\s+)?(?:final\s+)?answer\s+(?:is|=)\s*\S+",
        ),
        (
            "mc_option_correct",
            r"(?:[Oo]ption|[Cc]hoice)\s+\(?[A-E]\)?\s+(?:is|seems)\s+(?:correct|right|the answer)",
        ),
        ("mc_answer_letter", r"(?:answer|select|choose|pick)\s+(?:is\s+)?\(?[A-E]\)?"),
        ("equals_final", r"=\s*\S+\s*$"),
        ("i_choose", r"(?:I'?ll go with|I choose|I select|my answer is)\s+\S+"),
    ]
    for name, regex in regexes:
        n_matches = sum(
            1
            for entry in patterns
            if re.search(regex, entry["pattern"], re.IGNORECASE | re.MULTILINE)
        )
        print(f"  {name:<25} matches {n_matches:>5}/{len(patterns)} patterns")
        print(f"    regex: {regex}")

    return categories, regexes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract", action="store_true", help="Extract patterns from logs"
    )
    parser.add_argument("--log-dir", default="logs/pattern_study", help="Log directory")
    args = parser.parse_args()

    if args.extract:
        patterns, with_leaks, without_leaks = extract_patterns_from_logs(args.log_dir)
        print(f"Samples with leaks: {with_leaks}")
        print(f"Samples without leaks: {without_leaks}")
        print(f"Total patterns extracted: {len(patterns)}")

        if patterns:
            categorize_and_derive_regexes(patterns)
    else:
        print("Run the Inspect task first:")
        print("  inspect eval study_patterns.py@extract_leak_patterns --limit 200 \\")
        print("    --log-dir logs/pattern_study --max-connections 64")
        print()
        print("Then extract patterns:")
        print("  python study_patterns.py --extract --log-dir logs/pattern_study")


if __name__ == "__main__":
    main()
