"""Post-processing: 3-tier classification from SPEC 1.3 + V1-V3 validation.

Reads Step 1 and Step 2 logs and applies the classification logic:
  - ANSWER_LEAKED: R4 passes C2
  - REASONING_LEGIBLE: R4 fails C2 AND majority of {R1,R2,R3} pass C2
  - ILLEGIBLE: R4 fails C2 AND majority of {R1,R2,R3} fail C2

Plus validation analyses:
  - V1: Logistic regression (C2 success ~ legibility_class + surprisal)
  - V2: Reader agreement (pairwise Cohen's kappa)
  - V3: Answer-leakage rate by generator
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log, list_eval_logs
from inspect_ai.scorer import CORRECT

from config import LOG_DIR_GENERATION, LOG_DIR_READERS, LOG_DIR_FOREIGNNESS, RESULTS_DIR, FULL_READERS

LOG_DIR_LEAK_DETECTION = str(Path(__file__).parent / "logs" / "leak_detection")


# ---------------------------------------------------------------------------
# Log reading helpers
# ---------------------------------------------------------------------------

def _read_all_samples(log_dir: str) -> list[dict]:
    """Read all samples from all successful logs in a directory into flat dicts."""
    records = []
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        return records

    log_files = list_eval_logs(log_dir)
    for log_info in log_files:
        # Skip non-success logs
        if hasattr(log_info, "status") and log_info.status != "success":
            continue

        log = read_eval_log(log_info.name)

        # Double-check log status
        if hasattr(log, "status") and log.status != "success":
            continue

        task_name = log.eval.task if hasattr(log, "eval") and hasattr(log.eval, "task") else ""
        task_metadata = {}
        if hasattr(log, "eval") and hasattr(log.eval, "metadata") and log.eval.metadata:
            task_metadata = dict(log.eval.metadata)

        if not hasattr(log, "samples") or not log.samples:
            continue

        for sample in log.samples:
            sample_id = sample.id if hasattr(sample, "id") else None
            epoch = sample.epoch if hasattr(sample, "epoch") else 0

            # Extract score
            score_val = None
            score_metadata = {}
            if hasattr(sample, "scores") and sample.scores:
                for scorer_name, score in sample.scores.items():
                    if hasattr(score, "value"):
                        score_val = score.value
                    if hasattr(score, "metadata") and score.metadata:
                        score_metadata = dict(score.metadata)
                    break

            # Merge metadata sources
            sample_metadata = {}
            if hasattr(sample, "metadata") and sample.metadata:
                sample_metadata = dict(sample.metadata)

            # Build record with explicit fields first, then layer metadata.
            # Priority: score_metadata > sample_metadata > task_metadata
            # (score_metadata is most specific, task_metadata is broadest)
            merged = {}
            merged.update(task_metadata)
            merged.update(sample_metadata)
            merged.update(score_metadata)

            # Explicit fields override everything
            merged["sample_id"] = sample_id
            merged["epoch"] = epoch
            merged["task_name"] = task_name
            merged["score"] = score_val
            merged["correct"] = score_val == CORRECT

            records.append(merged)

    return records


# ---------------------------------------------------------------------------
# Build lookup tables from reader logs
# ---------------------------------------------------------------------------

def _build_reader_lookups(reader_records: list[dict]) -> dict:
    """Build structured lookups from reader evaluation records.

    Returns dict with keys:
        c1[reader_id][sample_id] -> {correct, predicted}
        c4[reader_id][sample_id] -> {correct, predicted}
        c2[reader_id][generator_id][(sample_id, epoch)] -> {correct, surprisal, predicted}
    """
    lookups = {
        "c1": defaultdict(dict),
        "c4": defaultdict(dict),
        "c2": defaultdict(lambda: defaultdict(dict)),
    }

    for rec in reader_records:
        condition = rec.get("condition", "")
        reader_id = rec.get("reader_id", "")
        if not reader_id:
            continue

        if condition == "C1":
            sid = rec.get("sample_id")
            if sid:
                lookups["c1"][reader_id][sid] = {
                    "correct": rec["correct"],
                    "predicted": rec.get("predicted"),
                }
        elif condition == "C4":
            sid = rec.get("sample_id")
            if sid:
                lookups["c4"][reader_id][sid] = {
                    "correct": rec["correct"],
                    "predicted": rec.get("predicted"),
                }
        elif condition == "C2":
            generator_id = rec.get("generator_id", "")
            epoch = rec.get("epoch", 0)
            original_id = rec.get("original_sample_id", rec.get("sample_id"))
            if not original_id:
                continue
            key = (original_id, epoch)
            lookups["c2"][reader_id][generator_id][key] = {
                "correct": rec["correct"],
                "surprisal": rec.get("surprisal"),
                "predicted": rec.get("predicted"),
            }

    return lookups


# ---------------------------------------------------------------------------
# Foreignness score extraction (replaces surprisal for V1 regression)
# ---------------------------------------------------------------------------

def extract_foreignness_scores(log_dir: str = LOG_DIR_FOREIGNNESS) -> dict:
    """Read foreignness evaluation logs and build a lookup table.

    Returns dict keyed by (original_sample_id, generator_id, epoch, reader_id)
    with values being the numeric foreignness score (1-5).

    These scores replace logprob-based surprisal as the distributional-shift
    covariate for V1 regression. See foreignness.py for details on why
    logprobs are unavailable from our reader model providers.
    """
    scores = {}
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        return scores

    log_files = list_eval_logs(log_dir)
    for log_info in log_files:
        if hasattr(log_info, "status") and log_info.status != "success":
            continue

        log = read_eval_log(log_info.name)
        if hasattr(log, "status") and log.status != "success":
            continue

        task_metadata = {}
        if hasattr(log, "eval") and hasattr(log.eval, "metadata") and log.eval.metadata:
            task_metadata = dict(log.eval.metadata)

        if not hasattr(log, "samples") or not log.samples:
            continue

        for sample in log.samples:
            # Extract score metadata
            score_metadata = {}
            if hasattr(sample, "scores") and sample.scores:
                for scorer_name, score in sample.scores.items():
                    if hasattr(score, "metadata") and score.metadata:
                        score_metadata = dict(score.metadata)
                    break

            foreignness_score = score_metadata.get("foreignness_score")
            if foreignness_score is None:
                continue

            # Build lookup key from score metadata (most specific) or task metadata
            reader_id = score_metadata.get("reader_id") or task_metadata.get("reader_id", "")
            generator_id = score_metadata.get("generator_id") or task_metadata.get("generator_id", "")
            original_sample_id = score_metadata.get("original_sample_id", "")
            epoch = score_metadata.get("epoch", 0)

            if reader_id and generator_id and original_sample_id:
                key = (original_sample_id, generator_id, epoch, reader_id)
                scores[key] = foreignness_score

    return scores


# ---------------------------------------------------------------------------
# Block-level leak detection results (from eval_leak.py)
# ---------------------------------------------------------------------------

def load_leak_detection_results(log_dir: str = LOG_DIR_LEAK_DETECTION) -> dict:
    """Load block-level leak detection results from eval_leak.py logs.

    Returns dict keyed by (original_sample_id, generator_id, epoch) with values:
        {last_safe_block: int, safe_truncation_pct: float}

    The last_safe_block is the highest block number (1-10) that does NOT
    leak the answer. safe_truncation_pct = last_safe_block / 10 * 100.
    If last_safe_block == 10, no leakage was detected.
    """
    results = {}
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        return results

    log_files = list_eval_logs(log_dir)
    for log_info in log_files:
        if hasattr(log_info, "status") and log_info.status != "success":
            continue

        log = read_eval_log(log_info.name)
        if hasattr(log, "status") and log.status != "success":
            continue

        if not hasattr(log, "samples") or not log.samples:
            continue

        for sample in log.samples:
            score_metadata = {}
            if hasattr(sample, "scores") and sample.scores:
                for scorer_name, score in sample.scores.items():
                    if hasattr(score, "metadata") and score.metadata:
                        score_metadata = dict(score.metadata)
                    break

            original_id = score_metadata.get("original_sample_id")
            generator_id = score_metadata.get("generator_id")
            epoch = score_metadata.get("epoch", 0)
            last_safe_block = score_metadata.get("last_safe_block")

            if original_id and generator_id and last_safe_block is not None:
                key = (original_id, generator_id, epoch)
                results[key] = {
                    "last_safe_block": last_safe_block,
                    "safe_truncation_pct": score_metadata.get("safe_truncation_pct"),
                }

    return results


# ---------------------------------------------------------------------------
# Classification logic (SPEC 1.3)
# ---------------------------------------------------------------------------

def classify_cots(
    gen_records: list[dict],
    reader_records: list[dict],
    r4_transform: str = "plain",
    leak_results: dict | None = None,
) -> list[dict]:
    """Apply SPEC 1.3 classification logic.

    Args:
        gen_records: generation log records
        reader_records: reader evaluation log records
        r4_transform: which CoT transform variant to use for all readers.
            Options: "plain", "_mask", "_t64", "_t5p", "_tleak".
            All transforms run across all readers (R1-R4).
        leak_results: optional dict from load_leak_detection_results().
            When provided, block-level leak detection replaces binary R4 C2
            for the ANSWER_LEAKED decision. Keyed by
            (sample_id, generator_id, epoch) -> {last_safe_block, safe_truncation_pct}.

    Returns list of classification records with:
        sample_id, epoch, generator_id, label, surprisal_R1/R2/R3,
        filter_reason (if filtered), and per-reader C2 results.
    """
    # Filter reader records by transform variant.
    # All transforms now run across all readers (R1-R4).
    # For C2 records, keep only those matching the selected transform.
    # "plain" transform matches records with transform "plain" or "none".
    filtered_reader_records = []
    for rec in reader_records:
        condition = rec.get("condition", "")
        transform = rec.get("cot_transform", "plain")

        # C1/C4: always include (not affected by CoT transforms)
        if condition in ("C1", "C4"):
            filtered_reader_records.append(rec)
            continue

        # C2: keep only the selected transform variant
        if r4_transform == "plain":
            if transform in ("plain", "none"):
                filtered_reader_records.append(rec)
        else:
            if transform == r4_transform:
                filtered_reader_records.append(rec)

    lookups = _build_reader_lookups(filtered_reader_records)
    results = []

    # Build generation lookup: (sample_id, generator_id, epoch) -> correct
    gen_lookup = {}
    for rec in gen_records:
        sid = rec.get("sample_id")
        gid = rec.get("generator_id")
        ep = rec.get("epoch", 0)
        if sid and gid:
            gen_lookup[(sid, gid, ep)] = rec["correct"]

    # Get unique (sample_id, generator_id, epoch) tuples from C2 data
    all_c2_keys = set()
    for rid in list(lookups["c2"].keys()):
        for gid in list(lookups["c2"][rid].keys()):
            for key in lookups["c2"][rid][gid]:
                sample_id, epoch = key
                all_c2_keys.add((sample_id, gid, epoch))

    for (sample_id, generator_id, epoch) in sorted(all_c2_keys):
        record = {
            "sample_id": sample_id,
            "generator_id": generator_id,
            "epoch": epoch,
            "label": None,
            "filter_reason": None,
            "surprisal_R1": None,
            "surprisal_R2": None,
            "surprisal_R3": None,
            "c2_results": {},
        }

        # Filter 1: generator got Q wrong with this CoT
        gen_correct = gen_lookup.get((sample_id, generator_id, epoch))
        if gen_correct is not None and not gen_correct:
            record["filter_reason"] = "generator_incorrect"
            record["label"] = "FILTERED"
            results.append(record)
            continue

        # Collect C1/C4 results for this question across full readers
        # (C1/C4 are per-question, not per-epoch)
        c1_passes = 0
        c4_passes = 0
        c1_total = 0
        c4_total = 0
        for rid in FULL_READERS:
            c1_data = lookups["c1"].get(rid, {}).get(sample_id)
            if c1_data is not None:
                c1_total += 1
                if c1_data["correct"]:
                    c1_passes += 1
            c4_data = lookups["c4"].get(rid, {}).get(sample_id)
            if c4_data is not None:
                c4_total += 1
                if c4_data["correct"]:
                    c4_passes += 1

        # Filter 2: majority of readers fail C1 (too hard)
        if c1_total > 0 and c1_passes < (c1_total / 2):
            record["filter_reason"] = "too_hard_c1"
            record["label"] = "FILTERED"
            results.append(record)
            continue

        # Filter 3: majority of readers pass C4 (too easy)
        if c4_total > 0 and c4_passes > (c4_total / 2):
            record["filter_reason"] = "too_easy_c4"
            record["label"] = "FILTERED"
            results.append(record)
            continue

        # Collect C2 results
        key = (sample_id, epoch)

        # Leak detection: prefer block-level results, fall back to binary R4
        leak_key = (sample_id, generator_id, epoch)
        leak_data = leak_results.get(leak_key) if leak_results else None
        r4_c2 = lookups["c2"].get("R4", {}).get(generator_id, {}).get(key)

        if leak_data is not None:
            # Block-level leak detection available
            last_safe_block = leak_data["last_safe_block"]
            n_blocks = 10
            answer_leaked = last_safe_block < n_blocks
            record["last_safe_block"] = last_safe_block
            record["safe_truncation_pct"] = leak_data.get("safe_truncation_pct")
            record["leak_source"] = "block_judge"
        elif r4_c2:
            # Fall back to binary R4 C2 result
            answer_leaked = r4_c2["correct"]
            record["leak_source"] = "r4_c2"
        else:
            answer_leaked = False
            record["leak_source"] = "none"

        # Full reader C2 results + confound #7 check:
        # For MCQA, require C2 answer differs from C4 answer to rule out
        # chance-level (25%) correct answers.
        full_reader_passes = 0
        full_reader_total = 0
        for rid in FULL_READERS:
            c2_data = lookups["c2"].get(rid, {}).get(generator_id, {}).get(key)
            if c2_data is not None:
                full_reader_total += 1
                c2_correct = c2_data["correct"]

                # SPEC confound #7: for MCQA, C2 correct only counts if
                # the C2 answer differs from the C4 answer (rules out
                # chance agreement without using the CoT).
                c4_data = lookups["c4"].get(rid, {}).get(sample_id)
                if c2_correct and c4_data is not None:
                    c2_pred = c2_data.get("predicted")
                    c4_pred = c4_data.get("predicted")
                    if c2_pred and c4_pred and c2_pred == c4_pred:
                        # C2 gave same answer as C4 — don't count as
                        # evidence the reader used the CoT
                        c2_correct = False

                if c2_correct:
                    full_reader_passes += 1
                record[f"surprisal_{rid}"] = c2_data.get("surprisal")
                record["c2_results"][rid] = c2_data["correct"]

        record["c2_results"]["R4"] = r4_c2["correct"] if r4_c2 else None
        r5_c2 = lookups["c2"].get("R5", {}).get(generator_id, {}).get(key)
        record["c2_results"]["R5"] = r5_c2["correct"] if r5_c2 else None
        majority_full_pass = full_reader_total > 0 and full_reader_passes > (full_reader_total / 2)

        # Classification
        if answer_leaked:
            record["label"] = "ANSWER_LEAKED"
        elif majority_full_pass:
            record["label"] = "REASONING_LEGIBLE"
        else:
            record["label"] = "ILLEGIBLE"

        record["cot_source"] = "truncated_at_leak" if r4_transform == "_tleak" else "original"
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# V1: Surprisal regression
# ---------------------------------------------------------------------------

def run_v1_surprisal_regression(
    classifications: list[dict],
    foreignness_scores: dict | None = None,
    perplexity_scores: dict | None = None,
) -> dict:
    """V1: Logistic regression — C2 success ~ legibility_class + covariate.

    Uses one-hot encoding for legibility class (nominal, not ordinal).

    The distributional-shift covariate is chosen automatically:
    - If surprisal values are available (from logprobs), use surprisal.
    - If perplexity_scores dict is provided (from NB10), use reader perplexity.
    - Otherwise, if foreignness_scores dict is provided, use the
      model-graded foreignness score (1-5) as a proxy.
    - If none are available, runs without the covariate.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return {"error": "scikit-learn not installed"}

    if foreignness_scores is None:
        foreignness_scores = {}
    if perplexity_scores is None:
        perplexity_scores = {}

    results = {}
    for reader_id in FULL_READERS:
        X_rows = []
        y_rows = []
        covariate_name = None

        for rec in classifications:
            if rec["label"] in ("FILTERED", None):
                continue
            c2_correct = rec["c2_results"].get(reader_id, False)

            # One-hot encode legibility class (nominal variable)
            is_legible = 1 if rec["label"] == "REASONING_LEGIBLE" else 0
            is_leaked = 1 if rec["label"] == "ANSWER_LEAKED" else 0

            # Try surprisal first, then perplexity, then foreignness
            surprisal = rec.get(f"surprisal_{reader_id}")
            if surprisal is not None:
                X_rows.append([is_legible, is_leaked, surprisal])
                y_rows.append(int(c2_correct))
                if covariate_name is None:
                    covariate_name = "surprisal"
            else:
                # Try perplexity from NB10
                pkey = (
                    rec["sample_id"],
                    rec["generator_id"],
                    rec["epoch"],
                    reader_id,
                )
                ppl_data = perplexity_scores.get(pkey)
                if ppl_data is not None:
                    ppl_val = ppl_data.get("reader_perplexity") if isinstance(ppl_data, dict) else float(ppl_data)
                    if ppl_val is not None:
                        X_rows.append([is_legible, is_leaked, float(ppl_val)])
                        y_rows.append(int(c2_correct))
                        if covariate_name is None:
                            covariate_name = "perplexity"
                        continue

                # Fall back to foreignness score
                fkey = (
                    rec["sample_id"],
                    rec["generator_id"],
                    rec["epoch"],
                    reader_id,
                )
                foreignness = foreignness_scores.get(fkey)
                if foreignness is not None:
                    X_rows.append([is_legible, is_leaked, float(foreignness)])
                    y_rows.append(int(c2_correct))
                    if covariate_name is None:
                        covariate_name = "foreignness"

        if covariate_name is None:
            covariate_name = "none"

        if len(X_rows) < 10:
            # Fall back to regression without covariate
            X_no_cov = []
            y_no_cov = []
            for rec in classifications:
                if rec["label"] in ("FILTERED", None):
                    continue
                c2_correct = rec["c2_results"].get(reader_id, False)
                is_legible = 1 if rec["label"] == "REASONING_LEGIBLE" else 0
                is_leaked = 1 if rec["label"] == "ANSWER_LEAKED" else 0
                X_no_cov.append([is_legible, is_leaked])
                y_no_cov.append(int(c2_correct))

            if len(X_no_cov) < 10:
                results[reader_id] = {"error": "insufficient data", "n": len(X_no_cov)}
                continue

            X = np.array(X_no_cov)
            y = np.array(y_no_cov)
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            results[reader_id] = {
                "n": len(y),
                "covariate": "none",
                "coefficients": {
                    "is_legible": float(model.coef_[0][0]),
                    "is_leaked": float(model.coef_[0][1]),
                },
                "intercept": float(model.intercept_[0]),
                "accuracy": float(model.score(X, y)),
            }
            continue

        X = np.array(X_rows)
        y = np.array(y_rows)

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        results[reader_id] = {
            "n": len(y),
            "covariate": covariate_name,
            "coefficients": {
                "is_legible": float(model.coef_[0][0]),
                "is_leaked": float(model.coef_[0][1]),
                covariate_name: float(model.coef_[0][2]),
            },
            "intercept": float(model.intercept_[0]),
            "accuracy": float(model.score(X, y)),
        }

    return results


# ---------------------------------------------------------------------------
# V2: Reader agreement (Cohen's kappa)
# ---------------------------------------------------------------------------

def run_v2_reader_agreement(classifications: list[dict]) -> dict:
    """V2: Pairwise Cohen's kappa on C2 pass/fail across readers."""
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return {"error": "scikit-learn not installed"}

    # Collect per-reader C2 results for non-filtered samples
    reader_results = defaultdict(dict)
    for rec in classifications:
        if rec["label"] in ("FILTERED", None):
            continue
        key = (rec["sample_id"], rec["generator_id"], rec["epoch"])
        for rid in FULL_READERS:
            if rid in rec.get("c2_results", {}):
                reader_results[rid][key] = int(rec["c2_results"][rid])

    # Pairwise kappa
    kappas = {}
    reader_list = FULL_READERS
    for i in range(len(reader_list)):
        for j in range(i + 1, len(reader_list)):
            r1, r2 = reader_list[i], reader_list[j]
            common_keys = set(reader_results[r1].keys()) & set(reader_results[r2].keys())
            if len(common_keys) < 10:
                kappas[f"{r1}_vs_{r2}"] = {"error": "insufficient overlap", "n": len(common_keys)}
                continue
            y1 = [reader_results[r1][k] for k in sorted(common_keys)]
            y2 = [reader_results[r2][k] for k in sorted(common_keys)]
            kappa = cohen_kappa_score(y1, y2)
            kappas[f"{r1}_vs_{r2}"] = {
                "kappa": float(kappa),
                "n": len(common_keys),
                "agreement": float(sum(a == b for a, b in zip(y1, y2)) / len(y1)),
            }

    # Per-reader pass rates
    pass_rates = {}
    for rid in reader_list:
        vals = list(reader_results[rid].values())
        if vals:
            pass_rates[rid] = {
                "pass_rate": float(sum(vals) / len(vals)),
                "n": len(vals),
            }

    return {"pairwise_kappa": kappas, "pass_rates": pass_rates}


# ---------------------------------------------------------------------------
# V3: Answer-leakage rate by generator
# ---------------------------------------------------------------------------

def run_v3_leakage_rates(classifications: list[dict]) -> dict:
    """V3: Fraction of CoTs classified as ANSWER_LEAKED per generator."""
    counts = defaultdict(lambda: defaultdict(int))
    for rec in classifications:
        gid = rec["generator_id"]
        label = rec["label"]
        if label is None:
            continue
        counts[gid][label] += 1
        counts[gid]["total"] += 1

    results = {}
    for gid in sorted(counts.keys()):
        total = counts[gid]["total"]
        non_filtered = total - counts[gid].get("FILTERED", 0)
        results[gid] = {
            "total": total,
            "filtered": counts[gid].get("FILTERED", 0),
            "answer_leaked": counts[gid].get("ANSWER_LEAKED", 0),
            "reasoning_legible": counts[gid].get("REASONING_LEGIBLE", 0),
            "illegible": counts[gid].get("ILLEGIBLE", 0),
        }
        if non_filtered > 0:
            results[gid]["leak_rate"] = counts[gid].get("ANSWER_LEAKED", 0) / non_filtered
            results[gid]["legible_rate"] = counts[gid].get("REASONING_LEGIBLE", 0) / non_filtered
            results[gid]["illegible_rate"] = counts[gid].get("ILLEGIBLE", 0) / non_filtered

    return results


# ---------------------------------------------------------------------------
# E1 truncation-sweep classification
# ---------------------------------------------------------------------------

LOG_DIR_E1_SWEEP = str(Path(__file__).parent / "logs" / "e1_truncation_sweep")


def classify_from_e1_sweep(
    leak_results: dict | None = None,
    e1_log_dir: str = LOG_DIR_E1_SWEEP,
    reader_log_dir: str = LOG_DIR_READERS,
    gen_log_dir: str = LOG_DIR_GENERATION,
) -> list[dict]:
    """Classify CoTs using E1 truncation-sweep data + block-level leak detection.

    For each (sample_id, generator_id, epoch):
      1. Use leak_results to determine safe_truncation_pct (where answer stops leaking)
      2. Find the closest E1 truncation level at or below the safe percentage
      3. Use {R1, R2, R3} C2 results at that truncation level for majority vote
      4. Apply C1/C4 filters from step2_readers logs (same as standard pipeline)

    Falls back to p100 when no leak data is available for a sample.

    Returns classification records in the same format as classify_cots().
    """
    print("  Reading E1 truncation-sweep logs...")
    e1_records = _read_all_samples(e1_log_dir)
    print(f"    {len(e1_records)} E1 records")

    print("  Reading C1/C4 from reader logs...")
    reader_records = _read_all_samples(reader_log_dir)
    c1c4_records = [r for r in reader_records if r.get("condition") in ("C1", "C4")]
    print(f"    {len(c1c4_records)} C1/C4 records")

    print("  Reading generation logs...")
    gen_records = _read_all_samples(gen_log_dir)
    print(f"    {len(gen_records)} generation records")

    # Build generation correctness lookup
    gen_lookup = {}
    for rec in gen_records:
        sid = rec.get("sample_id")
        gid = rec.get("generator_id")
        ep = rec.get("epoch", 0)
        if sid and gid:
            gen_lookup[(sid, gid, ep)] = rec["correct"]

    # Build C1/C4 lookups
    c1c4_lookups = _build_reader_lookups(c1c4_records)

    # Build E1 lookup: (original_sample_id, generator_id, epoch, reader_id, truncation_pct) -> {correct, predicted}
    e1_lookup = {}
    all_e1_keys = set()  # (original_sample_id, generator_id, epoch)
    for rec in e1_records:
        rid = rec.get("reader_id", "")
        gid = rec.get("generator_id", "")
        pct = rec.get("truncation_pct")
        original_id = rec.get("original_sample_id", "")
        # Parse epoch from sample_id: "bbh_0__G3__e3" -> epoch 3
        sid = rec.get("sample_id", "")
        epoch = rec.get("epoch", 0)
        if "__e" in sid:
            try:
                epoch = int(sid.rsplit("__e", 1)[1])
            except (ValueError, IndexError):
                pass

        if not (rid and gid and original_id and pct is not None):
            continue

        e1_lookup[(original_id, gid, epoch, rid, pct)] = {
            "correct": rec["correct"],
            "predicted": rec.get("predicted"),
        }
        all_e1_keys.add((original_id, gid, epoch))

    # Available truncation percentages (sorted descending for retreat)
    available_pcts = sorted(set(r.get("truncation_pct") for r in e1_records if r.get("truncation_pct") is not None), reverse=True)
    print(f"    Available truncation pcts: {available_pcts}")
    print(f"    Unique (sample, generator, epoch) tuples: {len(all_e1_keys)}")

    results = []
    for (sample_id, generator_id, epoch) in sorted(all_e1_keys):
        record = {
            "sample_id": sample_id,
            "generator_id": generator_id,
            "epoch": epoch,
            "label": None,
            "filter_reason": None,
            "surprisal_R1": None,
            "surprisal_R2": None,
            "surprisal_R3": None,
            "c2_results": {},
        }

        # Filter 1: generator got Q wrong
        gen_correct = gen_lookup.get((sample_id, generator_id, epoch))
        if gen_correct is not None and not gen_correct:
            record["filter_reason"] = "generator_incorrect"
            record["label"] = "FILTERED"
            results.append(record)
            continue

        # Filter 2: majority of readers fail C1 (too hard)
        c1_passes = 0
        c1_total = 0
        for rid in FULL_READERS:
            c1_data = c1c4_lookups["c1"].get(rid, {}).get(sample_id)
            if c1_data is not None:
                c1_total += 1
                if c1_data["correct"]:
                    c1_passes += 1

        if c1_total > 0 and c1_passes < (c1_total / 2):
            record["filter_reason"] = "too_hard_c1"
            record["label"] = "FILTERED"
            results.append(record)
            continue

        # Filter 3: majority of readers pass C4 (too easy)
        c4_passes = 0
        c4_total = 0
        for rid in FULL_READERS:
            c4_data = c1c4_lookups["c4"].get(rid, {}).get(sample_id)
            if c4_data is not None:
                c4_total += 1
                if c4_data["correct"]:
                    c4_passes += 1

        if c4_total > 0 and c4_passes > (c4_total / 2):
            record["filter_reason"] = "too_easy_c4"
            record["label"] = "FILTERED"
            results.append(record)
            continue

        # Determine safe truncation from leak detection
        leak_key = (sample_id, generator_id, epoch)
        leak_data = leak_results.get(leak_key) if leak_results else None

        if leak_data is not None:
            safe_pct = leak_data["safe_truncation_pct"]
            last_safe_block = leak_data["last_safe_block"]
            record["last_safe_block"] = last_safe_block
            record["safe_truncation_pct"] = safe_pct
            record["leak_source"] = "block_judge"

            # Find the highest available E1 truncation level <= safe_pct
            chosen_pct = None
            for pct in available_pcts:
                if pct <= safe_pct:
                    chosen_pct = pct
                    break
            if chosen_pct is None:
                # Even the lowest truncation level leaks
                record["label"] = "ANSWER_LEAKED"
                record["truncation_pct_used"] = 0
                results.append(record)
                continue
        else:
            # No leak data — use p100 (full CoT)
            chosen_pct = 100
            record["leak_source"] = "none"

        record["truncation_pct_used"] = chosen_pct

        # Collect C2 results at the chosen truncation level
        answer_leaked = leak_data is not None and leak_data["last_safe_block"] < 10
        if answer_leaked and chosen_pct is not None:
            # We found a safe level — not leaked at this truncation
            answer_leaked = False

        full_reader_passes = 0
        full_reader_total = 0
        for rid in FULL_READERS:
            c2_data = e1_lookup.get((sample_id, generator_id, epoch, rid, chosen_pct))
            if c2_data is not None:
                full_reader_total += 1
                c2_correct = c2_data["correct"]

                # Confound #7: C2 answer must differ from C4
                c4_data = c1c4_lookups["c4"].get(rid, {}).get(sample_id)
                if c2_correct and c4_data is not None:
                    c2_pred = c2_data.get("predicted")
                    c4_pred = c4_data.get("predicted")
                    if c2_pred and c4_pred and c2_pred == c4_pred:
                        c2_correct = False

                if c2_correct:
                    full_reader_passes += 1
                record["c2_results"][rid] = c2_data["correct"]

        # R5 result at chosen truncation
        r5_data = e1_lookup.get((sample_id, generator_id, epoch, "R5", chosen_pct))
        record["c2_results"]["R5"] = r5_data["correct"] if r5_data else None

        majority_full_pass = full_reader_total > 0 and full_reader_passes > (full_reader_total / 2)

        # Classification
        if answer_leaked:
            record["label"] = "ANSWER_LEAKED"
        elif majority_full_pass:
            record["label"] = "REASONING_LEGIBLE"
        else:
            record["label"] = "ILLEGIBLE"

        record["cot_source"] = "e1_truncation_sweep"
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Main classification pipeline
# ---------------------------------------------------------------------------

def run_classification(r4_transform: str = "_mask"):
    """Run full classification + validation pipeline. Save results to JSON.

    Args:
        r4_transform: which CoT transform to use across all readers.
            "_mask" (default) masks explicit answer patterns with pad tokens.
            "plain" uses raw CoTs (original behavior).
            "_t64" truncates last 64 tokens. "_t5p" truncates last 5%.
            "_tleak" truncates at the first answer-leaking pattern match.
    """
    print(f"R4 answer-leak detector: using transform={r4_transform!r}")
    print("Reading generation logs...")
    gen_records = _read_all_samples(LOG_DIR_GENERATION)
    print(f"  Found {len(gen_records)} generation records")

    print("Reading reader logs...")
    reader_records = _read_all_samples(LOG_DIR_READERS)
    print(f"  Found {len(reader_records)} reader records")

    print("Loading block-level leak detection results...")
    leak_results = load_leak_detection_results()
    if leak_results:
        print(f"  Found {len(leak_results)} leak detection results (block-level)")
        n_leaked = sum(1 for v in leak_results.values() if v["last_safe_block"] < 10)
        print(f"  Leaked: {n_leaked}, Clean: {len(leak_results) - n_leaked}")
        # Block-level detection replaces R4 -- use plain C2 for full readers
        if r4_transform != "plain":
            print(f"  NOTE: Overriding r4_transform={r4_transform!r} -> 'plain' "
                  f"(block judge handles leak detection)")
            r4_transform = "plain"
    else:
        print("  No leak detection logs found -- falling back to binary R4 C2")

    print("Classifying CoTs...")
    classifications = classify_cots(
        gen_records, reader_records,
        r4_transform=r4_transform,
        leak_results=leak_results if leak_results else None,
    )
    n_classified = sum(1 for r in classifications if r["label"] not in ("FILTERED", None))
    n_filtered = sum(1 for r in classifications if r["label"] == "FILTERED")
    print(f"  Classified: {n_classified}, Filtered: {n_filtered}")

    # Classify from E1 truncation sweep (covers BBH + additional GPQA/MATH data)
    e1_log_dir = Path(LOG_DIR_E1_SWEEP)
    if e1_log_dir.exists() and any(e1_log_dir.glob("*.eval")):
        print("\nClassifying from E1 truncation sweep...")
        e1_classifications = classify_from_e1_sweep(
            leak_results=leak_results if leak_results else None,
        )
        e1_classified = sum(1 for r in e1_classifications if r["label"] not in ("FILTERED", None))
        e1_filtered = sum(1 for r in e1_classifications if r["label"] == "FILTERED")
        print(f"  E1 classified: {e1_classified}, filtered: {e1_filtered}")

        # Merge: E1 results for samples not already in main classifications
        existing_keys = set()
        for r in classifications:
            existing_keys.add((r["sample_id"], r["generator_id"], r["epoch"]))

        n_new = 0
        for r in e1_classifications:
            key = (r["sample_id"], r["generator_id"], r["epoch"])
            if key not in existing_keys:
                classifications.append(r)
                existing_keys.add(key)
                n_new += 1
        print(f"  Merged {n_new} new records from E1 sweep")
        n_classified = sum(1 for r in classifications if r["label"] not in ("FILTERED", None))
        n_filtered = sum(1 for r in classifications if r["label"] == "FILTERED")
        print(f"  Total after merge: {len(classifications)} "
              f"(classified={n_classified}, filtered={n_filtered})")
    else:
        print("\nNo E1 truncation sweep logs found -- skipping")

    label_counts = Counter(r["label"] for r in classifications)
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count}")

    print("Loading foreignness scores...")
    foreignness_scores = extract_foreignness_scores()
    print(f"  Found {len(foreignness_scores)} foreignness scores")

    # Load perplexity-based distributional shift scores (from NB10)
    perplexity_scores = {}
    try:
        from phase2_utils import load_distributional_shift_scores
        perplexity_scores = load_distributional_shift_scores()
        print(f"  Found {len(perplexity_scores)} perplexity scores")
    except Exception:
        pass

    print("\nRunning V1: Distributional-shift regression...")
    v1 = run_v1_surprisal_regression(
        classifications,
        foreignness_scores=foreignness_scores,
        perplexity_scores=perplexity_scores,
    )
    for rid, data in v1.items():
        print(f"  {rid}: {data}")

    print("\nRunning V2: Reader agreement...")
    v2 = run_v2_reader_agreement(classifications)
    for pair, data in v2.get("pairwise_kappa", {}).items():
        print(f"  {pair}: {data}")

    print("\nRunning V3: Leakage rates...")
    v3 = run_v3_leakage_rates(classifications)
    for gid, data in v3.items():
        print(f"  {gid}: {data}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "classifications": classifications,
        "validation": {
            "v1_surprisal_regression": v1,
            "v2_reader_agreement": v2,
            "v3_leakage_rates": v3,
        },
        "summary": {
            "total": len(classifications),
            "classified": n_classified,
            "filtered": n_filtered,
            "label_counts": dict(label_counts),
            "r4_transform": r4_transform,
        },
    }

    # Save to both the transform-specific file and the canonical classifications.json
    transform_label = r4_transform.lstrip("_") if r4_transform != "plain" else "plain"
    output_path = RESULTS_DIR / f"classifications_{transform_label}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Also save as canonical classifications.json (used by Phase 2 notebooks)
    canonical_path = RESULTS_DIR / "classifications.json"
    with open(canonical_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Canonical copy saved to {canonical_path}")

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r4-transform", default="_mask",
        choices=["plain", "_mask", "_t64", "_t5p", "_tleak"],
        help="R4 CoT transform variant for answer-leak detection (default: _mask)",
    )
    args = parser.parse_args()
    run_classification(r4_transform=args.r4_transform)
