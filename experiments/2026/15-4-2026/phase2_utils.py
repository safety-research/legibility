"""Phase 2 shared utilities for activation analysis.

Provides data loading, model loading, activation extraction, probe training,
and plotting utilities used across all Phase 2 notebooks.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from config import (
    BASE_DIR,
    LOG_DIR_GENERATION,
    LOG_DIR_FOREIGNNESS,
    RESULTS_DIR,
    GENERATORS,
    FULL_READERS,
)

# ---------------------------------------------------------------------------
# Local model identifiers (HuggingFace) for activation extraction
# ---------------------------------------------------------------------------

LOCAL_MODELS = {
    "G1": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "n_layers": 64,
        "hidden_dim": 5120,
        "size_gb": 64,
    },
    "G3": {
        "hf_id": "Qwen/QwQ-32B",
        "n_layers": 64,
        "hidden_dim": 5120,
        "size_gb": 64,
    },
    "R2": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "n_layers": 80,
        "hidden_dim": 8192,
        "size_gb": 140,
        "quantize_4bit": True,
    },
    "R5": {
        "hf_id": "google/gemma-4-31b-it",
        "n_layers": 60,
        "hidden_dim": 5376,
        "size_gb": 62,
    },
}

ACTIVATIONS_DIR = BASE_DIR / "activations"
PHASE2_RESULTS_DIR = RESULTS_DIR / "phase2"


# ---------------------------------------------------------------------------
# Phase 1 data loading
# ---------------------------------------------------------------------------

def load_phase1_results(path: Optional[str] = None) -> dict:
    """Load classifications.json from Phase 1.

    Returns the full output dict with keys: classifications, validation, summary.
    """
    if path is None:
        path = RESULTS_DIR / "classifications.json"
    with open(path) as f:
        return json.load(f)


def get_labeled_cots(
    labels: list[str] | None = None,
    generator_ids: list[str] | None = None,
    results: dict | None = None,
) -> list[dict]:
    """Filter Phase 1 classifications to specific labels and generators.

    Args:
        labels: list of labels to include (e.g. ["REASONING_LEGIBLE", "ILLEGIBLE"]).
            If None, returns all non-filtered records.
        generator_ids: list of generator IDs (e.g. ["G1", "G3"]).
            If None, returns all generators.
        results: pre-loaded Phase 1 results dict. If None, loads from disk.

    Returns:
        List of classification record dicts.
    """
    if results is None:
        results = load_phase1_results()

    records = results["classifications"]
    filtered = []
    for rec in records:
        if rec["label"] in ("FILTERED", None):
            continue
        if labels is not None and rec["label"] not in labels:
            continue
        if generator_ids is not None and rec["generator_id"] not in generator_ids:
            continue
        filtered.append(rec)
    return filtered


def get_within_question_pairs(
    generator_id: str,
    results: dict | None = None,
) -> list[dict]:
    """Find (legible, illegible) pairs from same question + same generator.

    These are the best-controlled comparisons: same model, same question,
    same difficulty, different legibility outcome.

    Returns list of dicts with keys:
        sample_id, generator_id, legible_epoch, illegible_epoch
    """
    if results is None:
        results = load_phase1_results()

    # Group by (sample_id, generator_id)
    by_question = defaultdict(list)
    for rec in results["classifications"]:
        if rec["generator_id"] != generator_id:
            continue
        if rec["label"] in ("FILTERED", None, "ANSWER_LEAKED"):
            continue
        by_question[rec["sample_id"]].append(rec)

    pairs = []
    for sample_id, recs in by_question.items():
        legible = [r for r in recs if r["label"] == "REASONING_LEGIBLE"]
        illegible = [r for r in recs if r["label"] == "ILLEGIBLE"]
        if legible and illegible:
            for l_rec in legible:
                for i_rec in illegible:
                    pairs.append({
                        "sample_id": sample_id,
                        "generator_id": generator_id,
                        "legible_epoch": l_rec["epoch"],
                        "illegible_epoch": i_rec["epoch"],
                    })
    return pairs


def load_cot_texts() -> dict:
    """Load CoT texts, preferring pre-extracted JSON over log parsing.

    Tries results/cot_texts.json first (portable, no logs needed).
    Falls back to parsing Step 1 generation logs if the JSON doesn't exist.

    Returns dict keyed by (sample_id, generator_id, epoch) with values:
        {cot_text, generator_correct, input, target, metadata}
    """
    cached_path = RESULTS_DIR / "cot_texts.json"
    if cached_path.exists():
        with open(cached_path) as f:
            raw = json.load(f)
        # Convert string keys back to tuples
        cots = {}
        for key_str, v in raw.items():
            parts = key_str.split("|")
            sid, gid = parts[0], parts[1]
            epoch = int(parts[2]) if len(parts) > 2 else 0
            cots[(sid, gid, epoch)] = v
        return cots

    from data import extract_cots_from_logs
    return extract_cots_from_logs(LOG_DIR_GENERATION)


def load_foreignness_scores() -> dict:
    """Load foreignness scores, preferring pre-extracted JSON over log parsing.

    Falls back to distributional-shift (perplexity-based) scores when
    foreignness data is empty.

    Returns dict keyed by (original_sample_id, generator_id, epoch, reader_id)
    with values being the numeric foreignness score (1-5) or perplexity float.
    """
    cached_path = RESULTS_DIR / "foreignness_scores.json"
    if cached_path.exists():
        with open(cached_path) as f:
            raw = json.load(f)
        if raw:  # non-empty
            scores = {}
            for key_str, v in raw.items():
                parts = key_str.split("|")
                sid, gid = parts[0], parts[1]
                epoch = int(parts[2]) if len(parts) > 2 else 0
                rid = parts[3] if len(parts) > 3 else ""
                scores[(sid, gid, epoch, rid)] = v
            if scores:
                return scores

    # Try log parsing
    from classify import extract_foreignness_scores
    scores = extract_foreignness_scores(LOG_DIR_FOREIGNNESS)
    if scores:
        return scores

    # Fall back to perplexity-based distributional shift scores
    ds_scores = load_distributional_shift_scores()
    if ds_scores:
        # Convert to foreignness-compatible format: use reader_perplexity as the value
        compat = {}
        for key, data in ds_scores.items():
            compat[key] = data.get("reader_perplexity", data.get("kld_proxy", 0.0))
        return compat

    return {}


def load_distributional_shift_scores() -> dict:
    """Load perplexity-based distributional shift scores from NB10 output.

    Returns dict keyed by (sample_id, generator_id, epoch, reader_id) with
    values being dicts containing:
        reader_perplexity, reader_mean_logprob, reader_n_tokens,
        generator_perplexity, generator_mean_logprob, generator_n_tokens,
        kld_proxy, kld_proxy_per_char
    """
    cached_path = PHASE2_RESULTS_DIR / "distributional_shift_scores.json"
    if not cached_path.exists():
        return {}

    with open(cached_path) as f:
        raw = json.load(f)

    scores = {}
    for key_str, v in raw.items():
        parts = key_str.split("|")
        sid, gid = parts[0], parts[1]
        epoch = int(parts[2]) if len(parts) > 2 else 0
        rid = parts[3] if len(parts) > 3 else ""
        scores[(sid, gid, epoch, rid)] = v
    return scores


def join_cots_with_labels(
    labels: list[str] | None = None,
    generator_ids: list[str] | None = None,
) -> list[dict]:
    """Join Phase 1 labels with CoT text data.

    Returns list of dicts with both classification fields and cot_text.
    Records where CoT text is not found are excluded.
    """
    labeled = get_labeled_cots(labels=labels, generator_ids=generator_ids)
    cots = load_cot_texts()

    joined = []
    for rec in labeled:
        key = (rec["sample_id"], rec["generator_id"], rec["epoch"])
        cot_data = cots.get(key)
        if cot_data is None:
            continue
        merged = dict(rec)
        merged["cot_text"] = cot_data["cot_text"]
        merged["input"] = cot_data["input"]
        merged["target"] = cot_data["target"]
        merged["cot_metadata"] = cot_data["metadata"]
        joined.append(merged)
    return joined


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    model_key: str,
    dtype=None,
    quantize_4bit: bool = False,
    device_map: str = "auto",
):
    """Load a HuggingFace model and tokenizer for activation extraction.

    Args:
        model_key: key into LOCAL_MODELS (e.g. "G1", "G3", "R2")
        dtype: torch dtype (default: bfloat16)
        quantize_4bit: force 4-bit quantization via BitsAndBytes
        device_map: device placement strategy

    Returns:
        (model, tokenizer)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16

    model_info = LOCAL_MODELS[model_key]
    hf_id = model_info["hf_id"]
    use_4bit = quantize_4bit or model_info.get("quantize_4bit", False)

    print(f"Loading {model_key} ({hf_id})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": device_map, "torch_dtype": dtype}

    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        # Remove torch_dtype when using quantization config
        kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params/1e9:.1f}B params, 4bit={use_4bit}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Token-level log-probability extraction (for perplexity / KLD)
# ---------------------------------------------------------------------------

def compute_perplexity(token_logprobs: np.ndarray) -> float:
    """PPL = exp(-mean(logprobs)).

    Args:
        token_logprobs: 1-D array of per-token log-probabilities.

    Returns:
        Perplexity (scalar).
    """
    return float(np.exp(-np.mean(token_logprobs)))


def compute_token_logprobs(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 4096,
    batch_size: int = 1,
    show_progress: bool = True,
) -> list[dict]:
    """Compute per-token log-probabilities for a list of texts.

    For each text, tokenizes under the given model's tokenizer and runs a
    single forward pass to collect log P(token_t | context<t) for every
    token after the first.

    Args:
        model: HuggingFace CausalLM (already on GPU).
        tokenizer: matching tokenizer.
        texts: list of text strings (e.g. CoT traces).
        max_length: max tokenization length.
        batch_size: inference batch size (1 recommended for long CoTs).
        show_progress: display tqdm progress bar.

    Returns:
        List of dicts, one per text, with keys:
            token_logprobs: np.ndarray (n_tokens-1,) of log P(t | context)
            n_tokens: int — number of tokens (including first)
            mean_logprob: float — mean of token_logprobs
            perplexity: float — exp(-mean_logprob)
    """
    import torch
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    results = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing logprobs")

    for batch_idx in iterator:
        batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # outputs.logits: (batch, seq_len, vocab_size)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        for i in range(input_ids.shape[0]):
            seq_len = attention_mask[i].sum().item()
            # Shift: log_probs[t] predicts token at position t+1
            # So for positions 1..seq_len-1, gather log P(token[t] | context<t)
            shifted_logprobs = log_probs[i, :seq_len - 1, :]  # (seq_len-1, vocab)
            target_ids = input_ids[i, 1:seq_len]  # (seq_len-1,)

            token_lps = shifted_logprobs.gather(
                dim=-1, index=target_ids.unsqueeze(-1)
            ).squeeze(-1)  # (seq_len-1,)

            token_lps_np = token_lps.cpu().float().numpy()
            mean_lp = float(np.mean(token_lps_np))

            results.append({
                "token_logprobs": token_lps_np,
                "n_tokens": int(seq_len),
                "mean_logprob": mean_lp,
                "perplexity": compute_perplexity(token_lps_np),
            })

        del outputs, logits, log_probs
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_default_layer_indices(n_layers: int, stride: int = 4) -> list[int]:
    """Return default layer indices: every `stride`-th layer + last layer."""
    indices = list(range(0, n_layers, stride)) + [n_layers - 1]
    return sorted(set(indices))


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    layer_indices: list[int] | None = None,
    pooling: str = "last_token",
    batch_size: int = 1,
    max_length: int = 16384,
    show_progress: bool = True,
) -> dict[int, np.ndarray]:
    """Extract hidden state activations at specified layers.

    Adapted from linear_probe.py:15-94 with support for longer sequences
    and single-sample batching (CoTs range 100-16K tokens).

    Args:
        model: HuggingFace model with output_hidden_states support
        tokenizer: tokenizer for the model
        texts: list of input texts
        layer_indices: which layers to extract (default: every 4th + last)
        pooling: "last_token", "mean", or "full" (returns full sequence)
        batch_size: inference batch size (use 1 for long CoTs)
        max_length: max sequence length for tokenizer
        show_progress: show tqdm progress bar

    Returns:
        Dict mapping layer_index -> np.ndarray
        For "last_token"/"mean": shape (n_texts, hidden_dim)
        For "full": shape (n_texts,) of variable-length arrays
    """
    import torch
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    n_layers = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = get_default_layer_indices(n_layers)

    activations = {idx: [] for idx in layer_indices}
    n_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting activations")

    for batch_idx in iterator:
        batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states

        for layer_idx in layer_indices:
            if layer_idx >= len(hidden_states):
                continue

            layer_hidden = hidden_states[layer_idx]

            if pooling == "last_token":
                mask = inputs["attention_mask"]
                last_positions = mask.sum(dim=1) - 1
                pooled = torch.stack([
                    layer_hidden[i, last_positions[i], :]
                    for i in range(layer_hidden.shape[0])
                ])
                activations[layer_idx].append(pooled.cpu().float().numpy())

            elif pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (layer_hidden * mask).sum(dim=1) / mask.sum(dim=1)
                activations[layer_idx].append(pooled.cpu().float().numpy())

            elif pooling == "full":
                for i in range(layer_hidden.shape[0]):
                    seq_len = inputs["attention_mask"][i].sum().item()
                    seq_acts = layer_hidden[i, :seq_len, :].cpu().float().numpy()
                    activations[layer_idx].append(seq_acts)

            else:
                raise ValueError(f"Unknown pooling: {pooling}")

        # Free GPU memory
        del outputs, hidden_states
        torch.cuda.empty_cache()

    # Concatenate across batches
    if pooling in ("last_token", "mean"):
        return {
            idx: np.concatenate(acts, axis=0)
            for idx, acts in activations.items()
            if acts
        }
    else:
        # For "full" pooling, return list of arrays (variable length)
        return {idx: acts for idx, acts in activations.items() if acts}


def extract_activations_at_position(
    model,
    tokenizer,
    texts: list[str],
    position: str = "pre_think",
    layer_indices: list[int] | None = None,
    max_length: int = 16384,
    show_progress: bool = True,
) -> dict[int, np.ndarray]:
    """Extract activations at a specific token position.

    Args:
        model: HuggingFace model
        tokenizer: tokenizer
        texts: input texts (should include the question + start of CoT)
        position: "pre_think" extracts at the token just before <think>
        layer_indices: layers to extract
        max_length: max tokenization length

    Returns:
        Dict mapping layer_index -> np.ndarray of shape (n_texts, hidden_dim)
    """
    import torch
    from tqdm import tqdm

    model.eval()
    device = next(model.parameters()).device

    n_layers = model.config.num_hidden_layers
    if layer_indices is None:
        layer_indices = get_default_layer_indices(n_layers)

    activations = {idx: [] for idx in layer_indices}
    iterator = range(len(texts))
    if show_progress:
        iterator = tqdm(iterator, desc=f"Extracting at {position}")

    for i in iterator:
        text = texts[i]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Find position
        if position == "pre_think":
            # Find <think> token and take the position just before it
            think_tokens = tokenizer.encode("<think>", add_special_tokens=False)
            input_ids = inputs["input_ids"][0].tolist()
            pos = None
            for j in range(len(input_ids) - len(think_tokens) + 1):
                if input_ids[j:j + len(think_tokens)] == think_tokens:
                    pos = j - 1 if j > 0 else 0
                    break
            if pos is None:
                # Fallback: use position before the last quarter of tokens
                pos = max(0, len(input_ids) // 4 - 1)
        else:
            raise ValueError(f"Unknown position: {position}")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in layer_indices:
            if layer_idx >= len(outputs.hidden_states):
                continue
            act = outputs.hidden_states[layer_idx][0, pos, :].cpu().float().numpy()
            activations[layer_idx].append(act)

        del outputs
        torch.cuda.empty_cache()

    return {
        idx: np.stack(acts, axis=0)
        for idx, acts in activations.items()
        if acts
    }


# ---------------------------------------------------------------------------
# Activation I/O
# ---------------------------------------------------------------------------

def save_activations(
    activations: dict[int, np.ndarray],
    output_dir: str | Path,
    metadata: dict | None = None,
):
    """Save extracted activations to disk.

    Creates one .npy file per layer plus a metadata.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, acts in activations.items():
        if isinstance(acts, np.ndarray):
            np.save(output_dir / f"layer_{layer_idx}.npy", acts)
        elif isinstance(acts, list):
            # Variable-length sequences: save as list of arrays
            np.savez(
                output_dir / f"layer_{layer_idx}.npz",
                **{f"seq_{i}": a for i, a in enumerate(acts)},
            )

    if metadata is not None:
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    print(f"  Saved {len(activations)} layers to {output_dir}")


def load_activations(
    input_dir: str | Path,
    layer_indices: list[int] | None = None,
) -> dict[int, np.ndarray]:
    """Load activations from disk.

    Args:
        input_dir: directory containing layer_N.npy files
        layer_indices: which layers to load (default: all available)

    Returns:
        Dict mapping layer_index -> np.ndarray
    """
    input_dir = Path(input_dir)
    activations = {}

    for f in sorted(input_dir.glob("layer_*.npy")):
        layer_idx = int(f.stem.split("_")[1])
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        activations[layer_idx] = np.load(f)

    for f in sorted(input_dir.glob("layer_*.npz")):
        layer_idx = int(f.stem.split("_")[1])
        if layer_indices is not None and layer_idx not in layer_indices:
            continue
        data = np.load(f)
        activations[layer_idx] = [data[k] for k in sorted(data.files)]

    return activations


# ---------------------------------------------------------------------------
# Linear probe training
# ---------------------------------------------------------------------------

def train_binary_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 1000,
) -> dict:
    """Train a binary logistic regression probe with stratified CV.

    Uses sklearn Pipeline to fit the scaler inside each CV fold,
    preventing data leakage from validation samples into scaling statistics.

    Args:
        features: (n_samples, hidden_dim) activation features
        labels: (n_samples,) binary labels
        n_splits: number of CV folds
        seed: random seed
        max_iter: max iterations for LogisticRegression

    Returns:
        Dict with auroc, auroc_ci, cv_scores, probe_model, scaler, n_samples
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("probe", LogisticRegression(max_iter=max_iter, random_state=seed, solver="lbfgs")),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cv_scores = cross_val_score(
        pipeline, features, labels, cv=cv, scoring="roc_auc"
    )

    # Bootstrap 95% CI on mean AUROC
    ci_lo, ci_hi = _bootstrap_ci(cv_scores, n_boot=1000, seed=seed)

    # Final fit on all data (for inspection / downstream use)
    pipeline.fit(features, labels)
    scaler = pipeline.named_steps["scaler"]
    probe = pipeline.named_steps["probe"]
    probabilities = pipeline.predict_proba(features)[:, 1]

    return {
        "auroc": float(np.mean(cv_scores)),
        "auroc_std": float(np.std(cv_scores)),
        "auroc_ci": (float(ci_lo), float(ci_hi)),
        "cv_scores": cv_scores.tolist(),
        "train_auroc": float(roc_auc_score(labels, probabilities)),
        "probe_model": probe,
        "scaler": scaler,
        "pipeline": pipeline,
        "n_samples": len(labels),
        "n_features": features.shape[1],
    }


def permutation_test(
    features: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Run permutation test: shuffle labels and measure chance-level AUROC.

    Uses sklearn Pipeline to avoid scaler data leakage in each CV fold.

    Returns dict with null_aurocs, p_value, observed_auroc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)

    # Observed
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("probe", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
    observed = np.mean(cross_val_score(
        pipeline, features, labels, cv=cv, scoring="roc_auc"
    ))

    # Null distribution
    null_aurocs = []
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        cv_i = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + i)
        pipeline_i = Pipeline([
            ("scaler", StandardScaler()),
            ("probe", LogisticRegression(max_iter=1000, random_state=seed + i)),
        ])
        score = np.mean(cross_val_score(
            pipeline_i, features, shuffled, cv=cv_i, scoring="roc_auc"
        ))
        null_aurocs.append(score)

    null_aurocs = np.array(null_aurocs)
    p_value = float(np.mean(null_aurocs >= observed))

    return {
        "observed_auroc": float(observed),
        "null_mean": float(np.mean(null_aurocs)),
        "null_std": float(np.std(null_aurocs)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment)
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    Args:
        X: (n_samples, d1) activation matrix
        Y: (n_samples, d2) activation matrix

    Returns:
        Linear CKA similarity score in [0, 1].
    """
    # Center columns
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # HSIC-based linear CKA
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = np.trace(XtX @ YtY)  # equivalent to ||X^T Y||_F^2
    hsic_xx = np.trace(XtX @ XtX)
    hsic_yy = np.trace(YtY @ YtY)

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


# ---------------------------------------------------------------------------
# Bootstrap utilities
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval on the mean."""
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), \
           float(np.percentile(boot_means, 100 * (1 - alpha)))


def bootstrap_ci_metric(
    values: np.ndarray,
    metric_fn=np.mean,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a metric with bootstrap 95% CI.

    Returns (metric_value, ci_low, ci_high).
    """
    rng = np.random.RandomState(seed)
    observed = float(metric_fn(values))
    boot_vals = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_vals.append(float(metric_fn(sample)))
    boot_vals = np.array(boot_vals)
    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(boot_vals, 100 * alpha))
    ci_hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    return observed, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

def plot_layer_probe_curve(
    layer_results: dict[int, dict],
    title: str = "Probe AUROC by Layer",
    ylabel: str = "AUROC",
    figsize: tuple = (10, 5),
    chance_line: float = 0.5,
    save_path: str | None = None,
):
    """Plot layer-wise probe AUROC with error bars (bootstrap 95% CI).

    Args:
        layer_results: dict mapping layer_index -> {auroc, auroc_ci}
        title: plot title
        ylabel: y-axis label
        figsize: figure size
        chance_line: y-value for chance-level dashed line
        save_path: if provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    layers = sorted(layer_results.keys())
    aurocs = [layer_results[l]["auroc"] for l in layers]

    # Error bars from CI
    if "auroc_ci" in layer_results[layers[0]]:
        ci_los = [layer_results[l]["auroc_ci"][0] for l in layers]
        ci_his = [layer_results[l]["auroc_ci"][1] for l in layers]
        yerr_lo = [a - lo for a, lo in zip(aurocs, ci_los)]
        yerr_hi = [hi - a for a, hi in zip(aurocs, ci_his)]
        yerr = [yerr_lo, yerr_hi]
    elif "auroc_std" in layer_results[layers[0]]:
        stds = [layer_results[l]["auroc_std"] for l in layers]
        yerr = [stds, stds]
    else:
        yerr = None

    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(layers, aurocs, yerr=yerr, fmt="o-", capsize=4, capthick=1.5,
                linewidth=2, markersize=6)
    if chance_line is not None:
        ax.axhline(y=chance_line, color="gray", linestyle="--", alpha=0.7,
                   label=f"Chance ({chance_line})")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")

    return fig, ax


def plot_comparison_curves(
    results_dict: dict[str, dict[int, dict]],
    title: str = "Probe Comparison",
    ylabel: str = "AUROC",
    figsize: tuple = (10, 5),
    save_path: str | None = None,
):
    """Plot multiple layer-wise curves on the same axes.

    Args:
        results_dict: mapping name -> {layer_idx: {auroc, auroc_ci}}
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for name, layer_results in results_dict.items():
        layers = sorted(layer_results.keys())
        aurocs = [layer_results[l]["auroc"] for l in layers]

        if "auroc_ci" in layer_results[layers[0]]:
            ci_los = [layer_results[l]["auroc_ci"][0] for l in layers]
            ci_his = [layer_results[l]["auroc_ci"][1] for l in layers]
            yerr = [
                [a - lo for a, lo in zip(aurocs, ci_los)],
                [hi - a for a, hi in zip(aurocs, ci_his)],
            ]
        else:
            yerr = None

        ax.errorbar(layers, aurocs, yerr=yerr, fmt="o-", capsize=3,
                    linewidth=2, markersize=5, label=name)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


# ---------------------------------------------------------------------------
# Sequence resampling for CKA (variable-length -> fixed-length)
# ---------------------------------------------------------------------------

def resample_to_relative_positions(
    seq_activations: np.ndarray,
    n_positions: int = 100,
) -> np.ndarray:
    """Resample a variable-length activation sequence to fixed relative positions.

    Uses linear interpolation along the sequence dimension.

    Args:
        seq_activations: (seq_len, hidden_dim) array
        n_positions: number of output positions

    Returns:
        (n_positions, hidden_dim) array
    """
    from scipy.interpolate import interp1d

    seq_len, hidden_dim = seq_activations.shape
    if seq_len == n_positions:
        return seq_activations

    x_orig = np.linspace(0, 1, seq_len)
    x_new = np.linspace(0, 1, n_positions)

    resampled = np.zeros((n_positions, hidden_dim), dtype=np.float32)
    for d in range(hidden_dim):
        f = interp1d(x_orig, seq_activations[:, d], kind="linear")
        resampled[:, d] = f(x_new)

    return resampled


# ---------------------------------------------------------------------------
# Data summary / diagnostics
# ---------------------------------------------------------------------------

def print_phase1_summary(results: dict | None = None):
    """Print a summary of Phase 1 classification results."""
    if results is None:
        results = load_phase1_results()

    summary = results["summary"]
    print(f"Total records: {summary['total']}")
    print(f"Classified: {summary['classified']}, Filtered: {summary['filtered']}")
    print(f"R4 transform: {summary.get('r4_transform', 'unknown')}")
    print("Label counts:")
    for label, count in sorted(summary["label_counts"].items()):
        print(f"  {label}: {count}")

    # Per-generator breakdown
    v3 = results["validation"]["v3_leakage_rates"]
    print("\nPer-generator:")
    for gid, data in sorted(v3.items()):
        non_filtered = data["total"] - data["filtered"]
        print(f"  {gid}: {non_filtered} classified "
              f"(leaked={data.get('leak_rate', 0):.0%}, "
              f"legible={data.get('legible_rate', 0):.0%}, "
              f"illegible={data.get('illegible_rate', 0):.0%})")

    # Within-question pairs
    for gid in ["G1", "G3"]:
        pairs = get_within_question_pairs(gid, results=results)
        print(f"  {gid} within-Q pairs: {len(pairs)}")
