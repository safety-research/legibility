# Legibility

Which models are illegible under what conditions, and why? How does that impact monitorability?

## Inspect View

Evaluation logs are viewable at: **https://safety-research.github.io/legibility**

This site is auto-deployed via GitHub Pages using [`inspect view bundle`](https://inspect.aisi.org.uk/log-viewer.html#sec-publishing) on every push to `main`.

## Experiments

### 2026/15-4-2026 — CoT Legibility Phase 1: Classification

Systematic evaluation of whether external "reader" models can follow chain-of-thought reasoning produced by "generator" models. Tests whether reasoning legibility is a property of the CoT itself vs. the reader's capabilities.

**Pipeline:**
1. **Step 1 — CoT Generation**: 3 generators produce K=6 CoTs per question on GPQA-Diamond + MATH-500
2. **Step 2 — Reader Evaluation**: 4 readers evaluate under 3 conditions (self-CoT, crossfill, no-CoT)
3. **Step 3 — Classification**: each CoT classified as `ANSWER_LEAKED`, `REASONING_LEGIBLE`, or `ILLEGIBLE`

See [`experiments/2026/15-4-2026/SPEC.md`](experiments/2026/15-4-2026/SPEC.md) for the full protocol.

## Structure

```
experiments/
  2026/
    15-4-2026/
      eval.py        # @task definitions + eval_set() orchestration
      config.py      # Model IDs, constants, log directories
      data.py        # Dataset loading + CoT extraction from logs
      solvers.py     # Custom solvers: cot_generation, crossfill, self_cot, no_cot
      scorers.py     # Custom scorers: generator/reader correctness
      classify.py    # Post-processing: 3-tier classification + validation
      plot.py        # Visualization of classification results
      logs/          # Inspect eval logs (tracked with Git LFS)
```

## Development

```bash
pip install inspect-ai datasets scikit-learn matplotlib numpy

# Run Step 1 (CoT generation)
cd experiments/2026/15-4-2026
inspect eval-set eval.py@cot_gen_G1 eval.py@cot_gen_G2 eval.py@cot_gen_G3 \
  --log-dir logs/step1_generation

# View logs locally
inspect view --log-dir logs --recursive
```
