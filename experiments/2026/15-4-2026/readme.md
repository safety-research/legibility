# CoT Legibility Phase 1: Classification

Systematic evaluation of whether external "reader" models can follow chain-of-thought reasoning produced by "generator" models. Implements the Phase 1 (Classification) pipeline from SPEC.md.

## Pipeline

**Step 1 - CoT Generation**: 3 generators (DeepSeek-R1-Distill-Qwen-32B, DeepSeek-R1, QwQ-32B) produce K=6 CoTs per question at temperature 0.7 on GPQA-Diamond + MATH-500 (~350 questions). Total: 6,300 generation calls.

**Step 2 - Reader Evaluation**: 4 readers evaluate under 3 conditions:
- **C1 (Self)**: reader generates its own CoT (baseline capability)
- **C2 (Cross)**: reader is prefilled with generator's CoT and must answer (crossfill)
- **C4 (None)**: reader answers with no CoT (baseline without reasoning)

**Step 3 - Classification**: each CoT is classified as:
- `ANSWER_LEAKED` - tiny reader R4 passes C2 on **truncated** CoT (answer is trivially embedded in the body)
- `REASONING_LEGIBLE` - R4 fails truncated C2 but majority of {R1,R2,R3} pass C2 (reasoning is followable)
- `ILLEGIBLE` - R4 fails truncated C2 and majority of {R1,R2,R3} fail C2 (reasoning is opaque)

Three CoT transforms are applied to R4's input to prevent it from trivially reading off the final answer:
- **`trunc_last64`** (`_t64`): remove the last 64 whitespace-delimited tokens
- **`trunc_last5pct`** (`_t5p`): remove the last 5% of characters
- **`mask`** (`_mask`): replace answer-leaking patterns (e.g. "the answer is X", `\boxed{X}`, "option B is correct") with the reader's pad token. Patterns derived empirically from model-extracted analysis (see `study_patterns.py`)

## Files

| File | Purpose |
|------|---------|
| `config.py` | Model IDs, constants, log directories |
| `data.py` | Dataset loading (GPQA, MATH) + CoT extraction from logs |
| `solvers.py` | Custom solvers: cot_generation, crossfill, self_cot, no_cot |
| `scorers.py` | Custom scorers: generator/reader correctness with answer extraction |
| `eval.py` | @task definitions + eval_set() orchestration |
| `foreignness.py` | Model-graded CoT foreignness evaluation (9 tasks, replaces logprob surprisal) |
| `classify.py` | Post-processing: 3-tier classification + V1-V3 validation |
| `plot.py` | Visualization of classification results |
| `study_patterns.py` | Side Inspect task: model-based extraction of answer-leaking patterns |

## Running

```bash
# Full pipeline
inspect eval eval.py --model openrouter/deepseek/deepseek-r1

# Or from Python
python eval.py
```

```bash
# Step 1: CoT generation
inspect eval-set eval.py@cot_gen_G1 eval.py@cot_gen_G2 eval.py@cot_gen_G3 \
  --log-dir logs/step1_generation \
  --limit 256 \
  --max-connections 256 \
  --max-tasks 3 \
  -M "provider={'sort': 'throughput'}"
```

```bash
# Step 2: C1/C4 baselines
inspect eval-set eval.py@reader_c1_R1 eval.py@reader_c1_R2 eval.py@reader_c1_R3 \
  eval.py@reader_c4_R1 eval.py@reader_c4_R2 eval.py@reader_c4_R3 \
  --log-dir logs/step2_readers --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"

# Step 2: C2 crossfill (full readers R1-R3 x G1-G3)
inspect eval-set \
  eval.py@reader_c2_R1_G1 eval.py@reader_c2_R1_G2 eval.py@reader_c2_R1_G3 \
  eval.py@reader_c2_R2_G1 eval.py@reader_c2_R2_G2 eval.py@reader_c2_R2_G3 \
  eval.py@reader_c2_R3_G1 eval.py@reader_c2_R3_G2 eval.py@reader_c2_R3_G3 \
  --log-dir logs/step2_readers --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"

# Step 2: C2 R4 canonical (un-truncated, for comparison)
inspect eval-set eval.py@reader_c2_R4_G1 eval.py@reader_c2_R4_G2 eval.py@reader_c2_R4_G3 \
  --log-dir logs/step2_readers --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"

# Step 2: C2 R4 truncated — last 64 tokens removed
inspect eval-set eval.py@reader_c2_R4_t64_G1 eval.py@reader_c2_R4_t64_G2 eval.py@reader_c2_R4_t64_G3 \
  --log-dir logs/step2_readers --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"

# Step 2: C2 R4 truncated — last 5% of characters removed
inspect eval-set eval.py@reader_c2_R4_t5p_G1 eval.py@reader_c2_R4_t5p_G2 eval.py@reader_c2_R4_t5p_G3 \
  --log-dir logs/step2_readers --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"

# Step 2: C2 R4 masked — answer-leaking patterns replaced with pad token
inspect eval-set eval.py@reader_c2_R4_mask_G1 eval.py@reader_c2_R4_mask_G2 eval.py@reader_c2_R4_mask_G3 \
  --log-dir logs/step2_readers --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"
```

## Foreignness Evaluation (distributional-shift covariate)

SPEC 1.2 requires per-CoT surprisal to control for distribution shift, but our reader model providers do not return logprobs through OpenRouter (R1: 404, R2: 404, R3: unreliable all-zero, R4: 404). As a fallback, `foreignness.py` implements a self-judged foreignness task where each reader rates how "foreign" a generator's CoT feels to itself on a 1-5 scale.

Each reader is its own judge (not an external model) because foreignness is fundamentally about on/off-policy distance as perceived by the reader. An external judge can only match surface-level style markers; the reader captures whether the reasoning flow actually feels natural or alien to it.

9 tasks total: `foreignness_R{1,2,3}_G{1,2,3}`. The foreignness score is used as the distributional-shift covariate in the V1 regression (classify.py).

**Rubric (1-5):**
- 1 = On-policy (CoT matches reader's style)
- 2 = Mostly familiar
- 3 = Mixed signals
- 4 = Mostly foreign
- 5 = Definitely foreign (CoT clearly from a different model family)

```bash
# Run foreignness evaluation (9 tasks)
inspect eval-set \
  foreignness.py@foreignness_R1_G1 foreignness.py@foreignness_R1_G2 foreignness.py@foreignness_R1_G3 \
  foreignness.py@foreignness_R2_G1 foreignness.py@foreignness_R2_G2 foreignness.py@foreignness_R2_G3 \
  foreignness.py@foreignness_R3_G1 foreignness.py@foreignness_R3_G2 foreignness.py@foreignness_R3_G3 \
  --log-dir logs/foreignness --log-dir-allow-dirty \
  --limit 256 --max-connections 64 --max-tasks 3 \
  --retry-on-error 3 -M "provider={'sort': 'throughput'}"
```

## Models

### Generators
- **G1**: deepseek-r1-distill-qwen-32b (32B, distilled from R1)
- **G2**: deepseek-r1 (671B, RL + SFT)
- **G3**: qwq-32b (32B, native RL)

### Readers
- **R1**: qwen3-32b (reasoning reader)
- **R2**: llama-3.1-70b-instruct (non-reasoning reader)
- **R3**: deepseek-v3 (base model control)
- **R4**: qwen3-4b (answer-leakage detector, C2 only)

## Phase 2: Activation Analysis

Investigates **why** some CoTs are legible and others aren't by analyzing the internal representations of generator and reader models. All GPU code is in Jupyter notebooks for execution on a remote H200 (80GB).

### Classification Update

The original classification used plain R4, which classified ~96% of samples as ANSWER_LEAKED. Three R4 CoT transforms were tested to address this:

| R4 variant | G1 pass rate | G2 pass rate | G3 pass rate |
|---|---|---|---|
| Plain | 92.6% | 99.2% | 92.6% |
| Mask | 87.9% | 92.6% | 90.2% |
| t5p (5%) | 75.8% | 88.3% | 75.0% |
| **t64 (64 tok)** | **42.2%** | **66.4%** | **47.7%** |

Using `_t64` yields a workable label distribution: 148 leaked, 79 legible, 60 illegible, 277 filtered.

### Phase 2 Files

| File | Experiment | GPU | Purpose |
|------|-----------|-----|---------|
| `phase2_utils.py` | -- | -- | Shared utilities: data loading, model loading, activation extraction, probes, plotting |
| `notebooks/01_extract_activations_generators_h200.ipynb` | NB1 | Yes | Extract G3 + G1 activations (last-token, question-token, full-seq) |
| `notebooks/02_extract_activations_reader_h200_r2.ipynb` | NB2 | Yes | Extract R2 activations (4-bit quantized Llama-70B) |
| `notebooks/03_probe_legibility_generators_sklearn.ipynb` | NB3: A1+B | No | Linear probes for legibility and pre-CoT prediction |
| `notebooks/04_compare_geometry_generators_cka.ipynb` | NB4: A2 | No | CKA similarity between legible/illegible trajectories |
| `notebooks/05_analyze_attention_generators_heads.ipynb` | NB5: A3 | Yes | Attention entropy analysis |
| `notebooks/06_analyze_transition_generators_truncation.ipynb` | NB6: C | No | Within-CoT transition dynamics |
| `notebooks/07_analyze_activations_reader_r2.ipynb` | NB7: D | No | Reader-side activation analysis |
| `notebooks/08_compare_narration_g1g3_activations.ipynb` | NB8: F1/F2 | No | G1 vs G3 activation comparison |
| `notebooks/09_plot_results_phase2_summary.ipynb` | NB9 | No | Consolidated summary plots |

### Execution Order

```
Day 1:  NB1 (extract G3 + G1 activations)           ~3h GPU
Day 2:  NB2 (extract R2 activations)                 ~4h GPU
        NB3 (probes A1 + B -- CPU alongside NB2)     ~15 min
        NB4 (CKA A2 -- CPU)                          ~30 min
        NB6 (transition C -- CPU)                     ~30 min
Day 3:  NB5 (attention A3 -- GPU)                     ~1h GPU
        NB7 (reader D -- CPU)                         ~15 min
        NB8 (narration F1/F2 -- CPU)                  ~15 min
        NB9 (consolidated plots)                      ~10 min
```
