# Phase 2 Results: Activation Analysis of CoT Legibility

## Overview

Phase 2 investigates **why** some chain-of-thought traces are legible to external reader models and others are not. We extract activations from generator models (G1: DeepSeek-R1-Distill-Qwen-32B, G3: QwQ-32B) and a reader model (R2: Llama-3.1-70B-Instruct), then apply linear probes, CKA geometry analysis, attention entropy analysis, and truncation sweeps.

All experiments use Phase 1 classifications from the `_t64` transform (truncate last 64 tokens), which yields the most balanced label distribution: 278 ANSWER_LEAKED, 100 REASONING_LEGIBLE, 288 ILLEGIBLE, 619 FILTERED (n=666 classified).

### Key finding

Legibility is largely predetermined before reasoning begins. Question-token activations predict legibility with AUROC 0.76 -- and the signal manifests in the first 10% of generated tokens. Internally, legible and illegible CoTs diverge into radically different representational subspaces (CKA drops to near zero in middle layers), but no individual attention heads drive this divergence. The phenomenon is distributed and emergent, not localised.

---

## Experiment A1(b): Legibility Probes

**Question:** Can we linearly decode whether a CoT is legible vs illegible from the generator's last-token activations?

**Method:** Binary logistic regression probes at each layer, 5-fold stratified CV with bootstrap 95% CIs. Permutation test (1000 permutations) on the best layer.

### Results

| Model | Best layer | AUROC | 95% CI |
|-------|-----------|-------|--------|
| G3 (QwQ-32B) | 48 | 0.611 | [0.549, 0.675] |
| G1 (R1-Distill-32B) | 8 | 0.644 | [0.556, 0.731] |

G3 shows a slight rise in probe performance in late-middle layers (44-52), while G1 shows an early peak at layer 8 that plateaus through the middle layers (0.59-0.64) before declining at output layers.

### Interpretation

Legibility is **weakly** linearly decodable from generator activations. CIs overlap with chance (0.5) at most individual layers. This suggests the generator does not maintain a strong, explicit internal representation of "whether this reasoning will be followable by an external model." Legibility appears to be an emergent property of the reasoning trajectory rather than a deliberately tracked variable.

### Foreignness covariate

Adding foreignness scores as an additional probe feature does not change AUROC (0.674 with vs 0.674 without). Foreignness alone achieves only AUROC 0.589. Mean foreignness is nearly identical between groups (legible: 2.04, illegible: 1.96). This rules out distributional shift as a confound -- the probes are detecting content-level differences, not stylistic ones.

---

## Experiment A1(a): Answer Probes

**Question:** Can we decode the correct answer from last-token activations? Does this differ between legible and illegible CoTs?

**Method:** Multi-class logistic regression (4-way MCQA) at each layer, 5-fold stratified CV.

### Results

| Layer | Accuracy | Chance |
|-------|----------|--------|
| 32 (best) | 0.368 | 0.250 |
| 36 | 0.368 | 0.250 |

Split by legibility class (at best layer 32):
- REASONING_LEGIBLE: only 4 MCQA samples -- insufficient
- ILLEGIBLE: accuracy 0.230 (n=21) -- at chance

### Interpretation

Inconclusive. Only 38 MCQA samples have both activations and clear answer labels. The marginal above-chance performance (0.37 vs 0.25) cannot be reliably attributed to either legibility class. A larger MCQA-only evaluation would be needed to test whether illegible CoTs still compute the correct answer internally.

---

## Experiment B: Pre-CoT Illegibility Prediction

**Question:** Can we predict whether a CoT will be legible or illegible from activations at the *question token* -- before `<think>` is generated?

**Method:** Binary probes on question-token activations, 5-fold CV, permutation test on best layer.

### Results

| Model | Best layer | AUROC | 95% CI |
|-------|-----------|-------|--------|
| G3 (QwQ-32B) | 28 | **0.757** | [0.591, 0.893] |
| G1 (R1-Distill-32B) | -- | above 0.65 at multiple layers | -- |

G3 shows consistent above-threshold (>0.65) performance across many layers, with the strongest signal in layer 28. The signal is present from early layers (layer 0: AUROC 0.718) through the final layer (layer 63: AUROC 0.706), suggesting it is a robust, distributed property of how the model encodes the question.

### Interpretation

**This is the strongest and most practically significant finding.** Before the model generates a single reasoning token, the question representation already predicts whether the resulting CoT will be legible or illegible with 76% discriminability.

This implies that illegibility is partly determined by **what the question is** and how the model internally represents it, rather than being a stochastic outcome of the generation process. Some questions consistently elicit legible reasoning; others consistently elicit opaque reasoning.

**Practical implication:** A lightweight pre-generation monitor based on question-token activations could flag likely-illegible CoTs with moderate reliability, enabling selective regeneration, alternative prompting strategies, or flagging for human review.

---

## Experiment A2: CKA Representational Geometry

**Question:** Do legible and illegible CoTs occupy the same or different representational subspaces within the generator?

**Method:** Linear CKA between legible and illegible activation trajectories, resampled to 20 fixed relative positions per CoT. Bootstrap CIs (1000 resamples). Within-group self-CKA as a baseline for comparison.

### Results

**G3 (QwQ-32B):**

| Layer | Cross-CKA (leg vs ill) | 95% CI | Self-CKA (legible) |
|-------|----------------------|--------|-------------------|
| 0 | 0.928 | [0.943, 0.972] | 0.932 |
| 8 | **0.041** | [0.030, 0.286] | 0.132 |
| 16 | 0.082 | [0.092, 0.351] | 0.286 |
| 24 | 0.126 | [0.151, 0.433] | 0.364 |
| 32 | 0.189 | [0.226, 0.535] | 0.448 |
| 40 | 0.188 | [0.225, 0.533] | 0.456 |
| 48 | 0.389 | [0.437, 0.742] | 0.641 |
| 63 | 0.908 | [0.925, 0.963] | 0.945 |

**G1 (R1-Distill-32B):** Even more extreme divergence. CKA drops to 0.084 at layer 8 and remains near zero through layer 48, only partially recovering at layer 63 (0.422). Self-CKA within legible samples is also very low for G1 (0.009-0.042 in middle layers), suggesting the distilled model's representations are more variable overall.

### Interpretation

Legible and illegible CoTs undergo a **dramatic representational divergence** in early-to-middle layers, then reconverge toward the output. At layer 8, the two groups share essentially no linear representational structure (CKA ~0.04).

The U-shaped CKA curve reveals the model's architecture: the input layer and output layer are constrained (shared embedding/unembedding), but the middle layers -- where the core computation happens -- are free to develop entirely separate processing pathways for legible vs illegible reasoning.

In G3, self-CKA within legible samples is substantially higher than cross-CKA at every middle layer (e.g. 0.448 vs 0.189 at layer 32). This confirms the divergence is systematic: legible CoTs form a coherent cluster that is distinct from the illegible cluster.

G1's even lower self-CKA values suggest the distilled model has more heterogeneous internal representations, possibly reflecting the distillation process compressing diverse teacher strategies into a smaller parameter space.

---

## Experiment A3: Attention Entropy

**Question:** Do legible and illegible CoTs differ in attention concentration? Are there specific "legibility heads"?

**Method:** Per-head attention entropy for G3 (QwQ-32B), forward pass with `output_attentions=True`. Benjamini-Hochberg FDR correction across 2,560 tests (64 layers x 40 heads).

### Results

- **FDR-significant heads:** 0 out of 2,560
- **Top uncorrected:** Layer 44 head 20 (diff=+0.37, p=0.0017), layer 28 head 20 (diff=+0.34, p=0.0007)
- All top-10 differences are positive (legible has higher entropy than illegible)

Raw attention entropy is slightly higher for legible CoTs at most layers, but the effect is small (typically <0.3 nats) and none survive multiple comparison correction. Length-normalized entropy shows the same pattern.

### Interpretation

**No individual attention heads drive legibility.** This is informative alongside the CKA results: the dramatic representational divergence seen in A2 is **distributed** across the network rather than concentrated in specific attention circuits. Legibility is not a phenomenon that can be attributed to or controlled by manipulating individual heads.

The consistent (but individually non-significant) direction -- legible CoTs having slightly higher entropy -- tentatively suggests that legible reasoning involves slightly more diffuse attention (attending to broader context), while illegible reasoning is more narrowly focused. But this is speculative given the lack of statistical significance.

---

## Experiment C: Transition Dynamics

**Question:** At what point during CoT generation does the legibility signal emerge?

**Method:** Truncation sweep from 10% to 100% of the CoT in 5% increments. At each truncation point, extract the activation at that relative position and train a legibility probe. Sweep performed at layer 32 (middle of 64-layer model).

### Results

| Truncation % | AUROC |
|-------------|-------|
| 10% | **>0.65** (T* reached immediately) |
| 25% | moderate |
| 50% | moderate |
| 100% | moderate |

Multi-layer sweep (25%, 50%, 75%, 100% at all cached layers) shows the signal is present across layers and truncation points, without a sharp transition.

### Interpretation

T* = 10% -- the legibility probe exceeds the 0.65 threshold after reading just the first tenth of the CoT. There is no "transition point" because the signal is present from the very beginning.

This is fully consistent with Experiment B: if question-token activations already predict legibility (AUROC 0.76), then the information is available before generation starts and naturally persists into the first generated tokens. The CoT doesn't *become* legible or illegible at some midpoint -- it *starts* that way.

---

## Experiment D: Reader-Side Analysis (R2)

**Question:** What happens inside the reader model when processing legible vs illegible CoTs? Can we predict C2 success from reader activations?

**Method:** Probes on R2 (Llama-3.1-70B-Instruct, 4-bit quantized) activations. Three analyses: (1) predict C2 success from last-token activations, (2) predict C2 success from CoT boundary activations, (3) predict legibility class from reader activations. ANSWER_LEAKED samples excluded from all probes.

### Results

| Probe | Best layer | AUROC | 95% CI |
|-------|-----------|-------|--------|
| C2 success (last token) | 48 | 0.692 | ~[0.65, 0.73] |
| C2 success (CoT boundary) | 36 | 0.657 | ~[0.61, 0.70] |
| Legibility class | 36 | **0.702** | [0.652, 0.752] |

C2 success rates by Phase 1 label:
- ANSWER_LEAKED: 73.7% (excluded from probes)
- REASONING_LEGIBLE: 78.0%
- ILLEGIBLE: 20.5%

### Interpretation

The reader encodes legibility **more strongly** than the generator does (reader AUROC 0.70 vs generator 0.61-0.64). This makes sense: the reader is the model that must actually *use* the reasoning, so its activations should reflect whether the CoT is useful.

The last-token probe (0.692) outperforms the CoT boundary probe (0.657), indicating the reader continues to integrate and process the CoT content after the boundary marker. By the time the reader generates its answer, its internal state has a stronger legibility signal than at the transition point between CoT and answer prompt.

The legibility probe (0.702 at layer 36) peaks in the late-middle layers, suggesting R2 processes legibility information through a similar depth range as G3's CKA divergence.

---

## Experiment F1: G1 vs G3 Cosine Similarity

**Question:** When two different generators produce legible reasoning for the same question, how similar are their internal representations?

**Method:** Cosine similarity between matched legible CoT activations (same question, both generators produced legible reasoning and answered correctly). Only 9 matched pairs available.

### Results

| Layer | Cosine similarity | 95% CI |
|-------|------------------|--------|
| 0 | 0.647 | [0.402, 0.917] |
| 4 | 0.850 | [0.747, 0.936] |
| 8 | 0.813 | [0.710, 0.898] |
| 32 | 0.615 | [0.480, 0.752] |
| 63 | 0.868 | [0.819, 0.913] |

G1 activation norms are consistently ~1.4x larger than G3 (e.g. layer 63: G1=2070, G3=1531).

### Interpretation

The models share moderate representational similarity (cosine ~0.6-0.85), highest in early and final layers, lowest in middle layers (32: 0.615). This parallels the CKA U-curve: input/output representations are more constrained, while middle layers diverge. The models are solving the same problems but routing through different internal strategies.

---

## Experiment F2: Generator Identity Probe

**Question:** Can we distinguish G1 from G3 activations when both produce legible reasoning?

**Method:** Binary probe (G1=1, G3=0) on legible-only samples across all layers. Permutation test at best layer.

### Results

| Layer | AUROC |
|-------|-------|
| 0 | 0.976 |
| 4+ | **1.000** (perfect separation) |

Permutation test: p = 0.000

### Interpretation

Even when both models produce correct, legible reasoning for the same questions, their internal representations are **perfectly distinguishable** at every layer from 4 onward. The models employ entirely different computational strategies to arrive at the same externally-legible output.

This has implications for interpretability: "legible reasoning" does not correspond to a single internal computational strategy. Two models can both produce reasoning that humans and other models can follow, while implementing that reasoning through completely different neural pathways.

---

## Summary Table

| Exp | Question | Key metric | Result |
|-----|----------|-----------|--------|
| A1b | Can generators decode legibility? | AUROC | 0.61-0.64 (weak) |
| A1a | Can generators decode the answer? | Accuracy | 0.37 (inconclusive, n=38) |
| **B** | **Can legibility be predicted pre-CoT?** | **AUROC** | **0.757 (yes)** |
| A2 | Do leg/illeg CoTs share representational geometry? | CKA | 0.04 at layer 8 (no) |
| A3 | Are there legibility-specific attention heads? | FDR-significant | 0 of 2,560 (no) |
| C | When does the legibility signal emerge? | T* | 10% of CoT (immediately) |
| D | Can readers decode legibility? | AUROC | 0.70 (moderate) |
| F1 | Do generators share representations for legible CoTs? | Cosine | 0.6-0.85 (partial) |
| F2 | Can we tell generators apart during legible reasoning? | AUROC | 1.00 (perfectly) |

---

## Conclusions

### 1. Legibility is predetermined, not emergent during reasoning

The combination of Experiments B (pre-CoT AUROC 0.76) and C (T*=10%) establishes that legibility is largely determined before or at the very start of reasoning. The question's internal representation already carries the signal. This suggests certain questions or question-model interactions consistently elicit legible vs illegible reasoning patterns.

### 2. Legible and illegible CoTs are processed through different computational pathways

CKA analysis (A2) reveals near-zero representational overlap in middle layers, despite convergence at input and output layers. The model routes legible and illegible reasoning through fundamentally different internal subspaces. This divergence is distributed across the network (A3: no individual heads are responsible).

### 3. Readers are better legibility detectors than generators

Reader activations encode legibility more strongly (D: AUROC 0.70) than generator activations (A1b: AUROC 0.61-0.64). This is expected -- the reader is the one that must actually use the reasoning -- but it suggests that reader-side monitoring could be more effective than generator-side monitoring for detecting illegible CoTs.

### 4. Legibility is model-specific in implementation but cross-model in outcome

Different generators use completely different internal strategies (F2: perfect separation) even for identically-legible outputs. "Legible reasoning" is not a universal computational strategy but rather a property of the output that can be achieved through diverse internal mechanisms.

### 5. Distributional shift is not a confound

Foreignness scores do not differ between legible and illegible groups (mean 2.04 vs 1.96), and adding foreignness as a covariate does not change probe performance. The legibility differences we observe are content-level, not stylistic.

---

## Limitations

1. **Sample sizes are modest.** The legible class has only 30-33 samples per generator (100 total across generators). CIs are wide, and some analyses (A1a, F1) are severely underpowered.

2. **Only two generators were probed.** G2 (DeepSeek-R1, 671B) was too large for local activation extraction. The G1/G3 results may not generalise to G2.

3. **Linear probes only.** Non-linear decoders might find stronger legibility signals. The weak linear decodability (A1b) does not rule out strong non-linear decodability.

4. **Single reader model.** Only R2 (Llama-3.1-70B) was analysed on the reader side. R1 and R3 may show different patterns.

5. **Quantization effects.** R2 was loaded in 4-bit quantization for GPU memory constraints. This may attenuate activation signals.

6. **Phase 1 labels depend on R4 transform choice.** The `_t64` transform yields a specific label distribution; other transforms would change which samples are classified as ANSWER_LEAKED vs REASONING_LEGIBLE/ILLEGIBLE, potentially altering Phase 2 results.

---

## Notebooks

| Notebook | Experiment | GPU | Status |
|----------|-----------|-----|--------|
| NB1: `01_extract_activations_generators_h200.ipynb` | Activation extraction (G1, G3) | Yes | Complete |
| NB2: `02_extract_activations_reader_h200_r2.ipynb` | Activation extraction (R2) | Yes | Complete |
| NB3: `03_probe_legibility_generators_sklearn.ipynb` | A1a, A1b, B | No | Complete |
| NB4: `04_compare_geometry_generators_cka.ipynb` | A2 | No | Complete |
| NB5: `05_analyze_attention_generators_heads.ipynb` | A3 | Yes | Complete |
| NB6: `06_analyze_transition_generators_truncation.ipynb` | C | No | Complete |
| NB7: `07_analyze_activations_reader_r2.ipynb` | D | No | Complete |
| NB8: `08_compare_narration_g1g3_activations.ipynb` | F1, F2 | No | Complete |
| NB9: `09_plot_results_phase2_summary.ipynb` | Summary plots | No | Complete |