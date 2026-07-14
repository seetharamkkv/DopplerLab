# Physics vs CNN — direction comparison (mono, vehicle-only)

**Split:** EUSIPCO test  
**Task:** L2R vs R2L on vehicle clips only (n=1346)  
**Last updated:** 2026-07-02

This document compares the **physics-informed** track (`kinematic_v3` + logistic) against **CNN mel** baselines on **mono-left** audio. It does **not** compare against stereo cross-correlation (~99.9% bal acc) or mean-downmix mel — those use stereo or collapsed channels and are not fair monoaural baselines.

---

## Protocol

| Setting | Physics | CNN |
|---------|---------|-----|
| Classes | 2 (L2R, R2L) | 3 (L2R, R2L, no_vehicle) — metrics below are **vehicle-only** |
| Mono source | `left` (primary) / `right` (ablation) | `left` (primary) / `right` (reference) |
| Features | Hand-crafted kinematic scalars (`kinematic_v3`) | Log-mel spectrogram |
| Classifier | `LogisticRegression` + `StandardScaler` | Small CNN |
| Flip target | ≥70% **flip agreement** under time-reverse | Diagnostic only (not a training objective) |

**Flip metrics (vehicle L2R/R2L only):**

- **Flip agreement** — decision reverses: `pred_rev == 1 − pred_base` (pure mechanism).
- **Flip consistency** — prediction matches flipped true label: `pred_rev == 1 − true_label` (mechanism + skill; ceiling ≈ forward accuracy).

Artifacts: `outputs/physics/direction/<run>/eval_metrics.json`, `interventions.json`; `outputs/cnn/direction/<run>/eval_metrics.json`.

---

## Accuracy (vehicle-only balanced accuracy, L2R/R2L)

| Model | Run | Mono | Bal. acc | Macro F1 (2-class) | vs majority (53%) |
|-------|-----|------|----------:|-------------------:|--------------------:|
| Majority (always R2L) | — | — | 53.0% | — | — |
| **Physics v3_std (primary)** | `physics_lr_2class_left_v3_std` | left | **65.1%** | 65.0% | +12.1 pp |
| Physics v3_std (ablation) | `physics_lr_2class_right_v3_std` | right | **68.8%** | 68.8% | +15.8 pp |
| Physics v3 (flip-enforced) | `physics_lr_2class_left_v3` | left | 60.3% | 60.0% | +7.3 pp |
| Physics v2 (accuracy-first, linear) | `physics_lr_2class_left_v2` | left | 67.9% | 67.9% | +14.9 pp |
| Physics full + GBT | `physics_gbt_full_left` | left | 69.2% | 68.0% | +16.2 pp |
| **Physics full + MLP (best tabular)** | `physics_mlp_full_left` | left | **70.8%** | 70.0% | +17.8 pp |
| **Hybrid PINN (mel + kinematic_v3)** | `hybrid_mel_left_v3_ep60` | left | **76.3%** | 76.4% | +23.3 pp |
| CNN mel 3-class | `mel_3class_left` | left | **79.3%** | 79.5% | +26.3 pp |
| CNN mel 3-class (60 ep) | `mel_3class_left_ep60` | left | **~78.8%** | ~78.5% | +25.8 pp |
| CNN mel 3-class | `mel_3class_right` | right | **73.6%** | 73.2% | +20.6 pp |
| CNN mean downmix | `mel_3class` | (L+R)/2 | 81.5% | 81.3% | *not mono-fair* |

**Gap (primary):** CNN `mel_3class_left` still leads the hybrid PINN by **~3.0 pp** (79.3% vs 76.3%).
Hybrid closes **~5.5 pp** of the tabular physics gap (70.8% → 76.3%) but does not match the pure CNN.
Physics `v3_std` remains **~14 pp** below CNN.

CNN `mel_3class_left_ep60` (60 epochs, no early stop) did **not** improve over the original 40-epoch checkpoint; best val epoch was 23. Extra training is not required for this comparison.

---

## One-shot gap-closing attempt (rich features + nonlinear head)

Goal: match/exceed the CNN base (79.3%) with physics features by fixing **both** suspected
limits at once — feature starvation and the linear model.

- **`kinematic_full`** (27 features): union of every kinematic scalar the pipeline computes
  (envelope rise/fall/width/asymmetry at −3/−10 dB, peak offset, Doppler pre/post slopes +
  ratio + transition width, centroid mean/std/skew/span/Δt, plus antisym-derived terms).
  Accuracy-first, single ear, nothing dropped for flip symmetry.
- **Nonlinear heads:** `gbt` (HistGradientBoosting) and `mlp` (64→32).

| Model | Bal. acc | Δ vs linear v2 (67.9%) | Gap to CNN (79.3%) |
|-------|---------:|-----------------------:|-------------------:|
| Linear (v2) | 67.9% | — | −11.4 pp |
| GBT + full | 69.2% | +1.3 pp | −10.1 pp |
| MLP + full | **70.8%** | +2.9 pp | **−8.5 pp** |

**Verdict: the gap is a feature ceiling, not a model ceiling.** Throwing every feature and a
nonlinear model at the problem bought only ~3 pp (67.9% → 70.8%). Permutation importance on
the GBT shows the model leans on **envelope width** (`env_10db_width_s` ≈ 0.064,
`env_3db_width_s` ≈ 0.026) — a *duration/speed* proxy that is **direction-agnostic** — while
the genuinely directional antisym features contribute ≈ 0. Hand-crafted mono scalars simply
do not carry enough L2R/R2L information to reach the CNN, which reads the full 2-D mel.

**Implication:** tabular physics hit a ceiling at ~71%; the first **PINN/hybrid** run (`hybrid_mel_left_v3_ep60`,
60 epochs, no preempt) lifts that to **76.3%** by fusing mel CNN embeddings with `kinematic_v3`
conditioning. The remaining **~3 pp** to CNN likely needs stronger fusion (e.g. `kinematic_full`,
flip-consistency auxiliary loss) or training stability fixes (val loss was noisy across epochs).

---

## Interventions (test, vehicle L2R/R2L)

| Model | Bal. acc | **Flip agreement** | Flip consistency | Channel-swap agreement |
|-------|----------:|-------------------:|-----------------:|-----------------------:|
| Physics `v3_std` (left) | 65.1% | **77.3%** | 57.6% | 13.8% |
| Physics `v3_std` (right) | 68.8% | **88.4%** | 65.7% | 17.7% |
| Physics `v3` (antisym LR) | 60.3% | **81.2%** | 62.6% | — |
| CNN `mel_3class_left` | 79.3% | *not run* | — | 6.8% (legacy diag.) |
| CNN `mel_3class_left_ep60` | ~78.8% | **12.9%** | 30.1% | 6.7% |
| CNN `mel_3class_right` | 73.6% | *not run* | — | 11.1% (legacy diag.) |

Source files:

- Physics: [`physics_lr_2class_left_v3_std/eval_metrics.json`](direction/physics_lr_2class_left_v3_std/eval_metrics.json), [`interventions.json`](direction/physics_lr_2class_left_v3_std/interventions.json)
- CNN ep60: [`mel_3class_left_ep60/eval_metrics.json`](../cnn/direction/mel_3class_left_ep60/eval_metrics.json), [`interventions.json`](../cnn/direction/mel_3class_left_ep60/interventions.json)

---

## Interpretation

1. **Accuracy vs kinematics trade-off.** The CNN learns stronger monoaural direction cues (~79% vs ~65%) but those cues are **not time-reversible**: only **13%** of CNN decisions flip under `y[::-1]`, vs **77%** for physics `v3_std`. The accuracy gap is not evidence that physics is uniquely weak — the CNN is optimizing a different, non-kinematic representation.

2. **Flip agreement is the physics design target.** `kinematic_v3` uses strictly antisymmetric features; `logistic_antisym` enforces reversal by construction (81% agreement, −5 pp accuracy). Standard logistic on v3 (`v3_std`) is the recommended primary: clears the ≥70% bar with best accuracy/flip balance.

3. **Channel-swap is a weak probe for both tracks.** Opposite-ear audio is not a time-reversed copy (different envelope geometry). Low agreement (~7% CNN, ~14% physics) is expected and should not be read as “failed flip.”

4. **Ear asymmetry.** Left vs right mono show opposite biases on this dataset:
   - **CNN:** left 79.3% bal acc (R2L recall 87%); right 73.6% (R2L recall 96%, L2R recall 52%).
   - **Physics v3_std:** left 65.1% (balanced recalls ~67%/63%); right **68.8%** (+3.7 pp) with R2L recall 80% and flip agreement **88.4%**. Right ear carries stronger kinematic R2L cues for physics features; left remains the primary reported run for symmetry with CNN `mel_3class_left`.

---

## Recommended reporting (de-bloated)

For papers or slides, report **vehicle-only** metrics:

- **Physics primary:** `physics_lr_2class_left_v3_std` — 65.1% bal acc, 77.3% flip agreement.
- **CNN mono baseline:** `mel_3class_left` — 79.3% bal acc; cite 12.9% flip agreement from `mel_3class_left_ep60` intervention battery.
- **Do not** headline 3-class accuracy (~90%) without noting `no_vehicle` inflation (~10–13 pp).

---

## Checkpoints

| Step | Item | Status |
|------|------|--------|
| 5 | CNN intervention battery on `mel_3class_left_ep60` | done |
| 7 | This comparison doc | done |
| 6 | Right-channel physics ablation (`physics_lr_2class_right_v3_std`) | done (68.8% bal acc, 88.4% flip agreement) |

Related: [`ref_docs/checkpoints.md`](../../../ref_docs/checkpoints.md), [`outputs/shared/reports/monoaural_performance_report.md`](../shared/reports/monoaural_performance_report.md).
