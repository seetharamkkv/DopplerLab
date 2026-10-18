# Plan: CTC for kinematics-based retrieval

**Status:** phases 0–2 implemented in this folder. Train VS13 and DopplerSim separately (`scripts/run_phases.py`). Paths: `dataset_paths.json` or `DOPPLER_VS13` / `DOPPLER_SIM_TEST` / `DOPPLER_SIM_UNCONSTRAINED`.

**Starting point:** a CTC model that emits a **time-ordered kinematic token string** from pass-by audio, then retrieve other clips by that string (speed, distance, pass-by phase). Do not start with the full Acoustic Language Model.

Existing clip-level heads (`speed_estimation`, VS13 Ridge, length CNN, orbit CNN) predict **one number or one path per clip**. CTC is for the case where the useful object is a **sequence**: approach → CPA → recede, with speed and range changing along the event.

---

## Why CTC

Pass-by labels from DopplerSim are exact in time (`state_frames`: position, speed, range). Frame-level supervision is possible, but:

- Frame labels are noisy to define on real audio (no per-frame speed).
- The scientifically useful object is often a **short token sequence**, not a dense regression map.
- CTC trains a sequence model **without forcing a frame-to-token alignment**. That matches “this clip is a 20 m/s pass at ~15 m CPA” better than “frame 173 is 19.4 m/s”.

Retrieval then uses the decoded token string (or the CTC posterior lattice), not a single scalar.

---

## What we are *not* doing first

- Not ALM / language tokens for weather, type, or engine orders.
- Not real IDMT / VS13 as the training set (labels are clip-level, not sequences).
- Not full orbit polylines as CTC targets (too long, continuous). Orbit stays in `traj_reconstruction`.
- Not replacing the VS13 Ridge speed model. That remains the real-data baseline. CTC is the **sim sequence** track.

---

## Tokenization (v0)

One alphabet, one CTC head. Keep it small.

| Token class | Examples | Role |
|-------------|----------|------|
| Phase | `APR`, `CPA`, `REC`, `SIL` | Pass-by structure |
| Speed bin | `V10`, `V15`, … `V50` (5 m/s bins) | Kinematics |
| Range bin | `R05`, `R10`, … `R40` (CPA-centric or instantaneous range) | Distance |

**Target string (example):**  
`SIL APR V20 R30 APR V20 R20 CPA V20 R10 REC V20 R20 REC V20 R30 SIL`

Rules for v0:

- DopplerSim **constant-speed** clips: one `V*` token for the whole motion, plus phase and range that **do** change.
- Quantize from `state_frames` / filename `{speed}mps_{dist}m` onto the STFT hop grid, then **collapse repeats** for the CTC target (standard CTC).
- Blank token `_` is model-internal, not in the string we store for retrieval.
- Maximum target length: tens of tokens, not hundreds. If range is sampled every hop, downsample (e.g. one token per 100–200 ms) before collapse.

**v1 (later):** two-stream CTC (phase+speed vs range) only if v0 saturates or the joint alphabet explodes.

---

## Model (v0)

Small, audio-only, sim-only.

1. **Front-end:** log-mel or log-STFT on the same hop as `traj_reconstruction` (do not invent a second time grid).
2. **Encoder:** 1D CNN or small Conformer over time (not a giant transformer).
3. **CTC head:** linear projection to `|alphabet| + blank`.
4. **Decode:** greedy CTC for the first retrieval index; beam search only if greedy strings are unstable.

Train with CTC loss. Optional auxiliary clip-level MSE on mean speed / CPA distance as a **regularizer**, not the main metric.

Compare against:

- Clip-level classifier on the same speed/distance bins (no sequence).
- Frame-level CE on the uncollapsed bin sequence (forced alignment).
- Existing speed notebook regressor on the same batch (`speed_estimation`).

If CTC does not beat clip-level bins on **clip speed / CPA MAE after decoding**, stop and rethink tokenization before retrieval.

---

## Retrieval protocol

Index: CTC-decoded string (and optionally the collapsed phase-only string).

Queries:

| Query | Example | Success |
|-------|---------|---------|
| Speed | “find ~20 m/s pass-bys” | retrieved clips’ true speed in the same bin or ±5 m/s |
| Distance | “find close CPA (~10 m)” | true CPA in the same range bin |
| Joint | “20 m/s at 10 m CPA” | both |
| Phase shape | “multi-CPA / long recede” | family / duration agreement |

Metrics (sim, where GT exists):

- **Token error rate** (speed bins, range bins, phase) after CTC collapse.
- **Clip kinematics from decode:** parse first/majority `V*` and min `R*` → MAE vs filename / metadata.
- **Retrieval:** Recall@k and nDCG with relevance = same speed bin, same range bin, or both.
- **Hard negative:** same vehicle timbre, different speed (sim can hold source PSD fixed).

Do not use embedding cosine as the primary v0 metric. String match on kinematic tokens is the claim.

---

## Data

**Train / val / test:** DopplerSim 2.0 batch exports and/or `traj_reconstruction` Phase 1 packages (`dataset.csv`, `state_frames.npy`). Same leakage rules as other tracks: split by clip id, not random frames.

Minimum viable batch:

- Straight paths, constant speed, one mic, several vehicles.
- Speed grid covering the `V*` bins.
- CPA distance grid covering the `R*` bins.
- Enough clips that retrieval@k is not a toy (order of 10³, not 20).

**Later:** VS13 as a *probe only* — no per-frame tokens. Map CTC decode → clip speed / CPA vs VS13 labels. Expect failure until sim→real is addressed; still useful as a ceiling check.

Reuse loaders from `data_loading/` and Phase 1 readers from `traj_reconstruction`. Do not copy datasets into `ctc_exp/`.

---

## Phases

### Phase 0 — alphabet and labels (no neural net)

- Script: `state_frames` / metadata → collapsed token string.
- Sanity: constant-speed straight pass-by always contains `APR … CPA … REC`.
- Histogram of string lengths; reject alphabets that yield 1-token strings or 200-token strings.
- **Exit:** 20-clip inspection notebook or pytest on synthetic strings.

### Phase 1 — CTC kinematics model (this is the first *model*)

- Train CTC on sim audio → token string.
- Report TER and decoded speed / CPA MAE vs clip-level baselines.
- **Exit:** CTC decode recovers speed bin and CPA bin above chance and near the clip-level bin classifier.

### Phase 2 — retrieval

- Build an inverted index from decoded strings on a held-out sim gallery.
- Speed-only, distance-only, joint queries.
- **Exit:** Recall@10 clearly above random and above retrieval by clip-level predicted bins.

### Phase 3 — unconstrained sim paths + continuous tracking

Batch: `dopplersim_unconstrained` (`traj_reconstruction_1000`, `free_path_2d`).
Script: `scripts/run_track.py`.

Shared encoder predicts (1) class-balanced speed bins, (2) per-frame canonical `xy`, (3) per-frame velocity.
Position loss is **time-centered Huber / GT RMS** so a constant mean location does not help.
Velocity Huber plus finite-difference consistency (`v ≈ Δxy/Δt`).
Eval: speed-bin accuracy vs majority, orbit-aligned XY RMS, velocity MAE.

- Accelerating clips (`V*` changes inside the string).
- Freehand paths from DopplerSim 2.0 / orbit batches (range sequence richer; heading still out of scope).
- Optional second head: retrieve by **phase string only** vs full string.

### Phase 4 — real data (optional, late)

- Freeze sim CTC, decode VS13 / IDMT.
- Compare decoded speed to Ridge / VS13 labels.
- Decision: sim CTC as **supplement** (features / retrieval candidates) vs not ready as **proxy**.

---

## Folder layout (when code starts)

```
ctc_exp/
├── README.md
├── PLAN.md              ← this file
├── src/ctc_exp/         ← alphabet, label export, model, decode, retrieve
├── scripts/             ← export_tokens.py, train_ctc.py, retrieve.py
└── tests/               ← token collapse, CPA/APR/REC on a toy straight pass-by
```

Keep checkpoints and batch audio out of git.

---

## Success / fail

**Success (Phase 1–2):** from audio alone, a short CTC string retrieves other sim pass-bys with matching speed and CPA better than chance and better than a clip-level bin index.

**Fail:** decoded strings ignore range and only copy a global speed token. Then CTC is not doing alignment; fall back to clip-level models and do not expand the alphabet.

---

## Relation to other tracks

| Track | Role relative to CTC |
|-------|----------------------|
| `speed_estimation` | Clip-level speed baseline; same sim batches |
| `traj_reconstruction` | Time grid, `state_frames`, orbit (not CTC targets) |
| `length_estimation` / IDMT | Real probes in Phase 4 only |
| `engine_acoustics` | Source `s(t)`, not CTC labels |
| ALM (unbuilt) | CTC kinematics string can later be **one readout**, not the encoder |
