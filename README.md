# DopplerNet-SA+XPINN

**DopplerNet Self-Attention + XPINN** — JAX/Flax speed regression for DopplerSim acoustic pass-by clips. Combines a **2D CNN + Transformer** spectrogram encoder (from the DopplerNet / Self-Attention line) with **extended physics-informed (XPINN) regional losses** and a straight-path Doppler renderer.

Primary training artifact: [`DopplerNet_SA_XPINN.ipynb`](DopplerNet_SA_XPINN.ipynb)

---

## Name and lineage

| Term | Meaning |
|------|---------|
| **DopplerNet** | 2D log-CQT encoder family (per-bin z-score, CNN over time–frequency) |
| **SA (Self-Attention)** | Sinusoidal positional encoding + Transformer layers on the temporal axis after frequency compression |
| **XPINN** | Extended PINN-style **regional** losses (approach / CPA / recede) with a learned router and Doppler wing renderer |

This is **not** a wholesale PyTorch DopplerNet port. It is a hybrid: Self-Attention 2D backbone + physics regularizer + anti-collapse training (AttentionPool, phased curriculum).

---

## Repository layout

```
DopplerLab/
├── DopplerNet_SA_XPINN.ipynb   # train + evaluate (single source of truth)
├── README.md
├── model_plan_b1.md            # architecture history & ablations log
├── audit.md                      # pre-scale validation checklist
├── ref_docs/SelfAttn/            # PyTorch reference notebook
├── data/batch_outputs/           # symlink → attached disk (gitignored)
└── experiments/<run_name>/       # checkpoints + results (gitignored)
    ├── checkpoints/
    ├── setup_cache/
    └── results/
        ├── dataset/              # split histograms
        ├── training/             # curves + per-epoch .npy metrics
        ├── evaluation/             # pred plots, predictions
        └── metrics/                # json reports, resume state
```

---

## Quick start (GCP VM + GitHub)

```bash
git clone https://github.com/seetharamkkv/DopplerLab.git
cd DopplerLab
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip "jax[cuda12]" flax optax tqdm scikit-learn pandas matplotlib seaborn msgpack jupyter papermill

# Attach Rohitha's persistent disk (read-only data), then symlink:
ln -s /home/rohithas/DopplerSim/static/batch_outputs data/batch_outputs

# Optional: pick batch subfolder (else auto-select largest valid batch)
export DOPPLER_BATCH_NAME=model_test_1000000

# Smoke test (1 epoch) — edit RUN CONTROL in notebook first
papermill DopplerNet_SA_XPINN.ipynb outputs/smoke.ipynb --log-output
```

Verify GPU: `python -c "import jax; print(jax.devices())"`

---

## Task formulation

Given a clip's log-CQT spectrogram \(\mathbf{X} \in \mathbb{R}^{1 \times F \times T}\) (default \(F{=}84\), \(T{=}432\)), predict scalar **ground speed** \(v \in [0, v_{\max}]\) (m/s, default \(v_{\max}{=}50\)).

The model **never** sees CSV metadata, filenames, or simulator labels at inference — only \(\mathbf{X}\).

---

## Architecture

```
per-bin z-scored log-CQT  (1 × 84 × 432)
    │
    ▼
DopplerAttn2DEncoder
    2D CNN (×2 max-pool) → frequency compress → sinusoidal PE → Transformer × L
    │
    ▼
XPINN RegionRouter (softmax over 3 regions) + RegionAdapter
    │
    ▼
AttentionPool → shared embedding z
    │
    ├─► speed head  →  v̂, σ̂_v
    ├─► (optional) path / distance aux heads
    └─► nuisance head → d̂, t̂_CPA  →  physics renderer
```

**Default hyperparameters:** `attn_d_model=128`, `attn_n_layers=2`, `attn_n_heads=4`, `n_xpin_regions=3`.

### Input normalization

Per frequency bin \(f\), over time index \(t\):

\[
\tilde{X}_{f,t} = \frac{X_{f,t} - \mu_f}{\sigma_f + \epsilon}, \quad
\mu_f = \frac{1}{T}\sum_t X_{f,t}, \;\;
\sigma_f = \sqrt{\frac{1}{T}\sum_t (X_{f,t}-\mu_f)^2}
\]

Train-only gain jitter: \(\tilde{X} \leftarrow \tilde{X} \cdot g\), \(g \sim \mathcal{U}[0.85, 1.15]\).

### Positional encoding

For temporal index \(t\) and dimension \(i\):

\[
\mathrm{PE}_{t,2i} = \sin\left(t / 10000^{2i/d}\right), \quad
\mathrm{PE}_{t,2i+1} = \cos\left(t / 10000^{2i/d}\right)
\]

### Attention pooling

Learned query \(\mathbf{q}\): \(\alpha_t = \mathrm{softmax}_t(\mathbf{h}_t^\top \mathbf{q})\), \(\mathbf{z} = \sum_t \alpha_t \mathbf{h}_t\).

### Speed output

\[
\hat{v} = \sigma(\mathrm{MLP}(\mathbf{z})) \cdot v_{\max}, \quad
\hat{\sigma}_v = \mathrm{softplus}(\mathrm{MLP}(\mathbf{z})) + \epsilon
\]

---

## Physics renderer (straight-path pass-by)

Learned nuisances: distance at CPA \(\hat{d}\), CPA time \(\hat{t}_{\mathrm{CPA}}\), and speed \(\hat{v}\).

**Radial velocity** along line of sight (simplified geometry):

\[
v_r(t) = \frac{\hat{v}\,\hat{v}\,(t - \hat{t}_{\mathrm{CPA}})}
{\hat{v}^2 (t - \hat{t}_{\mathrm{CPA}})^2 + \hat{d}^2 + \epsilon}
\]

**Doppler log-frequency render** (fixed \(c = 343\) m/s):

\[
\log f_{\mathrm{render}}(t) = \log f_{\mathrm{ref}} +
\log\left(\frac{c}{c + v_r(t)}\right)
\]

Observed coarse log-frequency track (mean over frequency bins): \(\bar{X}_t\).

**Wing weights** (outside CPA neighborhood of width \(2\tau\)):

\[
w_{\mathrm{wing}}(t) = \mathbb{1}\big[|t - t_{\mathrm{CPA}}^{\mathrm{GT}}| > \tau\big]
\]

GT CPA time \(t_{\mathrm{CPA}}^{\mathrm{GT}}\) is used **only** for training masks, not at inference.

### XPINN regions

Three hard regions from GT CPA time (training masks):

\[
\mathcal{R}_{\mathrm{app}} = \{t : t - t_{\mathrm{CPA}}^{\mathrm{GT}} < -\tau\},\;
\mathcal{R}_{\mathrm{cpa}} = \{t : |t - t_{\mathrm{CPA}}^{\mathrm{GT}}| \le \tau\},\;
\mathcal{R}_{\mathrm{rec}} = \{t : t - t_{\mathrm{CPA}}^{\mathrm{GT}} > \tau\}
\]

Router output \(\mathbf{r}_t \in \Delta^2\) (3-way softmax) is trained to match these masks at the **CNN-downsampled** temporal resolution (\(T_{\mathrm{model}} \approx 108\)); physics losses use full \(T=432\).

---

## Loss function

### Supervised terms

**Huber loss** on normalized speed \(\tilde{v} = v / v_{\max}\):

\[
\mathcal{L}_{\mathrm{Huber}}(e) = \begin{cases}
\frac{1}{2} e^2 & |e| \le \delta \\
\delta(|e| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
\]

\[
\mathcal{L}_{\mathrm{sup}} =
\lambda_{\mathrm{sup}} \mathcal{L}_{\mathrm{Huber}}(\tilde{v} - \hat{\tilde{v}}) +
\lambda_{\mathrm{path}} \mathcal{L}_{\mathrm{CE}}(\mathrm{path}) +
\lambda_{\mathrm{dist}} \mathcal{L}_{\mathrm{Huber}}(\tilde{d} - \hat{\tilde{d}})
\]

Defaults: \(\lambda_{\mathrm{sup}}{=}1\), \(\lambda_{\mathrm{path}}{=}0.2\), \(\lambda_{\mathrm{dist}}{=}0.15\), \(\delta{=}0.5\).

**Heteroscedastic NLL** (Phase B+, optional weight \(\lambda_{\mathrm{nll}}\)):

\[
\mathcal{L}_{\mathrm{NLL}} = \frac{(\hat{v} - v)^2}{2\hat{\sigma}_v^2} + \log \hat{\sigma}_v
\]

### Physics terms (curriculum-gated)

**Wing Doppler MSE:**

\[
\mathcal{L}_{\mathrm{dopp}} =
\frac{\sum_t w_{\mathrm{wing}}(t)\,(\bar{X}_t - \log f_{\mathrm{render}}(t))^2}
{\sum_t w_{\mathrm{wing}}(t) + \epsilon}
\]

**CPA smoothness** on \(v_r(t)\):

\[
\mathcal{L}_{\mathrm{smooth}} =
\frac{\sum_t w_{\mathrm{cpa}}(t)\,(v_r(t+1) - v_r(t))^2}
{\sum_t w_{\mathrm{cpa}}(t) + \epsilon}
\]

**Spectral wing L1** (optional): mean absolute error between CQT and broadcast render, weighted by \(w_{\mathrm{wing}}\).

**Router alignment:**

\[
\mathcal{L}_{\mathrm{router}} = \|\mathbf{r} - \mathbf{r}_{\mathrm{GT}}\|_2^2
\]

**Physics cap** (prevents renderer dominating early training):

\[
\mathcal{L}_{\mathrm{phys}} \leftarrow \min\left(
\mathcal{L}_{\mathrm{phys}}^{\mathrm{raw}},\;
\eta \cdot \mathcal{L}_{\mathrm{sup}}
\right), \quad \eta = 0.3
\]

### Training curriculum

| Phase | Epochs | Active |
|-------|--------|--------|
| **A** | 1–20 | \(\mathcal{L}_{\mathrm{sup}}\) only |
| **B** | 21–60 | + Doppler wing, router, light contrastive |
| **C** | 61+ | + CPA smoothness, spectral physics |

Total loss: \(\mathcal{L} = \mathcal{L}_{\mathrm{sup}} + \lambda_{\mathrm{nll}}\mathcal{L}_{\mathrm{NLL}} + \mathcal{L}_{\mathrm{phys}} + \lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{con}}\).

---

## Data splits (no train/val/test leakage)

**Unit of split:** one DopplerSim scene = one `sample_*/` folder = one `clip_id`.

**Policy:** speed-stratified **70 / 15 / 15** within each 5 m/s bin \([10,15), [15,20), \ldots, [45,50]\).

1. Assign speed bin from label \(v\).
2. Within each bin: shuffle (seed **42**), allocate train/val/test counts (at least 1 val/test when bin has \(\ge 3\) clips).
3. Persist `setup_cache/split_indices.json` for reproducibility.

**Hard checks before training:**

- Zero overlap of `clip_id` and `sample_dir` across train / val / test
- `assert_split_no_leakage()` raises on any shared scene
- Loader disjointness verified on `clip_id`
- Augmentation (gain jitter) only on **train** loader
- Forward pass uses **CQT only** — no metadata leakage

---

## Metrics and saved artifacts

**Primary:** validation/test **MAE** (m/s), RMSE, \(R^2\).

**Per epoch** (saved under `results/training/`):

| File | Content |
|------|---------|
| `train_loss_per_epoch.npy` | Training loss |
| `train_mae_per_epoch.npy` | Training MAE |
| `train_sup_loss_per_epoch.npy` | Supervised loss component |
| `train_physics_loss_per_epoch.npy` | Physics loss (pre-cap) |
| `grad_norm_per_epoch.npy` | Gradient norm |
| `val_mae_per_epoch.npy` | Validation MAE |
| `val_r2_per_epoch.npy` | Validation \(R^2\) |
| `val_pred_std_per_epoch.npy` | Val prediction std (collapse detector) |
| `training_curves.png` | Combined plot at end of training |

**Evaluation:** ECE, PICP@90, Gaussian NLL, speed-bin confusion matrix, physics violation score \(V_{\mathrm{phys}}\).

---

## Configuration (notebook RUN CONTROL)

| Flag | Typical use |
|------|-------------|
| `EPOCHS` | 1 smoke / 80–250 full training |
| `RESUME_TRAINING` | `True` — load `checkpoints/checkpoint_latest.msgpack` |
| `START_FRESH` | `True` once for new architecture or clean run |
| `REFRESH_SETUP_CACHE` | `True` once after dataset or split change |
| `DATASET_BATCH_NAME` | Subfolder under `batch_outputs/` |

Experiment outputs: `experiments/test_dopplernet_sa_xpinn/` (change `experiment_name` in Config).

---

## Benchmark context

| Model | Dataset | Test MAE | Notes |
|-------|---------|----------|-------|
| DopplerNet 2D CNN (PyTorch) | model_test_1000 | ~6.5 m/s | Multitask baseline |
| Self-Attention 2D (PyTorch) | neurips_v1 | ~2.2 m/s | Different split |
| **DopplerNet-SA+XPINN** | model_test_1000 | ~2.7 m/s | Speed-stratified 70/15/15, hybrid |

See [`model_plan_b1.md`](model_plan_b1.md) for full ablation history and [`audit.md`](audit.md) for pre-1M scale validation gates.

---

## References

- DopplerSim batch layout: `dataset.csv`, `audio_clips/sample_*/Common/cqt.npy`, `B1_Speed/label_speed.npy`
- PyTorch reference: [`ref_docs/SelfAttn/SelfAttention.ipynb`](ref_docs/SelfAttn/SelfAttention.ipynb)

---

## Citation (placeholder)

If you use this model in publication, refer to it as **DopplerNet-SA+XPINN** (DopplerNet Self-Attention with Extended Physics-Informed Neural Network losses for Doppler speed regression).
