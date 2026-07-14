# 2-class direction comparison (L2R vs R2L)

**Metric:** balanced accuracy on **test** split, **vehicle clips only** (no `no_vehicle`).

## Baselines

| Phase | Run | Bal. acc | L2R recall | R2L recall | Flip agree. | Notes |
|-------|-----|----------|------------|------------|-------------|-------|
| ref | `mel_3class_left` | 79.3% | 71.2% | 87.4% | 6.8% | CNN baseline — mono left, 40 ep + preempt |
| ref | `mel_3class_right` | 73.6% | 51.6% | 95.7% | 11.1% | CNN baseline — mono right |
| ref | `mel_3class` | 81.5% | 86.6% | 76.5% | — | CNN baseline — stereo mean downmix (L+R)/2 |
| ref | `mel_3class_left_ep60` | 78.8% | 68.4% | 89.2% | 6.7% | CNN — 60 ep, no preempt (intervention baseline) |
| ref | `mel_3class_left_ep200` | 76.8% | 63.0% | 90.6% | 6.8% | CNN — 200 ep, no preempt |
| ref | `mel_3class_left_aug_v1` | 76.0% | 61.1% | 90.9% | 6.2% | Phase A — SpecAugment + focal + balanced sampler |
| ref | `physics_mlp_full_left` | 70.8% | 81.0% | 60.5% | — | Physics MLP — kinematic_full features, mono left |
| ref | `hybrid_mel_left_v3_ep60` | 76.3% | 62.2% | 90.5% | 8.3% | Archived hybrid PINN — mel + physics late fusion, 60 ep |
| ref | `fusion_cnn_baseline_2class` | 74.9% | 54.4% | 95.4% | 17.4% | CNN L+R late fusion (mel_3class left+right, w_L=0.15) |

## Phases B–D (100 epochs)

| Phase | Run | Bal. acc | L2R recall | R2L recall | Flip agree. | Notes |
|-------|-----|----------|------------|------------|-------------|-------|
| B | `deep_mel_2class_left_100ep` | 87.6% | 76.9% | 98.3% | 39.0% | Deep residual mel CNN, mono left |
| B | `deep_mel_2class_right_100ep` | 90.4% | 81.2% | 99.7% | 46.8% | Deep residual mel CNN, mono right |
| C | `fusion_2class_100ep` | 90.5% | 81.3% | 99.7% | 10.6% | Late fusion deep mel L+R (weight fit on valid) |
| D | `film_2class_left_100ep` | 77.5% | 79.3% | 75.6% | 15.6% | FiLM + flip-consistency loss, mono left (archived metrics; weights removed) |

## Targets (from plan)

| Phase | Target |
|-------|--------|
| B | >= 80.5% |
| C | >= 81.0% |
| D | accuracy + flip consistency |
