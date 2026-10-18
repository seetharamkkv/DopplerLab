# No-leakage compliance (Phase 6)

**Product:** monaural vehicle audio → observer-centered trajectory **orbit** (rotational ± mirror family).  
**Scope today:** simulated DopplerSim / in-package Tier-1 freehand data. Real GPS transfer is **explicitly deferred**.

## Rules (signed for demo / paper claims)

1. **Inference inputs only:** WAV and/or `spectrograms/stft.npy`.
2. **Never at inference:** `metadata/*`, labels, simulation JSON, vehicle/site IDs, path polyline, CPA/speed sidecars, GPS.
3. **Splits:** train/val/test clip IDs must not overlap; held-out path families used when claimed.
4. **Filename hygiene:** speeds/IDs in filenames are for humans only — not model features.
5. **Normalization:** sim clips may be peak-normalized; no site-calibrated mic gain tables are used. If absolute metric scale is uncertain, set `scale_ambiguous=true`.
6. **Ambiguity is part of the product:** `mirror_ambiguous=true`, `heading_absolute=false`.
7. **Headline metric:** orbit RMS after Procrustes rotation ± reflection about the mic — not raw world XY.

## Automated evidence

```bash
python scripts/audit_leakage.py --batch data/tier1_smoke --out outputs/leakage --wav-smoke
python scripts/package_release.py --out outputs/release
```

`leakage_audit.json` must report `"ok": true`. WAV-only smoke stages a folder containing **only** a `.wav` and still builds the Phase 5 orbit product.

## Real transfer

Deferred until controlled GPS-labeled flybys are available. When added: same `predict_orbit` API, GT for eval only, no site IDs as inputs.
