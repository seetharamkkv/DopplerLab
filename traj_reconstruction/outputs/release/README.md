# traj_reconstruction release (simulated)

Audio-only recovery of an **observer-centered trajectory orbit**
(full rotational family about the microphone). Absolute world heading is not identified.

## Input
- Monaural WAV, and/or
- `spectrograms/stft.npy`

## Output
- `orbit_product.html` — rotate / mirror the recovered path about the observer
- `orbit_product.json` — polyline + polar `(r, theta_rel)` + ambiguity flags
- `orbit_product.png` — static figure

## Ambiguity flags
- `mirror_ambiguous=true` (monaural default)
- `heading_absolute=false`
- `scale_ambiguous` when amplitude calibration is unknown

## Non-claims
- No absolute map heading
- No metadata (vehicle/site/CPA/speed labels) at inference
- Real roadside transfer is **deferred** unless a GPS appendix is attached

## Demo
```bash
python -m pip install -e .
python scripts/demo_orbit_product.py --wav /path/to/clip.wav --out release_demo
```

## Leakage
See `NO_LEAKAGE.md` and `leakage_audit.json` in this folder.
