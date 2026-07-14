# Engine acoustics

Physics-based **four-stroke engine acoustic synthesis** and **order-domain analysis** for automotive signal processing. Maps **RPM**, **cylinder count / layout**, and **load** to a time-varying audio signature with correct firing frequencies, harmonic orders, and per-cylinder phasing.

Designed to complement DopplerLab’s pass-by analysis tracks: use this package as a **source model** upstream of [DopplerSim 2.0](https://github.com/rohitharumugams/dopplersim_2.0), or validate synthesized audio with existing mel / envelope tooling.

---

## Physical model (summary)

### Crank and firing frequencies

For a **four-stroke** engine with `N` cylinders:

| Quantity | Formula |
|----------|---------|
| Crank (1×) frequency | `f_c = RPM / 60` Hz |
| Firing fundamental | `f_fire = f_c × (N / 2)` Hz |
| Engine order `k` | `f_k = k × f_c` Hz |

**Example — inline-4 @ 3000 RPM**

- 1× crank = 50 Hz (shaft / reciprocating imbalance)
- 2× = 100 Hz = **firing fundamental** (two power strokes per crank revolution)
- 4× = 200 Hz = 2nd harmonic of firing

Each cylinder fires once every **720°** of crank rotation. Cylinders are staggered by `720° / N` in crank angle; the **firing order** (e.g. 1-3-4-2) permutes which physical cylinder fires at each step.

### Why sound is not a pure tone

Engine noise is a **composite**:

1. **Combustion pressure pulses** — broadband, repeated at `f_fire`, with cycle-to-cycle variation
2. **Harmonic orders** — integer multiples of crank frequency (firing + mechanical harmonics)
3. **Layout phasing** — V/flat banks introduce asymmetric paths and weak half-orders
4. **Load scaling** — throttle / torque modulates pulse amplitude nonlinearly
5. **Resonances** — intake, exhaust, and block modes color the spectrum (modeled in `hybrid` synthesis)

---

## Install

```bash
cd engine_acoustics
pip install -r requirements.txt
pip install -e src
```

---

## Quick start

```python
from engine_acoustics import EngineConfig, EngineSynthesizer, SynthesisConfig
from engine_acoustics.synthesis import ramp_rpm

engine = EngineConfig(num_cylinders=4, firing_order=(1, 3, 4, 2))
synth = EngineSynthesizer(engine, SynthesisConfig(duration_s=2.0, load=0.8))

audio, meta = synth.synthesize(ramp_rpm(1000, 4000, duration_s=1.8), mode="hybrid")
# meta["rpm"], meta["crank_angle_rad"] — sidecars for order tracking
```

### Demo (WAV + plots)

```bash
python engine_acoustics/demo.py --output engine_acoustics/outputs/demo
```

---

## Synthesis modes

| Mode | Description |
|------|-------------|
| `pulse` | Crank-angle impulse train with per-cylinder firing angles |
| `additive` | Order-tracked sinusoids at configured harmonic amplitudes |
| `hybrid` | Pulse excitation + resonant orders + band-limited aspiration noise (default) |

### RPM profiles

Built-in factories in `synthesis.py`:

- `constant_rpm(rpm)`
- `ramp_rpm(rpm_start, rpm_end, duration_s)`
- `step_rpm([(t0, rpm0), (t1, rpm1), ...])`

Pass a NumPy array for recorded dyno / CAN bus traces.

---

## Order tracking

```python
from engine_acoustics import OrderTracker

tracker = OrderTracker(engine, sample_rate=22050)
spectrum = tracker.order_spectrum(audio, meta["rpm"], max_order=12.0)
tracks = tracker.extract_order_tracks(audio, meta["rpm"], orders=[1.0, 2.0, 4.0])
```

- **Angular resampling** — uniform crank-angle domain → order FFT
- **Order ridges** — `stft_order_ridge()` follows `f(t) = order × RPM(t)/60` in STFT
- **Model comparison** — measured vs configured `order_amplitudes`

---

## Configuration

`EngineConfig` fields:

| Field | Role |
|-------|------|
| `num_cylinders` | Cylinder count |
| `firing_order` | 1-based permutation (e.g. I4: 1-3-4-2) |
| `layout` | `inline`, `v`, `flat`, `boxer` |
| `order_amplitudes` | Dict `order → relative gain` (defaults provided) |
| `pulse_width_fraction` | Combustion pulse width vs firing interval |
| `reciprocating_imbalance` | 1× / 2× mechanical content |

`SynthesisConfig`: `sample_rate`, `load` (0–1), `broadband_noise`, `cycle_variation`.

---

## Integration with DopplerLab

| Downstream | How to connect |
|------------|----------------|
| **DopplerSim** | Export WAV @ 22.05/44.1 kHz as moving source input |
| **length_estimation** | Compare envelope / Doppler slopes of pass-by after simulation |
| **IDMT mel pipeline** | Use same `SR=22050`, `N_FFT=2048`, `HOP=512` for apples-to-apples spectrograms |

---

## Module map

```
engine_acoustics/src/engine_acoustics/
├── config.py          # EngineConfig, SynthesisConfig
├── physics.py         # f_c, f_fire, orders, cylinder angles
├── synthesis.py       # EngineSynthesizer, RPM profile helpers
├── order_tracking.py  # Angular resampling, order spectrum, ridges
└── __init__.py
```

---

## References

- Piston engines: firing interval = 720° crank / cylinder (four-stroke)
- Order domain: frequency = order × shaft frequency (SAE convention)
- Computed order tracking: angular resampling + FFT (Vold, Order Tracking)
