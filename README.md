# DopplerLab

Research workspace for vehicle pass-by audio: speed, length, direction, and related
Doppler / roadside experiments on simulated batches and real roadside recordings.

## This version

Adds a reusable **leakage-aware data-loading package** so models share one
discover → split → stats → verify path:

- scene-level holdout by default (`vehicle|speed`)
- mel / feature stats fit on the **fit** partition only
- hard leakage checks before train or eval
- opt-in LOVO (unknown car) and speed-gap spreading (±1 km/h near-twins)

**Full method, how to run, and expected outputs:**  
→ **[data_loading/README.md](data_loading/README.md)**

---

## Authors

**Seetharam Killivalavan**  
School of Computer Science, Carnegie Mellon University

---

## Acknowledgments

Carnegie Mellon University, the Language Technologies Institute, Bradley Warren, and Professor Bhiksha Raj for research guidance and support.
