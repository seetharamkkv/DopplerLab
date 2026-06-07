# Geolocation Pipeline — Architecture & Approach

**Last updated:** 2026-06-07  
**Scope:** Pretrained inference pipeline only (no model training).

**Verdict:** Promising engineering plan with the right shape (one primary geo branch, biological priors, weak scene cues, late fusion). It is **pipeline-only**, **good enough to implement**, but **not yet strong enough to claim precise coordinate geolocation** without caveats on checkpoint provenance, gallery quality, and heuristic confidence.

---

## 1. Purpose

Estimate geographic location from **audio recordings** (bird/nature soundscapes, urban parks) by fusing several pretrained signals. Default mode is **audio-only** — filenames and titles are ignored unless `--use-metadata` is passed.

**Outputs** (`outputs/predictions.json`):

- `latitude`, `longitude`, `radius_km` (uncertainty radius in km)
- `country`, `city`, `region` when inferable
- `granularity`, `confidence`, `confidence_by_level`
- `evidence[]` — which models contributed

Treat coordinates as **candidate estimates**, not ground truth. Prefer region/country when `granularity_fallback` is true.

---

## 2. Approach

### What is strong

- **Pipeline-only scope** — frozen weights, retrieval, fusion; no training loop.
- **Architecture shape** — one primary geo branch, one biological prior branch, one weaker scene branch, late fusion with fallback granularity. Appropriate when no single model is reliable alone.
- **GeoCLIP role** — used only as a **location encoder** for the retrieval index (official package/repo; frozen infrastructure). GeoCLIP is not treated as an audio geolocation model.

### Branch roles (honest wording)

1. **Primary geolocation hypothesis:** AG-CLIP-to-GeoCLIP retrieval over a **frozen gallery** yields a **candidate coordinate estimate**, subject to gallery-quality limits. This is cross-space projection retrieval (audio embedding → location embedding → nearest grid points), not classic matching against geo-tagged audio prototypes.

2. **Species priors:** BirdNET (+ optional Perch) and the AG-CLIP species head feed SINR range masking, eBird (optional), and GBIF fallback. These provide **exclusion and reweighting**, not direct high-resolution localization. On European clips they help eliminate impossible regions more than they pin a city.

3. **Auxiliary scene/context:** CLAP zero-shot prompts and urban/forest scene scores provide **auxiliary scene/context evidence** and fusion weight tweaks. CLAP must **not dominate** geographic inference; it is fallback-tier, not core geo.

4. **Fusion:** Spherical weighted mean of lat/lon hypotheses with a per-level confidence ladder (country → region → city → coordinates). Confidence is **heuristic**, not calibrated accuracy.

5. **Optional metadata path:** spaCy NER + Nominatim from titles/filenames — debug only, not for blind audio eval.

---

## 3. Model provenance

| Component | Grounding | Notes |
|-----------|-----------|--------|
| **GeoCLIP** | Externally verified | Published package (`geoclip`), location encoder used as frozen index infrastructure |
| **BirdNET** | Externally verified | Official pip package; checkpoints auto-download |
| **CLAP** | Externally verified | Hugging Face `laion/clap-htsat-fused`; general audio-text, **not** geo-trained |
| **AG-CLIP** | **Repo-local verified** | Paper: *Audio Geolocation: A Natural Sounds Benchmark* ([nat-sound2loc](https://github.com/cvl-umass/nat-sound2loc-code)). Checkpoints (`ag_clip.pt`, `sinr_geo_model.pt`) load and run in **this repo** via Google Drive; there is no equivalent `from_pretrained` Hugging Face release. Treat runtime contract (22 kHz mel, 512-d geo head) as **operationally verified here**, not as a fully public reproducibility guarantee until upstream documents a canonical release page. |
| **SINR geo model** | **Repo-local verified** | Bundled with AG-CLIP Drive release; same caveat as above |
| **Perch** | Externally verified | `perch-hoplite`; optional, heavy dependency |
| **eBird API** | Externally verified | Optional REST prior; requires API key |

**Constraint:** Current AG-CLIP dependency is operationally verified in this repo, but **external checkpoint provenance should be documented separately** (see `scripts/download_instructions.md`) before claiming public reproducibility beyond this workspace.

---

## 4. Architecture

```
                    ┌─────────────────────────────────────┐
                    │           INPUT: audio file          │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
   │  AG-CLIP*    │          │ BirdNET +    │          │ CLAP scene   │
   │  22 kHz mel  │          │ Perch        │          │ (auxiliary)  │
   └──────┬───────┘          └──────┬───────┘          └──────┬───────┘
          │                         │                         │
          │ geo 512-d               ▼                         │
          │ species 5547-d   ┌──────────────┐                 │
          │                  │ SINR / eBird │                 │
          │                  │ / GBIF prior │                 │
          ▼                  └──────┬───────┘                 │
   ┌──────────────┐                 │                         │
   │ GeoCLIP-index│  exclusion &   │                         │
   │ two-pass K-NN│  reweighting   │                         │
   └──────┬───────┘                 │                         │
          └─────────────────────────┼─────────────────────────┘
                                    ▼
                         ┌────────────────────┐
                         │ Spherical fusion   │
                         └─────────┬──────────┘
                                   ▼
                         outputs/predictions.json

* AG-CLIP: repo-local checkpoint; see §3
```

---

## 5. Gallery limitation (important)

The current gallery (`build_agclip_gallery.py`) is a **synthetic coordinate lattice** (~7k points: H3 grid + regional densification + hand-placed prompts), **not** geo-tagged iNatSounds audio prototypes.

Implications:

- Retrieval is **not** “find the nearest real recording at this habitat.”
- It is **AG-CLIP audio embedding → cosine match against GeoCLIP location embeddings of arbitrary lat/lon points** — a cross-space heuristic whose quality depends on how well the audio geo head aligns with that lattice.
- Upgrading to real GPS-tagged audio exemplars (`build_agclip_gallery.py` + HF metadata) would materially change what this branch can claim.

---

## 6. Project structure

```
geolocation/
├── configs/fusion_weights.yaml
├── data/                         # Input audio
├── cache/                        # Taxonomy, GBIF, Nominatim caches
├── models/                       # Weights & galleries (gitignored)
│   ├── ag_clip/models/           # ag_clip.pt, sinr_geo_model.pt
│   ├── faiss/                    # Retrieval galleries
│   └── clap/                     # HF cache for CLAP
├── outputs/                      # predictions.json, run_log.txt
├── ref_docs/progress.md          # This file
├── scripts/
│   ├── run_pipeline.py
│   ├── bootstrap.py
│   ├── download_instructions.md
│   └── pipeline/
└── requirements.txt
```

---

## 7. Major components

| Component | Module | Role |
|-----------|--------|------|
| AG-CLIP encoder | `agclip_encode.py` | Audio → 512-d geo embedding + species logits (repo-local weights) |
| GeoCLIP gallery | `geoclip_gallery.py` | Two-pass K-NN over **synthetic** location index |
| SINR range | `sinr_model.py`, `sinr_range.py` | Species-at-location **constraint** (exclusion / prior centroid) |
| Species ID | `species_id.py`, `birdnet_species.py` | BirdNET (verified); Perch optional |
| CLAP | `clap_fallback.py`, `retrieval.py` | **Auxiliary** scene + text-geo; must not dominate fusion |
| Fusion | `fusion.py` | Weighted spherical mean + heuristic confidence ladder |
| Metadata | `metadata_geocode.py` | Opt-in `--use-metadata` only |

---

## 8. Data flow (one clip)

1. Preprocess — mono mixdown, resample per model.
2. AG-CLIP encode — geo embedding + species checklist.
3. Species ID — BirdNET (+ Perch).
4. Range signals — SINR prior centroid (+ eBird if keyed, else GBIF when needed); **reweight / exclude**, not pin city.
5. Scene — CLAP urban/forest scores (auxiliary).
6. Retrieval — AG-CLIP two-pass over GeoCLIP index; CLAP secondary (low weight).
7. Fusion — JSON output with granularity fallback.

---

## 9. Required models & artifacts

| Item | Path | Provenance |
|------|------|------------|
| AG-CLIP | `models/ag_clip/models/ag_clip.pt` | Repo-local [Drive](https://drive.google.com/drive/folders/1USbpyxMxSXtNf6e_aKT3qO7FeqU7xeJo) |
| SINR | `models/ag_clip/models/sinr_geo_model.pt` | Same Drive folder |
| GeoCLIP gallery | `models/faiss/agclip_gallery_*` | Built locally (`build_agclip_gallery.py`) |
| Taxonomy cache | `cache/inatsounds_categories.json` | HF metadata; optional for BirdNET→SINR names |
| CLAP gallery | `models/faiss/gallery_*` | Built locally; auxiliary |

BirdNET, GeoCLIP (pip), and CLAP (HF) are externally installable. Extra `class_res*.pt` files in `models/ag_clip/models/` are not loaded by the current pipeline.

---

## 10. Setup & run

Run from **repo root** (`geolocation/`), not `models/`.

```bash
cd /d/Antigravity/DopplerLab/geolocation

export HF_HOME="$(pwd)/models/clap"
export TRANSFORMERS_CACHE="$(pwd)/models/clap"

pip install -r requirements.txt
python scripts/bootstrap.py          # if galleries missing
python scripts/run_pipeline.py --data data --out outputs
```

```powershell
cd D:\Antigravity\DopplerLab\geolocation
..\..\venv\Scripts\Activate.ps1
$env:HF_HOME = "D:\Antigravity\DopplerLab\geolocation\models\clap"
$env:TRANSFORMERS_CACHE = $env:HF_HOME
python scripts\run_pipeline.py --data data --out outputs
```

**Verify:**

```bash
python -c "import json; d=json.load(open('outputs/predictions.json')); print(len(d), 'clips', d[0]['input_mode'])"
```

Check `outputs/run_log.txt` for `ag_clip_encode_ok`, `ag_clip_two_pass_retrieval`. `eBird API: False` is normal without a key.

### Optional

| Item | How |
|------|-----|
| eBird range prior | `export EBIRD_API_KEY=...` |
| Metadata geocoding | `--use-metadata` + spaCy `en_core_web_lg` |

---

## 11. Current status (this machine)

| Item | Status |
|------|--------|
| `ag_clip.pt` + `sinr_geo_model.pt` | Present (repo-local) |
| GeoCLIP gallery (7,035 synthetic points) | Built |
| Taxonomy cache (5,255 classes) | Present |
| Pipeline run on 3 challenge clips | Complete |
| eBird API | Not configured (optional) |

---

## 12. Known constraints

- **Synthetic gallery** — not geo-tagged audio prototypes; limits semantic grounding of retrieval.
- **AG-CLIP checkpoints** — operationally verified in this repo; external public release less solid than BirdNET/GeoCLIP.
- **Species branch** — exclusion/reweighting prior, not a precise locator (especially not city-level in NL).
- **CLAP** — auxiliary; should not dominate geo inference.
- **Confidence** — heuristic ladder, not calibrated accuracy; do not treat as probability of being correct.
- **No eval harness** — inference only until labeled ground truth and ablations exist.

---

## 13. Extension points

| Extension | Impact |
|-----------|--------|
| iNatSounds GPS gallery (real audio prototypes) | Would strengthen primary retrieval claim |
| Document upstream AG-CLIP release URL | Improves reproducibility story |
| SINR as candidate reweight only (not fused centroid peer) | Honest species prior integration |
| Demote CLAP weight in fusion | Reduces scene-text contamination |
| `scripts/eval_predictions.py` + labeled manifest | Enables measured claims |
