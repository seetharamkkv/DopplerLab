# Manual direction test (monoaural clips)

Drop clips in [`data/`](data/) and run the **mel** direction model (mono log-mel, not stereo cross-correlation).

## Setup

```bash
cd IDMT_experiments
pip install -r requirements.txt
```

You need the trained weights locally (not in git):

`checkpoints/cnn/direction/mel_3class/best.pt`

For **video** inputs, `imageio-ffmpeg` (in `requirements.txt`) provides ffmpeg for audio extraction.

## Input types

| `--input-type` | Default `--ext` | What happens |
|----------------|-----------------|--------------|
| **`audio`** (default) | `wav` | Read audio files directly with librosa |
| **`video`** | `avi` | Extract the audio track with ffmpeg (full clip, no trim) |

Other extensions via `--ext`, for example `flac` (audio) or `mp4` (video).

## Add clips

**Audio (default):** `.wav` in `test/data/`

```
IDMT_experiments/test/data/
  my_clip_01.wav
```

**Video:** `.avi` (or `.mp4`, etc.) in `test/data/`

```
IDMT_experiments/test/data/
  my_clip_01.avi
```

**Clip length:** the full recording is used (no trimming). The mel CNN uses adaptive pooling, so variable lengths are fine.

## Run predictions

**WAV audio (default):**

```bash
python -m idmt_experiments.test_clips
```

**AVI video (extract audio, then predict):**

```bash
python -m idmt_experiments.test_clips --input-type video
```

**MP4 video:**

```bash
python -m idmt_experiments.test_clips --input-type video --ext mp4
```

### All options

| Option | Default | Meaning |
|--------|---------|---------|
| `--input-type` | `audio` | `audio` or `video` |
| `--ext` | `wav` / `avi` | Extension for the chosen input type |
| `--data-dir` | `test/data/` | Input folder |
| `--checkpoint` | `mel_3class/best.pt` | Mel (mono) direction model |
| `--output` | `test/outputs/predictions.csv` | Results table (+ summary row) |
| `--spectrogram-dir` | `test/outputs/` | Constant-Q PNG folder |
| `--no-spectrograms` | off | Skip PNG export |
| `--recursive` | off | Include subfolders |

Legacy: `--ext avi` alone still switches to video mode (same as `--input-type video --ext avi`).

## Outputs

**`test/outputs/predictions.csv`** (default)

Per-clip rows plus a final **`__OVERALL__`** row with accuracy. Also writes **`test/outputs/predictions_summary.json`**.

Place **`test/data/metadata.txt`** to score against manual labels (format: `clip.avi;count;left|right` where `left`→`L2R`, `right`→`R2L`).

Re-score an existing CSV without re-running inference:

```bash
python -m idmt_experiments.test_clips --score-only --input-type video
```

**Model used:** `checkpoints/cnn/direction/mel_3class/best.pt` — mel (mono log-mel) 3-class direction CNN trained on IDMT-Traffic (EUSIPCO split). Not the stereo CC model.

| Column | Meaning |
|--------|---------|
| `clip_name` | Filename |
| `clip_path` | Path relative to `data/` |
| `input_type` | `audio` or `video` |
| `label_raw` | `left` / `right` from metadata.txt |
| `label_true` | `L2R` / `R2L` mapped label |
| `correct` | `1` if prediction matches `label_true` |
| `model` | Checkpoint run name (`mel_3class`) |
| `checkpoint` | Path to `.pt` weights used |
| `duration_s` | Full decoded length (seconds) |
| `n_mel_frames` | Mel time bins used by the model |
| `prediction` | `L2R`, `R2L`, or `no_vehicle` |
| `prob_*` | Softmax scores |
| `spectrogram_png` | Path to CQT PNG under `test/` |

**`test/outputs/<clip_stem>.png`**

Constant-Q spectrogram of the **extracted / loaded** audio (same basename as the source file).

## Single file (optional)

```bash
python -m idmt_experiments.infer \
  --checkpoint checkpoints/cnn/direction/mel_3class/best.pt \
  --wav test/data/my_clip_01.wav
```
