# Real UAV footage

Train AeroJEPA on real drone video with the same `(frames, actions)` interface as
the synthetic generator — point a config at a folder of clips.

## Folder layout

```
data/flights/
├── flight_001.mp4
├── flight_001.csv      # optional: 6-DoF pose delta per frame
├── flight_002.mp4
├── flight_002.csv
└── ...
```

- Videos: `.mp4`, `.mov`, `.avi`, or `.mkv`. Any resolution/length — frames are
  resized to `data.img_size` (**64** in the current recipes) and sampled to
  `data.num_frames`. Preprocess often writes **128×128** files under
  `data/flights_128/` for storage; that is not the model input size. See
  [`docs/EVAL_PROTOCOL.md`](../docs/EVAL_PROTOCOL.md).
- Telemetry: a sibling `.csv` with the **same base name** as the video. Optional.
  If missing, actions default to zeros (the masked objective still trains; the
  action-conditioned world model needs telemetry to show a benefit).

## Telemetry format

One header row, then **one row per video frame**, columns in this exact order
(see [`telemetry.py`](../src/aerojepa/data/telemetry.py), `ACTION_COLUMNS`):

```csv
dx,dy,d_altitude,d_yaw,d_pitch,d_roll
0.01,-0.02,0.00,0.03,0.00,0.01
0.01,-0.01,0.01,0.02,0.00,0.00
...
```

Each row is the **motion that produced that frame** (pose delta from the previous
frame). Units can be mixed (metres, radians) — `normalize_actions` rescales them.
Extra trailing columns are ignored, and short/long logs are aligned to the
sampled frames automatically, so imperfect logging still works.

## Sampling modes

Set `data.window_mode` in the config:

- **`uniform`** — one clip per video, `num_frames` spread across the whole file.
  Best for a folder of many short, similar clips.
- **`sliding`** — many fixed-length windows per video (`window_stride` controls
  overlap). Best for a few long flights; multiplies a small corpus into many
  training samples. Clips shorter than `num_frames` are padded by repeating the
  last frame when `pad_short: true`.

Validation is split at the **video level** (`val_fraction`), so windows from one
flight never leak across train and val.

## Tello personal footage (one command)

After the Wilds fine-tune (`checkpoints/real_finetune_fast/latest.pt`):

```bash
chmod +x scripts/tello_workflow.sh

# Full pipeline: preflight + 4 maneuver clips + preprocess + train + report
./scripts/tello_workflow.sh all --duration 30

# Or step by step:
./scripts/tello_workflow.sh preflight
./scripts/tello_workflow.sh sessions --duration 30   # hover, forward, turn, altitude
./scripts/tello_workflow.sh preprocess               # -> data/flights_tello_128/
./scripts/tello_workflow.sh train                      # from real_finetune_fast
./scripts/tello_workflow.sh report                     # vs Wilds-only
```

Output folders:

```
data/raw/tello/           archived captures (dated)
data/flights_tello/       working .mp4 + .csv + .raw.csv
data/flights_tello_128/   128px training clips
checkpoints/real_finetune_tello/
results/tello_transfer_report.md
```

## Capture from DJI Tello

```bash
pip install djitellopy opencv-python

# 1) Health check (connect, print battery, disconnect — no recording, no flight):
python scripts/capture_tello.py --preflight

# 2) Record a 30s clip at 15 fps while a pilot flies manually:
python scripts/capture_tello.py --duration 30 --fps 15 --name-prefix hover --tags indoor
```

Each session writes timestamped files:

```
data/flights/hover_20260703_161500_indoor.mp4      # video
data/flights/hover_20260703_161500_indoor.csv      # training actions (ACTION_COLUMNS)
data/flights/hover_20260703_161500_indoor.raw.csv  # full flight log (provenance)
```

The `.raw.csv` records `t, vgx, vgy, vgz, height, yaw, pitch, roll, bat, tof, baro`;
the training `.csv` is derived from it (linear velocities + wrapped angular deltas).
See [`tello.py`](../src/aerojepa/data/tello.py).

### ⚠️ Hardware safety

- **This tool never flies the drone.** It records only; a human pilot flies
  manually. Capture and control are deliberately separate so a software bug here
  cannot move the aircraft.
- Fly in a **large, open area**; keep propellers clear of people, pets, and hands.
- Respect `--min-battery` (default 20%). A low battery risks an uncontrolled landing.
- Indoors: good lighting, minimal wind, watch downwash near walls and ceilings.
- Press **Ctrl-C** anytime — the video stream closes cleanly on exit.
- You are responsible for **airspace and regulatory compliance** in your area.

## Preprocess any footage → training format

Standardize Tello captures, phone video, or downloaded drone clips to a
consistent codec/frame rate (telemetry stays aligned):

```bash
# Inspect what you have first (dataset doctor — no writes):
python scripts/preprocess_real.py --probe --input-dir data/raw

# Standardize into data/flights/ at 15 fps, center-cropped square:
python scripts/preprocess_real.py --input-dir data/raw \
    --output-dir data/flights --target-fps 15 --square

# Trim long clips and downscale to shrink files:
python scripts/preprocess_real.py --input-dir data/raw --max-seconds 20 --square --resize 128
```

Frames are resized again at load time to `data.img_size`, so `--resize` is only
about on-disk size. `--square` avoids aspect-ratio distortion for wide footage.

## Train on real data

```bash
# From scratch on real clips:
python scripts/train.py --config configs/aerojepa_real.yaml

# Recommended: fine-tune the synthetic world model (far more sample efficient):
python scripts/train.py --config configs/aerojepa_finetune.yaml
#   equivalently:
python scripts/train.py --config configs/aerojepa_real.yaml \
    --init-checkpoint checkpoints/world_model/latest.pt
```

`--init-checkpoint` warm-starts encoder + EMA teacher + predictor weights and
tolerates minor architecture drift (`strict=False`), reporting anything it could
not load.

## Measure the synthetic-vs-real gap

```bash
python scripts/evaluate_real.py \
    --checkpoint checkpoints/world_model/latest.pt \
    --data-dir data/flights
```

This runs identical metrics (latent cosine, rollout) on the synthetic benchmark
and on your real clips, and reports the sim-to-real gap → `results/*_real_eval.json`.
`scripts/evaluate_all.sh` runs this automatically when `data/flights/` contains clips.

> Note: `data/` is git-ignored, so footage stays local. Commit configs and code,
> not raw flight video.
