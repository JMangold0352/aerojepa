# aerojepa-preprocess

Optional Rust helper for AeroJEPA video ingest. **Not required** for training
or research - without it, Python uses OpenCV.

Details: [`docs/NATIVE_PREPROCESS.md`](../../docs/NATIVE_PREPROCESS.md).

## Build

```bash
# Needs rustup + maturin
pip install maturin
cd native/aerojepa-preprocess
maturin develop --release
```

## What is implemented

`select_indices` only (frame sampling indices). Wired through
`aerojepa.data.preprocess_backend` with OpenCV fallback.

```bash
python scripts/preprocess_real.py --input-dir data/raw --backend auto
```

`standardize_video` / `probe_video` are stubs. Full decode/encode in Rust is
deferred.
