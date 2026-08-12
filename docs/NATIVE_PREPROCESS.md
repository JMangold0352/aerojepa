# Optional native preprocess (Rust)

Small optional accelerator for video frame-index selection during
`scripts/preprocess_real.py`. Training, models, and the prober never depend on it.

**Status:** Phase A only (`select_indices` + Python dispatch + parity tests).
Further decode/encode work is **deferred** — OpenCV remains the default path.

## Install (optional)

```bash
# Requires a Rust toolchain: https://rustup.rs
pip install -e ".[native]"
cd native/aerojepa-preprocess && maturin develop --release
```

```bash
python scripts/preprocess_real.py ... --backend auto   # rust if present, else opencv
python scripts/preprocess_real.py ... --backend opencv  # force OpenCV
```

## What exists

| Piece | Role |
| --- | --- |
| `native/aerojepa-preprocess/` | PyO3 crate (`select_indices`) |
| `src/aerojepa/data/preprocess_backend.py` | Dispatch + OpenCV fallback |
| `tests/test_preprocess_backend.py` | Backend selection / parity |

If the Rust module is missing or raises, preprocess falls back to OpenCV and
continues. Do not make training fail on a native import error.

## Out of scope (for now)

Full Rust decode → resize → write, directory-level Rayon parallelism, and CI
wheel builds are not scheduled. Revisit only if ingest wall-time becomes a
real bottleneck on large flight folders.
