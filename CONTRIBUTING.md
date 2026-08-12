# Contributing to AeroJEPA

Thanks for your interest. This is a research codebase; the priorities are
correctness, reproducibility, and readable code that a non-specialist can follow.

## Principles

- Prefer config-driven behavior (YAML with `_base_` inheritance) over hidden defaults.
- Keep comparisons fair: same recipe for baseline and variant.
- Comments and docs should explain intent and trade-offs, not narrate the code.
- No emojis in code, comments, or docs.

## Code style

- `from __future__ import annotations` at the top of every module; full type
  hints.
- Small, composable functions. The predictor is deliberately a reusable
  `BlockStack` so the looped variant can reuse it -- preserve that pattern.
- Keep optional dependencies (OpenCV, Gradio, scikit-learn, PyFlyt, djitellopy)
  behind lazy imports with clear error messages.

## Project layout

See the repository layout in the [README](README.md#repository-layout) and the
[technical report](docs/TECHNICAL_REPORT.md) for how the pieces fit together.

## Before opening a PR

```bash
python scripts/verify_install.py
pytest -q
```

- Add or update a test when you change model shapes, data, or training behavior.
- If you change results, update `REPORT.md` (and the README/model-card tables)
  with numbers you can regenerate.
- Keep commits focused and messages descriptive.

## Ideas for extensions

See `README.md` and `REPORT.md` for current limitations. Useful directions when
you pick them up: Tello fine-tune (needs footage), physics questions in
`research/prober/note.md`, and tighter closed-loop control in sim.
