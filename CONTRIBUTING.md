# Contributing to AeroJEPA

Thanks for your interest. This is a research codebase; the priorities are
correctness, reproducibility, and readable code that a non-specialist can follow.

## Principles

- **Config-driven.** New behavior should be reachable from a YAML config that
  inherits the base recipe via `_base_`. Avoid hidden defaults in code.
- **Honest science.** Report negative results. Do not tune the evaluation to
  flatter a method. Keep the baseline and the variant on the same recipe so
  comparisons are fair.
- **Readable voice.** Docstrings and comments explain *why* (intent, trade-offs,
  constraints), never narrate the obvious. Keep prose accessible; a smart reader
  who is not a deep-learning specialist should be able to follow.
- **No emojis** in code, comments, or docs.

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

## Good first extensions

The [Cursor kickoff prompts](docs/CURSOR_KICKOFF_PROMPTS.md) list concrete,
scoped tasks (real-data ingestion, public-dataset loaders, attention studies,
closing the planning loop) aligned with the [roadmap](docs/ROADMAP.md).
