# Model cards

Per-variant notes: what the model is, how it was trained, and what it is for.

| Card | Summary |
| --- | --- |
| [**aerojepa_base**](aerojepa_base.md) | Video encoder + feed-forward / looped predictor, masked objective. |
| [**aerojepa_world_model**](aerojepa_world_model.md) | Future-frame objective, recurrent predictor, optional 6-DoF action conditioning. |
| [**aerojepa_real_finetune**](aerojepa_real_finetune.md) | Unconditioned Wilds fine-tune - best representation numbers. |
| [**aerojepa_action_wilds**](aerojepa_action_wilds.md) | Action-conditioned Wilds + residual - default closed-loop stack. |

Numbers should match regenerable evals under `results/` / `visualizations/closed_loop/`
using [`docs/EVAL_PROTOCOL.md`](../docs/EVAL_PROTOCOL.md) and `REPORT.md`.
