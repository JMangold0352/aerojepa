# AeroJEPA + AeroProber — demo brief

**Author:** John Mangold  
**Date:** July 2026  
**Purpose:** Source notes for a short overview talk or NotebookLM session

---

## Pitch

**AeroJEPA** is a compact video world model for drones. It predicts future
*latents* (not pixels) from short egocentric clips, with optional 6-DoF action
conditioning. Models are ~3–5M parameters. Synthetic pretrain transfers to
Wilds Parrot footage; a PyFlyt closed-loop planner sits on top.

**AeroProber** is a small physics-structured head (~5k params) on a frozen
AeroJEPA. It maps latent rollouts + controls to metric state through a
differentiable control integrator.

---

## AeroJEPA numbers (protocol B)

| Metric | Value |
| --- | ---: |
| Real latent cosine (`real_finetune_fast`) | **0.974** |
| Synthetic latent cosine | **0.994** |
| Sim-to-real gap | **+0.019** |

Checkpoint: `checkpoints/real_finetune_fast/latest.pt`  
Protocol: `docs/EVAL_PROTOCOL.md`

Closed-loop default stack:
`action_conditioned_wilds` + `action_residual_wilds` (gradient planner,
`latent_smooth=0.05`).

---

## AeroProber (physics)

Freeze AeroJEPA. Train prober + `ControlIntegrator` on raw PyFlyt controls
`(vp, vq, vr, T)` so metric decoding does not leak GT velocity/attitude through
state-derived actions. Details and open questions:
[`research/prober/note.md`](../research/prober/note.md).

Leak-free v3 (5 seeds): structured pos **0.006 m**, att **2.28°** vs plain MLP
0.152 m / 2.84°. Looped vs regular: tie.

Wilds zero-control eval still has a large gap — that is the main open problem
for the physics collaboration, not a footnote.

---

## Closed-loop demos

```bash
python scripts/run_closed_loop_demo.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual-checkpoint checkpoints/action_residual_wilds/best.pt \
  --planner gradient --latent-smooth 0.05 \
  --task hover

# waypoint / recover: change --task
python scripts/stitch_closed_loop_demo.py --include-random
```

| Asset | Path |
| --- | --- |
| Demo reel | `docs/gallery/closed_loop_demo_reel.gif` |
| GIFs / metrics | `visualizations/closed_loop/` |

This is a research demo in PyFlyt, not a flight controller.

---

## Suggested talk outline (~8–10 min)

| Act | Content |
| --- | --- |
| 1 | What AeroJEPA predicts (latents, not pixels) |
| 2 | Sim → Wilds numbers + eval protocol |
| 3 | AeroProber + leak-free design (brief) |
| 4 | Closed-loop reel: hover → waypoint → recover |
| 5 | Open questions for physics collab; Tello when footage exists |

---

## Stack

```
egocentric video
      │
  AeroJEPA (frozen world model)
      │
      ├─ LatentPlanner + residual ──► PyFlyt closed loop
      │
      └─ AeroProber + ControlIntegrator ──► metric state
```
