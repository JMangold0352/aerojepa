"""Frozen-model rollout extraction for the prober.

Loads the frozen AeroJEPA (encoder + predictor) from a checkpoint and produces
per-frame latent rollouts for the future frames of each clip. These latents are
the prober's input.

Key design: the frozen predictor predicts target-frame latents one frame at a
time given the context frames (see ``src/aerojepa/eval/rollout.py``). We collect
one pooled latent vector per predicted future frame into a (B, T_pred, D)
sequence that the prober consumes alongside the corresponding actions.

The predictor mode (regular vs looped) is controlled by ``max_loops``: passing
``max_loops=1`` makes a looped checkpoint behave as single-pass; the
``LoopedVideoPredictor.forward(..., max_loops=...)`` signature lets us switch
without reloading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from aerojepa.eval import load_model
from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor

from aerojepa_research.prober.integrator import MetricState


@dataclass
class FrozenRollout:
    """One batch of frozen-model rollout outputs for the prober.

    Attributes
    ----------
    latents : (B, T_pred, encoder_dim)
        Pooled per-frame latent rollout for the predicted future frames.
    actions : (B, T_pred, 6)
        AeroJEPA-convention actions aligned to each predicted frame.
    init_state : MetricState
        The metric state at the last context frame -- the integrator's start.
    gt_states : (B, T_pred, 12)
        Ground-truth metric states for the predicted frames (supervision target).
    """

    latents: torch.Tensor
    actions: torch.Tensor
    init_state: MetricState
    gt_states: torch.Tensor


class FrozenRolloutExtractor:
    """Wraps a frozen AeroJEPA and extracts prober inputs from clips.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to a ``latest.pt`` checkpoint (stores its own config).
    device : torch.device
        Where to run the frozen model.
    context_frames : int | None
        Number of leading frames used as context. Defaults to half the clip.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
        context_frames: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model: AeroJEPA
        self.model, self.cfg = load_model(checkpoint_path, self.device)
        # Freeze everything.
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.num_temporal = self.model.encoder.num_temporal
        self.num_spatial = self.model.encoder.num_spatial
        self.context_frames = context_frames or max(1, self.num_temporal // 2)
        self.encoder_dim = self.cfg["encoder"]["embed_dim"]

        # Precompute the per-frame token index ranges.
        # Frame f occupies tokens [f*num_spatial, (f+1)*num_spatial).
        self._frame_token_slices = [
            torch.arange(f * self.num_spatial, (f + 1) * self.num_spatial, device=self.device)
            for f in range(self.num_temporal)
        ]

    @property
    def num_pred_frames(self) -> int:
        """How many future frames the rollout predicts."""
        return self.num_temporal - self.context_frames

    def _context_indices(self, batch_size: int) -> torch.Tensor:
        ctx = torch.cat([self._frame_token_slices[f] for f in range(self.context_frames)])
        return ctx.unsqueeze(0).expand(batch_size, -1)

    def _target_indices(self, frame: int) -> torch.Tensor:
        return self._frame_token_slices[frame].unsqueeze(0)

    def is_looped(self) -> bool:
        return isinstance(self.model.predictor, LoopedVideoPredictor)

    @torch.no_grad()
    def extract(
        self,
        clips: torch.Tensor,
        actions: torch.Tensor,
        metric_states: torch.Tensor,
        max_loops: int | None = None,
    ) -> FrozenRollout:
        """Run the frozen model to get per-frame latent rollouts for one batch.

        Parameters
        ----------
        clips : (B, T, C, H, W)
            Full video clips (context + future). T must equal num_temporal.
        actions : (B, T, 6)
            AeroJEPA-convention actions for every frame.
        metric_states : (B, T, 12)
            Ground-truth metric states [pos, vel, euler_att_deg, ang_vel] per frame.
        max_loops : int | None
            Override the looped predictor's recurrence depth. ``1`` = regular
            single-pass (even on a looped checkpoint). ``None`` = use the
            checkpoint's default. Ignored for non-looped checkpoints.

        Returns
        -------
        FrozenRollout
        """
        B, T = clips.shape[0], clips.shape[1]
        if T != self.num_temporal:
            raise ValueError(f"clip T={T} != encoder num_temporal={self.num_temporal}")
        clips = clips.to(self.device)
        actions = actions.to(self.device)
        metric_states = metric_states.to(self.device)

        ctx_indices = self._context_indices(B)
        # Encode context once.
        context_repr = self.model.encoder(clips, ctx_indices)

        pred_frame_indices = list(range(self.context_frames, self.num_temporal))
        per_frame_latents: list[torch.Tensor] = []
        for f in pred_frame_indices:
            tgt_indices = self._target_indices(f).expand(B, -1)
            if self.is_looped():
                pred_out = self.model.predictor(
                    context_repr, ctx_indices, tgt_indices,
                    actions=None, max_loops=max_loops,
                )
                if isinstance(pred_out, tuple):
                    pred_repr = pred_out[0]
                else:
                    pred_repr = pred_out
            else:
                pred_repr = self.model.predictor(context_repr, ctx_indices, tgt_indices, actions=None)
            # pred_repr: (B, num_spatial, encoder_dim). Mean-pool spatial tokens
            # to one vector per predicted frame.
            per_frame_latents.append(pred_repr.mean(dim=1))

        latents = torch.stack(per_frame_latents, dim=1)  # (B, T_pred, D)

        # Actions + GT states aligned to the predicted frames.
        pred_actions = actions[:, self.context_frames:self.num_temporal]
        pred_gt_states = metric_states[:, self.context_frames:self.num_temporal]
        # The integrator starts from the LAST context frame's state.
        init_state = MetricState(
            pos=metric_states[:, self.context_frames - 1, 0:3].clone(),
            vel=metric_states[:, self.context_frames - 1, 3:6].clone(),
            euler_att=metric_states[:, self.context_frames - 1, 6:9].clone(),
            ang_vel=metric_states[:, self.context_frames - 1, 9:12].clone(),
        )
        return FrozenRollout(
            latents=latents,
            actions=pred_actions,
            init_state=init_state,
            gt_states=pred_gt_states,
        )
