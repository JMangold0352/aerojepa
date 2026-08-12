from __future__ import annotations

import torch
import torch.nn as nn

from aerojepa.models.predictor import VideoPredictor


def exit_entropy_loss(exit_probs: torch.Tensor) -> torch.Tensor:
    """Penalize exit gates that collapse to always-stop or always-continue.

    Without this pressure the gate finds a degenerate shortcut (exit on step 1
    every time, or never), which throws away the whole point of adaptive depth.
    Maximizing the per-step Bernoulli entropy keeps the gate genuinely
    input-dependent.
    """
    p = exit_probs.clamp(min=1e-6, max=1.0 - 1e-6)
    entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
    return -entropy


def expected_loops_from_exit_probs(exit_probs: torch.Tensor) -> torch.Tensor:
    """Expected recurrence depth implied by per-step exit probabilities.

    ``exit_probs`` is ``(batch, max_loops)``. This is the average number of
    refinement steps the model would spend per sample -- our headline
    "how much compute did adaptivity actually save?" number.
    """
    batch_size, max_loops = exit_probs.shape
    survive = torch.ones(batch_size, device=exit_probs.device, dtype=exit_probs.dtype)
    expected = torch.zeros(batch_size, device=exit_probs.device, dtype=exit_probs.dtype)

    for loop_idx in range(max_loops):
        p_exit = exit_probs[:, loop_idx]
        loop_number = float(loop_idx + 1)
        expected += loop_number * p_exit * survive
        survive = survive * (1.0 - p_exit)

    expected += float(max_loops) * survive
    return expected


class LoopedVideoPredictor(nn.Module):
    """Weight-shared recurrent predictor for AeroJEPA.

    Re-applies the same ``VideoPredictor`` block stack for up to ``max_loops``
    steps, refining its guess about the target latents each time. Because the
    loop reuses one set of weights, this added depth costs *zero* extra
    parameters -- only compute grows. An optional per-loop exit gate makes that
    depth adaptive, so easy clips (say, a drone hovering over flat ground) stop
    early while hard clips (fast motion near obstacles) keep refining.

    Returns predicted target latents, or ``(pred, exit_probs)`` when the gate is
    on.
    """

    def __init__(
        self,
        base_predictor: VideoPredictor,
        max_loops: int = 4,
        use_exit_gate: bool = False,
    ) -> None:
        super().__init__()
        self.base_predictor = base_predictor
        self.max_loops = max_loops
        self.use_exit_gate = use_exit_gate
        hidden = base_predictor.predictor_dim
        self.exit_gate = nn.Linear(hidden, 1) if use_exit_gate else None

    def forward(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        actions: torch.Tensor | None = None,
        max_loops: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        loops = max_loops or self.max_loops
        x = self.base_predictor.build_sequence(
            context_repr, context_indices, target_indices, actions
        )
        n_ctx = context_indices.shape[1]

        exit_probs: list[torch.Tensor] = []
        for _ in range(loops):
            x = self.base_predictor.forward_stack(x)
            if self.use_exit_gate and self.exit_gate is not None:
                gate_in = x.mean(dim=1)
                exit_probs.append(torch.sigmoid(self.exit_gate(gate_in)).squeeze(-1))

        target_tokens = x[:, n_ctx:]
        target_tokens = self.base_predictor.output_proj(target_tokens)
        if self.use_exit_gate and exit_probs:
            return target_tokens, torch.stack(exit_probs, dim=1)
        return target_tokens

    def compute_total_loss(
        self,
        jepa_loss: torch.Tensor,
        exit_probs: torch.Tensor | None,
        beta: float = 0.01,
    ) -> torch.Tensor:
        if exit_probs is None or beta <= 0:
            return jepa_loss
        return jepa_loss + beta * exit_entropy_loss(exit_probs)
