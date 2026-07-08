from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor, expected_loops_from_exit_probs
from aerojepa.train import _prep_actions, stack_indices


@torch.no_grad()
def loop_metrics(
    model: AeroJEPA,
    loader: DataLoader,
    collator,
    device: torch.device,
    cfg: dict[str, Any],
    max_batches: int = 8,
) -> dict[str, Any]:
    """Interpretability metrics for the recurrent predictor.

    - ``per_loop_cosine``: prediction quality after 1, 2, ... refinement steps.
      A rising curve is direct evidence that recurrence *is* refining the guess.
    - ``exit_distribution``: how often the learned gate stops at each depth.
    - ``expected_loops``: average compute actually spent per clip.
    """
    if not isinstance(model.predictor, LoopedVideoPredictor):
        raise ValueError("loop_metrics requires a looped predictor.")

    model.eval()
    looped = model.predictor
    max_loops = looped.max_loops
    num_temporal = model.encoder.num_temporal
    use_actions = bool(cfg["predictor"].get("action_conditioning", False))

    per_loop_cos = [0.0 for _ in range(max_loops)]
    exit_hist = [0 for _ in range(max_loops)]
    expected_sum, total_samples, n_batches = 0.0, 0, 0

    for i, (clips, actions) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        b = clips.shape[0]
        masks = collator(b)
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        acts = _prep_actions(actions, num_temporal, device, cfg) if use_actions else None

        context_repr = model.encoder(clips, ctx)
        all_tokens = model.target_encoder.forward_all_patches(clips)
        idx = tgt.unsqueeze(-1).expand(-1, -1, all_tokens.size(-1))
        target_repr = torch.gather(all_tokens, 1, idx)

        for k in range(1, max_loops + 1):
            out = looped(context_repr, ctx, tgt, acts, max_loops=k)
            pred = out[0] if isinstance(out, tuple) else out
            per_loop_cos[k - 1] += float(F.cosine_similarity(pred, target_repr, dim=-1).mean().item())

        full = looped(context_repr, ctx, tgt, acts, max_loops=max_loops)
        if isinstance(full, tuple):
            exit_probs = full[1]  # (B, max_loops)
            expected_sum += float(expected_loops_from_exit_probs(exit_probs).sum().item())
            decided = exit_probs > 0.5
            for row in decided:
                nz = torch.nonzero(row)
                exit_idx = int(nz.min().item()) if nz.numel() else max_loops - 1
                exit_hist[exit_idx] += 1
            total_samples += b
        n_batches += 1

    n_batches = max(1, n_batches)
    per_loop_cosine = [c / n_batches for c in per_loop_cos]
    if total_samples:
        exit_distribution = [h / total_samples for h in exit_hist]
        expected_loops = expected_sum / total_samples
    else:
        exit_distribution = []
        expected_loops = float(max_loops)

    return {
        "per_loop_cosine": per_loop_cosine,
        "exit_distribution": exit_distribution,
        "expected_loops": expected_loops,
        "max_loops": max_loops,
    }
