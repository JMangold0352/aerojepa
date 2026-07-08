from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# Two masking strategies, one interface. Both decide which space-time tokens the
# encoder gets to see (context) and which the predictor must reconstruct
# (targets). Swapping the collator is what turns AeroJEPA from a representation
# learner into a forward world model -- the network architecture never changes.


@dataclass
class MaskBatch:
    context_indices: list[torch.Tensor]
    target_indices: list[torch.Tensor]


class SpatioTemporalMaskCollator:
    """Masked-prediction objective (V-JEPA style).

    Targets are a handful of space-time blocks scattered across random frames;
    the context is whatever is left. This teaches the encoder *what the scene
    is* by forcing it to infer hidden regions from visible ones -- strong,
    general-purpose representation learning.

    Token index convention: ``token = frame * num_spatial + spatial_patch``.
    """

    def __init__(
        self,
        grid_size: int,
        num_temporal: int,
        num_target_blocks: int = 4,
        target_scale: tuple[float, float] = (0.15, 0.25),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        target_tokens: int = 96,
        context_tokens: int = 128,
    ) -> None:
        self.grid_size = grid_size
        self.num_spatial = grid_size * grid_size
        self.num_temporal = num_temporal
        self.num_tokens = self.num_spatial * num_temporal
        self.num_target_blocks = num_target_blocks
        self.target_scale = target_scale
        self.aspect_ratio = aspect_ratio
        self.target_tokens = target_tokens
        self.context_tokens = context_tokens

    def _sample_spatial_block(self) -> set[int]:
        g = self.grid_size
        area = random.uniform(*self.target_scale) * self.num_spatial
        aspect = random.uniform(*self.aspect_ratio)
        h = max(1, min(g, int(round((area * aspect) ** 0.5))))
        w = max(1, min(g, int(round((area / aspect) ** 0.5))))
        top = random.randint(0, g - h)
        left = random.randint(0, g - w)
        return {r * g + c for r in range(top, top + h) for c in range(left, left + w)}

    def __call__(self, batch_size: int) -> MaskBatch:
        context_indices: list[torch.Tensor] = []
        target_indices: list[torch.Tensor] = []
        all_tokens = set(range(self.num_tokens))

        for _ in range(batch_size):
            targets: set[int] = set()
            for _ in range(self.num_target_blocks):
                frame = random.randrange(self.num_temporal)
                block = self._sample_spatial_block()
                targets |= {frame * self.num_spatial + s for s in block}

            targets = self._fix_count(targets, all_tokens, self.target_tokens)
            available = all_tokens - targets
            context = self._fix_count(set(), available, self.context_tokens)

            context_indices.append(torch.tensor(sorted(context), dtype=torch.long))
            target_indices.append(torch.tensor(sorted(targets), dtype=torch.long))

        return MaskBatch(context_indices=context_indices, target_indices=target_indices)

    @staticmethod
    def _fix_count(chosen: set[int], pool: set[int], count: int) -> set[int]:
        """Grow/trim ``chosen`` to exactly ``count`` tokens drawn from ``pool``."""
        chosen = set(chosen) & pool if chosen else set()
        if len(chosen) > count:
            return set(sorted(chosen)[:count])
        remaining = sorted(pool - chosen)
        need = count - len(chosen)
        if need > 0 and remaining:
            chosen |= set(random.sample(remaining, min(need, len(remaining))))
        return chosen


class FutureFrameMaskCollator:
    """Forward world-model objective.

    Context is *every* token of the first ``num_context_frames``; targets are
    *every* token of the remaining future frames. The predictor must roll the
    scene forward in latent space -- "given what I have seen, what happens next?"
    This is the objective that most directly supports predictive planning and
    obstacle anticipation.
    """

    def __init__(
        self,
        grid_size: int,
        num_temporal: int,
        num_context_frames: int = 4,
    ) -> None:
        self.num_spatial = grid_size * grid_size
        self.num_temporal = num_temporal
        if not 0 < num_context_frames < num_temporal:
            raise ValueError(
                f"num_context_frames must be in (0, {num_temporal}); got {num_context_frames}"
            )
        self.num_context_frames = num_context_frames

        split = num_context_frames * self.num_spatial
        self._context = torch.arange(0, split, dtype=torch.long)
        self._target = torch.arange(split, num_temporal * self.num_spatial, dtype=torch.long)

    def __call__(self, batch_size: int) -> MaskBatch:
        # Deterministic split, identical for every clip in the batch.
        return MaskBatch(
            context_indices=[self._context.clone() for _ in range(batch_size)],
            target_indices=[self._target.clone() for _ in range(batch_size)],
        )


def build_mask_collator(cfg: dict, grid_size: int, num_temporal: int):
    """Select the collator implementing the configured training objective."""
    mask_cfg = cfg.get("masking", {}) or {}
    objective = cfg.get("objective", "masked")
    if objective == "future":
        return FutureFrameMaskCollator(
            grid_size=grid_size,
            num_temporal=num_temporal,
            num_context_frames=mask_cfg.get("num_context_frames", num_temporal // 2),
        )
    return SpatioTemporalMaskCollator(
        grid_size=grid_size,
        num_temporal=num_temporal,
        num_target_blocks=mask_cfg.get("num_target_blocks", 4),
        target_scale=tuple(mask_cfg.get("target_scale", [0.15, 0.25])),
        target_tokens=mask_cfg.get("target_tokens", 96),
        context_tokens=mask_cfg.get("context_tokens", 128),
    )
