from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from aerojepa.eval import load_model
from aerojepa.data.synthetic import render_clip
from aerojepa.masking import build_mask_collator
from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor, expected_loops_from_exit_probs
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device


class DemoModel:
    """Thin wrapper that powers the Gradio demo.

    Loads a trained checkpoint when one is given, otherwise builds a small
    *untrained* model from the smoke config so the whole interface still runs
    end-to-end. The UI clearly flags when it is showing an untrained model.
    """

    def __init__(self, checkpoint: str | None = None, device: str = "auto") -> None:
        self.device = get_device(device)
        if checkpoint and Path(checkpoint).exists():
            self.model, self.cfg = load_model(checkpoint, self.device)
            self.trained = True
        else:
            self.cfg = load_config("configs/smoke_test.yaml")
            self.model = AeroJEPA.from_config(self.cfg).to(self.device).eval()
            self.trained = False

    @property
    def num_temporal(self) -> int:
        return self.model.encoder.num_temporal

    def sample_clip(self, seed: int) -> torch.Tensor:
        clip = render_clip(
            seed=seed,
            num_frames=self.cfg["data"]["num_frames"],
            img_size=self.cfg["data"]["img_size"],
        )
        return clip.frames

    @torch.no_grad()
    def future_quality(self, clip: torch.Tensor, context_frames: int, max_loops: int) -> dict:
        """Per-future-frame cosine between predicted and true latents."""
        if isinstance(self.model.predictor, LoopedVideoPredictor):
            self.model.predictor.max_loops = max_loops

        num_spatial = self.model.encoder.num_spatial
        num_temporal = self.num_temporal
        context_frames = max(1, min(context_frames, num_temporal - 1))

        clip_b = clip.unsqueeze(0).to(self.device)
        horizons, cosines = [], []
        for target_frame in range(context_frames, num_temporal):
            ctx = torch.arange(0, context_frames * num_spatial, device=self.device).unsqueeze(0)
            start = target_frame * num_spatial
            tgt = torch.arange(start, start + num_spatial, device=self.device).unsqueeze(0)
            out = self.model(clip_b, ctx, tgt)
            cos = F.cosine_similarity(out["pred_repr"], out["target_repr"], dim=-1).mean()
            horizons.append(target_frame - context_frames + 1)
            cosines.append(float(cos.item()))
        return {"horizon": horizons, "cosine": cosines, "context_frames": context_frames}

    @torch.no_grad()
    def loop_stats(self, clip: torch.Tensor, max_loops: int) -> dict | None:
        """Per-loop cosine and expected exit depth for one clip (looped only)."""
        if not isinstance(self.model.predictor, LoopedVideoPredictor):
            return None

        grid = self.cfg["data"]["img_size"] // self.cfg["data"]["patch_size"]
        collator = build_mask_collator(self.cfg, grid, self.num_temporal)
        masks = collator(1)
        ctx = masks.context_indices[0].unsqueeze(0).to(self.device)
        tgt = masks.target_indices[0].unsqueeze(0).to(self.device)

        context_repr = self.model.encoder(clip.unsqueeze(0).to(self.device), ctx)
        all_tokens = self.model.target_encoder.forward_all_patches(clip.unsqueeze(0).to(self.device))
        idx = tgt.unsqueeze(-1).expand(-1, -1, all_tokens.size(-1))
        target_repr = torch.gather(all_tokens, 1, idx)

        per_loop = []
        for k in range(1, max_loops + 1):
            out = self.model.predictor(context_repr, ctx, tgt, None, max_loops=k)
            pred = out[0] if isinstance(out, tuple) else out
            per_loop.append(float(F.cosine_similarity(pred, target_repr, dim=-1).mean().item()))

        full = self.model.predictor(context_repr, ctx, tgt, None, max_loops=max_loops)
        expected = None
        if isinstance(full, tuple):
            expected = float(expected_loops_from_exit_probs(full[1]).mean().item())
        return {"per_loop_cosine": per_loop, "expected_loops": expected}
