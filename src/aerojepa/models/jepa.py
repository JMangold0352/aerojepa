from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from aerojepa.models.encoder import VideoTransformerEncoder
from aerojepa.models.looped_predictor import LoopedVideoPredictor
from aerojepa.models.predictor import VideoPredictor
from aerojepa.models.vit import rms_norm_factory


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, momentum: float) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(momentum).add_(p.data, alpha=1.0 - momentum)


class AeroJEPA(nn.Module):
    """Video JEPA world model: context encoder + EMA target encoder + predictor.

    The student encoder sees only a subset of space-time tokens (the context).
    The EMA "teacher" -- a slowly updated copy of the student -- encodes the full
    clip and provides stop-gradient targets. The predictor then infers the
    teacher's latents at the held-out target positions.

    Whether those targets are *scattered masked patches* (representation
    learning) or *entire future frames* (a forward world model) is decided by the
    mask collator, not the model. That single design choice keeps AeroJEPA honest
    and reusable: the same weights support both objectives.

    Only the student encoder is kept for downstream use (probing, planning).
    """

    def __init__(
        self,
        encoder: VideoTransformerEncoder,
        target_encoder: VideoTransformerEncoder,
        predictor: VideoPredictor | LoopedVideoPredictor,
        ema_momentum: float = 0.996,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.ema_momentum = ema_momentum

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> AeroJEPA:
        data_cfg = cfg["data"]
        enc_cfg = cfg["encoder"]
        pred_cfg = cfg["predictor"]

        grid_size = data_cfg["img_size"] // data_cfg["patch_size"]
        num_spatial = grid_size ** 2
        num_frames = data_cfg["num_frames"]
        tokenizer = enc_cfg.get("tokenizer", "frame")
        num_temporal = num_frames // enc_cfg.get("tubelet_size", 2) if tokenizer == "tubelet" else num_frames

        encoder = VideoTransformerEncoder(
            img_size=data_cfg["img_size"],
            patch_size=data_cfg["patch_size"],
            in_chans=data_cfg.get("in_chans", 3),
            num_frames=num_frames,
            embed_dim=enc_cfg["embed_dim"],
            depth=enc_cfg["depth"],
            num_heads=enc_cfg["num_heads"],
            mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
            dropout=enc_cfg.get("dropout", 0.0),
            drop_path=enc_cfg.get("drop_path", 0.0),
            tokenizer=tokenizer,
            tubelet_size=enc_cfg.get("tubelet_size", 2),
        )
        target_encoder = copy.deepcopy(encoder)

        action_conditioning = bool(pred_cfg.get("action_conditioning", False))
        action_dim = int(pred_cfg.get("action_dim", 6))

        if pred_cfg.get("world_model", False):
            base_predictor = VideoPredictor.world_model(
                num_temporal=num_temporal,
                num_spatial=num_spatial,
                encoder_dim=enc_cfg["embed_dim"],
                predictor_dim=pred_cfg["embed_dim"],
                depth=pred_cfg["depth"],
                num_heads=pred_cfg["num_heads"],
                action_dim=action_dim,
                action_conditioning=action_conditioning,
            )
        else:
            norm_type = pred_cfg.get("norm", "layer").lower()
            norm_factory = rms_norm_factory if norm_type == "rms" else (lambda d: nn.LayerNorm(d))
            base_predictor = VideoPredictor(
                num_temporal=num_temporal,
                num_spatial=num_spatial,
                encoder_dim=enc_cfg["embed_dim"],
                predictor_dim=pred_cfg["embed_dim"],
                depth=pred_cfg["depth"],
                num_heads=pred_cfg["num_heads"],
                mlp_ratio=pred_cfg.get("mlp_ratio", 4.0),
                dropout=pred_cfg.get("dropout", 0.0),
                norm_factory=norm_factory,
                sandwich_norm=bool(pred_cfg.get("sandwich_norm", False)),
                action_dim=action_dim,
                action_conditioning=action_conditioning,
            )

        if pred_cfg.get("looped", False):
            predictor: VideoPredictor | LoopedVideoPredictor = LoopedVideoPredictor(
                base_predictor,
                max_loops=pred_cfg.get("max_loops", 4),
                use_exit_gate=pred_cfg.get("use_exit_gate", False),
            )
        else:
            predictor = base_predictor

        return cls(
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            ema_momentum=cfg["train"].get("ema_momentum_start", 0.996),
        )

    def train(self, mode: bool = True) -> AeroJEPA:
        super().train(mode)
        self.target_encoder.eval()
        return self

    def num_trainable_params(self) -> int:
        """Trainable param count excluding the frozen EMA copy."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predictor_is_action_conditioned(self) -> bool:
        """Whether the predictor consumes 6-DoF actions (needed for planning)."""
        pred = self.predictor
        base = pred.base_predictor if isinstance(pred, LoopedVideoPredictor) else pred
        return bool(getattr(base, "action_conditioning", False))

    def forward(
        self,
        clips: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        actions: torch.Tensor | None = None,
        teacher_clips: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context_repr = self.encoder(clips, context_indices)

        teacher_input = teacher_clips if teacher_clips is not None else clips
        with torch.no_grad():
            all_tokens = self.target_encoder.forward_all_patches(teacher_input)
            idx = target_indices.unsqueeze(-1).expand(-1, -1, all_tokens.size(-1))
            target_repr = torch.gather(all_tokens, 1, idx)

        pred_out = self.predictor(context_repr, context_indices, target_indices, actions)
        if isinstance(pred_out, tuple):
            pred_repr, exit_probs = pred_out
        else:
            pred_repr, exit_probs = pred_out, None

        # Smooth-L1 in latent space: robust to the occasional large target and
        # the standard JEPA reconstruction objective.
        pred_loss = F.smooth_l1_loss(pred_repr, target_repr)

        out: dict[str, torch.Tensor] = {
            "loss": pred_loss,
            "pred_loss": pred_loss.detach(),
            "pred_repr": pred_repr,
            "target_repr": target_repr,
        }
        if exit_probs is not None:
            out["exit_probs"] = exit_probs
        return out

    def update_target_encoder(self, momentum: float) -> None:
        update_ema(self.target_encoder, self.encoder, momentum)
