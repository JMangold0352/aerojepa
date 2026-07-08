from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from aerojepa.data.synthetic import build_synthetic_dataloaders
from aerojepa.data.telemetry import normalize_actions
from aerojepa.masking import build_mask_collator
from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor, expected_loops_from_exit_probs
from aerojepa.utils.logging import RunLogger
from aerojepa.utils.seed import set_seed


def ema_schedule_cosine(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine EMA momentum ramp (DINO / MoCo-v3 style)."""
    progress = min(1.0, step / max(1, total_steps))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay to zero."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def stack_indices(indices_list: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    return torch.stack(indices_list).to(device)


def adapt_actions(actions: torch.Tensor, num_temporal: int) -> torch.Tensor:
    """Match telemetry length to the token timeline.

    With the tubelet tokenizer the model has fewer temporal slots than frames,
    so we average the pose deltas within each tubelet -- the mean motion over the
    frames that tubelet summarizes.
    """
    if actions.shape[1] == num_temporal:
        return actions
    b, t, d = actions.shape
    group = max(1, t // num_temporal)
    trimmed = actions[:, : group * num_temporal]
    return trimmed.reshape(b, num_temporal, group, d).mean(dim=2)


def build_video_dataloaders(data_cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders over real footage in ``data_cfg['data_dir']``.

    The split is done at the *video* level (whole clips held out for validation)
    so sliding windows from the same flight never leak across train and val. With
    a single video the same clip is used for both (with a warning) so a tiny
    corpus still runs end to end.
    """
    from aerojepa.data.video_dataset import VideoClipDataset, discover_videos

    paths = discover_videos(data_cfg["data_dir"])
    if not paths:
        raise FileNotFoundError(
            f"No videos found in {data_cfg['data_dir']}. See data/README.md for the layout."
        )

    val_fraction = data_cfg.get("val_fraction", 0.1)
    n_val = int(len(paths) * val_fraction)
    if len(paths) > 1:
        n_val = max(1, n_val)
    val_paths = paths[:n_val]
    train_paths = paths[n_val:]
    if not train_paths or not val_paths:
        print(f"[data] only {len(paths)} video(s); using the same clip(s) for train and val.")
        train_paths = paths
        val_paths = paths

    common = dict(
        data_dir=data_cfg["data_dir"],
        num_frames=data_cfg["num_frames"],
        img_size=data_cfg["img_size"],
        mode=data_cfg.get("window_mode", "uniform"),
        stride=data_cfg.get("window_stride"),
        pad_short=data_cfg.get("pad_short", True),
    )
    train_ds = VideoClipDataset(video_paths=train_paths, **common)
    val_ds = VideoClipDataset(video_paths=val_paths, **common)
    print(f"[data] real video: {len(train_ds)} train / {len(val_ds)} val clips "
          f"from {len(train_paths)}/{len(val_paths)} videos (mode={common['mode']}).")

    def make(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=data_cfg["batch_size"],
            shuffle=shuffle,
            num_workers=data_cfg.get("num_workers", 0),
            drop_last=shuffle and len(ds) > data_cfg["batch_size"],
        )

    return make(train_ds, True), make(val_ds, False)


def build_dataloaders_from_cfg(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Synthetic loaders by default; real video loaders when configured."""
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")

    if source == "video":
        return build_video_dataloaders(data_cfg)

    return build_synthetic_dataloaders(
        batch_size=data_cfg["batch_size"],
        num_frames=data_cfg["num_frames"],
        img_size=data_cfg["img_size"],
        in_chans=data_cfg.get("in_chans", 3),
        num_train=data_cfg.get("num_train", 1024),
        num_val=data_cfg.get("num_val", 128),
        num_workers=data_cfg.get("num_workers", 0),
        seed=cfg.get("seed", 42),
        num_obstacles=data_cfg.get("num_obstacles", 5),
        max_speed=data_cfg.get("max_speed", 0.06),
    )


@torch.no_grad()
def evaluate_latent_cosine(
    model: AeroJEPA,
    loader: DataLoader,
    collator,
    device: torch.device,
    cfg: dict[str, Any],
    max_batches: int = 8,
) -> float:
    """Mean cosine similarity between predicted and teacher target latents.

    This is the headline "is the world model any good?" number: higher means the
    predictor's guess about hidden/future structure lines up with reality.
    """
    model.eval()
    num_temporal = model.encoder.num_temporal
    use_actions = bool(cfg["predictor"].get("action_conditioning", False))
    total, count = 0.0, 0
    for i, (clips, actions) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        masks = collator(clips.shape[0])
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        acts = _prep_actions(actions, num_temporal, device, cfg) if use_actions else None
        out = model(clips, ctx, tgt, actions=acts)
        cos = F.cosine_similarity(out["pred_repr"], out["target_repr"], dim=-1).mean()
        total += float(cos.item())
        count += 1
    return total / max(1, count)


def _prep_actions(
    actions: torch.Tensor, num_temporal: int, device: torch.device, cfg: dict[str, Any]
) -> torch.Tensor:
    actions = adapt_actions(actions, num_temporal).to(device)
    if cfg.get("data", {}).get("normalize_actions", True):
        actions = normalize_actions(actions)
    return actions


def train_epoch(
    model: AeroJEPA,
    loader: DataLoader,
    collator,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    global_step: int,
    total_steps: int,
    cfg: dict[str, Any],
    logger: RunLogger | None,
) -> tuple[float, int]:
    model.train()
    train_cfg = cfg["train"]
    beta = train_cfg.get("exit_entropy_beta", 0.01)
    ema_start = train_cfg.get("ema_momentum_start", 0.996)
    ema_end = train_cfg.get("ema_momentum_end", 1.0)
    use_actions = bool(cfg["predictor"].get("action_conditioning", False))
    num_temporal = model.encoder.num_temporal

    total_loss, num_batches = 0.0, 0
    for clips, actions in tqdm(loader, desc="train", leave=False):
        clips = clips.to(device)
        masks = collator(clips.shape[0])
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        acts = _prep_actions(actions, num_temporal, device, cfg) if use_actions else None

        out = model(clips, ctx, tgt, actions=acts)
        loss = out["loss"]
        if isinstance(model.predictor, LoopedVideoPredictor) and "exit_probs" in out:
            loss = model.predictor.compute_total_loss(loss, out["exit_probs"], beta=beta)

        optimizer.zero_grad()
        loss.backward()
        if train_cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        momentum = ema_schedule_cosine(global_step, total_steps, ema_start, ema_end)
        model.update_target_encoder(momentum)

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if logger and global_step % train_cfg.get("log_every", 50) == 0:
            metrics = {"train/loss": loss.item(), "train/ema_momentum": momentum}
            if "exit_probs" in out:
                probs = out["exit_probs"]
                metrics["train/expected_loops"] = float(
                    expected_loops_from_exit_probs(probs).mean().item()
                )
                for i in range(probs.shape[1]):
                    metrics[f"train/exit_prob_loop_{i + 1}"] = float(probs[:, i].mean().item())
            logger.log(metrics, global_step)

    return total_loss / max(1, num_batches), global_step


def save_checkpoint(
    path: Path,
    model: AeroJEPA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    step: int,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg,
        },
        path,
    )


def load_resume_state(
    checkpoint_path: str | Path,
    model: AeroJEPA,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int]:
    """Restore a full training checkpoint (model + optimizer + scheduler).

    Returns ``(completed_epochs, global_step)`` where ``completed_epochs`` is the
    1-based count of finished epochs (matching what :func:`save_checkpoint`
    writes). Resume training with ``range(completed_epochs, total_epochs)``.

    Raises a clear error if the file is weights-only (use
    :func:`load_pretrained_weights` / ``--init-checkpoint`` instead).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    required = ("model", "optimizer", "scheduler", "epoch", "step")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is missing {missing}. "
            "For weight-only warm-start use --init-checkpoint, not --resume."
        )

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    completed = int(ckpt["epoch"])
    step = int(ckpt["step"])
    print(f"[resume] restored {path} (completed {completed} epoch(s), step {step})")
    return completed, step


def load_pretrained_weights(
    model: AeroJEPA, checkpoint_path: str | Path, device: torch.device
) -> None:
    """Warm-start a model from another checkpoint for fine-tuning.

    Loads matching tensors (encoder, EMA teacher, and predictor) and tolerates
    architecture drift with ``strict=False`` -- e.g. fine-tuning a synthetic
    world model on real footage where only the data pipeline changed. Any keys
    that do not line up are reported rather than silently dropped so a genuine
    config mismatch is easy to spot.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    result = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)
    print(f"[finetune] loaded {loaded}/{len(state)} tensors from {checkpoint_path}")
    if result.missing_keys:
        print(f"[finetune]   {len(result.missing_keys)} missing (kept randomly initialized)")
    if result.unexpected_keys:
        print(f"[finetune]   {len(result.unexpected_keys)} unexpected (ignored)")


def train(
    cfg: dict[str, Any],
    device: torch.device,
    init_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    set_seed(cfg.get("seed", 42))

    train_loader, val_loader = build_dataloaders_from_cfg(cfg)
    model = AeroJEPA.from_config(cfg).to(device)
    print(f"Trainable parameters: {model.num_trainable_params():,}")

    train_cfg = cfg["train"]
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    epochs = train_cfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    if "warmup_epochs" in train_cfg:
        warmup_steps = int(train_cfg["warmup_epochs"] * steps_per_epoch)
    else:
        warmup_steps = train_cfg.get("warmup_steps", 200)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    start_epoch = 0
    global_step = 0

    resume_checkpoint = resume_checkpoint or train_cfg.get("resume_checkpoint")
    if resume_checkpoint:
        completed, global_step = load_resume_state(
            resume_checkpoint, model, optimizer, scheduler, device
        )
        start_epoch = completed
        if start_epoch >= epochs:
            print(
                f"[resume] checkpoint already at epoch {start_epoch} "
                f"but config requests only {epochs}; nothing to do."
            )
            return Path(resume_checkpoint)
    else:
        # CLI --init-checkpoint wins over config; skipped when resuming.
        init_checkpoint = init_checkpoint or train_cfg.get("init_checkpoint")
        if init_checkpoint:
            load_pretrained_weights(model, init_checkpoint, device)

    grid_size = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid_size, model.encoder.num_temporal)

    run_dir = Path(train_cfg.get("run_dir", "runs/default"))
    logger = RunLogger(run_dir, use_wandb=train_cfg.get("use_wandb", False))
    logger.init(cfg)
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/default"))

    eval_every = (cfg.get("eval", {}) or {}).get("eval_every_epochs", 0)

    for epoch in range(start_epoch, epochs):
        avg_loss, global_step = train_epoch(
            model, train_loader, collator, optimizer, scheduler,
            device, global_step, total_steps, cfg, logger,
        )
        print(f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
        logger.log({"epoch": epoch + 1, "train/epoch_loss": avg_loss}, global_step)
        save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scheduler, epoch + 1, global_step, cfg)

        is_final = epoch + 1 == epochs
        if (eval_every and (epoch + 1) % eval_every == 0) or is_final:
            cos = evaluate_latent_cosine(model, val_loader, collator, device, cfg)
            print(f"  [eval] epoch {epoch + 1}  latent_cosine={cos:.4f}")
            logger.log({"eval/latent_cosine": cos}, global_step)

    return ckpt_dir / "latest.pt"
