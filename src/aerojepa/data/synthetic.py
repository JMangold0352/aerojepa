from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Why a synthetic generator at all?
#
# Real drone corpora (MotionScape, AeroVerse, self-collected Tello footage) are
# large, licensed, and slow to iterate on. To make AeroJEPA reproducible from a
# clean checkout -- with *zero* downloads -- we ship a small procedural renderer
# that produces the two things the model needs: short video clips from a moving
# 6-DoF camera, and the per-frame motion ("telemetry") that produced them.
#
# The physics are deliberately simple (a camera flying over a textured ground
# plane with a few obstacles), but they contain the structure that matters for a
# world model: coherent egomotion, parallax-like scaling as altitude changes,
# and obstacles that loom as the camera approaches. The real-data loader in
# ``video_dataset.py`` is a drop-in replacement once footage is available.

ACTION_DIM = 6  # [dx, dy, d_altitude, d_yaw, d_pitch, d_roll]


@dataclass
class Clip:
    frames: torch.Tensor  # (T, C, H, W) in [0, 1]
    actions: torch.Tensor  # (T, ACTION_DIM); motion that produced each frame


def generate_world_texture(
    size: int,
    num_obstacles: int,
    rng: np.random.Generator,
    channels: int = 3,
) -> torch.Tensor:
    """Build a large, static overhead "world" the drone will fly over.

    We sum a few octaves of smooth value noise for a natural ground texture,
    give it an earthy tint, then stamp a handful of brightly colored obstacle
    disks at fixed world coordinates. Returns ``(C, size, size)`` in [0, 1].
    """
    field = torch.zeros(channels, size, size)
    for octave, freq in enumerate((4, 8, 16, 32)):
        low = torch.from_numpy(rng.random((1, channels, freq, freq)).astype(np.float32))
        up = F.interpolate(low, size=(size, size), mode="bilinear", align_corners=False)
        field += up[0] / (octave + 1)

    field = (field - field.amin()) / (field.amax() - field.amin() + 1e-6)
    # Earthy ground tint so the scene reads as terrain rather than TV static.
    tint = torch.tensor([0.45, 0.55, 0.35]).view(channels, 1, 1)
    field = 0.5 * field + 0.5 * field * tint

    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    for _ in range(num_obstacles):
        cx, cy = rng.integers(0, size, size=2)
        radius = int(rng.integers(max(2, size // 20), max(3, size // 8)))
        color = torch.from_numpy(rng.random(channels).astype(np.float32))
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
        field[:, disk] = color.view(channels, 1)

    return field.clamp(0.0, 1.0)


def _affine_theta(cx: float, cy: float, scale: float, yaw: float) -> torch.Tensor:
    """Camera pose -> affine matrix mapping the output frame into the world.

    ``scale`` is the half-window the camera sees (a proxy for altitude: higher
    altitude sees more ground), ``(cx, cy)`` is the look-at point in normalized
    world coordinates, and ``yaw`` rotates the view.
    """
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return torch.tensor(
        [[scale * c, -scale * s, cx], [scale * s, scale * c, cy]],
        dtype=torch.float32,
    ).unsqueeze(0)


def render_clip(
    seed: int,
    num_frames: int = 8,
    img_size: int = 64,
    in_chans: int = 3,
    world_size: int = 256,
    num_obstacles: int = 5,
    max_speed: float = 0.06,
) -> Clip:
    """Render one deterministic drone clip (and its telemetry) from a seed."""
    rng = np.random.default_rng(seed)
    world = generate_world_texture(world_size, num_obstacles, rng, in_chans).unsqueeze(0)

    cx, cy = rng.uniform(-0.4, 0.4, size=2)
    yaw = float(rng.uniform(-np.pi, np.pi))
    scale = float(rng.uniform(0.3, 0.5))

    vx, vy = rng.uniform(-max_speed, max_speed, size=2)
    v_yaw = float(rng.uniform(-0.12, 0.12))
    v_scale = float(rng.uniform(-0.02, 0.02))

    frames: list[torch.Tensor] = []
    actions: list[list[float]] = []
    for _ in range(num_frames):
        theta = _affine_theta(cx, cy, scale, yaw)
        grid = F.affine_grid(theta, (1, in_chans, img_size, img_size), align_corners=False)
        frame = F.grid_sample(world, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        frames.append(frame[0])

        # Small correlated pitch/roll from turning, so the 6-DoF vector is not
        # trivially zero on two of its axes.
        pitch = 0.3 * v_scale
        roll = 0.3 * v_yaw
        actions.append([float(vx), float(vy), float(v_scale), float(v_yaw), pitch, roll])

        cx += float(vx)
        cy += float(vy)
        yaw += v_yaw
        scale = float(np.clip(scale + v_scale, 0.2, 0.6))

        # Reflect off the world edges so the camera stays over valid terrain.
        limit = 1.0 - scale
        if abs(cx) > limit:
            cx = float(np.clip(cx, -limit, limit))
            vx = -vx
        if abs(cy) > limit:
            cy = float(np.clip(cy, -limit, limit))
            vy = -vy

    return Clip(frames=torch.stack(frames), actions=torch.tensor(actions, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Action-driven rendering.
#
# ``render_clip`` above rolls out its own random dynamics -- great for training
# data, but for the latent planner we need the opposite: drive the *same* camera
# with an explicit 6-DoF action sequence so we can visualize "what actually
# happens if the drone executes this plan." These helpers expose the camera model
# as a tiny controllable simulator (build a world, step a pose with an action,
# render poses to frames). It is the pixel-space ground truth that the planner's
# latent-space imagination is compared against in the demo.
# ---------------------------------------------------------------------------

# A camera pose: look-at point (cx, cy) in normalized world coords, view
# half-window ``scale`` (altitude proxy), and ``yaw`` rotation.
Pose = tuple[float, float, float, float]


def make_world(
    seed: int, world_size: int = 256, num_obstacles: int = 5, in_chans: int = 3
) -> torch.Tensor:
    """Build a static world texture with a ``(1, C, world_size, world_size)`` shape."""
    rng = np.random.default_rng(seed)
    return generate_world_texture(world_size, num_obstacles, rng, in_chans).unsqueeze(0)


def step_pose(pose: Pose, action: torch.Tensor | list[float]) -> Pose:
    """Advance a camera pose by one 6-DoF action (``[dx, dy, d_altitude, d_yaw, ...]``).

    Only the first four components move the camera; pitch/roll are cosmetic in
    this simple model. The camera reflects off the world edges so it always stays
    over valid terrain -- the same guard ``render_clip`` uses.
    """
    cx, cy, scale, yaw = pose
    dx, dy, d_alt, d_yaw = (float(action[0]), float(action[1]), float(action[2]), float(action[3]))
    cx += dx
    cy += dy
    yaw += d_yaw
    scale = float(np.clip(scale + d_alt, 0.2, 0.6))
    limit = 1.0 - scale
    cx = float(np.clip(cx, -limit, limit))
    cy = float(np.clip(cy, -limit, limit))
    return (cx, cy, scale, yaw)


def integrate_actions(init_pose: Pose, actions: torch.Tensor) -> list[Pose]:
    """Roll an action sequence ``(T, >=4)`` forward into a list of ``T`` poses."""
    poses: list[Pose] = []
    pose = init_pose
    for a in actions:
        pose = step_pose(pose, a)
        poses.append(pose)
    return poses


def render_poses(
    world: torch.Tensor, poses: list[Pose], img_size: int = 64, in_chans: int = 3
) -> torch.Tensor:
    """Render a list of camera poses over ``world`` to a clip ``(T, C, H, W)``."""
    frames: list[torch.Tensor] = []
    for cx, cy, scale, yaw in poses:
        theta = _affine_theta(cx, cy, scale, yaw)
        grid = F.affine_grid(theta, (1, in_chans, img_size, img_size), align_corners=False)
        frame = F.grid_sample(
            world, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )
        frames.append(frame[0])
    return torch.stack(frames)


def sample_context(
    seed: int,
    context_frames: int,
    img_size: int = 64,
    in_chans: int = 3,
    world_size: int = 256,
    num_obstacles: int = 5,
    drift: float = 0.05,
) -> tuple[torch.Tensor, Pose, torch.Tensor]:
    """Render a short context clip and return ``(frames, end_pose, world)``.

    The end pose and world are handed to the planner so it can render the
    *consequences* of a chosen action plan from exactly where the context left
    off -- giving a seamless "observed -> planned" visualization.
    """
    rng = np.random.default_rng(seed)
    world = generate_world_texture(world_size, num_obstacles, rng, in_chans).unsqueeze(0)
    init: Pose = (
        float(rng.uniform(-0.3, 0.3)),
        float(rng.uniform(-0.3, 0.3)),
        float(rng.uniform(0.3, 0.5)),
        float(rng.uniform(-np.pi, np.pi)),
    )
    vx, vy = rng.uniform(-drift, drift, size=2)
    ctx_actions = torch.tensor(
        [[float(vx), float(vy), 0.0, 0.0, 0.0, 0.0]] * context_frames, dtype=torch.float32
    )
    poses = integrate_actions(init, ctx_actions)
    frames = render_poses(world, poses, img_size, in_chans)
    return frames, poses[-1], world


class SyntheticDroneClips(Dataset):
    """A reproducible dataset of procedurally rendered drone clips.

    Each index maps deterministically to one clip via a per-sample seed, so the
    dataset is fully reproducible and needs no files on disk.
    """

    def __init__(
        self,
        num_clips: int = 1024,
        num_frames: int = 8,
        img_size: int = 64,
        in_chans: int = 3,
        seed: int = 0,
        world_size: int = 256,
        num_obstacles: int = 5,
        max_speed: float = 0.06,
    ) -> None:
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.img_size = img_size
        self.in_chans = in_chans
        self.seed = seed
        self.world_size = world_size
        self.num_obstacles = num_obstacles
        self.max_speed = max_speed

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip = render_clip(
            seed=self.seed * 100_003 + idx,
            num_frames=self.num_frames,
            img_size=self.img_size,
            in_chans=self.in_chans,
            world_size=self.world_size,
            num_obstacles=self.num_obstacles,
            max_speed=self.max_speed,
        )
        return clip.frames, clip.actions


def build_synthetic_dataloaders(
    batch_size: int = 32,
    num_frames: int = 8,
    img_size: int = 64,
    in_chans: int = 3,
    num_train: int = 1024,
    num_val: int = 128,
    num_workers: int = 0,
    seed: int = 0,
    num_obstacles: int = 5,
    max_speed: float = 0.06,
) -> tuple[DataLoader, DataLoader]:
    """Train / val loaders over disjoint synthetic clip seeds."""
    train = SyntheticDroneClips(
        num_clips=num_train,
        num_frames=num_frames,
        img_size=img_size,
        in_chans=in_chans,
        seed=seed,
        num_obstacles=num_obstacles,
        max_speed=max_speed,
    )
    val = SyntheticDroneClips(
        num_clips=num_val,
        num_frames=num_frames,
        img_size=img_size,
        in_chans=in_chans,
        seed=seed + 9973,  # disjoint seed range from train
        num_obstacles=num_obstacles,
        max_speed=max_speed,
    )
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader
