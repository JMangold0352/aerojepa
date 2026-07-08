from __future__ import annotations

import torch

from aerojepa.viz import plots


def test_plot_functions_write_files(tmp_path) -> None:
    plots.plot_rollout_curve({"horizon": [1, 2, 3], "cosine": [0.9, 0.8, 0.7]}, tmp_path / "rollout.png")
    plots.plot_per_loop_cosine({"per_loop_cosine": [0.7, 0.8]}, tmp_path / "loop.png")
    plots.plot_exit_distribution(
        {"exit_distribution": [0.5, 0.5], "expected_loops": 1.5}, tmp_path / "exit.png"
    )
    plots.plot_latent_trajectory(torch.randn(6, 16), tmp_path / "traj.png")
    plots.plot_clip_frames(torch.rand(4, 3, 32, 32), tmp_path / "clip.png")
    plots.plot_attention_matrix(torch.rand(20, 20), num_temporal=4, num_context=8, path=tmp_path / "attn.png")

    for name in ("rollout", "loop", "exit", "traj", "clip", "attn"):
        assert (tmp_path / f"{name}.png").exists()


def test_generate_all_figures_fast(tmp_path) -> None:
    import torch as _torch

    from aerojepa.train import train
    from aerojepa.utils.config import load_config
    from visualizations.generate_all_figures import generate_all

    cfg = load_config("configs/smoke_test.yaml")
    cfg["train"]["run_dir"] = str(tmp_path / "runs")
    cfg["train"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    ckpt = train(cfg, _torch.device("cpu"))

    out_dir = tmp_path / "figures"
    generate_all(str(ckpt), str(out_dir), device="cpu", fast=True)
    assert (out_dir / "00_clip.png").exists()
    assert (out_dir / "01_rollout.png").exists()
