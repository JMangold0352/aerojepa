"""Registry and download helpers for released pretrained checkpoints."""
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PLACEHOLDER = "PLACEHOLDER_URL"

_ENV_PRIMARY = {
    "world_model": "AEROJEPA_WEIGHT_URL_WORLD_MODEL",
    "real_finetune_fast": "AEROJEPA_WEIGHT_URL_REAL_FINETUNE_FAST",
}


def find_repo_root() -> Path:
    """Locate repository root (contains ``released_weights/`` and ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "released_weights").is_dir():
            return parent
    return here.parents[3]


ROOT = find_repo_root()
URLS_YAML = ROOT / "released_weights" / "urls.yaml"


@dataclass(frozen=True)
class ReleasedWeight:
    """Metadata for one published checkpoint."""

    key: str
    checkpoint: str
    hf_filename: str
    description: str

    @property
    def checkpoint_path(self) -> Path:
        return ROOT / self.checkpoint


RELEASED_WEIGHTS: dict[str, ReleasedWeight] = {
    "world_model": ReleasedWeight(
        key="world_model",
        checkpoint="checkpoints/world_model/latest.pt",
        hf_filename="world_model.pt",
        description="Synthetic future-frame world model (Gradio default)",
    ),
    "real_finetune_fast": ReleasedWeight(
        key="real_finetune_fast",
        checkpoint="checkpoints/real_finetune_fast/latest.pt",
        hf_filename="real_finetune_fast.pt",
        description="Unconditioned Wilds fine-tune (representation / gap tables)",
    ),
}


class WeightDownloadError(RuntimeError):
    """Raised when a checkpoint URL is missing or download fails."""


def list_released_weights() -> list[str]:
    return list(RELEASED_WEIGHTS.keys())


def get_released_weight(name: str) -> ReleasedWeight:
    try:
        return RELEASED_WEIGHTS[name]
    except KeyError as exc:
        known = ", ".join(RELEASED_WEIGHTS)
        raise KeyError(f"Unknown model {name!r}. Choose from: {known}") from exc


def _load_url_map() -> dict[str, dict[str, str]]:
    if not URLS_YAML.is_file():
        return {}
    with URLS_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    return {str(k): dict(v or {}) for k, v in data.items()}


def _is_placeholder(url: str | None) -> bool:
    if not url:
        return True
    u = url.strip()
    return not u or u == PLACEHOLDER or u.upper().startswith("TODO")


def resolve_download_url(name: str) -> str | None:
    """Return huggingface / primary URL, or None if still a placeholder."""
    spec = get_released_weight(name)
    env_primary = os.environ.get(_ENV_PRIMARY.get(spec.key, ""), "").strip()
    yaml_entry = _load_url_map().get(spec.key, {})
    hf = env_primary or yaml_entry.get("huggingface") or yaml_entry.get("url")
    return None if _is_placeholder(hf) else hf


def _download_via_hf_hub(url: str, dest: Path) -> bool:
    """Try huggingface_hub.hf_hub_download when URL looks like a Hub resolve link."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    # https://huggingface.co/<repo_id>/resolve/main/<filename>
    marker = "/resolve/"
    if "huggingface.co/" not in url or marker not in url:
        return False
    try:
        after_host = url.split("huggingface.co/", 1)[1]
        repo_id, rest = after_host.split(marker, 1)
        revision, filename = rest.split("/", 1)
        cached = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, dest)
        return True
    except Exception:
        return False


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise WeightDownloadError(f"Failed to download {url} -> {dest}: {exc}") from exc


def download_released_weight(
    name: str,
    *,
    force: bool = False,
    root: Path | None = None,
) -> Path:
    """Download one released checkpoint if URLs are configured. Idempotent."""
    spec = get_released_weight(name)
    dest = (root or ROOT) / spec.checkpoint
    if dest.is_file() and not force:
        return dest.resolve()

    url = resolve_download_url(name)
    if url is None:
        raise WeightDownloadError(
            f"No download URL configured for {name!r} (PLACEHOLDER_URL). "
            f"Set URLs in {URLS_YAML.relative_to(ROOT)} or env "
            f"{_ENV_PRIMARY.get(name)}. See released_weights/README.md. "
            f"weights not published yet; train or pass --checkpoint"
        )

    if _download_via_hf_hub(url, dest):
        return dest.resolve()

    # Fallback: curl via urllib (same role as shell curl)
    try:
        _download_file(url, dest)
        return dest.resolve()
    except WeightDownloadError:
        # Last resort: curl binary if present
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            subprocess.run(
                ["curl", "-fL", "--retry", "3", "-o", str(tmp), url],
                check=True,
            )
            tmp.replace(dest)
            return dest.resolve()
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise WeightDownloadError(
                f"All download attempts failed for {name!r} ({url}): {exc}"
            ) from exc


def ensure_checkpoint(
    name: str,
    *,
    pretrained: bool = True,
    root: Path | None = None,
) -> Path:
    """Resolve local path; optionally download when ``pretrained=True``."""
    spec = get_released_weight(name)
    repo = root or ROOT
    path = repo / spec.checkpoint
    if path.is_file():
        return path.resolve()
    if not pretrained:
        raise FileNotFoundError(
            f"Checkpoint missing for {name!r}: {path}\n"
            f"weights not published yet; train or pass --checkpoint"
        )
    return download_released_weight(name, root=repo)


def checkpoint_status(name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Return registry metadata plus whether the local checkpoint file exists."""
    spec = get_released_weight(name)
    path = (root or ROOT) / spec.checkpoint
    url = resolve_download_url(name)
    return {
        "name": name,
        "checkpoint": str(path.relative_to(root or ROOT)),
        "hf_filename": spec.hf_filename,
        "present": path.is_file(),
        "huggingface_url": url,
        "urls_configured": url is not None,
        "description": spec.description,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``scripts/download_weights.sh``."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Download released AeroJEPA checkpoints into checkpoints/"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print registry entries and local checkpoint status",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when the checkpoint file already exists",
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Registry keys (default: all). Example: world_model",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(f"Registry ({len(RELEASED_WEIGHTS)} models) — root: {ROOT}\n")
        for key in list_released_weights():
            st = checkpoint_status(key)
            url_note = "URLs set" if st["urls_configured"] else "PLACEHOLDER (set urls.yaml or env)"
            present = "present" if st["present"] else "missing"
            print(f"  {key:22}  {present:7}  {url_note}")
            print(f"    {st['description']}")
            print(f"    -> {st['checkpoint']}  (HF: {st['hf_filename']})")
        return 0

    targets = args.models or list_released_weights()
    for key in targets:
        try:
            path = download_released_weight(key, force=args.force)
            print(f"OK  {key} -> {path}")
        except (KeyError, WeightDownloadError) as exc:
            print(f"ERROR  {key}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
