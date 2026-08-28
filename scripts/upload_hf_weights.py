#!/usr/bin/env python3
"""Upload CPU release checkpoints + model card to Hugging Face.

Creates JMangold0352/aerojepa (model) if needed, uploads world_model.pt and
real_finetune_fast.pt from released_weights/_export/, and HF_MODEL_CARD.md as
README.md. After a real upload, writes resolve/main URLs into urls.yaml.

If huggingface_hub / login is missing, prints the install commands and STOP.
Does not invent URLs.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "released_weights" / "_export"
CARD = ROOT / "released_weights" / "HF_MODEL_CARD.md"
URLS_YAML = ROOT / "released_weights" / "urls.yaml"
HF_REPO = "JMangold0352/aerojepa"
HF_FILES = {
    "world_model": "world_model.pt",
    "real_finetune_fast": "real_finetune_fast.pt",
}

INSTALL_HINT = """pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create JMangold0352/aerojepa --type model --yes
"""


def _stop_missing_cli() -> int:
    print("huggingface-cli / token is missing. Run:", file=sys.stderr)
    print(INSTALL_HINT, end="")
    return 1


def _write_urls(uploaded: dict[str, str]) -> None:
    import yaml

    data = {}
    if URLS_YAML.is_file():
        with URLS_YAML.open() as f:
            data = yaml.safe_load(f) or {}
    for key, filename in HF_FILES.items():
        url = uploaded.get(key)
        entry = dict(data.get(key) or {})
        if url:
            entry["huggingface"] = url
        data[key] = entry
    with URLS_YAML.open("w") as f:
        f.write(
            "# Pretrained checkpoint URLs (not committed to git).\n"
            "# Override any entry with an environment variable (see released_weights/README.md).\n"
            "#\n"
            "# Canonical shape after upload:\n"
            "#   https://huggingface.co/JMangold0352/aerojepa/resolve/main/<filename>\n\n"
        )
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote URLs -> {URLS_YAML}")


def main() -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download, whoami
    except ImportError:
        return _stop_missing_cli()

    try:
        whoami()
    except Exception:
        return _stop_missing_cli()

    export_paths: dict[str, Path] = {}
    for key, filename in HF_FILES.items():
        src = EXPORT_DIR / f"{key}.pt"
        if src.is_file():
            export_paths[key] = src
        else:
            print(f"ERROR  missing export {src} — run scripts/export_release_ckpt.py first", file=sys.stderr)

    if not export_paths:
        print("ERROR  nothing to upload", file=sys.stderr)
        return 1
    if not CARD.is_file():
        print(f"ERROR  missing model card {CARD}", file=sys.stderr)
        return 1

    api = HfApi()
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True, private=False)
    except Exception as exc:
        print(f"ERROR  could not create/open repo {HF_REPO}: {exc}", file=sys.stderr)
        return _stop_missing_cli()

    uploaded: dict[str, str] = {}
    for key, src in export_paths.items():
        filename = HF_FILES[key]
        print(f"Uploading {src} -> {HF_REPO}/{filename}")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=filename,
            repo_id=HF_REPO,
            repo_type="model",
        )
        uploaded[key] = f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"

    api.upload_file(
        path_or_fileobj=str(CARD),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        repo_type="model",
    )
    print(f"Uploaded model card -> {HF_REPO}/README.md")

    _write_urls(uploaded)
    # Touch download so the cache is warm (optional; ignore failures)
    for key, filename in HF_FILES.items():
        if key not in uploaded:
            continue
        try:
            hf_hub_download(repo_id=HF_REPO, filename=filename)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
