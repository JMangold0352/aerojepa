# Released pretrained weights

Two AeroJEPA checkpoints are documented here. **Weights are not stored in git**
(see `.gitignore`). Download with [`scripts/download_weights.sh`](../scripts/download_weights.sh)
or load via Python `load_pretrained` (see below).

Until URLs in [`urls.yaml`](urls.yaml) are filled in (replace `PLACEHOLDER_URL`),
download will fail loudly — train locally or wait for the Hugging Face host.

| Registry key | Local path after download | Hugging Face file | What it is for |
| --- | --- | --- | --- |
| `world_model` | `checkpoints/world_model/latest.pt` | `world_model.pt` | Gradio default; synthetic future-frame world model |
| `real_finetune_fast` | `checkpoints/real_finetune_fast/latest.pt` | `real_finetune_fast.pt` | Unconditioned Wilds fine-tune; representation / sim-to-real tables |

**Not released:** `action_conditioned_wilds`, `action_residual_*`, and `*_v2`.
Action-conditioned Wilds is not released because counterfactuals fail
(true ≈ zero ≈ shuffle). Those stay local.

**License:** [MIT](../LICENSE) (same as the repository).

## Download URLs

URLs live in [`urls.yaml`](urls.yaml). Canonical shape:

```text
https://huggingface.co/JMangold0352/aerojepa/resolve/main/<filename>
```

### Environment overrides

- `AEROJEPA_WEIGHT_URL_WORLD_MODEL`
- `AEROJEPA_WEIGHT_URL_REAL_FINETUNE_FAST`

## Shell download

```bash
# List registry entries and whether checkpoints are already present
./scripts/download_weights.sh --list

# Download all released weights (skips files that already exist)
./scripts/download_weights.sh

# Download one variant
./scripts/download_weights.sh world_model
```

## Python loader

```python
import torch
from aerojepa.eval import load_pretrained

model, cfg = load_pretrained("world_model", torch.device("cpu"))
# also: "real_finetune_fast"
```

If `pretrained=True` and the checkpoint is missing locally, the loader downloads
when URLs are configured. With placeholder URLs you get a short error:
train or pass `--checkpoint`.

## Export + upload (maintainers)

```bash
# Strip optimizer/scaler/EMA extras → released_weights/_export/<key>.pt
python scripts/export_release_ckpt.py              # both keys that exist locally
python scripts/export_release_ckpt.py world_model  # one key

# Upload to Hugging Face (needs huggingface_hub + login)
python scripts/upload_hf_weights.py
```

If the CLI / token is missing, the upload script prints:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create JMangold0352/aerojepa --type model --yes
```

and stops. It does not invent URLs.
