# Embedding Extraction Examples

This folder contains a small CSV with bundled satellite images, street-view images, geolocations, and satellite image bounding boxes.

Run from the repository root:

```bash
CHECKPOINT=$(hf download PingL/GAIR checkpoint.pth)

python scripts/demo_extract_example_embeddings.py \
  --checkpoint "$CHECKPOINT"
```

The script saves one `.npy` file per embedding type and example under `outputs/example_embeddings/`:

- `{id}_rs.npy`
- `{id}_sv.npy`
- `{id}_loc.npy`
- `{id}_localized_rs.npy`

Each embedding has shape `(1, 768)`.
