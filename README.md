# Official Implementation of "GAIR: Location-Aware Self-Supervised Contrastive Pre-Training with Geo-Aligned Implicit Representations"

GAIR is a location-aware self-supervised pre-training framework for learning geo-aligned representations from remote sensing imagery, street-view imagery, and geographic coordinates.

This repository currently releases the **inference code** for GAIR. This repo includes

- loading the pretrained GAIR checkpoint
- extracting **remote sensing (RS)** embeddings
- extracting **street-view (SV)** embeddings
- extracting **location** embeddings
- querying **localized RS embeddings** with the **NILI** module

The **pretraining dataset** and **pretraining code** are still being organized and will be released later.

## Installation

```bash
conda create -n gair python=3.10 -y
conda activate gair
python -m pip install --upgrade pip
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` should print `True`.

If you need a different CUDA wheel, replace only the PyTorch install command above and keep the remaining steps unchanged.


## Quick Start

### Python API

```python
import torch

from huggingface_hub import hf_hub_download
from gair import GAIRModel, load_rs_image, load_sv_image, preprocess_rs_array, preprocess_sv_array

checkpoint = hf_hub_download(repo_id="PingL/GAIR", filename="checkpoint.pth")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = GAIRModel.from_checkpoint(checkpoint, device=device, query_mode="nili")

rs = preprocess_rs_array(load_rs_image("path/to/rs_crop.tif")).to(device)
sv = preprocess_sv_array(load_sv_image("path/to/street_view.jpg")).to(device)
coords = torch.tensor([[116.397, 39.908]], device=device)  # [lon, lat]
bbox = torch.tensor([[116.387, 39.918, 116.407, 39.898]], device=device)  # bbox of the rs image, [lon_min, lat_max, lon_max, lat_min]

with torch.inference_mode():
    rs_embedding = model.encode_rs(rs, normalize=True)
    sv_embedding = model.encode_sv(sv, normalize=True)
    loc_embedding = model.encode_location(coords, normalize=True)
    rs_global, localized_rs_embedding = model.query_localized_rs(
        rs,
        coords,
        bbox,
        mode="nili",
        normalize=True,
    )
```

### Command Line Inference

```bash
CHECKPOINT=$(hf download PingL/GAIR checkpoint.pth)

python scripts/extract_embeddings.py \
  --checkpoint "$CHECKPOINT" \
  --rs-image path/to/rs_crop.tif \
  --sv-image path/to/street_view.jpg \
  --lon 116.397 \
  --lat 39.908 \
  --rs-bbox 116.387 39.918 116.407 39.898 \
  --query-mode nili \
  --normalize \
  --output-dir outputs/demo
```

The script exports:

- `rs_embedding.npy`
- `sv_embedding.npy`
- `loc_embedding.npy`
- `localized_rs_embedding.npy`
- `rs_embedding_from_query.npy`
- `summary.json`

## Retrieval Demo

A tiny localized RS-to-street-view retrieval demo is included in [examples/rs_to_sv_retrieval/README.md](examples/rs_to_sv_retrieval/README.md).

Run:

```bash
CHECKPOINT=$(hf download PingL/GAIR checkpoint.pth)

python scripts/demo_localized_rs_to_sv_retrieval.py \
  --checkpoint "$CHECKPOINT"
```

This demo uses 3 remote sensing queries and 3 street-view gallery images bundled in the repository, and saves one preview image per query under `outputs/rs_to_sv_retrieval_demo/`.

## Input Conventions

- RS crop bbox order: `[lon_min, lat_max, lon_max, lat_min]`
- Coordinate order: `[lon, lat]`
- `--rs-image` should already be the RS crop corresponding to `--rs-bbox`
- RS inputs should be 10-band Sentinel-2 image (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12).



## Query Modes

- `nili`: learned NILI query used by GAIR
- `bilinear`: deterministic interpolation baseline
- `bicubic`: deterministic interpolation baseline

## Citation

The citation entry will be added after the paper is officially published.
