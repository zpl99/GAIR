# Localized RS to Street-View Retrieval Demo

This folder contains a tiny retrieval demo built from 3 remote sensing query images and 3 street-view gallery images.

It follows the same retrieval setup used in `SatMAE_new/misc/retrieve_rs_sv_liif.py`: localized RS query embeddings are matched against street-view gallery embeddings computed with the SV encoder plus `optical_proj`.

- Query images: `queries/*.tif`
- Gallery images: `gallery/*.jpg`
- Pre-rendered previews: [previews/](previews)

Bundled example ids:

- `detroit`
- `sacramento`
- `tokyo`

Run the demo from the repository root:

```bash
python scripts/demo_localized_rs_to_sv_retrieval.py \
  --checkpoint path/to/checkpoint.pth
```

The script:

- computes a localized RS embedding for each query point with NILI
- computes street-view gallery embeddings with the GAIR SV encoder and `optical_proj`
- ranks gallery images by cosine similarity
- saves one preview image per query under `outputs/rs_to_sv_retrieval_demo/`

Expected behavior for this tiny demo: each query retrieves the street-view image with the same id as the top-1 match.
