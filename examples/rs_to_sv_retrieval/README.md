# Localized RS to Street-View Retrieval Demo

This folder contains a tiny retrieval demo built from 3 remote sensing query images and 3 street-view gallery images. We can use the RS encoder in GAIR to extract the localized RS embedding, then use this embedding the retrive the corresponding SV images.


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
  --checkpoint "$CHECKPOINT"
```
