# GAIR

This repository provides the code and pretrained checkpoints for **GAIR** and **Croma** models.

---

##  Checkpoints

- **Croma:** [Download Link](https://drive.google.com/file/d/1Q_nPIzOS7Jwe9JKyD-Ow5iV3MfIoOP1o/view?usp=sharing)
- **GAIR:** [Download Link](https://drive.google.com/file/d/1ychCvJ9DO8s7Svvn_9REMxHoqouDk5PS/view?usp=sharing)

---

##  Training GAIR

To train GAIR from scratch using distributed training, run:

```bash
python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py
```

---

##  Finetuning GAIR

To finetune a pretrained GAIR model, run:

```bash
python finetune.py
```

---

##  Model Implementation

- The main implementation of GAIR can be found in **model\_cl\_RS\_sc.py**.
- See the class `RSSV_LIIF_style` for core model details.

---

##  Using Only the Street View Encoder

To use only the street view encoder from GAIR, run the following code, the image should of size 224*224:

```python
import torch
from transformers import ViTModel, ViTConfig

checkpoint_path = 'path/to/your/checkpoint.pth' # downloaded from google drive
config = ViTConfig(
    image_size=224,
    patch_size=16,
    num_labels=1,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    layer_norm_eps=1e-12,
    dropout_rate=0.1,
    attention_probs_dropout_prob=0.1
)
sv_encoder = ViTModel(config)
sv_encoder.load_state_dict(torch.load(checkpoint_path))
```

