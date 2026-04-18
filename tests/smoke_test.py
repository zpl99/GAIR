#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gair import GAIRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAIR checkpoint smoke test.")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint path. If omitted, runs with random init.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.checkpoint is not None:
        model = GAIRModel.from_checkpoint(args.checkpoint.resolve(), device=device)
        load_info = getattr(model, "load_info", {})
    else:
        model = GAIRModel().to(device).freeze()
        load_info = {}

    rs = torch.rand(2, 10, 96, 96, device=device)
    sv = torch.rand(2, 3, 224, 224, device=device)
    coords = torch.tensor([[116.397, 39.908], [121.4737, 31.2304]], dtype=torch.float32, device=device)
    bboxes = torch.tensor(
        [
            [116.387, 39.918, 116.407, 39.898],
            [121.4637, 31.2404, 121.4837, 31.2204],
        ],
        dtype=torch.float32,
        device=device,
    )

    with torch.inference_mode():
        rs_embedding = model.encode_rs(rs)
        sv_embedding = model.encode_sv(sv)
        loc_embedding = model.encode_location(coords)
        rs_global_nili, localized_nili = model.query_localized_rs(rs, coords, bboxes, mode="nili")
        rs_global_bilinear, localized_bilinear = model.query_localized_rs(rs, coords, bboxes, mode="bilinear")
        rs_global_bicubic, localized_bicubic = model.query_localized_rs(rs, coords, bboxes, mode="bicubic")

    outputs = {
        "rs_embedding": tuple(rs_embedding.shape),
        "sv_embedding": tuple(sv_embedding.shape),
        "loc_embedding": tuple(loc_embedding.shape),
        "rs_global_nili": tuple(rs_global_nili.shape),
        "localized_nili": tuple(localized_nili.shape),
        "rs_global_bilinear": tuple(rs_global_bilinear.shape),
        "localized_bilinear": tuple(localized_bilinear.shape),
        "rs_global_bicubic": tuple(rs_global_bicubic.shape),
        "localized_bicubic": tuple(localized_bicubic.shape),
    }

    for name, tensor in [
        ("rs_embedding", rs_embedding),
        ("sv_embedding", sv_embedding),
        ("loc_embedding", loc_embedding),
        ("localized_nili", localized_nili),
        ("localized_bilinear", localized_bilinear),
        ("localized_bicubic", localized_bicubic),
    ]:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"{name} contains non-finite values.")

    print(
        json.dumps(
            {
                "device": str(device),
                "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
                "load_info": load_info,
                "outputs": outputs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
