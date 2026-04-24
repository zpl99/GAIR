#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gair import GAIRModel, load_rs_image, load_sv_image, preprocess_rs_array, preprocess_sv_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GAIR RS/SV/location embeddings and localized RS embeddings.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the GAIR checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device. Falls back to CPU if unavailable.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs", help="Directory for .npy outputs.")
    parser.add_argument("--rs-image", type=Path, help="RS image path (.tif or .npy). Must be a single crop, not a full mosaic.")
    parser.add_argument("--sv-image", type=Path, help="Street-view RGB image path.")
    parser.add_argument("--lon", type=float, help="Longitude of the query point.")
    parser.add_argument("--lat", type=float, help="Latitude of the query point.")
    parser.add_argument(
        "--rs-bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MAX", "LON_MAX", "LAT_MIN"),
        help="BBox of the RS crop, using the order [lon_min, lat_max, lon_max, lat_min].",
    )
    parser.add_argument(
        "--query-mode",
        choices=["nili", "bilinear", "bicubic"],
        default="nili",
        help="Method used for localized RS querying.",
    )
    parser.add_argument("--normalize", action="store_true", help="L2-normalize all exported embeddings.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def save_embedding(path: Path, tensor: torch.Tensor) -> list[int]:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(path, array)
    return list(array.shape)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GAIRModel.from_checkpoint(
        args.checkpoint.resolve(),
        device=device,
        query_mode=args.query_mode,
    )

    saved_outputs: list[tuple[str, Path, list[int]]] = []

    rs_tensor = None
    coord_tensor = None
    bbox_tensor = None

    with torch.inference_mode():
        if args.rs_image is not None:
            rs_array = load_rs_image(args.rs_image)
            rs_tensor = preprocess_rs_array(rs_array).to(device=device, dtype=torch.float32)
            rs_embedding = model.encode_rs(rs_tensor, normalize=args.normalize)
            out_path = output_dir / "rs_embedding.npy"
            saved_outputs.append(("rs_embedding", out_path, save_embedding(out_path, rs_embedding)))

        if args.sv_image is not None:
            sv_array = load_sv_image(args.sv_image)
            sv_tensor = preprocess_sv_array(sv_array).to(device=device, dtype=torch.float32)
            sv_embedding = model.encode_sv(sv_tensor, normalize=args.normalize)
            out_path = output_dir / "sv_embedding.npy"
            saved_outputs.append(("sv_embedding", out_path, save_embedding(out_path, sv_embedding)))

        if args.lon is not None or args.lat is not None:
            if args.lon is None or args.lat is None:
                raise ValueError("Both --lon and --lat must be provided together.")
            coord_tensor = torch.tensor([[args.lon, args.lat]], device=device, dtype=torch.float32)
            loc_embedding = model.encode_location(coord_tensor, normalize=args.normalize)
            out_path = output_dir / "loc_embedding.npy"
            saved_outputs.append(("loc_embedding", out_path, save_embedding(out_path, loc_embedding)))

        if args.rs_bbox is not None:
            bbox_tensor = torch.tensor([args.rs_bbox], device=device, dtype=torch.float32)

        if rs_tensor is not None and coord_tensor is not None and bbox_tensor is not None:
            _, localized = model.query_localized_rs(
                rs_tensor,
                coord_tensor,
                bbox_tensor,
                mode=args.query_mode,
                normalize=args.normalize,
            )
            out_path = output_dir / "localized_rs_embedding.npy"
            saved_outputs.append(("localized_rs_embedding", out_path, save_embedding(out_path, localized)))

    for name, path, shape in saved_outputs:
        print(f"{name}: {path} shape={shape}")


if __name__ == "__main__":
    main()
