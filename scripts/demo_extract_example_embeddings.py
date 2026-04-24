#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gair import GAIRModel, load_rs_image, load_sv_image, preprocess_rs_array, preprocess_sv_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GAIR embeddings for bundled satellite/SV/geolocation examples.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a GAIR checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device. Falls back to CPU if unavailable.")
    parser.add_argument(
        "--examples",
        type=Path,
        default=PROJECT_ROOT / "examples" / "extract_embeddings" / "examples.csv",
        help="CSV with rs_image, sv_image, lon/lat, and RS bbox columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "example_embeddings",
        help="Directory for exported .npy embeddings.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def load_examples(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_embedding(path: Path, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(path, array)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    examples = load_examples(args.examples.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GAIRModel.from_checkpoint(args.checkpoint.resolve(), device=device, query_mode="nili")

    with torch.inference_mode():
        for example in examples:
            example_id = example["id"]
            rs = preprocess_rs_array(load_rs_image(project_path(example["rs_image"]))).to(device=device, dtype=torch.float32)
            sv = preprocess_sv_array(load_sv_image(project_path(example["sv_image"]))).to(device=device, dtype=torch.float32)
            coord = torch.tensor([[float(example["lon"]), float(example["lat"])]], device=device, dtype=torch.float32)
            bbox = torch.tensor(
                [
                    [
                        float(example["rs_bbox_lon_min"]),
                        float(example["rs_bbox_lat_max"]),
                        float(example["rs_bbox_lon_max"]),
                        float(example["rs_bbox_lat_min"]),
                    ]
                ],
                device=device,
                dtype=torch.float32,
            )

            rs_embedding = model.encode_rs(rs, normalize=True)
            sv_embedding = model.encode_sv(sv, normalize=True)
            loc_embedding = model.encode_location(coord, normalize=True)
            _, localized_rs_embedding = model.query_localized_rs(rs, coord, bbox, mode="nili", normalize=True)

            outputs = {
                "rs": rs_embedding,
                "sv": sv_embedding,
                "loc": loc_embedding,
                "localized_rs": localized_rs_embedding,
            }
            for name, embedding in outputs.items():
                save_embedding(output_dir / f"{example_id}_{name}.npy", embedding)

            print(
                f"{example_id}: "
                f"rs={tuple(rs_embedding.shape)} "
                f"sv={tuple(sv_embedding.shape)} "
                f"loc={tuple(loc_embedding.shape)} "
                f"localized_rs={tuple(localized_rs_embedding.shape)}"
            )

    print(f"Saved embeddings to: {output_dir}")


if __name__ == "__main__":
    main()
