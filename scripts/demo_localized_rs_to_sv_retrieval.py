#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gair import GAIRModel, load_rs_image, load_sv_image, preprocess_rs_array, preprocess_sv_array
from gair.preprocessing import select_rs_bands


EXAMPLE_ROOT = PROJECT_ROOT / "examples" / "rs_to_sv_retrieval"
GALLERY = [
    {"id": "detroit", "image": EXAMPLE_ROOT / "gallery" / "detroit_sv.jpg", "lon": -83.258332, "lat": 42.482289},
    {"id": "sacramento", "image": EXAMPLE_ROOT / "gallery" / "sacramento_sv.jpg", "lon": -121.474012, "lat": 38.570138},
    {"id": "tokyo", "image": EXAMPLE_ROOT / "gallery" / "tokyo_sv.jpg", "lon": 139.750090, "lat": 35.657349},
]
QUERIES = [
    {
        "id": "detroit",
        "rs_image": EXAMPLE_ROOT / "queries" / "detroit_rs.tif",
        "lon": -83.258332,
        "lat": 42.482289,
        "expected": "detroit",
    },
    {
        "id": "sacramento",
        "rs_image": EXAMPLE_ROOT / "queries" / "sacramento_rs.tif",
        "lon": -121.474012,
        "lat": 38.570138,
        "expected": "sacramento",
    },
    {
        "id": "tokyo",
        "rs_image": EXAMPLE_ROOT / "queries" / "tokyo_rs.tif",
        "lon": 139.750090,
        "lat": 35.657349,
        "expected": "tokyo",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundled localized RS -> street-view retrieval example.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a GAIR checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device. Falls back to CPU if unavailable.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rs_to_sv_retrieval_demo",
        help="Directory for the rendered demo previews.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def load_sv_gallery_embeddings(model: GAIRModel, device: torch.device) -> torch.Tensor:
    sv_batch = []
    for item in GALLERY:
        sv_batch.append(preprocess_sv_array(load_sv_image(item["image"])).to(device=device, dtype=torch.float32))
    sv_batch = torch.cat(sv_batch, dim=0)
    with torch.inference_mode():
        sv_embeddings = model.sv_encoder(sv_batch)["pooler_output"]
        sv_embeddings = model.optical_proj(sv_embeddings)
        sv_embeddings = F.normalize(sv_embeddings, dim=1)
    return sv_embeddings


def rs_bbox_from_geotiff(path: Path) -> list[float]:
    with rasterio.open(path) as ds:
        bounds = ds.bounds
    return [float(bounds.left), float(bounds.top), float(bounds.right), float(bounds.bottom)]


def rs_preview(path: Path) -> Image.Image:
    rs = select_rs_bands(load_rs_image(path))
    rgb = np.stack([rs[2], rs[1], rs[0]], axis=-1).astype(np.float32)
    lo = np.percentile(rgb, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(rgb, 98, axis=(0, 1), keepdims=True)
    rgb = (rgb - lo) / np.maximum(hi - lo, 1e-6)
    rgb = np.clip(rgb, 0.0, 1.0)
    return Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")


def sv_preview(path: Path) -> Image.Image:
    return Image.fromarray(load_sv_image(path).astype(np.uint8), mode="RGB")


def normalized_query_xy(lon: float, lat: float, bbox: list[float]) -> tuple[float, float]:
    lon_min, lat_max, lon_max, lat_min = bbox
    x = (lon - lon_min) / max(lon_max - lon_min, 1e-12)
    y = (lat_max - lat) / max(lat_max - lat_min, 1e-12)
    return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))


def draw_query_marker(image: Image.Image, x_norm: float, y_norm: float) -> Image.Image:
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x = x_norm * max(marked.width - 1, 1)
    y = y_norm * max(marked.height - 1, 1)
    radius = max(8, round(min(marked.width, marked.height) * 0.025))
    draw.ellipse((x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2), outline="white", width=4)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="red", width=3)
    draw.line((x - radius - 4, y, x + radius + 4, y), fill="red", width=3)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill="red", width=3)
    draw.text((min(x + radius + 8, marked.width - 90), max(y - radius - 18, 4)), "query point", fill="red")
    return marked


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(6371.0088 * 2.0 * np.arcsin(np.sqrt(a)))


def compose_visualization(
    query_id: str,
    query_image: Image.Image,
    query_lon: float,
    query_lat: float,
    query_xy_norm: tuple[float, float],
    ranked_items: list[dict[str, object]],
    save_path: Path,
) -> None:
    panel_w = 320
    panel_h = 320
    caption_h = 92
    margin = 20
    total_w = margin * 5 + panel_w * 4
    total_h = margin * 2 + panel_h + caption_h
    canvas = Image.new("RGB", (total_w, total_h), color="white")
    draw = ImageDraw.Draw(canvas)

    def paste_panel(image: Image.Image, x: int, title: str, lines: list[str], accent: tuple[int, int, int]) -> None:
        thumb = image.copy()
        thumb.thumbnail((panel_w, panel_h))
        bg = Image.new("RGB", (panel_w, panel_h), color=(245, 245, 245))
        bg.paste(thumb, ((panel_w - thumb.width) // 2, (panel_h - thumb.height) // 2))
        y = margin
        canvas.paste(bg, (x, y))
        draw.rectangle([x, y, x + panel_w, y + panel_h], outline=accent, width=4)
        text_y = y + panel_h + 10
        draw.text((x, text_y), title, fill=accent)
        for idx, line in enumerate(lines, start=1):
            draw.text((x, text_y + idx * 18), line, fill=(30, 41, 59))

    paste_panel(
        draw_query_marker(query_image, query_xy_norm[0], query_xy_norm[1]),
        margin,
        f"Query RS: {query_id}",
        [f"lon={query_lon:.5f}", f"lat={query_lat:.5f}"],
        (37, 99, 235),
    )

    accents = [(22, 163, 74), (245, 158, 11), (220, 38, 38)]
    for idx, item in enumerate(ranked_items[:3], start=1):
        gallery_item = GALLERY[int(item["gallery_index"])]
        image = sv_preview(gallery_item["image"])
        lines = [
            f"id={item['gallery_id']}",
            f"score={item['score']:.4f}",
            f"err={item['distance_km']:.3f} km",
        ]
        if item["is_expected"]:
            lines.append("expected match")
        paste_panel(
            image,
            margin * (idx + 1) + panel_w * idx,
            f"Top-{idx}",
            lines,
            accents[idx - 1],
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = GAIRModel.from_checkpoint(args.checkpoint.resolve(), device=device, query_mode="nili")
    gallery_embeddings = load_sv_gallery_embeddings(model, device)

    summary = []
    with torch.inference_mode():
        for query in QUERIES:
            rs_path = query["rs_image"]
            lon = float(query["lon"])
            lat = float(query["lat"])
            bbox_list = rs_bbox_from_geotiff(rs_path)
            bbox = torch.tensor([bbox_list], device=device, dtype=torch.float32)
            coords = torch.tensor([[lon, lat]], device=device, dtype=torch.float32)
            rs_tensor = preprocess_rs_array(load_rs_image(rs_path)).to(device=device, dtype=torch.float32)

            _, localized = model.query_localized_rs(rs_tensor, coords, bbox, mode="nili", normalize=False)
            localized = F.normalize(localized, dim=1)
            scores = (localized @ gallery_embeddings.T).squeeze(0).detach().cpu().numpy()
            order = np.argsort(-scores)

            ranked_items = []
            for rank_idx in order[:3]:
                gallery_item = GALLERY[int(rank_idx)]
                ranked_items.append(
                    {
                        "gallery_index": int(rank_idx),
                        "gallery_id": gallery_item["id"],
                        "score": float(scores[int(rank_idx)]),
                        "distance_km": haversine_km(lon, lat, float(gallery_item["lon"]), float(gallery_item["lat"])),
                        "is_expected": gallery_item["id"] == query["expected"],
                    }
                )

            top1 = ranked_items[0]
            summary.append(
                {
                    "query_id": query["id"],
                    "top1_id": str(top1["gallery_id"]),
                    "score": float(top1["score"]),
                    "distance_km": float(top1["distance_km"]),
                    "is_correct": bool(top1["gallery_id"] == query["expected"]),
                }
            )

            compose_visualization(
                query_id=str(query["id"]),
                query_image=rs_preview(rs_path),
                query_lon=lon,
                query_lat=lat,
                query_xy_norm=normalized_query_xy(lon, lat, bbox_list),
                ranked_items=ranked_items,
                save_path=output_dir / f"{query['id']}.png",
            )

    correct = 0
    for item in summary:
        correct += int(item["is_correct"])
        status = "OK" if item["is_correct"] else "MISS"
        print(
            f"{item['query_id']}: top1={item['top1_id']} "
            f"score={item['score']:.4f} err_km={item['distance_km']:.3f} [{status}]"
        )

    print(f"Top-1 accuracy on bundled demo: {correct}/{len(QUERIES)} = {correct / len(QUERIES):.3f}")
    print(f"Saved previews to: {output_dir}")


if __name__ == "__main__":
    main()
