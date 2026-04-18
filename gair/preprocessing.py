from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from pyproj import Transformer
from rasterio.transform import rowcol


SENTINEL_MEAN = np.array(
    [1172.9397, 1378.0846, 1509.3327, 1750.0275, 2073.2758, 2207.3283, 2245.1629, 2284.6384, 2231.7283, 1899.1932],
    dtype=np.float32,
)
SENTINEL_STD = np.array(
    [706.6250, 720.1862, 783.1424, 707.1962, 714.2782, 748.1827, 852.3585, 762.0849, 690.0165, 669.9036],
    dtype=np.float32,
)
SV_MEAN = np.array([135.47184384, 142.28621995, 145.12271992], dtype=np.float32)
SV_STD = np.array([59.19978436, 59.35279219, 72.02295734], dtype=np.float32)
DEFAULT_KEEP_BAND_IDX = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11], dtype=np.int64)


def _to_chw(array: np.ndarray, valid_channels: tuple[int, ...]) -> np.ndarray:
    if array.ndim == 3:
        if array.shape[0] in valid_channels:
            return array
        if array.shape[-1] in valid_channels:
            return np.moveaxis(array, -1, 0)
    if array.ndim == 4:
        if array.shape[1] in valid_channels:
            return array
        if array.shape[-1] in valid_channels:
            return np.moveaxis(array, -1, 1)
    raise ValueError(f"Could not infer channel dimension for shape={array.shape}")


def select_rs_bands(array: np.ndarray, keep_band_idx: np.ndarray = DEFAULT_KEEP_BAND_IDX) -> np.ndarray:
    chw = _to_chw(array, valid_channels=(10, 12, 13))
    channel_dim = 0 if chw.ndim == 3 else 1
    if chw.shape[channel_dim] == 10:
        return chw.astype(np.float32, copy=False)
    if chw.shape[channel_dim] <= keep_band_idx.max():
        raise ValueError(
            f"Input has {chw.shape[channel_dim]} channels but keep_band_idx expects at least {keep_band_idx.max() + 1}"
        )
    if chw.ndim == 3:
        return chw[keep_band_idx].astype(np.float32, copy=False)
    return chw[:, keep_band_idx].astype(np.float32, copy=False)


def preprocess_rs_array(
    array: np.ndarray,
    input_size: int = 96,
    keep_band_idx: np.ndarray = DEFAULT_KEEP_BAND_IDX,
) -> torch.Tensor:
    rs = select_rs_bands(array, keep_band_idx=keep_band_idx)
    if rs.ndim == 3:
        rs = rs[None, ...]
    min_value = (SENTINEL_MEAN - 2.0 * SENTINEL_STD).reshape(1, -1, 1, 1)
    max_value = (SENTINEL_MEAN + 2.0 * SENTINEL_STD).reshape(1, -1, 1, 1)
    rs = (rs - min_value) / (max_value - min_value) * 255.0
    rs = np.clip(rs, 0.0, 255.0).astype(np.uint8)
    tensor = torch.from_numpy(rs).float().div_(255.0)
    tensor = F.interpolate(tensor, size=(input_size, input_size), mode="bicubic", align_corners=False)
    return tensor


def preprocess_sv_array(array: np.ndarray, input_size: int = 224) -> torch.Tensor:
    if array.ndim != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape={array.shape}")
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape={array.shape}")
    image = array.astype(np.float32)
    image = (image - SV_MEAN) / SV_STD
    tensor = torch.from_numpy(np.moveaxis(image, -1, 0)).float().unsqueeze(0)
    tensor = F.interpolate(tensor, size=(input_size, input_size), mode="bicubic", align_corners=False)
    return tensor


def load_rs_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    with rasterio.open(path) as ds:
        return ds.read()


def load_sv_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def compute_centered_window(
    row: int,
    col: int,
    patch_size: int,
    *,
    height: int,
    width: int,
) -> tuple[int, int, bool]:
    max_row0 = max(0, height - patch_size)
    max_col0 = max(0, width - patch_size)
    half = patch_size // 2
    row0 = int(np.clip(row - half, 0, max_row0))
    col0 = int(np.clip(col - half, 0, max_col0))
    shifted = row0 != row - half or col0 != col - half
    return row0, col0, shifted


def extract_rs_patch_with_bbox(
    tif_path: str | Path,
    *,
    x: float,
    y: float,
    patch_size: int = 224,
    coord_crs: str = "EPSG:4326",
    keep_band_idx: np.ndarray = DEFAULT_KEEP_BAND_IDX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as ds:
        query_x, query_y = x, y
        if coord_crs != str(ds.crs):
            transformer = Transformer.from_crs(coord_crs, ds.crs, always_xy=True)
            query_x, query_y = transformer.transform(x, y)

        row, col = rowcol(ds.transform, query_x, query_y)
        row = int(np.clip(row, 0, ds.height - 1))
        col = int(np.clip(col, 0, ds.width - 1))
        row0, col0, shifted = compute_centered_window(row, col, patch_size, height=ds.height, width=ds.width)
        row1 = row0 + patch_size
        col1 = col0 + patch_size

        full_img = ds.read()
        patch = full_img[keep_band_idx, row0:row1, col0:col1]

        a = float(ds.transform.a)
        e = float(ds.transform.e)
        c = float(ds.transform.c)
        f = float(ds.transform.f)
        bbox = np.array(
            [
                c + col0 * a,
                f + row0 * e,
                c + col1 * a,
                f + row1 * e,
            ],
            dtype=np.float32,
        )
        query_coord = np.array([query_x, query_y], dtype=np.float32)
        return patch, bbox, query_coord, shifted
