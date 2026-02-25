from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from mintpy.utils.writefile import write_isce_file
from osgeo import gdal


def bbox_to_sar_indices(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    lat_data: np.ndarray,
    lon_data: np.ndarray,
) -> list[int]:
    """Convert geographic bbox to rounded SAR row/col indices."""
    data_map = (
        (lon_data >= lon_min)
        & (lon_data <= lon_max)
        & (lat_data >= lat_min)
        & (lat_data <= lat_max)
    )
    region_list = np.argwhere(data_map)
    if region_list.size == 0:
        raise ValueError("No SAR pixels found inside requested geographic bounding box.")
    return [
        10 * math.floor(region_list[:, 0].min() / 10),
        10 * math.ceil(region_list[:, 0].max() / 10),
        10 * math.floor(region_list[:, 1].min() / 10),
        10 * math.ceil(region_list[:, 1].max() / 10),
    ]


def read_isce_file(file_path: str | Path) -> np.ndarray:
    """Read an ISCE/GDAL-readable raster as (H, W, 1) float32 array."""
    file_path = str(file_path)
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(f"Unable to open file: {file_path}")

    ext = Path(file_path).suffix
    band = ds.GetRasterBand(2 if ext == ".unw" else 1)
    if band is None:
        raise RuntimeError(f"Unable to read raster band from: {file_path}")

    data = np.expand_dims(band.ReadAsArray(), 2).astype(np.float32)
    return data


def write_gdal_file(arr: np.ndarray, output_filepath: str | Path, data_type=gdal.GDT_Float32) -> None:
    """Write a 2D or 3D array to ENVI format using GDAL."""
    output_filepath = str(output_filepath)
    if arr.ndim == 2:
        rows, cols = arr.shape
        data_to_write = arr
    elif arr.ndim == 3:
        rows, cols, _ = arr.shape
        data_to_write = arr[:, :, 0]
    else:
        raise ValueError("Array must be 2D or 3D.")

    driver = gdal.GetDriverByName("ENVI")
    dataset = driver.Create(output_filepath, cols, rows, 1, data_type)
    if dataset is None:
        raise RuntimeError(f"Could not create file: {output_filepath}")

    dataset.GetRasterBand(1).WriteArray(data_to_write)
    dataset.FlushCache()


def write_array_to_isce(arr: np.ndarray, output_filepath: str | Path) -> None:
    """Write array to ISCE or ENVI file according to extension."""
    output_filepath = str(output_filepath)
    ext = Path(output_filepath).suffix
    type_map = {
        ".unw": "MOD_isce_unw",
        ".cor": "isce_cor",
        ".rdr": "isce_cor",
        ".int": "isce_int",
        ".full": "envi",
    }
    if ext == ".full":
        write_gdal_file(arr, output_filepath)
        return
    if ext not in type_map:
        raise ValueError(f"Unsupported file extension: {ext}")

    if arr.ndim == 2:
        data = arr
    elif arr.ndim == 3:
        data = arr[:, :, 0]
    else:
        raise ValueError("Array must be 2D or 3D.")

    write_isce_file(data=data, out_file=output_filepath, file_type=type_map[ext])


def find_first_existing_file(parent: str | Path, candidates: Iterable[str]) -> Path:
    base = Path(parent)
    for name in candidates:
        path = base / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of candidate files exist under {base}: {list(candidates)}")
