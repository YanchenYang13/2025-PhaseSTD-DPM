from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import bbox_to_sar_indices, read_isce_file, write_array_to_isce


@dataclass
class CropConfig:
    base_path: Path
    geom_reference_path: Path
    output_base_path: Path
    lat_min: float = 42.625
    lat_max: float = 42.635
    lon_min: float = 13.28
    lon_max: float = 13.30


def _resolve_lat_lon_files(geom_reference_path: Path) -> tuple[Path, Path]:
    lat_candidates = ["lat.rdr"] + sorted([p.name for p in geom_reference_path.glob("*lat*") if p.is_file()])
    lon_candidates = ["lon.rdr"] + sorted([p.name for p in geom_reference_path.glob("*lon*") if p.is_file()])

    lat_file = next((geom_reference_path / f for f in lat_candidates if (geom_reference_path / f).exists()), None)
    lon_file = next((geom_reference_path / f for f in lon_candidates if (geom_reference_path / f).exists()), None)
    if lat_file is None or lon_file is None:
        raise FileNotFoundError("lat/lon lookup file not found in geom_reference_path")
    return lat_file, lon_file


def crop_single_band_file(file_path: Path, y_min: int, y_max: int, x_min: int, x_max: int, out_file: Path) -> None:
    data = read_isce_file(file_path)
    data_height, data_width = data.shape[0], data.shape[1]
    y_max = min(y_max, data_height)
    x_max = min(x_max, data_width)
    data_crop = data[y_min:y_max, x_min:x_max, :]
    write_array_to_isce(data_crop, out_file)


def batch_crop_filt_fine_cor(config: CropConfig) -> list[Path]:
    config.output_base_path.mkdir(parents=True, exist_ok=True)

    lat_file, lon_file = _resolve_lat_lon_files(config.geom_reference_path)
    lat_data = read_isce_file(lat_file)
    lon_data = read_isce_file(lon_file)

    y_min, y_max, x_min, x_max = bbox_to_sar_indices(
        config.lat_min,
        config.lat_max,
        config.lon_min,
        config.lon_max,
        lat_data,
        lon_data,
    )

    write_array_to_isce(lat_data[y_min:y_max, x_min:x_max], config.output_base_path / "lat_cropped.rdr")
    write_array_to_isce(lon_data[y_min:y_max, x_min:x_max], config.output_base_path / "lon_cropped.rdr")

    cor_files = sorted(config.base_path.rglob("filt_fine.cor"))
    outputs: list[Path] = []
    for cor_file in cor_files:
        date_match = re.search(r"(\d{8}_\d{8})", str(cor_file))
        date_str = date_match.group(1) if date_match else cor_file.parent.name
        output_file = config.output_base_path / f"{date_str}_filt_fine.cor"
        if output_file.exists():
            outputs.append(output_file)
            continue
        crop_single_band_file(cor_file, y_min, y_max, x_min, x_max, output_file)
        outputs.append(output_file)

    return outputs
