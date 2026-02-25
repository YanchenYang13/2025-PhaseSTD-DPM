from __future__ import annotations

import datetime as dt
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_utils import read_isce_file


@dataclass
class DatasetConfig:
    cropped_dir: Path
    output_dir: Path
    event_date: dt.datetime = dt.datetime(2016, 8, 24)
    data_mode: str = "amplitude"


def find_cor_files_sorted(cropped_dir: Path) -> list[tuple[dt.datetime, str, Path]]:
    file_infos = []
    for path in cropped_dir.rglob("*filt_fine.cor"):
        m = re.search(r"(\d{8}_\d{8})", path.name)
        if not m:
            continue
        date_str = m.group(1)
        start_str, _ = date_str.split("_")
        start_dt = dt.datetime.strptime(start_str, "%Y%m%d")
        file_infos.append((start_dt, date_str, path))
    return sorted(file_infos, key=lambda x: x[0])


def build_insar_timeseries_from_cor(
    file_infos: list[tuple[dt.datetime, str, Path]],
    use: str = "amplitude",
) -> tuple[np.ndarray, list[str]]:
    if len(file_infos) < 2:
        raise RuntimeError("Need at least 2 cor files to build timeseries and target.")

    first_data = read_isce_file(file_infos[0][2])
    h, w = first_data.shape[:2]
    t = len(file_infos)

    timeseries = np.zeros((h, w, t - 1), dtype=np.float32)
    dates: list[str] = []

    for i, (_, date_str, path) in enumerate(file_infos[:-1]):
        arr = read_isce_file(path)
        if np.iscomplexobj(arr):
            if use == "amplitude":
                arr_use = np.abs(arr)
            elif use == "phase":
                arr_use = np.angle(arr)
            elif use == "real":
                arr_use = np.real(arr)
            else:
                raise ValueError(f"Unknown mode: {use}")
        else:
            arr_use = arr

        timeseries[:, :, i] = arr_use[:, :, 0].astype(np.float32)
        dates.append(date_str)

    return timeseries, dates


def calculate_std_from_cor(cor: np.ndarray, chunk_size: int = 50) -> np.ndarray:
    rows, cols, bands = cor.shape
    result = np.full_like(cor, np.nan, dtype=np.float32)
    epsilon = 1e-8

    for i in range(0, rows, chunk_size):
        for j in range(0, cols, chunk_size):
            end_i = min(i + chunk_size, rows)
            end_j = min(j + chunk_size, cols)
            chunk = cor[i:end_i, j:end_j, :]
            denominator = chunk**2
            valid_mask = denominator > epsilon
            std_chunk = np.where(valid_mask, np.sqrt((1 - denominator) / (2 * denominator)), 0.0)
            result[i:end_i, j:end_j, :] = std_chunk

    result[np.isnan(cor)] = np.nan
    result[cor == 0] = 0.0
    return result


def save_dataset(output_subfolder: Path, timeseries: np.ndarray, dates: list[str], geninue_data: np.ndarray) -> None:
    output_subfolder.mkdir(parents=True, exist_ok=True)

    np.save(output_subfolder / "data.npy", timeseries)
    with open(output_subfolder / "dates.pkl", "wb") as f:
        pickle.dump(dates, f)

    np.save(output_subfolder / "geninue.npy", geninue_data)
    np.save(output_subfolder / "data_std.npy", calculate_std_from_cor(timeseries))

    if geninue_data.ndim == 2:
        geninue_data = np.expand_dims(geninue_data, axis=-1)
    np.save(output_subfolder / "geninue_std.npy", calculate_std_from_cor(geninue_data))


def build_and_save_dataset(config: DatasetConfig) -> Path:
    file_infos = find_cor_files_sorted(config.cropped_dir)
    train_file_infos = [info for info in file_infos if info[0] < config.event_date]

    timeseries, dates = build_insar_timeseries_from_cor(train_file_infos, use=config.data_mode)

    _, _, geninue_path = file_infos[-1]
    geninue_data = np.abs(read_isce_file(geninue_path)[:, :, 0])

    output_subfolder = config.output_dir / "dataset"
    save_dataset(output_subfolder, timeseries, dates, geninue_data)
    return output_subfolder
