# Notebook Decomposition and Consolidation Guide

## Objective
This document defines a structured refactoring strategy for transforming three monolithic notebooks (`Part1`, `Part2`, and `Part3`) into:
1. A modular Python package containing reusable, testable components; and
2. A single orchestration notebook that retains only explanatory narrative, parameter configuration, and concise function invocations.

The central design principle is to separate **computational logic** from **experimental narration** so that maintenance, reproducibility, and collaborative development are substantially improved.

## Architectural Decomposition

### 1. Low-level I/O and format utilities
- **File:** `insar_pipeline/io_utils.py`
- **Responsibilities:**
  - Reading ISCE/GDAL-compatible raster products;
  - Geographic bounding-box to SAR index conversion;
  - Writing NumPy arrays back to ISCE/ENVI formats.

### 2. Preprocessing and cropping (former Part 1)
- **File:** `insar_pipeline/preprocess.py`
- **Responsibilities:**
  - Resolving latitude/longitude lookup rasters;
  - Batch cropping of `filt_fine.cor` products;
  - Exporting cropped geometry rasters (`lat_cropped.rdr`, `lon_cropped.rdr`).

### 3. Dataset construction (former Part 1)
- **File:** `insar_pipeline/dataset_builder.py`
- **Responsibilities:**
  - Chronological discovery of `*_filt_fine.cor` files;
  - Construction of `data.npy`, `dates.pkl`, and `geninue.npy`;
  - Generation of uncertainty-derived products (`data_std.npy`, `geninue_std.npy`).

### 4. Learning and forecasting (former Part 2)
- **File:** `insar_pipeline/modeling.py`
- **Responsibilities:**
  - Definition of `InSARDataset` and `InSARLSTM`;
  - Model training, validation, checkpointing, and inference;
  - Export of forecast output (`future_predictions.npy`).

### 5. Score generation (former Part 2)
- **File:** `insar_pipeline/scoring.py`
- **Responsibilities:**
  - Normalized difference computation between reference and prediction;
  - Export of `score.npy`.

### 6. Product generation and geospatial export (former Part 3)
- **File:** `insar_pipeline/output_products.py`
- **Responsibilities:**
  - Conversion from `.npy` to `.cor`;
  - Invocation of `geocode.py`, `subset.py`, and `save_gdal.py`;
  - Production of geocoded outputs and GeoTIFF deliverables.

### 7. End-to-end orchestration
- **File:** `insar_pipeline/pipeline.py`
- **Responsibility:**
  - Sequential composition of all pipeline stages through `run_full_pipeline(...)`.

## Recommended Execution Commands

### A. Stage-wise execution (recommended for debugging and method development)
```bash
python - <<'PY'
from pathlib import Path
from insar_pipeline import (
    CropConfig, batch_crop_filt_fine_cor,
    DatasetConfig, build_and_save_dataset,
    TrainingConfig, run_training_and_prediction,
    ScoreConfig, compute_and_save_score,
    OutputConfig, generate_geocoded_outputs,
)

base = Path('/data6/WORKDIR/AmatriceSenDT22/merged/interferograms')
geom = Path('/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference')

batch_crop_filt_fine_cor(
    CropConfig(base_path=base, geom_reference_path=geom, output_base_path=base/'cropped')
)

dataset_dir = build_and_save_dataset(
    DatasetConfig(cropped_dir=base/'cropped', output_dir=base/'cropped')
)

predict_dir = run_training_and_prediction(
    TrainingConfig(dataset_dir=dataset_dir, output_dir=base/'cropped', next_date='20160821_20160902')
)

compute_and_save_score(
    ScoreConfig(dataset_dir=dataset_dir, predict_dir=predict_dir)
)

generate_geocoded_outputs(
    OutputConfig(
        predict_dir=predict_dir,
        lat_file=base/'cropped'/'lat_cropped.rdr',
        lon_file=base/'cropped'/'lon_cropped.rdr',
    )
)
PY
```

### B. One-command execution (recommended after validation)
```bash
python - <<'PY'
from pathlib import Path
from insar_pipeline import run_full_pipeline

result = run_full_pipeline(
    base_dir=Path('/data6/WORKDIR/AmatriceSenDT22/merged/interferograms'),
    geom_reference_dir=Path('/data6/WORKDIR/AmatriceSenDT22/merged/geom_reference'),
    next_date='20160821_20160902',
)
print(result)
PY
```

## Best Practices for Ongoing Development
- Use `InSAR_End_to_End_Workflow.ipynb` for experiment narration, parameter adjustment, and lightweight visualization.
- Implement all algorithmic modifications in `insar_pipeline/*.py` rather than reintroducing long logic blocks in notebooks.
- Keep notebook outputs minimal to reduce version-control noise and improve reviewability.
- As a next step, externalize hyperparameters (e.g., epochs, batch size, spatial bounds) into a YAML configuration for reproducible experiments.
