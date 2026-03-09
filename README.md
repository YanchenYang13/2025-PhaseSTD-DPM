# Damage Proxy Mapping with Time Series Prediction of InSAR Phase Standard Deviation

## Overview

This project generates **Damage Proxy Maps (DPMs)** from multi-temporal **InSAR** observations by modeling phase-derived time-series behavior and highlighting post-event anomalies.

At a high level, the method:
- builds pixel-wise time-series from interferometric products,
- learns temporal patterns with an LSTM-based predictor,
- estimates expected post-event behavior under a no-disaster baseline,
- compares predicted and observed post-event signals,
- and exports geocoded outputs for mapping/interpretation.

Compared with earlier notebook-centric workflows, this repository now provides a modular Python package (`insar_pipeline`) with a step-wise CLI and reusable modules for crop → dataset → train/predict → score → output.

---

## Scientific/Method Context

The DPM logic in this repo follows the same core idea used in the earlier workflow documentation, with additional implementation flexibility:

1. **Preprocessing & Interferometric Inputs**
   - Start from co-registered multi-temporal SAR data and interferometric products.
   - Crop region-of-interest products and geolocation rasters (`lat/lon`) for downstream consistency.

2. **Time-Series Construction**
   - Assemble chronological per-pixel sequences from interferogram-derived coherence/phase-STD-like signals.
   - Include temporal ordering and date features used by the prediction model.

3. **Prediction Baseline (LSTM)**
   - Train an LSTM-based predictor on pre-event temporal behavior.
   - Produce expected post-event values under a no-disaster baseline.

4. **Damage Score Computation**
   - Compare observed vs predicted post-event values.
   - Use normalized-difference-style scoring to emphasize anomalous changes.

5. **Geocoding & Export**
   - Convert prediction/score products to geocoded outputs.
   - Subset to target bounding boxes and export final map products (e.g., GeoTIFF).

---

## Input and Output Summary

### Input

| Stage | Input Data | Description |
|-------|------------|-------------|
| Interferogram Generation | Registered multi-temporal SAR images | Co-registered Sentinel-1 stack covering pre/post event periods |
| Time Series Construction | Sequential interferograms | Pixel-wise interferogram-derived signals from adjacent date pairs |
| Prediction Model | Pre-event sequences + timestamps | Chronologically arranged values with temporal context |

### Output

| Stage | Output Data | Description |
|-------|-------------|-------------|
| Time Series Prediction | Hypothetical post-event signal | Predicted no-disaster baseline |
| Damage Mapping | Continuous DPM score | Pixel-wise contrast between predicted and observed post-event values |
| Final Delivery | Geocoded subset products | GIS-ready outputs for interpretation/analysis |

---

## Current Repository Capabilities (GitHub Codebase)

The `insar_pipeline` package currently supports:

- **Two data ingestion modes**
  - `input_source='cor'`: read coherence directly from `.cor`
  - `input_source='stack_int'`: read ISCE stack interferograms (`.int`) and derive coherence

- **Multiple coherence paths for stack inputs**
  - `coherence_source='isce'`
  - `coherence_source='computed_phsig'`
  - `coherence_source='computed_crlb'`

- **Step-wise CLI execution**
  - `load_data | crop | build_dataset | train_predict | score | output | full`

- **End-to-end orchestration**
  - Programmatic `run_full_pipeline(...)` helper in `insar_pipeline/pipeline.py`

- **Output chain**
  - Generates score products, geocodes via MintPy tools, subsets, then exports raster outputs.

---

## Package Structure

- `insar_pipeline/app.py` — CLI entrypoint and argument parsing
- `insar_pipeline/pipeline.py` — high-level full workflow orchestrator
- `insar_pipeline/preprocess.py` — cropping and target file collection
- `insar_pipeline/dataset_builder.py` — observation collection and dataset serialization
- `insar_pipeline/isce_stack.py` — stack pair discovery and `.int` access helpers
- `insar_pipeline/coherence.py` — coherence estimation/mapping utilities
- `insar_pipeline/io_utils.py` — raster read/write + bbox/index helpers
- `insar_pipeline/modeling.py` — LSTM dataset/model/training/prediction
- `insar_pipeline/scoring.py` — score computation
- `insar_pipeline/output_products.py` — geocoded output production

---

## CLI Usage (Step-Wise)

```bash
python -m insar_pipeline.app -h
```

### 1) Crop

```bash
python -m insar_pipeline.app --step crop \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference
```

### 2) Build dataset (CRLB example)

```bash
python -m insar_pipeline.app --step build_dataset \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --input-source stack_int \
  --stack-root /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --coherence-source computed_crlb
```

### 3) Train & predict

```bash
python -m insar_pipeline.app --step train_predict \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped
```

### 4) Score

```bash
python -m insar_pipeline.app --step score \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped
```

### 5) Output (geocode/subset/export)

```bash
python -m insar_pipeline.app --step output \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --lat-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lat_cropped.rdr \
  --lon-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lon_cropped.rdr
```

---

## Notebooks in This Repository

- `CRLB_InSAR_Workflow_Tutorial.ipynb`  
  English tutorial focused on the CRLB path, with visualization for each generated artifact (including `.npy` via `matplotlib`).

- `Part1_Input_Dataset_Construction.ipynb`  
  Input and dataset construction reference.

- `Part2_Prediction_DPM_Generation.ipynb`  
  Prediction and DPM generation reference.

- `Part3_Output.ipynb`  
  Output/geocoding reference.

---

## Detailed Workflow Notes (Legacy Explanation + Current Implementation)

### A) Data Preparation

- Multi-temporal SAR data are processed into interferometric products (typically through external InSAR stack tooling such as ISCE topsStack pipelines).
- The repo then crops relevant interferometric files and geolocation rasters to a target AOI.
- Cropped `lat/lon` rasters are used by the geocode stage.

### B) Time-Series Modeling

- The dataset builder serializes tensors such as `data.npy`, `data_std.npy`, `geninue.npy`, `geninue_std.npy`, and `dates.pkl`.
- The modeling module trains an LSTM baseline and produces `future_predictions.npy`.

### C) Damage Proxy Scoring

- The score module computes a normalized contrast between observed and predicted post-event maps and writes `score.npy`.

### D) Geospatial Output Generation

- The output module writes intermediate coherence-like rasters, invokes geocoding/subsetting tools, and exports final map files.
- This stage depends on your MintPy/ISCE runtime setup.

---

## Typical Artifact Locations

- Cropped files and geometry: `<output_dir>/` (e.g., `lat_cropped.rdr`, `lon_cropped.rdr`)
- Dataset artifacts: `<output_dir>/dataset/`
- Prediction and score artifacts: `<output_dir>/predict/`

---

## Environment Requirements

Common dependencies include:
- `numpy`, `matplotlib`, `torch`
- `GDAL` (`osgeo.gdal`)
- MintPy utilities (`geocode.py`, `subset.py`, `save_gdal.py`, `mintpy.utils.writefile`)
- ISCE-compatible inputs for stack workflows

If these tools are missing, run the pipeline inside your configured InSAR/MintPy environment.

---

## Acknowledgement

If this repository is used for operational or publication workflows, please add a project license and include required attribution for data sources and upstream toolchains.
