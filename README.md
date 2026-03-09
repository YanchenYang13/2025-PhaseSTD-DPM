# Damage Proxy Mapping with InSAR Time-Series Prediction

This repository provides a modular **InSAR damage proxy mapping (DPM)** workflow built around a unified Python package: `insar_pipeline`.

It supports both:
- classic coherence-based inputs (`.cor`), and
- stack-based inputs (`.int`) with optional coherence estimation using either **ISCE-like phsig** or **CRLB-inspired** mappings.

The end goal is to generate geocoded damage proxy products (e.g., GeoTIFF) from time-series modeling of phase-derived signals.

---

## What this project does

The workflow covers:
1. **Cropping** interferograms and geometry lookup rasters (`lat/lon`).
2. **Dataset construction** for time-series learning.
3. **Model training + prediction** (LSTM-based baseline implementation).
4. **Score generation** from observed vs predicted phase-STD-like products.
5. **Geocoding/subsetting/export** to final map products.

---

## Package layout

Main modules under `insar_pipeline/`:

- `app.py`: CLI entrypoint with step-wise and full-run modes.
- `pipeline.py`: high-level orchestration (`run_full_pipeline`).
- `preprocess.py`: cropping utilities.
- `dataset_builder.py`: data collection and dataset serialization.
- `coherence.py`: coherence estimation from interferometric inputs.
- `isce_stack.py`: ISCE stack discovery/read helpers.
- `io_utils.py`: raster I/O, bbox indexing, ISCE/GDAL write helpers.
- `modeling.py`: LSTM training and prediction.
- `scoring.py`: score computation.
- `output_products.py`: geocoded output generation.

---

## CLI quick start

Run help:

```bash
python -m insar_pipeline.app -h
```

Available steps:

- `load_data`
- `crop`
- `build_dataset`
- `train_predict`
- `score`
- `output`
- `full`

Example (step-wise):

```bash
# 1) Crop
python -m insar_pipeline.app --step crop \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference

# 2) Build dataset from stack .int using CRLB coherence
python -m insar_pipeline.app --step build_dataset \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --input-source stack_int \
  --stack-root /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --coherence-source computed_crlb

# 3) Train + predict
python -m insar_pipeline.app --step train_predict \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped

# 4) Score
python -m insar_pipeline.app --step score \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped

# 5) Output (geocode/subset/save)
python -m insar_pipeline.app --step output \
  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
  --lat-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lat_cropped.rdr \
  --lon-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lon_cropped.rdr
```

---

## Notebooks

This repository currently includes:

- `CRLB_InSAR_Workflow_Tutorial.ipynb`  
  English tutorial for the **CRLB path** with per-stage artifact visualization (`.npy` plots and raster previews).

- `InSAR_End_to_End_Workflow.ipynb`  
  End-to-end workflow demonstration.

- `Part1_Input_Dataset_Construction.ipynb`  
  Dataset/input construction reference notebook.

- `Part2_Prediction_DPM_Generation.ipynb`  
  Prediction and DPM generation reference notebook.

- `Part3_Output.ipynb`  
  Output/geocoding reference notebook.

---

## Important environment dependencies

This project depends on geospatial and InSAR tooling. Typical requirements include:

- Python scientific stack (`numpy`, `matplotlib`, `torch`, etc.)
- GDAL (`osgeo.gdal`)
- MintPy utilities (`geocode.py`, `subset.py`, `save_gdal.py`, `mintpy.utils.writefile`)
- ISCE-compatible data products for stack workflows

Ensure these tools are available in your runtime environment before running `--step output`.

---

## Notes on data products

- Cropped geometry files (`lat_cropped.rdr`, `lon_cropped.rdr`) are generated for geocoding.
- Prediction artifacts are saved under `<output_dir>/predict` (e.g., `future_predictions.npy`, `score.npy`).
- Dataset artifacts are saved under `<output_dir>/dataset` (e.g., `data.npy`, `data_std.npy`, `dates.pkl`, `geninue*.npy`).

---

## License / attribution

Please add your project license and any required data/tool attribution policies if this repository is used in production or publication workflows.
