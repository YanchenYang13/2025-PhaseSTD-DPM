 # Damage Proxy Mapping with Time Series Prediction of InSAR Phase Standard Deviation
 
 ## Overview
 
-This project focuses on generating **Damage Proxy Maps (DPMs)** using time series data derived from **Interferometric Synthetic Aperture Radar (InSAR)**. By leveraging predictions of InSAR phase standard deviation (STD) through a Long Short-Term Memory (LSTM) model, the project identifies surface displacements post-earthquake, which serve as proxies for assessing the extent of damage. The workflow involves preprocessing multi-temporal SAR images, constructing time series data, training a model for phase STD prediction, and calculating a damage score to map areas of significant change.
+This project generates **Damage Proxy Maps (DPMs)** from multi-temporal **InSAR** observations by modeling phase-derived time-series behavior and highlighting post-event anomalies.
+
+At a high level, the method:
+- builds pixel-wise time-series from interferometric products,
+- learns temporal patterns with an LSTM-based predictor,
+- estimates expected post-event behavior under a no-disaster baseline,
+- compares predicted and observed post-event signals,
+- and exports geocoded outputs for mapping/interpretation.
+
+Compared with earlier notebook-centric workflows, this repository now provides a modular Python package (`insar_pipeline`) with a step-wise CLI and reusable modules for crop → dataset → train/predict → score → output.
 
 ---
 
-## Input and Output
+## Scientific/Method Context
+
+The DPM logic in this repo follows the same core idea used in the earlier workflow documentation, with additional implementation flexibility:
+
+1. **Preprocessing & Interferometric Inputs**
+   - Start from co-registered multi-temporal SAR data and interferometric products.
+   - Crop region-of-interest products and geolocation rasters (`lat/lon`) for downstream consistency.
+
+2. **Time-Series Construction**
+   - Assemble chronological per-pixel sequences from interferogram-derived coherence/phase-STD-like signals.
+   - Include temporal ordering and date features used by the prediction model.
+
+3. **Prediction Baseline (LSTM)**
+   - Train an LSTM-based predictor on pre-event temporal behavior.
+   - Produce expected post-event values under a no-disaster baseline.
+
+4. **Damage Score Computation**
+   - Compare observed vs predicted post-event values.
+   - Use normalized-difference-style scoring to emphasize anomalous changes.
+
+5. **Geocoding & Export**
+   - Convert prediction/score products to geocoded outputs.
+   - Subset to target bounding boxes and export final map products (e.g., GeoTIFF).
+
+---
 
-The overall procedure for generating the Damage Proxy Map follows a clear input–output structure:
+## Input and Output Summary
 
 ### Input
 
 | Stage | Input Data | Description |
 |-------|------------|-------------|
-| Interferogram Generation | Registered multi-temporal SAR images | Co-registered Sentinel-1 SAR image stack covering pre- and post-disaster periods |
-| Time Series Construction | Sequential interferograms | Pixel-wise phase STD images derived from adjacent SAR image pairs |
-| Prediction Model | Pre-disaster phase STD sequences + Encoded timestamps | Chronologically arranged phase STD values with temporal features (year, month, day) |
+| Interferogram Generation | Registered multi-temporal SAR images | Co-registered Sentinel-1 stack covering pre/post event periods |
+| Time Series Construction | Sequential interferograms | Pixel-wise interferogram-derived signals from adjacent date pairs |
+| Prediction Model | Pre-event sequences + timestamps | Chronologically arranged values with temporal context |
 
 ### Output
 
 | Stage | Output Data | Description |
 |-------|-------------|-------------|
-| Time Series Prediction | Hypothetical post-disaster phase STD | Predicted phase STD under the assumption that no disaster occurred |
-| Damage Mapping | Continuous Damage Proxy Map (DPM) | Pixel-wise damage score derived from the contrast between predicted and observed post-disaster phase STD |
-| Thresholding | Masked damage map | Binary or classified map for rapid delineation of impacted areas |
+| Time Series Prediction | Hypothetical post-event signal | Predicted no-disaster baseline |
+| Damage Mapping | Continuous DPM score | Pixel-wise contrast between predicted and observed post-event values |
+| Final Delivery | Geocoded subset products | GIS-ready outputs for interpretation/analysis |
 
-### Workflow Diagram
+---
 
-```
-┌─────────────────────────────────────────────────────────────────────────────┐
-│                              INPUT                                          │
-│  ┌─────────────────────────────────────────────────────────────────────┐   │
-│  │  Registered Multi-temporal SAR Images                                │   │
-│  └─────────────────────────────────────────────────────────────────────┘   │
-└─────────────────────────────────────────────────────────────────────────────┘
-                                    │
-                                    ▼
-                    ┌───────────────────────────────┐
-                    │  Sequential Interferograms    │
-                    │  (Adjacent Image Pairs)       │
-                    └───────────────────────────────┘
-                                    │
-                                    ▼
-                    ┌───────────────────────────────┐
-                    │  Pixel-wise Phase STD Images  │
-                    └───────────────────────────────┘
-                                    │
-                    ┌───────────────┴───────────────┐
-                    │                               │
-                    ▼                               ▼
-    ┌───────────────────────────┐   ┌───────────────────────────┐
-    │  Pre-disaster Phase STD   │   │  Post-disaster Phase STD  │
-    │  Sequences + Timestamps   │   │  (Observed)               │
-    └───────────────────────────┘   └───────────────────────────┘
-                    │                               │
-                    ▼                               │
-    ┌───────────────────────────┐                   │
-    │  LSTM Prediction Model    │                   │
-    └───────────────────────────┘                   │
-                    │                               │
-                    ▼                               │
-    ┌───────────────────────────┐                   │
-    │  Predicted Phase STD      │                   │
-    │  (No-disaster Assumption) │                   │
-    └───────────────────────────┘                   │
-                    │                               │
-                    └───────────────┬───────────────┘
-                                    │
-                                    ▼
-                    ┌───────────────────────────────┐
-                    │  Contrast / Comparison        │
-                    └───────────────────────────────┘
-                                    │
-                                    ▼
-┌─────────────────────────────────────────────────────────────────────────────┐
-│                              OUTPUT                                         │
-│  ┌─────────────────────────────────────────────────────────────────────┐   │
-│  │  Continuous Damage Proxy Map (DPM)                                   │   │
-│  └─────────────────────────────────────────────────────────────────────┘   │
-│                                    │                                        │
-│                                    ▼                                        │
-│  ┌─────────────────────────────────────────────────────────────────────┐   │
-│  │  Masked Damage Map (Thresholded)                                     │   │
-│  └─────────────────────────────────────────────────────────────────────┘   │
-└─────────────────────────────────────────────────────────────────────────────┘
-```
+## Current Repository Capabilities (GitHub Codebase)
+
+The `insar_pipeline` package currently supports:
+
+- **Two data ingestion modes**
+  - `input_source='cor'`: read coherence directly from `.cor`
+  - `input_source='stack_int'`: read ISCE stack interferograms (`.int`) and derive coherence
+
+- **Multiple coherence paths for stack inputs**
+  - `coherence_source='isce'`
+  - `coherence_source='computed_phsig'`
+  - `coherence_source='computed_crlb'`
+
+- **Step-wise CLI execution**
+  - `load_data | crop | build_dataset | train_predict | score | output | full`
+
+- **End-to-end orchestration**
+  - Programmatic `run_full_pipeline(...)` helper in `insar_pipeline/pipeline.py`
+
+- **Output chain**
+  - Generates score products, geocodes via MintPy tools, subsets, then exports raster outputs.
+
+---
+
+## Package Structure
+
+- `insar_pipeline/app.py` — CLI entrypoint and argument parsing
+- `insar_pipeline/pipeline.py` — high-level full workflow orchestrator
+- `insar_pipeline/preprocess.py` — cropping and target file collection
+- `insar_pipeline/dataset_builder.py` — observation collection and dataset serialization
+- `insar_pipeline/isce_stack.py` — stack pair discovery and `.int` access helpers
+- `insar_pipeline/coherence.py` — coherence estimation/mapping utilities
+- `insar_pipeline/io_utils.py` — raster read/write + bbox/index helpers
+- `insar_pipeline/modeling.py` — LSTM dataset/model/training/prediction
+- `insar_pipeline/scoring.py` — score computation
+- `insar_pipeline/output_products.py` — geocoded output production
 
 ---
 
-## Key Features
+## CLI Usage (Step-Wise)
+
+```bash
+python -m insar_pipeline.app -h
+```
 
-1. **Data Preprocessing**:
+### 1) Crop
 
-   * The **Sentinel-1 SAR data** undergoes preprocessing steps including **baseline estimation**, **multi-looking**, **image registration**, **filtering**, and **geocoding**. These steps prepare the raw SAR data for interferometric processing and subsequent analysis.
-   * **Interferometric processing** is performed to calculate the pixel-level **phase standard deviation (STD)** for each interferogram formed from adjacent image pairs.
+```bash
+python -m insar_pipeline.app --step crop \
+  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
+  --geom-reference-dir /data6/WORKDIR/AmatriceSenDT22/merged/geom_reference
+```
 
-2. **InSAR Time Series Construction**:
+### 2) Build dataset (CRLB example)
 
-   * A time series is constructed by arranging the phase STD values chronologically for each pixel. This series, along with relevant temporal features (e.g., year, month, day), serves as input for the **LSTM network**.
-   * The **LSTM model** is trained to predict phase STD values under an assumed **post-disaster scenario**, simulating conditions as if no disaster had occurred. This provides a baseline for detecting anomalies caused by the disaster.
+```bash
+python -m insar_pipeline.app --step build_dataset \
+  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
+  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
+  --input-source stack_int \
+  --stack-root /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
+  --coherence-source computed_crlb
+```
 
-3. **Damage Score Calculation**:
+### 3) Train & predict
 
-   * The **damage score** is constructed by comparing the **LSTM-predicted phase STD** with the observed **post-disaster phase STD**. This score quantifies the extent of displacement or change and acts as an indicator of damage.
-   * **Normalized difference** calculations are used to highlight discrepancies between predicted and observed values, helping identify areas of potential damage.
+```bash
+python -m insar_pipeline.app --step train_predict \
+  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
+  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped
+```
 
-4. **Geocoding and Subsetting**:
+### 4) Score
 
-   * After generating the damage scores, the data is **geocoded** using the `geocode.py` tool to associate it with geographic coordinates (latitude and longitude) using mintpy (https://github.com/insarlab/MintPy).
-   * A **subset extraction** is then performed, focusing on regions of interest, and the results are saved as **GeoTIFF** files for visualization and further analysis.
+```bash
+python -m insar_pipeline.app --step score \
+  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
+  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped
+```
 
-5. **Damage Proxy Mapping (DPM)**:
+### 5) Output (geocode/subset/export)
 
-   * The final output includes high-precision **damage proxy maps** (DPMs) that visualize surface displacement and standard deviation, enabling disaster response teams to efficiently assess the damage.
+```bash
+python -m insar_pipeline.app --step output \
+  --base-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms \
+  --output-dir /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped \
+  --lat-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lat_cropped.rdr \
+  --lon-file /data6/WORKDIR/AmatriceSenDT22/merged/interferograms/cropped/lon_cropped.rdr
+```
 
 ---
 
-## Project Workflow
+## Notebooks in This Repository
 
-https://github.com/YanchenYang13/PhaseSigma-TS-LSTM/blob/main/Part1_Input_Dataset_Construction.ipynb
+- `CRLB_InSAR_Workflow_Tutorial.ipynb`  
+  English tutorial focused on the CRLB path, with visualization for each generated artifact (including `.npy` via `matplotlib`).
 
-The first step in the workflow involves the collection and preprocessing of multi-temporal SAR images, which includes baseline estimation, multi-looking, image registration, filtering, and geocoding. This is performed using the `stackSentinel.py` script (https://github.com/isce-framework/isce2/blob/main/contrib/stack/topsStack/stackSentinel.py), which automates the entire process:
+- `Part1_Input_Dataset_Construction.ipynb`  
+  Input and dataset construction reference.
 
-* **SAR Image Collection**: Multiple Sentinel-1 SAR images are collected for the desired time period.
-* **Preprocessing**: This includes operations such as multi-looking to reduce the speckle noise, baseline estimation to determine the geometric relationship between the image pairs, and image registration to align the images for interferometric analysis.
-* **Geocoding**: The SAR data is geocoded to associate the interferogram with geographic coordinates, allowing for easier interpretation in GIS tools.
+- `Part2_Prediction_DPM_Generation.ipynb`  
+  Prediction and DPM generation reference.
 
-After preprocessing, **interferometric processing** is applied to the SAR data. The **pixel-level phase STD** for each interferogram formed from adjacent image pairs is calculated. The phase STD values are then arranged chronologically to construct an **InSAR time series** for each pixel.
+- `Part3_Output.ipynb`  
+  Output/geocoding reference.
 
-* The time series includes both **pre-disaster** and **post-disaster** phase STD data, with temporal features such as the year, month, and day incorporated into the input for the LSTM model.
+---
+
+## Detailed Workflow Notes (Legacy Explanation + Current Implementation)
 
-https://github.com/YanchenYang13/PhaseSigma-TS-LSTM/blob/main/Part2_Prediction_DPM_Generation.ipynb
+### A) Data Preparation
 
-An **LSTM neural network** is used to model the temporal relationships in the InSAR data. The model is trained on the phase STD values and their associated temporal features to predict the phase STD under a **post-disaster scenario** (i.e., simulating conditions as if no disaster had occurred). This predicted phase STD represents the expected surface displacement under normal conditions.
+- Multi-temporal SAR data are processed into interferometric products (typically through external InSAR stack tooling such as ISCE topsStack pipelines).
+- The repo then crops relevant interferometric files and geolocation rasters to a target AOI.
+- Cropped `lat/lon` rasters are used by the geocode stage.
 
-* **Training the LSTM Model**: The LSTM model learns to predict the future phase STD values, which are then compared to the observed post-disaster values to construct the damage score.
+### B) Time-Series Modeling
 
-The **damage score** is derived by comparing the **predicted phase STD** (from the LSTM model) with the **observed post-disaster phase STD**. The normalized difference score is used to highlight discrepancies, where large differences indicate significant changes or damage.
+- The dataset builder serializes tensors such as `data.npy`, `data_std.npy`, `geninue.npy`, `geninue_std.npy`, and `dates.pkl`.
+- The modeling module trains an LSTM baseline and produces `future_predictions.npy`.
 
-* **Normalized Difference**: The normalized difference score is calculated for each pixel in the interferogram, providing a quantitative measure of damage.
+### C) Damage Proxy Scoring
 
-https://github.com/YanchenYang13/PhaseSigma-TS-LSTM/blob/main/Part3_Output.ipynb
+- The score module computes a normalized contrast between observed and predicted post-event maps and writes `score.npy`.
 
-Once the damage scores are calculated, the next step involves **geocoding** the data using latitude and longitude files, ensuring that each data point is associated with geographic coordinates.
+### D) Geospatial Output Generation
 
-* **Subset Extraction**: Geographic subsets are extracted based on the desired bounding box coordinates, focusing on regions of interest.
-* **Saving as GeoTIFF**: The subsetted data is saved as **GeoTIFF files** for visualization and further analysis in GIS applications.
+- The output module writes intermediate coherence-like rasters, invokes geocoding/subsetting tools, and exports final map files.
+- This stage depends on your MintPy/ISCE runtime setup.
 
+---
 
-The final step is the generation of **damage proxy maps (DPMs)**, which visualize the damage score and surface displacement across the region of interest. These maps serve as important tools for disaster response, allowing for a visual representation of the areas most affected by the disaster.
+## Typical Artifact Locations
 
-* **Visualization**: The damage proxy maps are visualized using **Matplotlib** and can be saved as images or GeoTIFF files for GIS analysis.
+- Cropped files and geometry: `<output_dir>/` (e.g., `lat_cropped.rdr`, `lon_cropped.rdr`)
+- Dataset artifacts: `<output_dir>/dataset/`
+- Prediction and score artifacts: `<output_dir>/predict/`
 
 ---
 
-## File Structure
+## Environment Requirements
 
-```
-/WORKDIR/
-├── merged/
-│   ├── interferograms/
-│   │   ├── cropped/    # Contains cropped coherence files
-│   │   ├── predict/    # Contains prediction results
-│   │   ├── dataset/    # Contains pre-processed InSAR time series data
-│   │   └── geom_reference/  # Geometry reference files (lat, lon, etc.)
-│   └── dataset/        # Final processed InSAR time series dataset
-```
+Common dependencies include:
+- `numpy`, `matplotlib`, `torch`
+- `GDAL` (`osgeo.gdal`)
+- MintPy utilities (`geocode.py`, `subset.py`, `save_gdal.py`, `mintpy.utils.writefile`)
+- ISCE-compatible inputs for stack workflows
+
+If these tools are missing, run the pipeline inside your configured InSAR/MintPy environment.
+
+---
+
+## Acknowledgement
+
+If this repository is used for operational or publication workflows, please add a project license and include required attribution for data sources and upstream toolchains.
