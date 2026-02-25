# Notebook 拆分与合并指南

## 目标
- 将原本体积较大的 3 个 `ipynb`（Part1/Part2/Part3）中“可复用逻辑”迁移到 Python 模块。
- Notebook 只保留：流程说明、参数设置、函数调用。
- 最终统一为单个入口 Notebook：`InSAR_End_to_End_Workflow.ipynb`。

## 拆分方案

### 1. I/O 与底层格式读写
- 文件：`insar_pipeline/io_utils.py`
- 责任：
  - ISCE/GDAL 文件读取
  - bbox -> SAR 索引转换
  - numpy -> ISCE/ENVI 写入

### 2. Part1：裁剪预处理
- 文件：`insar_pipeline/preprocess.py`
- 责任：
  - 自动定位 `lat/lon` 文件
  - 批量裁剪 `filt_fine.cor`
  - 输出 `lat_cropped.rdr`, `lon_cropped.rdr`

### 3. Part1：数据集构建
- 文件：`insar_pipeline/dataset_builder.py`
- 责任：
  - 按日期排序读取 `*_filt_fine.cor`
  - 构建 `data.npy / dates.pkl / geninue.npy`
  - 计算 `data_std.npy / geninue_std.npy`

### 4. Part2：训练与预测
- 文件：`insar_pipeline/modeling.py`
- 责任：
  - `InSARDataset`、`InSARLSTM`
  - 训练、验证、加载最佳模型
  - 预测并输出 `future_predictions.npy`

### 5. Part2：评分
- 文件：`insar_pipeline/scoring.py`
- 责任：
  - 归一化差异计算
  - 输出 `score.npy`

### 6. Part3：输出产品
- 文件：`insar_pipeline/output_products.py`
- 责任：
  - `.npy -> .cor`
  - 调用 `geocode.py / subset.py / save_gdal.py`
  - 输出地理编码与 GeoTIFF

### 7. 一键编排
- 文件：`insar_pipeline/pipeline.py`
- 责任：
  - 串联完整流程 `run_full_pipeline(...)`

## 推荐执行指令

### A. 分步骤（便于调试）
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

batch_crop_filt_fine_cor(CropConfig(base_path=base, geom_reference_path=geom, output_base_path=base/'cropped'))
dataset_dir = build_and_save_dataset(DatasetConfig(cropped_dir=base/'cropped', output_dir=base/'cropped'))
predict_dir = run_training_and_prediction(TrainingConfig(dataset_dir=dataset_dir, output_dir=base/'cropped', next_date='20160821_20160902'))
compute_and_save_score(ScoreConfig(dataset_dir=dataset_dir, predict_dir=predict_dir))
generate_geocoded_outputs(OutputConfig(predict_dir=predict_dir, lat_file=base/'cropped'/'lat_cropped.rdr', lon_file=base/'cropped'/'lon_cropped.rdr'))
PY
```

### B. 一键执行
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

## Notebook 使用建议
- 日常实验：用 `InSAR_End_to_End_Workflow.ipynb` 做参数管理与可视化。
- 逻辑改动：只改 `insar_pipeline/*.py`，不要在 Notebook 重复粘贴长函数。
- 版本控制：Notebook 尽量只保留少量可读输出，减少 diff 噪声。
