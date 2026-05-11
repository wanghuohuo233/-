# 公开数据接入说明

本项目已经提供 `dataset/prepare_public_dataset.py`，用于把公开二维/晶体数据库整理成统一 JSON 格式。更详细的数据扩展和重训方案见 [dataset_expansion_retraining.md](dataset_expansion_retraining.md)。

## 推荐数据源

1. 2DMatPedia
   - 适合二维材料生成训练。
   - 下载结构数据后，用 `--source 2dmatpedia` 转换。

2. C2DB
   - 适合二维材料稳定性、电性、磁性等性质筛选。
   - 下载 JSON/CSV 后，用 `--source c2db` 转换。

3. Materials Project
   - 适合通用晶体结构学习。
   - 需要官方 API key：设置 `MP_API_KEY` 后运行 `--source materials-project`。

4. JARVIS / JARVIS-C2DB / 2DMatPedia mirror
   - 适合把训练集从 494 条扩展到 1000 到 6000 条以上。
   - 需要 `jarvis-tools`、`pandas`、`pyarrow`，已写入 `requirements.txt`。
   - 支持 `--source jarvis --jarvis-dataset dft_2d/c2db/twod_matpd`。
   - 本机已使用 ColabFit/Hugging Face 的 `JARVIS_C2DB` 镜像完成下载和重训。

## 转换命令

```powershell
cd "F:\机器学习面试\her_diffusion_2d_materials"

# 通用 JSON / JSONL 文件
.\.venv\Scripts\python.exe dataset\prepare_public_dataset.py `
  --input path\to\public_dataset.json `
  --source 2dmatpedia `
  --output data\public_2d_materials.json

# Materials Project API
$env:MP_API_KEY="your_api_key"
.\.venv\Scripts\python.exe dataset\prepare_public_dataset.py `
  --source materials-project `
  --output data\mp_2d_like_materials.json `
  --max-entries 1000

# JARVIS official Figshare datasets
.\.venv\Scripts\python.exe dataset\prepare_public_dataset.py `
  --source jarvis `
  --jarvis-dataset dft_2d `
  --store-dir data\jarvis_cache `
  --output data\jarvis_dft_2d.json `
  --max-entries 2000

# Merge for expanded retraining
.\.venv\Scripts\python.exe scripts\merge_material_datasets.py `
  --inputs data\c2dm_public_2d.json data\jarvis_dft_2d.json `
  --output data\expanded_2d_materials.json

# ColabFit/Hugging Face JARVIS_C2DB parquet mirror, already used in this run
.\.venv\Scripts\python.exe dataset\prepare_public_dataset.py `
  --input data\hf_colabfit_jarvis_c2db\co\co_0.parquet `
  --source c2db `
  --output data\jarvis_c2db_colabfit_3520.json `
  --max-entries 5000
```

## 训练

```powershell
.\.venv\Scripts\python.exe train.py --data data\public_2d_materials.json
```

扩展数据后的推荐重训命令：

```powershell
.\.venv\Scripts\python.exe train.py `
  --data data\expanded_2d_materials_supported.json `
  --epochs 420 `
  --batch-size 96 `
  --samples 240 `
  --device cuda `
  --graph-hidden-dim 128 `
  --denoiser-hidden-dim 320 `
  --timesteps 140 `
  --lr 8e-4
```

## 统一 JSON 格式

```json
{
  "name": "MoS2",
  "formula": "MoS2",
  "prototype": "MX2",
  "elements": ["Mo", "S", "S"],
  "positions": [[0, 0, 0], [1.59, 0.92, 1.56], [1.59, 0.92, -1.56]],
  "lattice": [[3.18, 0, 0], [-1.59, 2.75, 0], [0, 0, 18]],
  "properties": {
    "energy_above_hull": 0.01,
    "formation_energy_per_atom": -1.2
  }
}
```

如果公开数据已经包含真实 DFT 标签，可以把这些字段放进 `properties`。当前训练脚本会优先保留外部标签，并补充本项目的 surrogate 指标。
