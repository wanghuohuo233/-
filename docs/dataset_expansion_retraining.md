# 数据集扩展与保守重训方案

## 结论

原始 `data/c2dm_public_2d.json` 有 494 条 C2DM 结构，足够证明工程链路可以运行，但对扩散模型来说偏小。当前已经通过 ColabFit/Hugging Face 镜像下载 `JARVIS_C2DB`，转换出 3520 条 C2DB 配置；和原始 C2DM 合并后得到 3967 条结构，其中 1378 条完全落在当前 HER 生成器支持的元素表内，并已完成 GPU 扩展参数重训。

优先级不是直接堆更大的网络，而是：

1. 扩大公开二维材料数据。
2. 保留 GNN 条件扩散结构，只小幅增加 `timesteps`、`hidden_dim` 和训练轮数。
3. 用 QE/DFT 验证筛掉 surrogate 虚高的候选。

## 推荐公开数据源

| 数据源 | 规模 | 接入方式 | 用途 |
|---|---:|---|---|
| C2DM/C2DB 当前项目数据 | 494 | 已下载到 `data/c2dm.db` 并转成 JSON | 最小可复现实验 |
| JARVIS `dft_2d` | 1109 | `jarvis.db.figshare.data(dataset="dft_2d")` | 二维材料结构与 DFT 标签 |
| JARVIS `dft_2d_2021` | 1079 | `jarvis.db.figshare.data(dataset="dft_2d_2021")` | 备用二维材料集 |
| JARVIS `c2db` | 3514 | `jarvis.db.figshare.data(dataset="c2db")` | C2DB 镜像数据 |
| JARVIS `twod_matpd` | 6351 | `jarvis.db.figshare.data(dataset="twod_matpd")` | 2DMatPedia 扩展结构 |
| C2DB 2024 官方库 | 16789 | 2DHub/C2DB 页面导出或官方数据包 | 更大规模二维材料筛选 |

官方参考：

- JARVIS-Tools databases: https://pages.nist.gov/jarvis/databases/
- JARVIS-Tools example API: https://pages.nist.gov/jarvis/
- C2DB official page: https://www.2dhub.org/c2db/c2db.html

## 已做的项目改动

- `dataset/prepare_public_dataset.py` 增加 `--source jarvis` 和 `--jarvis-dataset`。
- `scripts/merge_material_datasets.py` 增加多数据集去重合并。
- `train_torch.py` 增加可调模型参数：`--graph-hidden-dim`、`--denoiser-hidden-dim`、`--timesteps`、`--lr`、`--guidance-scale` 等。
- `requirements.txt` 增加 `jarvis-tools`。

## JARVIS/C2DB 转换命令

所有缓存都放在 F 盘项目目录：

```powershell
cd "F:\机器学习面试\her_diffusion_2d_materials"

$env:TEMP="F:\机器学习面试\tmp"
$env:TMP="F:\机器学习面试\tmp"
$env:PIP_CACHE_DIR="F:\机器学习面试\pip_cache"

.\.venv\Scripts\python.exe dataset\prepare_public_dataset.py `
  --source jarvis `
  --jarvis-dataset dft_2d `
  --store-dir data\jarvis_cache `
  --output data\jarvis_dft_2d.json `
  --max-entries 2000

.\.venv\Scripts\python.exe dataset\prepare_public_dataset.py `
  --source jarvis `
  --jarvis-dataset c2db `
  --store-dir data\jarvis_cache `
  --output data\jarvis_c2db.json `
  --max-entries 4000
```

如果当前网络对 Figshare 返回 `403 Forbidden`，就在浏览器下载官方 zip 后放入 `data\jarvis_cache`。`jarvis-tools` 会优先读取已经存在的本地 zip。

## 合并数据

```powershell
.\.venv\Scripts\python.exe scripts\merge_material_datasets.py `
  --inputs data\c2dm_public_2d.json data\jarvis_dft_2d.json data\jarvis_c2db.json `
  --output data\expanded_2d_materials.json `
  --metadata data\expanded_2d_materials_metadata.json
```

## 保守重训参数

当前 494 条数据下不建议继续放大模型；扩展到 1000 条以上后，建议用以下参数重训：

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
  --lr 8e-4 `
  --guidance-scale 0.035 `
  --checkpoint checkpoints\torch_gnn_diffusion_expanded.pt
```

参数解释：

- `graph-hidden-dim 96 -> 128`：小幅增强 GNN 表达能力。
- `denoiser-hidden-dim 256 -> 320`：小幅增强扩散去噪器，不直接翻倍。
- `timesteps 100 -> 140`：增加生成平滑度。
- `epochs 260 -> 420`：扩大数据后多训练一些轮数。
- `lr 1e-3 -> 8e-4`：放大模型后略降学习率，减少震荡。
- `guidance-scale 0.04 -> 0.035`：降低后处理牵引，减少模式坍缩。

## 本机状态

本机已经安装 `jarvis-tools`、`pandas` 和 `pyarrow` 到项目虚拟环境：

```text
F:\机器学习面试\her_diffusion_2d_materials\.venv
```

直接访问 JARVIS Figshare 时，Python 请求返回 `403 Forbidden`。因此本机使用 ColabFit/Hugging Face 的公开镜像：

```text
https://huggingface.co/datasets/colabfit/JARVIS_C2DB
```

已下载文件：

```text
data\hf_colabfit_jarvis_c2db\README.md
data\hf_colabfit_jarvis_c2db\ds.parquet
data\hf_colabfit_jarvis_c2db\co\co_0.parquet
```

已转换和合并：

```text
data\jarvis_c2db_colabfit_3520.json       # 3520 条
data\expanded_2d_materials.json           # 3967 条，完整合并
data\expanded_2d_materials_supported.json # 1378 条，当前模型支持元素子集
```

扩展参数重训已完成：

```text
checkpoint: checkpoints\torch_gnn_diffusion_expanded.pt
outputs: results_expanded\
dataset records: 1378
epochs: 420
batch size: 96
graph hidden dim: 128
denoiser hidden dim: 320
timesteps: 140
learning rate: 8e-4
final loss: 0.099830
```
