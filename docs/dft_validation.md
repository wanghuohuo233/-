# DFT / 声子 / AIMD 验证接口

项目已补充 Quantum ESPRESSO 输入文件生成器，并且已经在 WSL + conda-forge QE 7.5 环境中完成了一轮真实 QE 验证。实算结果见 [dft_validation_results.md](dft_validation_results.md)，机器可读汇总见 [results/dft_validation_summary.json](../results/dft_validation_summary.json)。

生成 QE 输入文件：

```powershell
cd "F:\机器学习面试\her_diffusion_2d_materials"
.\.venv\Scripts\python.exe validation\qe_workflow.py `
  --materials results\generated_materials.json `
  --output-dir validation_inputs\qe `
  --top-k 5
```

每个候选会生成：

- `01_relax.in`：二维材料结构弛豫
- `02_scf.in`：弛豫后 SCF 能量
- `03_h_ads_relax.in`：顶部位点吸氢结构弛豫，用于估计 HER `ΔG_H`
- `04_gamma_phonon.in`：Gamma 点声子稳定性检查
- `05_aimd_300K.in`：300K 短程 AIMD 模板

如果目标机器已经安装 Quantum ESPRESSO：

```powershell
.\.venv\Scripts\python.exe validation\qe_workflow.py --run
```

## HER 真实计算公式

```text
ΔG_H = ΔE_H + ΔE_ZPE - TΔS_H
ΔE_H = E(surface + H) - E(surface) - 1/2 E(H2)
```

经验近似常用：

```text
ΔG_H ≈ ΔE_H + 0.24 eV
```

真实版本需要额外计算：

1. 干净表面总能 `E(surface)`
2. 吸氢表面总能 `E(surface + H)`
3. 氢气分子总能 `E(H2)`
4. 零点能和熵修正

## 稳定性真实验证

热力学稳定：

```text
E_hull = E_candidate - convex_hull(composition)
```

动力学稳定：

- 声子谱无虚频
- AIMD 300K/500K 下结构不崩塌
- 必要时 NEB 计算关键扩散/反应路径势垒

当前代码中的 surrogate 指标用于机器学习闭环，不替代以上验证。

## 本机已完成的真实 QE 验证

本机可运行方案：

```text
WSL UbuntuQE + conda-forge qe=7.5
输出目录：validation_outputs/qe/
```

已经完成：

- H2 reference SCF：`E(H2) = -2.32383790 Ry`
- `candidate_01_NbNSe`：relax、SCF、H adsorption、Gamma phonon
- `candidate_NbS2`：relax、SCF、两个 H adsorption 位点、Gamma phonon
- `candidate_NbSSe`：relax、SCF、Se/S 两个 H adsorption 位点、Gamma phonon

关键结论：

| Candidate | Best approximate DeltaG_H | Stability |
|---|---:|---|
| NbNSe | 0.948 eV | 大虚频，拒绝 |
| NbS2 | 0.614 eV | Gamma 点稳定 |
| NbSSe | 0.567 eV | 小虚频，需更高精度复核 |

这说明当前 surrogate 模型可以生成合理二维结构，但真实 HER 标签仍需要 DFT 回填后再训练。
