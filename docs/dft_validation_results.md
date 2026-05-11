# QE 实算验证结果

## 运行环境

Windows 原生 MSYS2 版 QE 和 Ubuntu apt 版 QE 都在最小 H2 测试中崩溃。最终可稳定运行的方案是：

```text
WSL UbuntuQE + conda-forge qe=7.5
WSL 虚拟磁盘：F:\机器学习面试\wsl\UbuntuQE\ext4.vhdx
项目输出：F:\机器学习面试\her_diffusion_2d_materials\validation_outputs\qe
```

不可避免在 C 盘使用的是 Windows/WSL 系统组件，例如：

```text
C:\Windows\System32\wsl.exe
C:\Users\17654\AppData\Local\Programs\Python\Python311
```

这些不是项目运行数据；项目代码、虚拟环境、QE 输出、WSL VHD 和缓存主体都保存在 `F:\机器学习面试`。

## QE 设置

```text
Quantum ESPRESSO: conda-forge qe=7.5
赝势：PSLibrary RRKJUS UPF
ecutwfc / ecutrho: 35 / 280 Ry
k-points: 4x4x1
2D 处理：assume_isolated='2D'
H2 reference energy: -2.32383790 Ry
```

HER 近似公式：

```text
DeltaG_H ~= E(surface + H) - E(surface) - 1/2 E(H2) + 0.24 eV
```

## 实算结果

| Candidate | Best site | DeltaE_H (eV) | Approx DeltaG_H (eV) | Gamma phonon result | Conclusion |
|---|---:|---:|---:|---|---|
| NbNSe | top | 0.708 | 0.948 | large imaginary modes: -275.8 cm-1 | reject: dynamically unstable |
| NbS2 | S_top | 0.374 | 0.614 | all positive at Gamma | stable but HER weak |
| NbSSe | S_top | 0.327 | 0.567 | small negative acoustic-like modes: -9.0 cm-1 | borderline stable, HER still positive |

完整机器可读结果见：

```text
results/dft_validation_summary.json
```

完整 QE 输入输出已经复制到：

```text
validation_outputs/qe/
```

## 解释

这一步很重要，因为它暴露了 surrogate 和真实 PBE/QE 验证之间的差距。模型生成的 top 候选在 surrogate 上 `DeltaG_H` 接近 0 eV，但 QE 实算后：

- `NbNSe`：HER 不够理想，并且有明显虚频，不能作为稳定候选。
- `NbS2`：Gamma 声子稳定，但氢吸附偏弱。
- `NbSSe`：S 面吸附优于 Se 面，HER 比 `NbS2` 略好，但仍不接近 0 eV；小虚频需要更高 cutoff、更密 k 点和完整声子谱复核。

因此，下一轮真正应该做的是把这些 QE 标签回填为训练数据，进行主动学习式重训，而不是只依赖 surrogate 排名。

## 后续严谨验证

当前验证已经满足“QE 能跑通、能计算 H2/clean surface/H adsorption/Gamma phonon”的工程闭环，但科研级结论还需要：

1. 更高精度收敛测试：例如 `ecutwfc >= 50 Ry`、更密 k 点。
2. 多吸附位点搜索：top、bridge、hollow、边/缺陷位点。
3. 完整 phonon dispersion，而不只是 Gamma 点。
4. AIMD 300K/500K 长时稳定性。
5. formation energy / energy above hull。
