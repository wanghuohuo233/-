# 扩展重训候选 QE 轻量筛选

## 已完成范围

扩展重训后的 top-5 候选已经完成轻量 QE 筛选。筛选使用 `01_relax` 的正常 BFGS 收敛结构作为干净表面；只有 clean relax 正常收敛时，才继续生成并运行 H 吸附 relax 与 Gamma phonon。

输出目录：

```text
validation_outputs/qe_expanded/
```

## 计算设置

```text
Quantum ESPRESSO: conda-forge qe=7.5
Functional: PBE
Pseudopotentials: PSLibrary RRKJUS
Clean ecutwfc/ecutrho: 35/280 Ry
H adsorption ecutwfc/ecutrho: 35/420 Ry
k-points: 4x4x1
H2 reference: -2.32383790 Ry
```

HER 近似：

```text
DeltaG_H ~= E(surface + H) - E(surface) - 0.5 * E(H2) + 0.24 eV
```

## 严格解析规则

为避免从 QE 输出日志中误读坐标，本轮脚本采用以下规则：

- 只有检测到明确的 `Begin final coordinates` / `End final coordinates` 和 `ATOMIC_POSITIONS` 块时才解析结构。
- 原子坐标行必须是元素符号 + 3 个浮点数。
- 如果 clean relax 没有正常 BFGS 收敛，即使 QE 写出 `JOB DONE`，也不 fallback 生成后续 SCF/H/phonon 输入。
- 未正常收敛的候选标记为 `relax_not_converged`。

## Top-5 结果

| Rank | Candidate | Clean relax | Approx DeltaG_H (eV) | Gamma phonon | Decision |
|---:|---|---|---:|---|---|
| 1 | candidate_01_VCSe | converged | 0.423 | large imaginary modes: -432.7, -96.3 cm^-1 | rejected |
| 2 | candidate_02_VCSe | relax_not_converged | n/a | skipped | rejected |
| 3 | candidate_03_VNSe | relax_not_converged | n/a | skipped | rejected |
| 4 | candidate_04_VSSe | converged | 0.771 | large imaginary modes: -917.3, -386.4 cm^-1 | rejected |
| 5 | candidate_05_VNSe | relax_not_converged | n/a | skipped | rejected |

## 关键候选细节

### candidate_01_VCSe

```text
E(surface) = -178.3948940708 Ry
E(surface + H) = -179.5433468023 Ry
DeltaE_H = 0.183 eV
DeltaG_H ~= 0.423 eV
Gamma phonon = -432.7, -432.7, -96.3, -96.3, 40.7, 481.1, 597.9, 597.9, 1195.7 cm^-1
```

结论：HER 比早期 `NbS2` / `NbSSe` 更接近目标，但仍偏正；同时 Gamma 点有大虚频，动力学稳定性不合格。

### candidate_04_VSSe

```text
E(surface) = -187.3867850071 Ry
E(surface + H) = -188.5096559524 Ry
DeltaE_H = 0.531 eV
DeltaG_H ~= 0.771 eV
Gamma phonon = -917.3, -386.4, -386.4, 140.6, 174.6, 174.6, 289.1, 839.9, 839.9 cm^-1
```

结论：HER 明显偏弱，并且存在大虚频，淘汰。

## 总结

expanded surrogate top-5 没有得到同时满足 `DeltaG_H` 接近 0 eV 与 Gamma 点动力学稳定的最终候选。这个结果对面试题是有价值的：它证明了工程闭环不是只看 surrogate 排名，而是把真实 QE 标签反馈回来，暴露并量化了 surrogate 与 DFT 之间的差距。

下一轮最合理的改进是把这些 QE 结果作为 hard negative 加入训练/筛选闭环：未收敛结构和大虚频结构降低稳定性标签；真实 `DeltaG_H` 替换或校准 surrogate HER 标签；然后再做主动学习式扩展生成。

机器可读汇总：

```text
results_expanded/dft_screening_summary.json
```
