# 创新点简述

本项目面向 HER 二维材料生成，已经从轻量 demo 升级为 GPU GNN 条件扩散训练流程。

核心创新点：

1. 真实公开数据接入：下载并转换 DTU/CAMD C2DM 公开二维材料数据库，共 494 条结构记录；进一步下载 ColabFit/Hugging Face 的 JARVIS_C2DB 镜像，转换 3520 条 C2DB 配置并完成扩展重训。
2. GNN 条件扩散：使用 dense graph convolution 对原子图做消息传递编码，再将 graph embedding、时间步和目标性质一起输入 denoiser。
3. 多目标条件生成：条件向量包含 `dG_H`、稳定性和可合成性，使生成方向直接面向 HER 催化目标。
4. 智能优化：扩散模型给出候选后，使用局部突变、多目标排序和多样性筛选继续优化。
5. 工程闭环完整：训练、测试、权重保存、结构文件、可视化、baseline 对比和 QE 验证输入全部自动生成。
6. 可迁移到真实科研验证：`validation/qe_workflow.py` 可以生成 relax/scf/H-adsorption/phonon/AIMD 输入，便于在有 QE/VASP 的机器上继续算真实 `dG_H` 和稳定性。

当前结果仍然使用 surrogate 指标排序，适合面试展示机器学习系统设计与可运行交付。最终材料结论必须经过 DFT、声子谱、AIMD 和实验可合成性验证。
