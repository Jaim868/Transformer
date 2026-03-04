# QSM-Transformer-Lab（脑部 MRI QSM 重建学习项目）

这个项目专门为你“学习 Transformer 并落地到脑部 MRI 的 QSM（定量磁化率映射）重建”而设计，目标对应你提出的 4 个方向：

1. **聚焦病灶磁化率特征与可解释性**：提供可控的合成病灶数据生成流程，便于观察重建误差分布。
2. **CNN + 注意力机制融合**：同时实现 U-Net、ResNet、Transformer 变体，支持统一训练与比较。
3. **PyTorch 统一实验框架**：可批量训练并比较精度（MAE/PSNR）与噪声鲁棒性。
4. **针对病态反问题的正则化策略**：内置复合损失（重建 + 梯度一致性 + 平滑先验）提升稳定性与泛化。

---

## 项目结构

```text
.
├── configs/
│   └── base.yaml                # 默认训练配置
├── scripts/
│   └── benchmark_models.py      # 自动跑 U-Net/ResNet/Transformer 对比
├── src/qsm_transformer/
│   ├── data/synthetic_qsm.py    # 合成 QSM 数据（含简化 dipole 前向模型 + 噪声）
│   ├── losses/qsm_losses.py     # QSM 病态反问题复合损失
│   ├── models/
│   │   ├── unet.py
│   │   ├── resnet_qsm.py
│   │   ├── transformer_qsm.py
│   │   └── factory.py           # 统一模型构建入口
│   ├── utils/metrics.py         # MAE/PSNR
│   └── train.py                 # 统一训练脚本
├── pyproject.toml
└── requirements.txt
```

---

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 2) 训练 Transformer 基线

```bash
python -m qsm_transformer.train --config configs/base.yaml
```

### 3) 一键比较三类模型

```bash
python scripts/benchmark_models.py
```

> 脚本会自动生成 `configs/unet.yaml`、`configs/resnet.yaml`、`configs/transformer.yaml` 并依次训练。

---

## 如何围绕 Transformer 学习（建议路径）

1. **先跑通 ResNet/U-Net**：理解局部感受野、跳连对 QSM 边缘恢复的影响。
2. **再看 Patch Transformer**：理解 patchify、位置编码、自注意力为何能建模远距离磁化率依赖。
3. **噪声鲁棒实验**：提高 `noise_std`（如 0.03 → 0.08），比较三种模型 PSNR 下降幅度。
4. **病态反问题正则化实验**：调节 `loss.beta/gamma`，观察病灶边缘锐度与伪影权衡。
5. **可解释性分析**：保存中间注意力权重（可在 `transformer_qsm.py` 中扩展）并关联病灶区域。

---

## 下一步可扩展点

- 将 2D 扩展为 **3D QSM**（使用 3D patch + 3D UNet）。
- 加入 **物理一致性损失**：把预测 susceptibility 再经过 dipole 前向算子回到 field domain 做闭环约束。
- 引入 **混合模型**：CNN 提取局部纹理 + Transformer 进行全局建模。
- 在真实数据上加入 **mask、background field removal** 与临床评价指标。

如果你愿意，我下一步可以直接帮你继续升级成：
- `Swin-Transformer QSM` 版本，
- 带物理约束（physics-informed）的训练版本，
- 以及可视化 notebook（病灶区域 attention + error map）。
