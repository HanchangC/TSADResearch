# 时间序列异常检测研究项目

本项目收集并实现了多种先进的时间序列异常检测算法：

## 项目结构

```
TSADResearch/
├── Anomaly-Transformer/ - ICLR 2022 Spotlight论文实现
│   ├── data_factory/ - 数据加载和处理
│   ├── dataset/ - 多个基准数据集
│   ├── model/ - 模型实现
│   ├── scripts/ - 运行脚本
│   └── utils/ - 工具函数
│
└── MEMTO/ - NeurIPS 2023论文实现
    ├── data/ - 基准数据集
    ├── data_factory/ - 数据加载和处理
    ├── model/ - 模型实现
    └── utils/ - 工具函数
```

## 包含的算法

### 1. Anomaly-Transformer

Anomaly Transformer是一种基于关联差异(Association Discrepancy)的时间序列异常检测方法，发表于ICLR 2022（Spotlight）。该模型引入了：

- 一种固有的可区分标准：**关联差异(Association Discrepancy)**
- 新的**异常注意力(Anomaly-Attention)**机制来计算关联差异
- **极小极大(minimax)策略**来放大关联差异的正常-异常可区分性

### 2. MEMTO

MEMTO (Memory-guided Transformer)是一种用于多变量时间序列异常检测的基于记忆的Transformer模型，发表于NeurIPS 2023。该模型特点包括：

- 新颖的记忆模块，可以学习每个记忆项应该根据输入数据更新的程度
- 两阶段训练范式，使用K-means聚类初始化记忆项
- 双维度偏差(bi-dimensional deviation)的检测标准，同时考虑输入空间和潜在空间的异常分数

## 数据集

本项目包含多个常用的时间序列异常检测基准数据集：

- MSL (Mars Science Laboratory)
- PSM (Pooled Server Metrics)
- SMAP (Soil Moisture Active Passive satellite)
- SMD (Server Machine Dataset)
- SWaT (Secure Water Treatment)

## 使用方法

### Anomaly-Transformer

```bash
cd Anomaly-Transformer
bash ./scripts/Start.sh  # 或选择特定数据集的脚本
```

### MEMTO

```bash
cd MEMTO
bash test.sh
```

## 环境要求

- Python 3.6+
- PyTorch >= 1.4.0
- 其他依赖请参考各模型目录下的requirements.txt文件

## 引用

如果您在研究中使用了这些模型，请引用相应的论文：

### Anomaly-Transformer
```
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

### MEMTO
```
@inproceedings{
anonymous2023memto,
title={{MEMTO}: Memory-guided Transformer for Multivariate Time Series Anomaly Detection},
author={Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=UFW67uduJd}
}
```
