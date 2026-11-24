# 🎵 轻量级流匹配音频分离 - 项目总览

**项目名称**: Lightweight Flow-Matching Audio Separation
**合并日期**: 2024年11月24日
**状态**: ✅ 完整合并，所有文件已去重并整理

---

## 📂 快速导航

### 首先阅读
1. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 详细的项目结构说明
2. **[MERGE_SUMMARY.md](MERGE_SUMMARY.md)** - 合并过程和去重结果
3. **[README.md](README.md)** - 项目概述

### 重要文件
- **requirements.txt** - Python依赖包
- **team_members.xlsx** - 团队成员信息
- **数据来源.txt** - 数据集来源和说明
- **提交材料说明.md** - 提交指南

---

## 🏗️ 项目结构速览

```
Lightweight_flow-matching_audio_separation/
│
├── 📊 数据集
│   ├── week2/              # 数据生成脚本
│   └── week2_data/         # ESC50 2mix 音频数据集
│
├── 🧠 模型实现 (周期迭代)
│   ├── week3/              # 流匹配 v1 (Self-Attention)
│   ├── week4/              # 流匹配 v2 (Mamba架构)
│   ├── week5/              # 波形微调
│   └── week6/              # 最终版本 + 后处理
│
├── 🎯 结果和评估
│   ├── checkpoint/         # 训练的模型检查点
│   ├── results/            # 评估指标 (CSV/JSON)
│   └── samples_demo/       # 分离效果演示 (4个样例)
│
├── 📝 文档和报告
│   ├── report_source/      # 论文源文件和图表
│   ├── weekly_reports/     # 周期性总结报告
│   └── Memory.md           # 研究记录和笔记
│
└── 📋 配置文件
    ├── requirements.txt
    ├── README.md
    └── 其他说明文档
```

---

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 查看演示样本
演示样本位于 `samples_demo/` 目录：
- `demo_00_vacuum_cleaner/` - 真空吸尘器与其他声音分离
- `demo_01_door_wood_knock/` - 门敲击声分离
- `demo_02_rain/` - 雨声分离
- `demo_03_car_horn/` - 汽车喇叭分离

每个演示包含：
- `mixture.wav` - 原始混合音频
- `target.wav` - 目标音源
- `prediction.wav` - 模型分离结果
- `spectrogram_triplet.png` - 频谱可视化

### 3. 查看训练结果
评估结果存储在 `results/` 文件夹：
- CSV 文件: 按类别的详细指标
- JSON 文件: 训练摘要和统计信息

### 4. 运行代码
参考各 week 文件夹内的脚本：
```bash
# 例如：生成数据集
python week2/scripts/run_week2_pipeline.py

# 训练模型
python week3/scripts/train_minidit.py

# 评估模型
python week3/scripts/evaluate_minidit.py
```

---

## 📊 项目统计

| 指标 | 值 |
|------|-----|
| 总文件数 | 15,000+ |
| 代码行数 | ~10,000 |
| 总大小 | ~2.6GB* |
| Python脚本 | 50+ |
| 配置文件 | 5+ |
| 模型检查点 | 多个 |

*不含完整训练数据。完整数据集约 25GB (week2_data/)

---

## 🔄 周期迭代过程

| 周次 | 主要工作 | 模型 | 关键改进 |
|------|---------|------|---------|
| Week 2 | 数据生成 | - | ESC50 2mix 数据集 |
| Week 3 | 流匹配基础 | Self-Attention | 初始实现 |
| Week 4 | 架构升级 | Mamba | 序列建模效率提升 |
| Week 5 | 微调优化 | Mamba FT | 波形域微调 |
| Week 6 | 最终训练 | IRM/TFMask | 最终评估和后处理 |

---

## 📋 文件清单

### 核心脚本
- `train_*.py` - 各周期的训练脚本
- `evaluate_*.py` - 评估脚本
- `sample_*.py` - 推理采样脚本
- `dataset.py` - 数据加载器
- `modeling_*.py` - 模型定义

### 配置文件
- `configs/*.yaml` - YAML 格式的模型配置

### 数据
- `week2_data/clean_sources/` - 清洁音源 (train/val/test)
- `week2_data/rendered_mixtures/` - 生成的混合音频
- `week2_data/metadata/` - 元数据

### 结果
- `checkpoint/week6_best.pt` - 最佳模型检查点
- `results/*.csv` - 详细的评估结果
- `results/*.json` - 训练摘要

---

## 🔗 相关信息

### 团队成员
查看 `team_members.xlsx` 了解项目团队

### 数据来源
详见 `数据来源.txt` - ESC50 数据集信息和使用说明

### 提交说明
参考 `提交材料说明.md` - 项目提交方式和要求

---

## ✅ 合并完整性检查

- ✓ 所有代码文件完整
- ✓ 所有数据集完整
- ✓ 所有配置文件完整
- ✓ 所有训练结果完整
- ✓ 所有演示样本完整
- ✓ 所有文档完整
- ✓ 去重处理完成

---

## 💡 建议

1. **首次使用**: 阅读 `PROJECT_STRUCTURE.md` 了解详细结构
2. **查看成果**: 浏览 `samples_demo/` 中的演示样本
3. **理解流程**: 参考各 week 文件夹了解迭代过程
4. **深入研究**: 查阅 `report_source/` 中的论文和图表
5. **再现结果**: 按照各脚本的说明重新训练或评估

---

**最后更新**: 2024年11月24日  
**维护者**: 团队全体成员  
**许可**: 请参考原项目文档

