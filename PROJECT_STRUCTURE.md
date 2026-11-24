# Lightweight Flow-Matching Audio Separation - 项目结构说明

## 目录总览

本项目是轻量级流匹配音频分离系统的完整实现，包含数据处理、模型训练、评估和演示。

### 核心文件夹

| 文件夹 | 说明 |
|--------|------|
| `week2` | 数据集生成脚本和配置 |
| `week2_data` | ESC50 2mix 音频数据集 (train/val/test) |
| `week3` | 初始流匹配模型 (Self-Attention) |
| `week4` | 改进的流匹配模型 (Mamba架构) |
| `week5` | 波形微调阶段 |
| `week6` | 最终模型和后处理 |
| `checkpoint` | 训练的模型检查点 |
| `results` | 评估结果 (CSV/JSON) |
| `samples_demo` | 演示样本 (分离后的音频) |
| `report_source` | 论文源文件和图表 |
| `weekly_reports` | 每周总结报告 |

### 重要文件

| 文件 | 说明 |
|------|------|
| `requirements.txt` | Python依赖 |
| `README.md` | 项目概述 |
| `Memory.md` | 研究记录 |
| `team_members.xlsx` | 团队成员信息 |
| `数据来源.txt` | 数据集来源说明 |
| `提交材料说明.md` | 提交指南 |

### 周期迭代说明

- **Week 2**: 生成2混音数据集
- **Week 3**: 实现流匹配模型（Self-Attention）
- **Week 4**: 升级为Mamba架构
- **Week 5**: 波形域微调
- **Week 6**: 最终训练和后处理

### 数据结构

```
week2_data/
└── clean_sources/
    ├── train/     # 训练集
    ├── val/       # 验证集
    └── test/      # 测试集
```

### 模型配置

各周配置文件位置：
- `week3/configs/minidit_self_attention.yaml`
- `week4/configs/minidit_mamba.yaml`
- `week5/configs/mamba_waveform_finetune.yaml`

### 评估结果

结果存储在 `results/` 目录：
- CSV 格式: 详细的按类别指标
- JSON 格式: 训练摘要和统计

### 演示样本

`samples_demo/` 包含4个演示案例，每个包含：
- `mixture.wav` - 混合音频
- `target.wav` - 目标音源
- `prediction.wav` - 模型分离结果
- `spectrogram_triplet.png` - 频谱可视化

