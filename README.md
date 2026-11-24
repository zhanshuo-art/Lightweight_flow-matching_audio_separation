# 基于高效序列建模的轻量级流匹配音频分离模型研究

本项目目标是在课程算力限制下，完成一个可训练、可评估、可试听的轻量级二源音频分离系统，并比较 self-attention 与 Mamba-style SSM 作为 Mini-DiT 底层序列混合算子的效果。

## 项目主线

| 阶段 | 内容 | 作用 |
| --- | --- | --- |
| Week1 | 确定选题和技术路线 | 明确“轻量 Flow Matching 音频分离”问题 |
| Week2 | 构造 ESC-50 2-mix 数据集 | 建立监督训练数据 |
| Week3 | 训练 self-attention Mini-DiT baseline | 得到可复现对照组 |
| Week4 | 替换底层 mixer 为 Mamba-style SSM | 完成主要创新实验 |
| Week5 | 从 Week4 Mamba 继续训练，加入 waveform L1，并补齐 target + residual 输出 | 形成完整二源输出和试听分析 |
| Week6 | 最终整合、复现清单、报告和展示稿 | 完成课程作业验收材料 |

## 数据集

数据来源为 ESC-50，选取 10 个类别：

```text
car_horn, chirping_birds, clock_tick, dog, door_wood_knock,
glass_breaking, keyboard_typing, rain, siren, vacuum_cleaner
```

数据构造规则：

- 音频统一为 `16 kHz`、单声道、`5 秒`。
- 使用 ESC-50 官方 fold：`1/2/3 -> train`，`4 -> val`，`5 -> test`。
- 每个 mixture 由两个 clean source 合成。
- 每个 mixture 生成两条 target records，即两个声源轮流作为 target。

最终规模：

```text
train: 10,000 mixtures -> 20,000 target records
val:    1,000 mixtures ->  2,000 target records
test:   1,000 mixtures ->  2,000 target records
```

数据入口：

```text
week2/data_esc50_2mix/
week2/data_esc50_2mix/metadata/manifest_train.jsonl
week2/data_esc50_2mix/metadata/manifest_val.jsonl
week2/data_esc50_2mix/metadata/manifest_test.jsonl
```

## 方法流程

本项目使用条件 Flow Matching 形式的轻量 Mini-DiT 分离器：

```text
mixture waveform
-> log-magnitude STFT
-> Mini-DiT / Mamba Mini-DiT
-> prediction target spectrogram
-> iSTFT with mixture phase
-> prediction target waveform
-> prediction residual = mixture - prediction target
```

训练输入包括：

```text
x_t + mixture_feature + t + target_label
```

其中 `x_t` 是从 mixture spectrogram 到 target spectrogram 的 Flow Matching 中间状态。推理时给定 mixture spectrogram 和 target label，从 mixture 端沿预测 velocity 前进一步，得到 target spectrogram。Week5 的 residual 不由模型单独预测，而是用 `mixture - prediction_target` 得到。

## 最终主结果

所有主结果均为 test split、log-magnitude STFT、1-step inference。

| Model | 主干 | 参数量 | 训练策略 | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Week3 baseline | self-attention | 3,102,529 | 从零训练 | 0.03539 | 0.03203 | -0.419 dB | - |
| Week4 recheck | self-attention | 3,102,529 | 同 batch 重训 | 0.03600 | 0.03378 | -1.418 dB | - |
| Week4 Mamba | Mamba-style SSM | 3,014,977 | 从零训练 | 0.03047 | 0.02815 | 0.642 dB | 1.743 dB |
| Week5 Mamba FT | Mamba-style SSM | 3,014,977 | 从 Week4 Mamba 继续训练 + waveform L1 | 0.03097 | 0.02721 | 1.417 dB | 1.975 dB |

结论边界：

> 在当前 ESC-50 2-mix、轻量 Mini-DiT、log-magnitude STFT、1-step inference 设置下，Mamba-style SSM 比 self-attention 对照取得更好的分离指标；Week5 加入 waveform L1 后，Wave L1 和 SI-SDR 改善，但频谱类指标略退。

该结论只针对本项目当前数据、模型规模和训练协议，不扩展为“所有 Mamba 都优于所有 attention”。

## 关键交付物

| 内容 | 路径 |
| --- | --- |
| Week3 baseline checkpoint | `week3/runs/minidit_self_attention_v3/checkpoints/best.pt` |
| Week4 Mamba checkpoint | `week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt` |
| Week5 Mamba FT checkpoint | `week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt` |
| Week5 试听页 | `week5/analysis/week5_listening_demo/index.html` |
| Week5 验收报告 | `week5/analysis/week5_acceptance_report.md` |
| 最终结果汇总 | `week6/final_results_summary.md` |
| 复现清单 | `week6/reproducibility_checklist.md` |
| 最终报告草稿 | `week6/final_report.md` |
| 最终展示稿 | `week6/展示ppt生成/final_project_show.pdf` |

## 复现建议

默认不需要重训模型。课程验收建议优先复现：

1. 查看 `week6/final_results_summary.md`。
2. 打开 `week5/analysis/week5_listening_demo/index.html` 试听样例。
3. 按 `week6/reproducibility_checklist.md` 运行轻量评估命令。
4. 使用 `week6/展示ppt生成/final_project_show.pdf` 进行最终展示。

本地当前环境为 CPU 版 PyTorch，完整 test split 推理会较慢；已有 full test 结果已经保存。
