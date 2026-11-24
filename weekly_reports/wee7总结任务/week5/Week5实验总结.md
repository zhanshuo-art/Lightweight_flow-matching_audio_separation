# Week5 实验总结

## 实验目标

Week5 固定 Week4 的 Mamba-style Mini-DiT 主干，从 `week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt` 继续训练。目标是把 Week4 的 target-only 预测推进到更完整的二源分离输出，并验证 waveform L1 辅助 loss 是否改善最终音频质量。

## 实验设置

```text
base checkpoint: week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
run dir:         week5/runs/mamba_waveform_ft_v1
loss:            velocity loss + endpoint loss + 0.1 * waveform L1
learning rate:   5e-5
epochs:          10
batch size:      192
flow_start:      mixture
inference:       1-step
```

注：原计划默认 batch size 为 24。正式训练服务器为 98GB 显存 GPU，为了在 1 小时租期内完成 10 epoch，将 batch size 提高到 192；模型结构、起点 checkpoint、loss 设计和评估协议保持不变。

## 测试集结果

| Model | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual Wave L1 ↓ | Residual SI-SDR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Week4 Mamba | 0.03047 | 0.02815 | 0.642 dB | 0.02815 | 1.743 dB |
| Week5 Mamba FT | 0.03097 | 0.02721 | 1.417 dB | 0.02721 | 1.975 dB |

## 分析要点

- Week5 target Wave L1 从 0.02815 降到 0.02721，target SI-SDR 从 0.642 dB 提升到 1.417 dB，达到验收条件。
- Target Spec L1 从 0.03047 略升到 0.03097，说明频谱域 L1 和 waveform/SI-SDR 并不完全一致。
- Residual 第一版使用 `prediction_residual = mixture - prediction_target`，不代表模型已经学习了双输出头；它是利用二源混合的一致性补齐第二路输出。

## 补充分析

### 1. 推理步数对比

由于本地环境为 CPU 版 PyTorch，推理步数对比使用 test split 前 512 条 target records，四个步数使用同一批样本。

| Steps | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.03076 | 0.02709 | 1.413 dB | 1.934 dB |
| 2 | 0.03057 | 0.02708 | 1.376 dB | 1.914 dB |
| 4 | 0.03047 | 0.02710 | 1.301 dB | 1.868 dB |
| 8 | 0.03040 | 0.02715 | 1.173 dB | 1.792 dB |

结论：多步推理让 Spec L1 略有下降，但 SI-SDR 反而降低。当前 1-step 是更合适的默认推理方式，瓶颈更可能来自特征表示、phase 重建或模型能力，而不是采样步数不够。

### 2. Multi-resolution STFT 离线指标

在全 test split 上，使用 `n_fft = 256 / 512 / 1024` 的 `log1p magnitude L1` 平均值作为 MR-STFT 指标：

| Model | Target MR-STFT L1 ↓ | Residual MR-STFT L1 ↓ |
| --- | ---: | ---: |
| Week4 Mamba | 0.07983 | 0.08248 |
| Week5 Mamba FT | 0.08103 | 0.08338 |

结论：MR-STFT 指标也显示 Week5 的多尺度频谱距离略差于 Week4。这与 Spec L1 略退一致，但与 Wave L1 / SI-SDR 改善形成对照，说明不同指标关注的误差并不完全相同。

### 3. 案例分析

已从 20 个试听样例中整理成功/失败/边界样例，见：

```text
week5/analysis/week5_case_study.md
```

成功样例通常同时满足 target SI-SDR 改善、residual SI-SDR 为正；失败样例多与瞬态声音、类别重叠或 mixture phase 重建误差有关。

## 交付物

```text
week5/runs/mamba_waveform_ft_v1/
week5/analysis/week5_test_summary.json
week5/analysis/week5_classwise_metrics.csv
week5/analysis/week5_listening_demo/index.html
week5/analysis/inference_steps/step_ablation_summary.md
week5/analysis/mrstft_metrics/mrstft_comparison_summary.md
week5/analysis/week5_case_study.md
```
