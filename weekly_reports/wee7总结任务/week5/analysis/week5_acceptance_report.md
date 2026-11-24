# Week5 验收报告

## 1. 本周目标

Week5 固定 Week4 已经验证更好的 Mamba-style Mini-DiT 主干，不再继续换算子。任务是把 Week4 的 target-only 分离器推进到更完整的二源输出：

- 从 Week4 Mamba `best.pt` 继续训练。
- 保留原来的频谱域 Flow Matching loss。
- 额外加入 `0.1 * waveform L1`，让训练目标更贴近最终音频。
- 输出 `prediction_target.wav` 和 `prediction_residual.wav`。
- 在 test split 上重新评估 target 与 residual，并生成试听页。

## 2. 训练设置

```text
checkpoint: week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
run dir:    week5/runs/mamba_waveform_ft_v1
lr:         5e-5
epochs:     10
batch size: 192
loss:       velocity_loss + endpoint_loss + 0.1 * waveform_l1
inference:  1-step
```

说明：原计划 batch size 为 24。正式训练时服务器显存为 98GB，为了在 1 小时租期内完成 10 epoch，将 batch size 提高到 192；模型结构、起点 checkpoint、loss 设计和评估协议没有改变。

## 3. Test Split 结果

| Model | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual Wave L1 ↓ | Residual SI-SDR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Week4 Mamba | 0.03047 | 0.02815 | 0.642 dB | 0.02815 | 1.743 dB |
| Week5 Mamba FT | 0.03097 | 0.02721 | 1.417 dB | 0.02721 | 1.975 dB |

结论：

- `Target Wave L1` 从 0.02815 降到 0.02721。
- `Target SI-SDR` 从 0.642 dB 提升到 1.417 dB。
- `Residual SI-SDR` 从 1.743 dB 提升到 1.975 dB。
- `Target Spec L1` 从 0.03047 略升到 0.03097，说明频谱 L1 与最终音频质量指标不完全一致。

Week5 的验收条件满足：target 的 waveform 指标和 SI-SDR 至少一项优于 Week4，同时补齐了 residual 输出和试听页。

## 4. 补充证据链

### 推理步数对比

本地 CPU 环境下，对 test split 前 512 条 target records 跑了 `1 / 2 / 4 / 8-step inference`：

| Steps | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.03076 | 0.02709 | 1.413 dB | 1.934 dB |
| 2 | 0.03057 | 0.02708 | 1.376 dB | 1.914 dB |
| 4 | 0.03047 | 0.02710 | 1.301 dB | 1.868 dB |
| 8 | 0.03040 | 0.02715 | 1.173 dB | 1.792 dB |

结论：更多推理步数没有带来更好的最终音频指标。1-step 仍是当前默认设置。

### Multi-resolution STFT 指标

全 test split 的 MR-STFT 离线指标如下：

| Model | Target MR-STFT L1 ↓ | Residual MR-STFT L1 ↓ |
| --- | ---: | ---: |
| Week4 Mamba | 0.07983 | 0.08248 |
| Week5 Mamba FT | 0.08103 | 0.08338 |

结论：Week5 的多尺度频谱距离略差于 Week4，和 Spec L1 略退一致；但 Wave L1 / SI-SDR 变好，说明单一指标不能完整代表最终音频质量。

### 案例分析

已补充 `3` 个成功样例、`3` 个失败样例和 `2` 个边界样例：

```text
week5/analysis/week5_case_study.md
```

这部分用于答辩时解释：模型在连续背景声或较稳定目标上更容易成功，在瞬态声音、强重叠类别、mixture phase 重建误差明显的样例上更容易失败。

## 5. 生成文件

```text
week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt
week5/runs/mamba_waveform_ft_v1/metrics.csv
week5/analysis/week5_comparison_summary.json
week5/analysis/week5_classwise_metrics.csv
week5/analysis/week5_listening_demo/index.html
week5/analysis/figures/week5_training_curves.png
week5/analysis/figures/week5_test_metrics_comparison.png
week5/analysis/figures/week5_classwise_si_sdr.png
week5/analysis/figures/week5_classwise_wave_l1.png
week5/analysis/inference_steps/step_ablation_summary.md
week5/analysis/mrstft_metrics/mrstft_comparison_summary.md
week5/analysis/week5_case_study.md
week5/analysis/figures/week5_inference_steps_comparison.png
week5/analysis/figures/week5_mrstft_comparison.png
```

## 6. 可以展示的讲法

Week4 解决的是“在当前小模型和同一数据设置下，Mamba 算子是否比 self-attention 更适合”。Week5 接着做的是“固定 Mamba 主干以后，让模型输出更接近一个完整音频分离系统”。

因此 Week5 的重点不是再证明 Mamba 好，而是补齐三个实际环节：音频域 loss、target/residual 两路输出、可试听和可分析的结果。

补充分析后的讲法可以更具体：Week5 的 waveform 和 SI-SDR 指标改善，但频谱类指标没有同步改善；多步推理也没有解决这个问题。因此当前瓶颈更可能在幅度谱/mixture phase 重建方式和模型表达能力，而不是训练轮数或采样步数。
