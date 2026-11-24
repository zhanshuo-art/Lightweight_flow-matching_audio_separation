# Week5 说明

## 当前起点

Week4 已完成底层算子替换实验。正式结论是：在相同数据、相同 batch size、相同 1-step inference 协议下，Mamba-style SSM 的测试集分离指标优于 self-attention 对照。

Week5 固定使用 Week4 Mamba 主干作为起点：

```text
week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
```

## Week5 主线

Week5 的目标不是继续证明 Mamba 是否优于 self-attention。这个问题 Week4 已经用同 batch 对照回答了。

Week5 要回答的是：**固定 Week4 已经更好的 Mamba 主干后，如果把训练目标和输出形式改得更贴近真实音频分离任务，它还能提升到什么程度。**

要补齐三件事：

1. **训练目标更贴近音频听感**
   - 保留 Week4 的 Flow Matching loss。
   - 额外加入 waveform L1 辅助 loss。
   - 从 Week4 best checkpoint 小学习率 fine-tuning。
   - 这里的 fine-tuning 指 continued training：从已有 Mamba checkpoint 继续训练全部参数，不是 LoRA。

2. **输出 target + residual 两路音频**
   - 当前模型只显式输出 target prediction。
   - Week5 先使用：
     ```text
     prediction_residual = mixture - prediction_target
     ```
   - 采样和评估时同时导出 target / residual 两路 wav。

3. **让结果可解释、可试听**
   - 输出全 test split 指标。
   - 做按类别统计。
   - 生成试听页，展示 mixture、target、residual、prediction target、prediction residual。

## 第一版实验设定

```text
base checkpoint: week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
learning rate:   5e-5
epochs:          10
batch size:      192
loss:            velocity loss + endpoint loss + 0.1 * waveform L1
```

注：原计划 batch size 为 24。正式训练服务器显存较大，为了在 1 小时租期内完成 10 epoch，实际使用 batch size 192；模型结构、checkpoint、loss 和评估协议保持不变。

## 当前完成情况

Week5 正式训练已经完成，并在同一 test split 上与 Week4 Mamba 做了 1-step inference 对比。

| Model | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual Wave L1 ↓ | Residual SI-SDR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Week4 Mamba | 0.03047 | 0.02815 | 0.642 dB | 0.02815 | 1.743 dB |
| Week5 Mamba FT | 0.03097 | 0.02721 | 1.417 dB | 0.02721 | 1.975 dB |

主要结论：Week5 的频谱 L1 略差，但 waveform L1 和 SI-SDR 更好，说明加入时域辅助项后，最终音频质量指标有改善。

## 轻量补充分析

为增强课程报告中的证据链，Week5 额外补了三个不需要重训的分析：

1. **推理步数对比**
   - 使用 test split 前 512 条 target records。
   - 对比 `1 / 2 / 4 / 8-step inference`。
   - 结果：步数增加会让 Spec L1 略好，但 Wave L1 和 SI-SDR 没有稳定变好，1-step 的 Target SI-SDR 最高。

2. **Multi-resolution STFT 离线指标**
   - 在全 test split 上计算 `n_fft = 256 / 512 / 1024` 的 `log1p magnitude L1`。
   - Week5 的 MR-STFT L1 略高于 Week4，和 Spec L1 略退一致。
   - 这说明 Week5 的提升主要体现在 Wave L1 / SI-SDR，而不是所有频谱指标都同步提升。

3. **成功/失败案例分析**
   - 从现有 20 个试听样例中选出 `3` 个成功样例、`3` 个失败样例和 `2` 个边界样例。
   - 重点解释：哪些样例 target 和 residual 都比较合理，哪些样例受到瞬态声音、类别重叠或 mixture phase 重建限制影响。

新增文件：

```text
week5/analysis/inference_steps/step_ablation_summary.md
week5/analysis/mrstft_metrics/mrstft_comparison_summary.md
week5/analysis/week5_case_study.md
week5/analysis/figures/week5_inference_steps_comparison.png
week5/analysis/figures/week5_mrstft_comparison.png
```

与 Week3 / Week4 的关系：

```text
Week3 self-attention best.pt: 保留作 baseline，不作为 Week5 训练起点。
Week4 Mamba best.pt: Week5 的唯一继续训练起点。
```

这样主线更集中：Week4 先确定更好的主干，Week5 再把这个主干补成更完整的二源分离流程。

## 验收标准

- 能完成 smoke train，确认 waveform loss 可反传。
- 能生成新的 fine-tuned checkpoint。
- 能导出 `prediction_target.wav` 和 `prediction_residual.wav`。
- 能在 test split 上给出 target 和 residual 指标。
- 能生成 20-40 个样例的试听页。
- 与 Week4 Mamba 对比时，target Wave L1 或 SI-SDR 至少一项改善。

以上验收标准已完成。详细验收结果见：

```text
week5/analysis/week5_acceptance_report.md
week5/Week5实验总结.md
```

详细执行计划见：

```text
week5/Week5方案计划.md
```
