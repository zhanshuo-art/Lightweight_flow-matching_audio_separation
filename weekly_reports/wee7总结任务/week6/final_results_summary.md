# 最终结果汇总

## 1. 实验口径

主结果均使用同一套数据和评估协议：

```text
dataset: ESC-50 2-mix
split: test
test target records: 2,000
audio: 16 kHz, mono, 5 seconds
feature: log-magnitude STFT
flow_start: mixture
inference: 1-step
```

其中 Week5 的 residual 使用：

```text
prediction_residual = mixture - prediction_target
```

## 2. 主结果表

| Model | 主干 | 参数量 | 训练策略 | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ | 备注 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Week3 baseline | self-attention | 3,102,529 | 从零训练 | 0.03539 | 0.03203 | -0.419 dB | - | 原始 Mini-DiT baseline |
| Week4 recheck | self-attention | 3,102,529 | batch size 48 重训 | 0.03600 | 0.03378 | -1.418 dB | - | 与 Week4 Mamba 对齐 batch |
| Week4 Mamba | Mamba-style SSM | 3,014,977 | 从零训练 | 0.03047 | 0.02815 | 0.642 dB | 1.743 dB | 算子替换主结果 |
| Week5 Mamba FT | Mamba-style SSM | 3,014,977 | 从 Week4 Mamba 继续训练 + waveform L1 | 0.03097 | 0.02721 | 1.417 dB | 1.975 dB | 最终 target + residual 输出 |

## 3. 主要结论

1. Week4 的 Mamba-style SSM 在当前设置下优于 self-attention 对照。
   - 相比 Week4 self-attention recheck，Week4 Mamba 的 Target Spec L1、Wave L1 和 SI-SDR 都更好。
   - 两者 batch size 均为 48，因此 batch 差异不是该结果的主要解释。

2. Week5 的 continued training 改善了最终音频指标。
   - Target Wave L1 从 0.02815 降到 0.02721。
   - Target SI-SDR 从 0.642 dB 提升到 1.417 dB。
   - Residual SI-SDR 从 1.743 dB 提升到 1.975 dB。

3. Week5 的频谱类指标略退。
   - Target Spec L1 从 0.03047 升到 0.03097。
   - Multi-resolution STFT L1 也略高于 Week4。
   - 这说明单一频谱 L1 与最终波形指标并不完全一致。

4. 当前默认推理步数仍为 1-step。
   - 512 条 test records 的步数对比显示，更多步数让 Spec L1 略好，但 SI-SDR 下降。
   - 当前瓶颈更可能在特征表示、mixture phase 重建或模型能力，而不是采样步数不够。

## 4. 严谨表述

推荐最终报告和答辩使用以下表述：

> 在当前 ESC-50 2-mix、轻量 Mini-DiT、log-magnitude STFT、1-step inference 设置下，Mamba-style SSM 比 self-attention 对照取得更好的分离指标；Week5 加入 waveform L1 后，Wave L1 和 SI-SDR 改善，但频谱类指标略退。

不建议表述为：

```text
Mamba 一定比 attention 更好。
Week5 在所有指标上都优于 Week4。
当前模型已经达到成熟音频分离系统水平。
```

## 5. 结果来源

```text
week3/runs/minidit_self_attention_v3/eval_test_full_1step/summary.json
week4/runs/minidit_self_attention_recheck_bs48/eval_test_1step/summary.json
week4/runs/minidit_mamba_v1_bs48/eval_test_1step/summary.json
week5/analysis/eval_week5_mamba_ft_1step/summary.json
week5/analysis/mrstft_metrics/mrstft_comparison_summary.md
week5/analysis/inference_steps/step_ablation_summary.md
```
