# Week6 实验总结：TF-mask 后处理与 IRM 辅助微调

## 1. 实验目标

Week6 固定使用 Week5 已展示完成的 Mamba FT 分离器，不更换主干模型。目标是在 Week5 的 `prediction_target` 后加入 TF-mask / mixture consistency 后处理，并在 oracle IRM 显示仍有空间后，进一步做小成本 IRM auxiliary continued training。

实验起点：

```text
week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt
week5/configs/mamba_waveform_finetune.yaml
test split, 1-step inference
```

## 2. 后处理实验

首先不重训模型，只在 Week5 预测结果后做后处理。

| Method | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| --- | ---: | ---: | ---: | ---: |
| Week5 direct | 0.03097 | 0.02721 | 1.417 dB | 1.975 dB |
| TF-mask clip | 0.03067 | 0.02718 | 1.431 dB | 1.984 dB |
| Wiener p=1 | 0.03067 | 0.02718 | 1.431 dB | 1.984 dB |
| Wiener p=2 | 0.02982 | 0.02761 | 0.422 dB | 1.648 dB |
| Oracle IRM | 0.00955 | 0.00817 | 14.869 dB | 14.362 dB |

结论：

- `TF-mask clip` 和 `Wiener p=1` 相比 Week5 direct 有小幅提升，Target SI-SDR 提升约 `+0.013 dB`。
- `Wiener p=2` 虽然降低了 Spec L1，但 Wave L1 和 SI-SDR 变差，说明更强的功率谱 mask 会过度抑制 target。
- `Oracle IRM` 显示理论上仍有很大空间，因此启动 IRM auxiliary continued training 是合理的。

## 3. IRM auxiliary continued training

训练配置：

```text
init checkpoint: week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt
run dir:         week6/runs/mamba_irm_ft_v1
epochs:          5
learning rate:   3e-5
batch size:      128
num_workers:     8
loss:            flow_loss + 0.1 * waveform_l1 + 0.1 * irm_l1
best epoch:      2
```

训练时一开始使用 `batch_size=48`，GPU 利用率约 40%。随后调整到 `batch_size=128`、`num_workers=8`，显存约 29GB，GPU 利用率约 90%，训练速度明显提升。

验证集 best checkpoint：

```text
week6/runs/mamba_irm_ft_v1/checkpoints/best.pt
best_val_loss = 0.0288701109
best_epoch = 2
```

## 4. IRM FT 测试结果

对 Week6 IRM FT best checkpoint 做 full test 评估：

| Method | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| --- | ---: | ---: | ---: | ---: |
| IRM FT direct | 0.03080 | 0.02665 | 2.528 dB | 1.245 dB |
| IRM FT + TF-mask clip | 0.03064 | 0.02663 | 2.535 dB | 1.273 dB |
| IRM FT + Wiener p=1 | 0.03064 | 0.02663 | 2.535 dB | 1.273 dB |
| IRM FT + Wiener p=2 | 0.02957 | 0.02692 | 2.170 dB | 0.449 dB |
| Oracle IRM | 0.00955 | 0.00817 | 14.869 dB | 14.362 dB |

相对 Week5 Mamba FT：

| Model | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| --- | ---: | ---: | ---: | ---: |
| Week5 Mamba FT | 0.03097 | 0.02721 | 1.417 dB | 1.975 dB |
| Week6 IRM FT + TF-mask | 0.03064 | 0.02663 | 2.535 dB | 1.273 dB |

结论：

- Week6 明显提升 target 预测：Target SI-SDR 从 `1.417 dB` 提升到 `2.535 dB`，Target Wave L1 从 `0.02721` 降到 `0.02663`。
- Residual SI-SDR 从 `1.975 dB` 降到 `1.273 dB`。这说明当前 IRM 辅助项主要强化了 target 方向，没有同步优化 residual 质量。
- 因为 residual 仍然由 `mixture - prediction_target` 得到，target 变好不必然意味着 residual SI-SDR 也变好。Week7 汇报时要把 Week6 说成“target 质量优化”，不要说成“完整二源输出全面提升”。

## 5. 交付物

```text
week6/scripts/
week6/analysis/tf_mask_postprocess/summary.md
week6/analysis/tf_mask_postprocess/week5_vs_tfmask_metrics.png
week6/analysis/tf_mask_postprocess_irm_ft/summary.md
week6/analysis/tf_mask_postprocess_irm_ft/week5_vs_tfmask_metrics.png
week6/analysis/tf_mask_postprocess_irm_ft/listening_demo/index.html
week6/runs/mamba_irm_ft_v1/checkpoints/best.pt
week6/runs/mamba_irm_ft_v1/metrics.csv
week6/logs/mamba_irm_ft_v1.log
```

## 6. Week7 建议结论

推荐 Week7 主结论写法：

```text
Week6 在 Week5 Mamba FT 基础上加入 TF-mask 后处理和 IRM 辅助微调。
结果显示，TF-mask 后处理本身只能带来很小提升；
IRM 辅助微调能明显改善 target 分离质量，但 residual 指标下降。
这说明 mask 约束对 target 预测有效，但当前单 target 输出 + residual subtraction 的结构仍然不是最优的二源联合建模方式。
```

后续工作建议写：

```text
将模型改成 target/residual 双输出，或在训练时同时约束 target 与 residual 的 SI-SDR / mask consistency。
```
