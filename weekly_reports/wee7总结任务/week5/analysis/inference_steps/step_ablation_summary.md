# Week5 推理步数对比

本实验使用 test split 前 512 条 target records。

| Steps | Samples | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.03076 | 0.02709 | 1.413 dB | 1.934 dB |
| 2 | 512 | 0.03057 | 0.02708 | 1.376 dB | 1.914 dB |
| 4 | 512 | 0.03047 | 0.02710 | 1.301 dB | 1.868 dB |
| 8 | 512 | 0.03040 | 0.02715 | 1.173 dB | 1.792 dB |

## 结论

- Target SI-SDR 最好的步数是 `1`，数值为 `1.413 dB`。
- Target Wave L1 最好的步数是 `2`，数值为 `0.02708`。
- Target Spec L1 最好的步数是 `8`，数值为 `0.03040`。
- 如果多步没有稳定优于 1-step，说明当前瓶颈主要不在采样步数，而更可能在特征表示、phase 重建或模型容量。
