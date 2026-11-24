# Week5 Multi-resolution STFT 离线指标

该指标不参与训练，只在 test split 上评估最终音频波形。计算方式为 `n_fft = 256 / 512 / 1024` 的 `log1p magnitude L1` 平均值。

| Model | Samples | Target MR-STFT L1 ↓ | Residual MR-STFT L1 ↓ | Mixture Target MR-STFT L1 ↓ |
| --- | ---: | ---: | ---: | ---: |
| Week4 Mamba | 2000 | 0.07983 | 0.08248 | 0.15405 |
| Week5 Mamba FT | 2000 | 0.08103 | 0.08338 | 0.15405 |

## 结论

- Week5 相对 Week4 的 target MR-STFT L1 变化为 `+0.00120`。
- Week5 相对 Week4 的 residual MR-STFT L1 变化为 `+0.00089`。
- 这个指标从多尺度频谱角度补充 Spec L1 / Wave L1 / SI-SDR，帮助解释单一频谱 L1 与最终音频指标不完全一致的情况。
