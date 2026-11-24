# Week6 TF-mask 后处理结果

## 主结果

| Method | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ | Clip ratio | Mask mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Week5 direct | 0.03097 | 0.02721 | 1.417 | 1.975 | 0.1379 | 0.5587 |
| TF-mask clip | 0.03067 | 0.02718 | 1.431 | 1.984 | 0.1379 | 0.5587 |
| Wiener p=1 | 0.03067 | 0.02718 | 1.431 | 1.984 | 0.1379 | 0.5587 |
| Wiener p=2 | 0.02982 | 0.02761 | 0.422 | 1.648 | 0.1379 | 0.5490 |
| Oracle IRM | 0.00955 | 0.00817 | 14.869 | 14.362 | nan | 0.4846 |

## 结论口径

- 最好的非 oracle 方法是 `tfmask_clip_0_1`，Target SI-SDR 相比 Week5 direct 变化 +0.013 dB，Residual SI-SDR 变化 +0.010 dB。
- `oracle_irm` 只作为诊断上界。它相对 Week5 direct 的 Target SI-SDR 差值为 +13.452 dB。
- oracle gap 达到 0.5 dB 阈值，可以启动 Week6 IRM auxiliary continued training。

## 文件

- `summary.csv`: 方法级别平均指标。
- `samples.csv`: 每条 test record、每个方法的逐样本指标。
- `by_label.csv`: 按 target label 的类别统计。
- `week5_vs_tfmask_metrics.png`: 指标对比图。
