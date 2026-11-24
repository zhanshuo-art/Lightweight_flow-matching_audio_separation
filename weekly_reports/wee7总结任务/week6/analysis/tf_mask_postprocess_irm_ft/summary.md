# Week6 TF-mask 后处理结果

## 主结果

| Method | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ | Clip ratio | Mask mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Week5 direct | 0.03080 | 0.02665 | 2.528 | 1.245 | 0.1753 | 0.5857 |
| TF-mask clip | 0.03064 | 0.02663 | 2.535 | 1.273 | 0.1753 | 0.5857 |
| Wiener p=1 | 0.03064 | 0.02663 | 2.535 | 1.273 | 0.1753 | 0.5857 |
| Wiener p=2 | 0.02957 | 0.02692 | 2.170 | 0.449 | 0.1753 | 0.5854 |
| Oracle IRM | 0.00955 | 0.00817 | 14.869 | 14.362 | nan | 0.4846 |

## 结论口径

- 最好的非 oracle 方法是 `tfmask_clip_0_1`，Target SI-SDR 相比 Week5 direct 变化 +0.007 dB，Residual SI-SDR 变化 +0.028 dB。
- `oracle_irm` 只作为诊断上界。它相对 Week5 direct 的 Target SI-SDR 差值为 +12.341 dB。
- oracle gap 达到 0.5 dB 阈值，可以启动 Week6 IRM auxiliary continued training。

## 文件

- `summary.csv`: 方法级别平均指标。
- `samples.csv`: 每条 test record、每个方法的逐样本指标。
- `by_label.csv`: 按 target label 的类别统计。
- `week5_vs_tfmask_metrics.png`: 指标对比图。
