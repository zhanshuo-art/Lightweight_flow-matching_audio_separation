# 成功/失败案例分析

本文件基于 Week5 已生成的 20 个试听样例整理，不重新采样、不重新训练。完整试听页位于：

```text
week5/analysis/week5_listening_demo/index.html
```

## 1. 案例总表

| Type | Sample | Mixture | Target | Residual | Week4 Target SI-SDR | Week5 Target SI-SDR | Week5 Residual SI-SDR | 说明 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| success | sample_10 / test_000006_t0 | rain + dog | rain | dog | 2.51 | 3.51 | 9.40 | target 与 residual 都较稳定，Week5 target 比 Week4 提升 1.00 dB。 |
| success | sample_04 / test_000003_t0 | keyboard_typing + vacuum_cleaner | keyboard_typing | vacuum_cleaner | -0.22 | 0.32 | 4.15 | Week5 能保留键盘敲击目标，同时 residual 仍有正向 SI-SDR。 |
| success | sample_13 / test_000007_t1 | keyboard_typing + door_wood_knock | door_wood_knock | keyboard_typing | 1.39 | 2.15 | 1.68 | 敲门声 target 有改善，residual 也没有明显崩坏。 |
| failure | sample_00 / test_000001_t0 | clock_tick + keyboard_typing | clock_tick | keyboard_typing | -4.40 | -8.01 | 8.38 | clock_tick 是短瞬态声音，预测幅度和 mixture phase 的误差容易被放大。 |
| failure | sample_06 / test_000004_t0 | glass_breaking + car_horn | glass_breaking | car_horn | -1.61 | -4.27 | 12.56 | glass_breaking 瞬态强，target 本身被削弱，但 residual 反而较容易保留。 |
| failure | sample_02 / test_000002_t0 | rain + siren | rain | siren | 4.40 | 2.51 | 4.99 | rain 和 siren 在时间/频谱上重叠，target 质量相对 Week4 下降。 |
| borderline | sample_01 / test_000001_t1 | clock_tick + keyboard_typing | keyboard_typing | clock_tick | -6.34 | 0.80 | -24.45 | Spec L1 略差但 SI-SDR 大幅提升，说明单一频谱 L1 与最终音频质量可能不同向。 |
| borderline | sample_07 / test_000004_t1 | glass_breaking + car_horn | car_horn | glass_breaking | 2.98 | 5.23 | -9.29 | target 明显变好，但 residual 很差，说明 `mixture - target` 的 residual 补全仍有局限。 |

## 2. 现象归纳

### 成功样例

成功样例通常有两个特征：

- target label 能提供比较明确的条件信息。
- Week5 target SI-SDR 提升时，residual SI-SDR 也保持为正。

这些样例说明 Week5 的 continued training 和 waveform L1 对最终波形质量有帮助，尤其是在 target 和 residual 不完全互相覆盖时更明显。

### 失败样例

失败样例主要集中在三类情况：

- **瞬态声音**：如 `clock_tick`、`glass_breaking`，声音持续时间短，幅度预测和 phase 重建误差会更明显。
- **类别重叠**：如 `rain + siren`，连续背景声与警报类声音在时间/频谱上有重叠，分离边界不清晰。
- **residual 由减法得到**：当前 residual 不是模型单独预测，target 一旦有偏差，residual 会直接继承这个偏差。

### 边界样例

边界样例说明两个事实：

- Spec L1 和 SI-SDR 可能不同向。频谱 L1 略差，不代表最终听感或信号质量一定变差。
- target 变好不代表 residual 一定变好。当前 residual 由 `mixture - prediction_target` 得到，还不是双输出模型。

## 3. 失败原因总结

当前模型的主要边界是：

1. **mixture phase 重建限制**
   - 模型只预测幅度谱，音频重建仍使用 mixture phase。
   - 当 target 和 residual 的 phase 差异较大时，波形质量会受影响。

2. **瞬态声源建模较难**
   - 短促声音在 STFT 上占用帧数少。
   - 预测稍有偏差就会明显影响 SI-SDR。

3. **residual 不是独立输出**
   - Week5 第一版使用 `prediction_residual = mixture - prediction_target`。
   - 这保证了 mixture consistency，但不能保证 residual 本身最优。

4. **数据规模仍然偏课程级**
   - 当前数据来自 ESC-50 10 类 synthetic 2-mix。
   - 泛化结论需要更大数据和更强 baseline 才能支持。

## 4. 音频路径

所有样例都可以在试听页中直接播放：

```text
week5/analysis/week5_listening_demo/index.html
```

单个样例目录格式：

```text
week5/analysis/week5_listening_demo/assets/sample_XX/
  mixture.wav
  target.wav
  residual_target.wav
  week4_prediction_target.wav
  week5_prediction_target.wav
  week5_prediction_residual.wav
```

推荐展示样例：

```text
成功: sample_10, sample_04, sample_13
失败: sample_00, sample_06, sample_02
边界: sample_01, sample_07
```
