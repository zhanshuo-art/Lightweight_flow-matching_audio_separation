# Week 3 总览

## 1. 最终结论

Week 3 已完成。经过云端正式训练和公平复评后，当前**正式 baseline** 保留为：

```text
log-magnitude STFT
-> mixture-to-target Flow Matching
-> self-attention Mini-DiT
-> 1-step inference
```

复数频谱版本也已经实现、训练并完整评估，但它在当前训练目标下没有超过幅度谱 baseline，因此被保留为**扩展实验**，暂不作为 Week 4 的默认起点。

## 2. 为什么最终没有把复数谱硬设成主线

复数 STFT 的信息更完整，这个判断本身没错；但“信息更完整”不等于“在同一套损失函数下必然更好学”。

同样使用 `1 step` 推理时：

| 模型 | Test waveform L1 | Test SI-SDR |
|---|---:|---:|
| magnitude baseline | `0.03203` | `-0.419 dB` |
| complex-STFT experiment | `0.03665` | `-0.937 dB` |

复数版虽然能直接重建相位，但当前只用特征域 L1 训练，模型对相位更敏感，最终没有换来更好的总体分离质量。这个结果值得保留，但不值得把较弱模型当成正式 baseline。

## 3. 一个很重要的修正：正式推理改为 1 step

我们现在的 flow 是：

```text
mixture -> target
```

而且训练路径本身就是线性插值，所以验证后发现 `1 step` 比 `100 steps` 更合适。继续做很多步 Euler 反而会累计误差。

以验证集为例：

| 模型 | 1 step SI-SDR | 100 steps SI-SDR |
|---|---:|---:|
| magnitude baseline | `-0.439 dB` | `-1.123 dB` |
| complex experiment | `-0.976 dB` | `-1.978 dB` |

因此，后续 Week 4 的公平比较也应固定使用 `1 step` 推理。

## 4. 目录阅读顺序

| 目的 | 文件 |
|---|---|
| 看最终训练与复现实验方式 | `README_week3_training.md` |
| 看建模方法、loss 和输出含义 | `方法及输出解释.md` |
| 看 Week 3 最终交付状态 | `第三周交付说明.md` |
| 看复数频谱实验为何暂未成为主线 | `复数频谱改造说明.md` |
| 看本地 CPU 验收 | `本地验收记录.md` |
| 看最终正式结果分析 | `正式结果分析_20260518.md` |
| 看旧版本说明、旧 notebook 与旧分析 | `history/README.md` |

## 5. 当前目录怎么读

```text
week3/
  configs/
    minidit_self_attention.yaml              # 当前正式 baseline：magnitude
    minidit_complex_self_attention.yaml      # 复数频谱扩展实验
    minidit_magnitude_self_attention.yaml    # 显式保留的 magnitude 配置副本
  scripts/
    ...                                      # 训练 / 采样 / 评估代码
  spectrogram/
    sigma_stats.json                         # magnitude baseline 统计量
    complex_stats.json                       # complex experiment 统计量
  runs/
    minidit_self_attention_v3/              # 当前正式 baseline 结果
    minidit_complex_self_attention/          # 复数频谱扩展实验结果
    _archive/                                # smoke tests、旧评估与远端打包件
  history/
    ...                                      # 旧 notebook 与历史分析
```

## 6. 进入 Week 4 前固定下来的东西

Week 4 应固定：

1. 数据集与官方 split；
2. `mixture -> target` flow 设定；
3. self-attention Mini-DiT 的规模；
4. 训练损失；
5. `1 step` 推理协议；
6. validation / test 评估脚本。

然后只替换内部 mixer：

```text
self-attention
-> Mamba
-> hybrid
```

这样后续差异才主要来自底层算子本身，而不是数据、采样步数或评估口径变化。
