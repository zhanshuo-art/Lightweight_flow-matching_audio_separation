# Week 3 工作指南：Self-Attention Mini-DiT Baseline

## 1. 最终目标

Week 3 的正式交付是先建立一个可复现、可训练、可评估的基础模型：

```text
ESC-50 2-mix waveform
    -> log-magnitude STFT
    -> mixture-to-target Flow Matching tuple
    -> self-attention Mini-DiT
    -> checkpoint / sample / evaluation
```

当前正式 baseline 采用 `log_magnitude` 表示。我们也完成了 `complex_signed_log` 扩展实验，但它在同等协议下暂未超过正式 baseline，因此不作为 Week 4 的默认比较起点。

## 2. 正式输入表示

对 16 kHz、5 秒单声道音频做：

```text
STFT(n_fft=1024, hop_length=256, win_length=1024)
```

再取：

```text
log1p(|STFT|)
```

最终输入大小为：

```text
[1, 513, 313]
```

统计量来自：

```text
week3/spectrogram/sigma_stats.json
```

正式配置：

```text
week3/configs/minidit_self_attention.yaml
```

## 3. Flow Matching 设定

对 mixture 特征 `x0` 和 target 特征 `x1`：

```text
t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
v* = x1 - x0
v_pred = v_theta(xt, t, mixture_feature, label)
x1_pred = xt + (1 - t) * v_pred
loss = L1(v_pred, v*) + L1(x1_pred, x1)
```

这里的 flow 是 `mixture -> target`，不是 `noise -> target`。由于路径是线性的，验证后把正式推理协议固定为：

```text
1 step Euler inference
```

这比原先沿用的 100 步采样更合适，也更快。

## 4. Mini-DiT baseline

正式模型：

| 项目 | 数值 |
|---|---:|
| 输入通道 | `2`，即 `xt` 与 `mixture_feature` 拼接 |
| 输出通道 | `1` |
| patch size | `16 x 16` |
| hidden size | `192` |
| depth | `4` |
| heads | `6` |
| 参数量 | `3,102,529` |

## 5. 复现命令

正式训练：

```bash
python week3/scripts/train_minidit.py \
  --config week3/configs/minidit_self_attention.yaml \
  --device cuda
```

完整验证集评估：

```bash
python week3/scripts/evaluate_minidit.py \
  --config week3/configs/minidit_self_attention.yaml \
  --checkpoint week3/runs/minidit_self_attention_v3/checkpoints/best.pt \
  --device cuda \
  --split val \
  --batch-size 16 \
  --num-workers 4 \
  --num-steps 1 \
  --output-dir week3/runs/minidit_self_attention_v3/eval_val_full_1step
```

最终测试集评估：

```bash
python week3/scripts/evaluate_minidit.py \
  --config week3/configs/minidit_self_attention.yaml \
  --checkpoint week3/runs/minidit_self_attention_v3/checkpoints/best.pt \
  --device cuda \
  --split test \
  --batch-size 16 \
  --num-workers 4 \
  --num-steps 1 \
  --output-dir week3/runs/minidit_self_attention_v3/eval_test_full_1step
```

采样示例：

```bash
python week3/scripts/sample_minidit.py \
  --config week3/configs/minidit_self_attention.yaml \
  --checkpoint week3/runs/minidit_self_attention_v3/checkpoints/best.pt \
  --device cuda \
  --split test \
  --num-samples 16 \
  --num-steps 1 \
  --output-dir week3/runs/minidit_self_attention_v3/samples_test_16_1step
```

## 6. 复数频谱扩展实验

复数版本配置：

```text
week3/configs/minidit_complex_self_attention.yaml
```

它使用：

```text
[real(STFT), imag(STFT)]
```

两通道 signed-log 特征，模型输出也为两通道，可直接 iSTFT。完整结果保存在：

```text
week3/runs/minidit_complex_self_attention/
```

它的重要价值是告诉我们：若后续要认真使用复数频谱，不能只把输入通道换成 real / imag，还需要重新考虑 waveform-aware loss、相位一致性或 STFT consistency 等问题。

## 7. Week 4 接口

Week 4 先固定：

- 数据集；
- magnitude 输入表示；
- `mixture -> target` flow；
- 训练损失；
- `1 step` 推理；
- validation / test 协议。

只替换：

```text
SelfAttentionMixer
```

再比较：

```text
self-attention
vs Mamba
vs hybrid
```
