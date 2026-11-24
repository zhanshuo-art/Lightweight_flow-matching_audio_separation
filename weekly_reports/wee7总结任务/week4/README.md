# Week 4：底层算子替换实验

## 1. 本周固定不动的协议

Week 4 从 Week 3 正式 baseline 出发，只替换 Mini-DiT block 内部的 mixer。以下内容保持不变：

```text
ESC-50 2-mix 数据
log-magnitude STFT
mixture -> target Flow Matching
velocity L1 + endpoint L1
1-step inference
validation / test 评估脚本
```

Week 3 正式 baseline 结果仍作为主对照：

```text
../week3/runs/minidit_self_attention_v3/
```

## 2. 已新增代码

```text
week4/scripts/
  modeling_minidit.py        # 已抽象 mixer，并支持 self_attention / mamba / hybrid
  train_minidit.py           # 记录 mixer_type 和每层 backend
  sample_minidit.py          # 沿用 Week 3 采样协议
  evaluate_minidit.py        # 沿用 Week 3 评估协议
  ...

week4/configs/
  minidit_self_attention_recheck.yaml
  minidit_mamba.yaml
  minidit_hybrid.yaml
```

## 3. 三组模型

| 配置 | mixer | 说明 |
|---|---|---|
| `minidit_self_attention_recheck.yaml` | self-attention | Week4 代码重跑校准用 |
| `minidit_mamba.yaml` | Mamba-style selective SSM | 纯 Mamba 替换实验 |
| `minidit_hybrid.yaml` | self-attention / Mamba 交替 | 混合结构实验 |

`mamba_backend: auto` 会优先使用云端已安装的 `mamba-ssm`。如果没有安装，会退回到无额外依赖的 PyTorch selective-SSM 实现，便于 smoke test；正式训练建议优先使用 `mamba-ssm`。

## 4. 本地/云端 smoke test

先跑极小检查：

```bash
python week4/scripts/train_minidit.py \
  --config week4/configs/minidit_mamba.yaml \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --overfit-samples 4 \
  --max-train-steps 1 \
  --max-val-steps 1 \
  --dry-run \
  --output-dir week4/runs/smoke_mamba
```

Hybrid 检查：

```bash
python week4/scripts/train_minidit.py \
  --config week4/configs/minidit_hybrid.yaml \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --overfit-samples 4 \
  --max-train-steps 1 \
  --max-val-steps 1 \
  --dry-run \
  --output-dir week4/runs/smoke_hybrid
```

## 5. 正式训练命令

Mamba：

```bash
python week4/scripts/train_minidit.py \
  --config week4/configs/minidit_mamba.yaml \
  --device cuda
```

Hybrid：

```bash
python week4/scripts/train_minidit.py \
  --config week4/configs/minidit_hybrid.yaml \
  --device cuda
```

## 6. 评估命令

固定 1-step validation：

```bash
python week4/scripts/evaluate_minidit.py \
  --config week4/configs/minidit_mamba.yaml \
  --checkpoint week4/runs/minidit_mamba_v1/checkpoints/best.pt \
  --device cuda \
  --split val \
  --batch-size 16 \
  --num-workers 4 \
  --num-steps 1 \
  --output-dir week4/runs/minidit_mamba_v1/eval_val_full_1step
```

最终 test 只在选定模型后跑一次：

```bash
python week4/scripts/evaluate_minidit.py \
  --config week4/configs/minidit_mamba.yaml \
  --checkpoint week4/runs/minidit_mamba_v1/checkpoints/best.pt \
  --device cuda \
  --split test \
  --batch-size 16 \
  --num-workers 4 \
  --num-steps 1 \
  --output-dir week4/runs/minidit_mamba_v1/eval_test_full_1step
```

采样音频：

```bash
python week4/scripts/sample_minidit.py \
  --config week4/configs/minidit_mamba.yaml \
  --checkpoint week4/runs/minidit_mamba_v1/checkpoints/best.pt \
  --device cuda \
  --split test \
  --num-samples 16 \
  --num-steps 1 \
  --output-dir week4/runs/minidit_mamba_v1/samples_test_16_1step
```

## 7. 结果比较口径

至少比较：

```text
spectrogram L1
waveform L1
SI-SDR
参数量
每 epoch 训练时间
1-step 推理速度
```

结论表述要保持严谨：如果 SI-SDR 仍未超过 mixture，就只能说结构带来了某些指标或效率改善，不能说已经得到成熟高质量分离器。
