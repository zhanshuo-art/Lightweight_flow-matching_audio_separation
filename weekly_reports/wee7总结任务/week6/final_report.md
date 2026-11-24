# 最终报告草稿

题目：基于高效序列建模的轻量级流匹配音频分离模型研究

## 1. 研究背景与问题

音频分离的目标是从混合音频中恢复指定声源。传统方法常使用频谱掩码或端到端波形模型；近年来，扩散模型和 Flow Matching 也被用于生成式音频分离。课程项目的目标不是追求大模型规模，而是在有限算力下搭建一个轻量、可复现、可试听的音频分离系统。

本项目关注的问题是：

```text
在轻量 Mini-DiT + Flow Matching 框架下，使用更高效的序列建模算子是否能改善环境声音二源分离？
```

为此，我们从 self-attention Mini-DiT baseline 出发，在 Week4 将底层 token mixing 算子替换为 Mamba-style SSM，并在 Week5 补齐 target + residual 两路输出和更贴近音频质量的 waveform L1 辅助项。

## 2. 数据构造

数据来源为 ESC-50。我们选取 10 个类别：

```text
car_horn, chirping_birds, clock_tick, dog, door_wood_knock,
glass_breaking, keyboard_typing, rain, siren, vacuum_cleaner
```

所有音频统一处理为：

```text
16 kHz, mono, 5 seconds, 80000 samples
```

按照 ESC-50 官方 fold 划分：

```text
fold 1/2/3 -> train
fold 4     -> val
fold 5     -> test
```

每个 mixture 由两个 clean source 合成。每个 mixture 生成两条训练记录：

```text
record 1: stem_0 as target, stem_1 as residual
record 2: stem_1 as target, stem_0 as residual
```

最终规模：

```text
train: 10,000 mixtures -> 20,000 target records
val:    1,000 mixtures ->  2,000 target records
test:   1,000 mixtures ->  2,000 target records
```

这种构造方式让模型学习条件分离任务：给定 mixture 和 target label，输出对应目标声源。

## 3. 方法：STFT + Conditional Flow Matching + Mini-DiT

项目使用 log-magnitude STFT 表示音频。核心流程为：

```text
mixture waveform
-> log-magnitude STFT
-> conditional Flow Matching
-> Mini-DiT / Mamba Mini-DiT
-> target spectrogram
-> iSTFT with mixture phase
-> prediction target waveform
```

训练时，模型输入包括：

```text
x_t + mixture_feature + t + target_label
```

其中 `x_t` 是从 mixture spectrogram 到 target spectrogram 的中间状态，`t` 是流匹配时间，`target_label` 提供目标类别条件。模型预测 velocity，使 `x_t` 沿着 mixture 到 target 的方向移动。

推理时，模型从 mixture spectrogram 出发，使用 1-step inference 得到 target spectrogram，再用 mixture phase 进行 iSTFT 重建音频。

## 4. Week3 self-attention baseline

Week3 搭建了 self-attention Mini-DiT baseline。输入为 mixture 的 log-magnitude STFT，条件为 target label，输出为 target spectrogram 方向的 velocity。最终音频使用预测 magnitude 和 mixture phase 重建。

Week3 full test split 1-step 结果：

```text
Target Spec L1 = 0.03539
Target Wave L1 = 0.03203
Target SI-SDR  = -0.419 dB
```

这个 baseline 已经在 Spec L1 和 Wave L1 上优于直接使用 mixture，但 SI-SDR 仍然较弱，说明模型学到了一定分离趋势，但还不是成熟的高质量分离器。

## 5. Week4 Mamba-style SSM 算子替换

Week4 的核心实验是保持数据、输入输出、loss 和推理协议基本不变，只更换 Mini-DiT block 中的序列混合算子：

```text
self-attention mixer -> Mamba-style SSM mixer
```

为了避免 batch size 影响结论，Week4 还补跑了 batch size 48 的 self-attention recheck。结果如下：

| Model | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ |
| --- | ---: | ---: | ---: |
| Week4 self-attention recheck | 0.03600 | 0.03378 | -1.418 dB |
| Week4 Mamba | 0.03047 | 0.02815 | 0.642 dB |

在当前 ESC-50 2-mix、轻量 Mini-DiT、log-magnitude STFT、1-step inference 设置下，Mamba-style SSM 的测试指标优于 self-attention 对照。

需要强调的是，这个结论只针对当前实验设置，不代表所有 Mamba 模型都一定优于所有 attention 模型。

## 6. Week5 waveform L1 与 target/residual 输出

Week5 固定 Week4 已经更好的 Mamba 主干，从 Week4 Mamba checkpoint 继续训练：

```text
base checkpoint: week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
learning rate:   5e-5
epochs:          10
loss:            velocity_loss + endpoint_loss + 0.1 * waveform_l1
```

这里的 fine-tuning 指 continued training，即从已有 Mamba 权重继续训练全部参数，不是 LoRA 或 adapter。

Week5 同时补齐二源输出：

```text
prediction_target = model(mixture, target_label)
prediction_residual = mixture - prediction_target
```

第一版 residual 没有单独输出头，而是利用二源混合一致性得到。

## 7. 实验结果

最终主结果如下：

| Model | 主干 | 参数量 | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Week3 baseline | self-attention | 3,102,529 | 0.03539 | 0.03203 | -0.419 dB | - |
| Week4 recheck | self-attention | 3,102,529 | 0.03600 | 0.03378 | -1.418 dB | - |
| Week4 Mamba | Mamba-style SSM | 3,014,977 | 0.03047 | 0.02815 | 0.642 dB | 1.743 dB |
| Week5 Mamba FT | Mamba-style SSM | 3,014,977 | 0.03097 | 0.02721 | 1.417 dB | 1.975 dB |

从 Week4 到 Week5：

- Target Wave L1 从 0.02815 降到 0.02721。
- Target SI-SDR 从 0.642 dB 提升到 1.417 dB。
- Residual SI-SDR 从 1.743 dB 提升到 1.975 dB。
- Target Spec L1 从 0.03047 略升到 0.03097。

补充分析显示，Week5 的 MR-STFT L1 也略高于 Week4；同时 1/2/4/8-step 推理对比中，更多步数没有提升 SI-SDR。因此当前瓶颈更可能来自特征表示、phase 重建或模型表达能力，而不是采样步数不够。

## 8. 成功/失败案例

Week5 生成了可试听页面：

```text
week5/analysis/week5_listening_demo/index.html
```

成功样例包括：

- `sample_10`: rain + dog，target 为 rain，Week5 target SI-SDR 从 2.51 dB 提升到 3.51 dB。
- `sample_04`: keyboard_typing + vacuum_cleaner，target 为 keyboard_typing，Week5 target 和 residual 都保持正向 SI-SDR。
- `sample_13`: keyboard_typing + door_wood_knock，target 为 door_wood_knock，target 和 residual 都较稳定。

失败样例包括：

- `sample_00`: clock_tick + keyboard_typing，clock_tick 是短瞬态声音，Week5 target SI-SDR 下降。
- `sample_06`: glass_breaking + car_horn，glass_breaking 瞬态强，target 容易被削弱。
- `sample_02`: rain + siren，两个声源有明显时间/频谱重叠。

案例分析说明：模型在稳定或较连续的声音上更容易获得可用分离结果；在瞬态声音、强重叠类别和 phase 重建误差明显的样例上更容易失败。

## 9. 局限与后续工作

当前项目仍有明显局限：

1. 数据规模偏课程级。
   - 只使用 ESC-50 10 类 synthetic 2-mix。
   - 结论不能直接推广到大规模真实混合音频。

2. 相位建模不足。
   - 模型只预测 magnitude。
   - 音频重建依赖 mixture phase。

3. residual 不是独立预测。
   - 当前 residual 使用 `mixture - prediction_target`。
   - target 的误差会直接传递到 residual。

4. baseline 仍不够强。
   - 当前主要比较 self-attention Mini-DiT 和 Mamba Mini-DiT。
   - 若要推进论文级工作，需要加入 Conv-TasNet、U-Net、SepFormer 等更强对照。

后续可以考虑：

- 增加 target/residual 双输出头。
- 加入 mixture consistency loss。
- 引入 phase-aware 或 complex STFT 目标。
- 使用 multi-resolution STFT loss 参与训练。
- 扩大数据集和 baseline 范围。

## 10. 结论

本项目完成了从数据构造、baseline、算子替换、fine-tuning 到试听分析的完整课程作业闭环。

最终结论是：

> 在当前 ESC-50 2-mix、轻量 Mini-DiT、log-magnitude STFT、1-step inference 设置下，Mamba-style SSM 比 self-attention 对照取得更好的分离指标；Week5 加入 waveform L1 后，Wave L1 和 SI-SDR 改善，但频谱类指标略退。

该项目已经达到课程大作业的完整性要求，但仍应谨慎表述，不扩展成通用的 Mamba 优于 attention 或成熟音频分离系统结论。
