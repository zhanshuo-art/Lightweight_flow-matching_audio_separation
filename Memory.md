# 项目记忆摘要

这份文件用于新聊天框快速恢复上下文。

## 项目主题

项目题目：**基于高效序列建模的轻量级流匹配音频分离模型研究**。

核心目标：

- 在课程算力限制下完成一个轻量级音频分离系统。
- Week3 已用 `self-attention Mini-DiT` 建立可复现 baseline。
- Week4 已完成底层序列混合算子替换实验：`self-attention` 对比 `Mamba-style SSM`，并补跑同 batch 对照。
- Week5 已完成：从 Week4 Mamba `best.pt` 继续训练，加入 waveform L1 辅助项，并补齐 target + residual 两路输出、测试集评估和试听页。

参考思路：

- `SamAudio`：DiT + Flow Matching 的音频分离框架。
- `DavisFlow`：直接在频谱图上做轻量 Flow Matching。
- `HIVE`：重视高纯度 clean source，并通过合成方式构造 mixture 数据。

## Week1：选题与方案

Week1 完成选题、文献方向和实验方案设计。

核心设想：

- 音频先从 waveform 转为 STFT / spectrogram。
- 主干网络使用轻量 Mini-DiT。
- Week4 已完成 `self-attention` 与 `Mamba-style SSM` 的底层算子对照，`hybrid` 配置已准备但不是正式主结果。

相关文件：

```text
week1/第一次汇报展示.pdf
```

## Week2：ESC-50 到 2-mix 数据构造

Week2 已完成数据构造闭环。

数据来源：ESC-50。

选取 10 个类别：

```text
car_horn
chirping_birds
clock_tick
dog
door_wood_knock
glass_breaking
keyboard_typing
rain
siren
vacuum_cleaner
```

官方 fold 划分：

```text
fold 1/2/3 -> train
fold 4     -> val
fold 5     -> test
```

clean source 处理：

- 统一为 `16 kHz`
- `mono`
- `5 秒`
- RMS 归一化到目标响度 `0.1`
- 控制 peak，避免 clipping
- 过滤低 RMS / 静音异常样本

2-mix 构造规则：

- 使用 compatibility pair list 控制哪些类别可以混合。
- 两个声源之间采样相对增益，范围为 `[-5, 5] dB`。
- 每个 mixture 保存 `mixture`、`stem_0`、`stem_1`。
- 每个 mixture 生成两条训练记录：
  - target 为 `stem_0`，residual 为 `stem_1`
  - target 为 `stem_1`，residual 为 `stem_0`

最终数据规模：

```text
train: 10,000 mixtures -> 20,000 target records
val:    1,000 mixtures ->  2,000 target records
test:   1,000 mixtures ->  2,000 target records
clean source: 400 条
```

训练格式对齐 SamAudio 的思想：

```text
(x_mix, x_tgt, x_res, c)
```

其中：

- `x_mix`：混合音频
- `x_tgt`：目标声源
- `x_res`：剩余声源
- `c`：目标类别条件

Week2 检查工作：

- 数量统计
- fold / split 泄漏检查
- mixture 可加性检查
- RMS 检查
- clipping 检查
- 类别均衡性检查
- 频谱图可视化检查

重要文件：

```text
week2/README_week2_pipeline.md
week2/数据构造详细说明.md
week2/第二次汇报要点.md
week2/展示ppt生成/deep_learning_week2_show.Rmd
```

## Week3：频谱图 + Mini-DiT Baseline

Week3 已完成一个可训练、可评估、可复现的基础模型。

正式 baseline：

```text
ESC-50 2-mix waveform
-> log-magnitude STFT
-> mixture-to-target Flow Matching
-> self-attention Mini-DiT
-> 1-step inference
```

### 输入特征

对 16 kHz、5 秒音频做 STFT：

```text
n_fft = 1024
hop_length = 256
win_length = 1024
```

取：

```text
log(1 + |STFT|)
```

最终输入尺寸：

```text
[1, 513, 313]
```

含义：

- `1`：单通道幅度谱
- `513`：频率 bin 数
- `313`：时间帧数

正式 baseline 没有直接建模相位。重建波形时使用：

```text
predicted magnitude + mixture phase -> iSTFT -> waveform
```

这是当前模型的一个重要限制。

### Flow Matching 训练逻辑

设：

```text
x0 = mixture feature
x1 = target feature
```

训练时：

```text
t ~ Uniform(0, 1)
xt = (1 - t) * x0 + t * x1
v* = x1 - x0
v_pred = model(xt, t, mixture_feature, label)
x1_pred = xt + (1 - t) * v_pred
loss = L1(v_pred, v*) + L1(x1_pred, x1)
```

含义：

- `velocity loss`：方向要预测对。
- `endpoint loss`：沿预测方向走到终点后要接近 target。

重点：当前 flow 是 `mixture -> target`，不是从纯噪声到 target。

### Mini-DiT 配置

正式模型配置：

```text
mixer_type: self_attention
in_channels: 2
out_channels: 1
hidden_size: 192
depth: 4
num_heads: 6
patch_size: [16, 16]
parameters: 3,102,529
```

输入通道为 2，因为模型同时看：

```text
xt + mixture_feature
```

### 关键修正：1-step inference

最开始沿用了 100-step Euler 推理，后来发现不适合当前任务。

当前路径是线性的：

```text
mixture -> target
```

多步推理会累积误差，因此正式协议改为：

```text
1-step inference
```

验证集 SI-SDR 对比：

```text
100 steps: -1.123 dB
1 step:   -0.439 dB
```

这是 Week3 很重要的工程修正。

### 正式训练结果

正式结果目录：

```text
week3/runs/minidit_self_attention_v3/
```

最优 checkpoint：

```text
week3/runs/minidit_self_attention_v3/checkpoints/best.pt
```

最优 epoch：

```text
epoch 39
best val loss = 0.013755
```

完整测试集 `1-step` 结果：

```text
Mixture:
spectrogram L1 = 0.05708
waveform L1    = 0.04200
SI-SDR         = 0.004 dB

Prediction:
spectrogram L1 = 0.03539
waveform L1    = 0.03203
SI-SDR         = -0.419 dB
```

结论：

- 频谱 L1 改善明显。
- 波形 L1 改善明显。
- SI-SDR 没有超过 mixture。

严谨表述：

```text
当前 baseline 已经学到部分分离趋势，但还不是成熟的高质量分离器。
```

### 复数频谱实验

也实现并训练了 complex-STFT 版本。

配置：

```text
week3/configs/minidit_complex_self_attention.yaml
```

结果目录：

```text
week3/runs/minidit_complex_self_attention/
```

复数谱表示：

```text
[real(STFT), imag(STFT)]
```

测试结果：

```text
magnitude baseline:
waveform L1 = 0.03203
SI-SDR      = -0.419 dB

complex experiment:
waveform L1 = 0.03665
SI-SDR      = -0.937 dB
```

结论：

```text
复数谱信息更完整，但只把输入从 magnitude 换成 real / imag，并不会自动变强。
```

原因：

- 当前 loss 仍主要在特征域。
- 没有 waveform-domain loss。
- 没有 phase-sensitive loss。
- 没有 STFT consistency 约束。

因此复数谱实验保留为扩展实验，不作为 Week4 主 baseline。

## Week3 当前交付文件

核心说明文件：

```text
week3/README.md
week3/README_week3_training.md
week3/第三周交付说明.md
week3/方法及输出解释.md
week3/正式结果分析_20260518.md
week3/复数频谱改造说明.md
week3/Week3展示讲解稿_技术小白版.md
```

代码目录：

```text
week3/scripts/
  audio_features.py
  common.py
  dataset.py
  evaluate_minidit.py
  estimate_sigma.py
  flow_matching.py
  modeling_minidit.py
  plot_minidit_results.py
  sample_minidit.py
  train_minidit.py
```

正式配置：

```text
week3/configs/minidit_self_attention.yaml
```

复数谱配置：

```text
week3/configs/minidit_complex_self_attention.yaml
```

Week3 汇报 Beamer：

```text
week3/展示ppt生成/deep_learning_week3_show.Rmd
week3/展示ppt生成/deep_learning_week3_show.pdf
```

注意：Week3 Beamer 已删除个人姓名，作者只保留：

```text
深度学习自选题小组
```

## Week4：底层算子替换实验已完成

Week4 的实验边界：

```text
固定：ESC-50 2-mix 数据、官方 split、log-magnitude STFT、
      mixture -> target Flow Matching、1-step inference、test 指标。
变量：Mini-DiT block 内部的序列混合算子。
```

已完成的代码改动：

- `week4/scripts/modeling_minidit.py`
  - `MiniDiTConfig` 支持 `mixer_type`、`mamba_backend`、`mamba_d_state`、`mamba_d_conv`、`mamba_expand`、`hybrid_start`。
  - 支持 `self_attention`、`mamba`、`hybrid` 三类 mixer。
  - `mamba_backend=auto` 时优先尝试官方 `mamba_ssm.Mamba`，不可用时回退到项目内 `TorchSelectiveSSMMixer`。
  - `mixer_summary()` 会记录每个 block 实际使用的 mixer 后端。
- `week4/scripts/train_minidit.py`
  - 支持 Week4 配置项并写出 `run_summary.json`。
- 主要配置：
  - `week4/configs/minidit_self_attention_recheck.yaml`
  - `week4/configs/minidit_mamba.yaml`
  - `week4/configs/minidit_hybrid.yaml`

正式训练结果已同步到本地：

```text
week4/runs/minidit_mamba_v1_bs48/
week4/runs/minidit_self_attention_recheck_bs48/
```

Mamba-style SSM 正式结果：

```text
backend = torch_selective_ssm
parameters = 3,014,977
best epoch = 40
best val loss = 0.019782

test, 1-step:
spectrogram L1 = 0.03047
waveform L1    = 0.02815
SI-SDR         = 0.642 dB
wave L1 improved ratio = 0.7045
```

同 batch self-attention 对照：

```text
backend = self_attention
parameters = 3,102,529
best epoch = 38
best val loss = 0.018675

test, 1-step:
spectrogram L1 = 0.03600
waveform L1    = 0.03378
SI-SDR         = -1.418 dB
wave L1 improved ratio = 0.6275
```

Mixture baseline：

```text
spectrogram L1 = 0.05708
waveform L1    = 0.04200
SI-SDR         = 0.004 dB
```

Week4 结论：

- 在相同 batch size 48、相同数据和相同推理协议下，Mamba-style SSM 的测试集分离指标优于 self-attention 对照。
- batch size 不是 Mamba 指标更好的主要解释，因为 self-attention 已按 batch 48 补跑。
- 当前 Mamba 正式训练使用 PyTorch fallback `TorchSelectiveSSMMixer`，官方 `mamba-ssm` CUDA kernel 未在无外网云机上快速安装成功。
- 运行时间口径要谨慎：fallback Mamba 训练约 13.21 小时，self-attention 对照约 0.28 小时。Week4 可以主张质量提升，不主张当前实现速度更快。

官方 `mamba-ssm` CUDA kernel 未使用的原因：

- 云服务器无外网，包源/镜像只拿到了 source dist。
- `causal-conv1d`、`mamba-ssm` 编译链较重。
- 默认 shell 环境未直接暴露完整 CUDA 编译链，尝试编译耗时过长后中止。
- 后续若要验证速度收益，需要一台能联网或预装好 CUDA build toolchain 的服务器。

Week4 收尾材料：

```text
week4/收尾材料/week4_acceptance_report.md
week4/收尾材料/week4_operator_comparison_table.csv
week4/收尾材料/week4_training_curves.png
week4/收尾材料/week4_test_metrics_comparison.png
week4/收尾材料/week4_runtime_comparison.png
week4/收尾材料/week5_handoff.md
```

Week4 课堂展示稿：

```text
week4/展示ppt生成/deep_learning_week4_show.Rmd
week4/展示ppt生成/deep_learning_week4_show.pdf
```

Beamer 已重新编译成功，共 18 页。为了降低机械感，已删改“不是...而是...”“结论口径”“证明了什么”这类表述。

## Week5：完整二源输出与音频域微调已完成

Week5 固定 Week4 已验证更好的 Mamba-style Mini-DiT 主干，不再继续换算子。核心做法：

```text
base checkpoint: week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
run dir:         week5/runs/mamba_waveform_ft_v1
loss:            velocity loss + endpoint loss + 0.1 * waveform L1
learning rate:   5e-5
epochs:          10
batch size:      192
flow_start:      mixture
inference:       1-step
```

说明：原计划 batch size 为 24；正式训练服务器显存为 98GB，为了在 1 小时租期内完成 10 epoch，实际使用 batch size 192。模型结构、起点 checkpoint、loss 设计和评估协议保持不变。

Week5 输出：

```text
prediction_target.wav
prediction_residual.wav = mixture.wav - prediction_target.wav
```

第一版 residual 没有改模型为双输出头，而是利用二源混合一致性通过减法得到。

测试集结果：

```text
Week4 Mamba:
target spec L1      = 0.03047
target wave L1      = 0.02815
target SI-SDR       = 0.642 dB
residual wave L1    = 0.02815
residual SI-SDR     = 1.743 dB

Week5 Mamba FT:
target spec L1      = 0.03097
target wave L1      = 0.02721
target SI-SDR       = 1.417 dB
residual wave L1    = 0.02721
residual SI-SDR     = 1.975 dB
```

结论：Week5 的 `Target Spec L1` 略退，但 `Target Wave L1`、`Target SI-SDR`、`Residual SI-SDR` 均改善。可以解释为频谱 L1 和最终音频质量指标不完全一致；加入 waveform L1 后，模型更贴近最终音频输出。

Week5 已补充轻量证据链：

```text
推理步数对比：test 前 512 条 target records，1/2/4/8-step
1-step: target Spec L1 0.03076, Wave L1 0.02709, SI-SDR 1.413 dB
2-step: target Spec L1 0.03057, Wave L1 0.02708, SI-SDR 1.376 dB
4-step: target Spec L1 0.03047, Wave L1 0.02710, SI-SDR 1.301 dB
8-step: target Spec L1 0.03040, Wave L1 0.02715, SI-SDR 1.173 dB
```

推理步数结论：步数增加使 Spec L1 略好，但 Wave L1 / SI-SDR 没有改善，当前默认 1-step 更合适。

```text
MR-STFT 离线指标，全 test split:
Week4 Mamba target MR-STFT L1 = 0.07983, residual MR-STFT L1 = 0.08248
Week5 Mamba target MR-STFT L1 = 0.08103, residual MR-STFT L1 = 0.08338
```

MR-STFT 结论：多尺度频谱距离也略退，与 Spec L1 一致；Week5 的收益主要体现在 Wave L1 / SI-SDR。

案例分析已完成：

```text
success: 3 个
failure: 3 个
borderline: 2 个
week5/analysis/week5_case_study.md
```

Week5 主要文件：

```text
week5/README.md
week5/Week5方案计划.md
week5/Week5实验总结.md
week5/analysis/week5_acceptance_report.md
week5/analysis/week5_comparison_summary.json
week5/analysis/week5_classwise_metrics.csv
week5/analysis/week5_listening_demo/index.html
week5/analysis/inference_steps/step_ablation_summary.md
week5/analysis/mrstft_metrics/mrstft_comparison_summary.md
week5/analysis/week5_case_study.md
week5/展示ppt生成/deep_learning_week5_show.Rmd
week5/展示ppt生成/deep_learning_week5_show.pdf
```

Week5 Beamer 已使用 `week5/展示ppt生成/` 里的人大 beamer 模板：

```text
beamer_default.tex
ruc.sty
ruc_background.png
ruc_logo.png
ruc_title.png
_output.yaml
```

当前 PDF 共 19 页，已补充推理步数、MR-STFT 和成功/失败案例页，并抽查新增页面。

## Week6：最终整合材料已完成

Week6 没有再做大规模新模型，重点放在最终整合、误差分析和答辩收尾：

```text
week6/Week6任务设计.md
```

已完成交付：

- 汇总 Week3 / Week4 / Week5 的最终结果表。
- 从试听页中挑选成功和失败案例，写失败原因分析。
- 生成最终报告 `final_report.md`。
- 复用 `week5/展示ppt生成/` 的 beamer 模板生成最终展示稿。
- 检查脚本、配置、checkpoint、指标和试听页路径，完成复现清单。

Week6 关键文件：

```text
README.md
week6/final_results_summary.md
week6/final_results_summary.csv
week6/reproducibility_checklist.md
week6/failure_case_analysis.md
week6/final_report.md
week6/展示ppt生成/final_project_show.Rmd
week6/展示ppt生成/final_project_show.pdf
```

最终 Beamer：

```text
week6/展示ppt生成/final_project_show.pdf
```

该 PDF 已使用 `week6/展示ppt生成/` 中从 Week5 复制的人大 Beamer 模板编译成功，共 16 页。已抽查标题页、最终主结果页、局限页和最终交付页，中文显示正常。

最终主结果表：

```text
Week3 baseline:       Spec L1 0.03539, Wave L1 0.03203, SI-SDR -0.419 dB
Week4 self-attn:      Spec L1 0.03600, Wave L1 0.03378, SI-SDR -1.418 dB
Week4 Mamba:          Spec L1 0.03047, Wave L1 0.02815, SI-SDR  0.642 dB, residual SI-SDR 1.743 dB
Week5 Mamba FT:       Spec L1 0.03097, Wave L1 0.02721, SI-SDR  1.417 dB, residual SI-SDR 1.975 dB
```

最终推荐结论口径：

```text
在当前 ESC-50 2-mix、轻量 Mini-DiT、log-magnitude STFT、1-step inference 设置下，
Mamba-style SSM 比 self-attention 对照取得更好的分离指标；
Week5 加入 waveform L1 后，Wave L1 和 SI-SDR 改善，但频谱类指标略退。
```

## 当前项目最重要的判断

1. Week2 数据构造已经完成，可以继续使用。
2. Week3 self-attention baseline 已完成，正式协议是 magnitude + `1-step inference`。
3. Week4 Mamba-style SSM 算子替换实验已完成，质量指标优于同 batch self-attention。
4. 当前正式 Mamba 使用 `torch_selective_ssm` fallback，速度结论不能夸大。
5. Week5 已把 Mamba 分离器补成 target + residual 两路输出，并且音频域指标有改善。
6. Week6 最终整合材料已完成；后续只需要按需微调文字或展示页，不建议再新增模型训练。
