# Week7 总结工作的指南

## 1. Week7 的任务边界

Week7 默认不再训练模型，也不再新增方法实验。主要工作是把 Week1 到 Week6 的项目整理成结课汇报：

- 讲清楚项目为什么做。
- 讲清楚数据怎么构造。
- 讲清楚 Week3 baseline、Week4 Mamba 替换、Week5 后训练、Week6 TF-mask/IRM 的逻辑。
- 汇总最终结果，说明指标提升和边界。
- 展望如果结课后继续做，技术路线应该怎样升级。

一句话概括本项目：

> 在 ESC-50 二源混合数据上，我们用 STFT 幅度谱作为输入，搭建 Conditional Flow Matching + Mini-DiT 声源分离框架；Week4 将 self-attention 换成 Mamba-style SSM 后效果提升，Week5 和 Week6 进一步补充 waveform/IRM 监督与 TF-mask 后处理，使 target 分离指标继续改善。

## 2. 建议汇报结构

建议控制在 10 到 14 页，不要把每周所有细节都放进去。

1. 项目目标
   - 输入：两个环境声音混合后的 mixture。
   - 输出：指定 target label 对应的目标声源。
   - residual 可以由 `mixture - prediction_target` 得到，但主指标仍然看 target 分离。

2. 数据构造
   - ESC-50 选 10 类。
   - 2-mix 构造：两段 5 秒音频混合。
   - 训练集约 10000 条 mixture，对应 20000 条 target record；验证/测试各约 1000 条 mixture，对应 2000 条 target record。
   - 每条 mixture 可以轮流把两个源设为 target，因此我们主要讨论 target prediction。

3. Week3 baseline
   - 方法：STFT log-magnitude + Conditional Flow Matching + Mini-DiT self-attention。
   - 输入：mixture 幅度谱、当前状态 `x_t`、时间步 `t`、target label。
   - 输出：target 幅度谱估计，再用 mixture phase 做 iSTFT 得到音频。

4. Week4 Mamba 替换
   - 问题：self-attention 在小数据、小模型设置下效果一般。
   - 做法：保持 Flow Matching 和 Mini-DiT 框架不变，只把主干 block 中的序列建模算子换成 Mamba-style SSM。
   - 结论：在当前轻量参数量和同一数据设定下，Mamba 版本优于 self-attention 对照。

5. Week5 后训练
   - 从 Week4 Mamba best checkpoint 继续训练。
   - 保留原来的 flow loss。
   - 增加 waveform L1 辅助项，让训练目标更贴近最终音频。
   - 补齐 target/residual 两路 wav 输出和全测试集评估。

6. Week6 TF-mask 与 IRM
   - TF-mask：推理阶段的幅度谱后处理，不改模型权重。
   - IRM auxiliary fine-tuning：训练阶段新增 mask 监督项，继续训练 Week5 模型。
   - Oracle IRM：只作为上界诊断，用真实 target/residual 计算理想 mask，不能当成模型结果。

7. 最终结果表
   - 表格只放关键模型，不要放所有烟测和中间尝试。
   - 推荐主线：Week3 self-attention、Week4 Mamba、Week5 Mamba FT、Week6 IRM FT + TF-mask。

8. 成功与失败案例
   - 成功：持续噪声或纹理明显的类别通常更容易分离。
   - 失败：瞬态声、重叠频段、target 能量弱、phase 不匹配会影响效果。

9. 局限与后续工作
   - 课程项目做到这里已经闭环。
   - 如果继续研究，重点不应再只是小修 loss，而应升级数据规模、表示空间和分离架构。

## 3. 最终结果建议表

下面数值来自现有 full test split 结果，Week7 可以直接引用。不同表中若小数位略有差异，以 `week6/final_results_summary.csv` 为准。

| Model | 主干 | 训练/处理策略 | Target Spec L1 | Target Wave L1 | Target SI-SDR | Residual SI-SDR |
|---|---|---|---:|---:|---:|---:|
| Week3 baseline | self-attention Mini-DiT | Flow Matching | 0.03539 | 0.03203 | -0.419 dB | 未作为主结果 |
| Week4 Mamba | Mamba-style Mini-DiT | Flow Matching | 0.03047 | 0.02815 | 0.642 dB | 1.743 dB |
| Week5 Mamba FT | Mamba-style Mini-DiT | Flow + waveform L1 | 0.03097 | 0.02721 | 1.417 dB | 1.975 dB |
| Week6 final | Mamba-style Mini-DiT | IRM FT + TF-mask | 0.03064 | 0.02663 | 2.535 dB | 1.273 dB |

推荐结论写法：

> 在当前 ESC-50 2-mix、轻量 Mini-DiT、log-magnitude STFT、1-step inference 设置下，Mamba-style SSM 相比 self-attention baseline 取得更好的 target 分离指标；Week5/Week6 的后训练和 mask 约束进一步改善了 Target Wave L1 与 Target SI-SDR。Residual 指标不是最终主目标，因为当前 residual 是由 mixture 减去 target 估计得到的。

## 4. Week6 方法怎么讲

### TF-mask 后处理

模型先预测 target 幅度谱 `pred_mag`。TF-mask 后处理把这个预测改写成一个 0 到 1 之间的比例：

```text
mask = clip(pred_mag / (mixture_mag + eps), 0, 1)
final_target_mag = mask * mixture_mag
```

直观解释：

- 每个时间-频率格子都问一句：这个格子里有多少比例属于 target？
- `clip(mask, 0, 1)` 表示比例不能小于 0，也不能大于 1。
- 它只修正幅度谱，不学习相位；最后仍然使用 mixture phase 还原 waveform。

讲法建议：

> Week6 的第一类改动是推理后处理。模型直接输出的 target 幅度谱可能局部超过 mixture 幅度谱，TF-mask 把它约束成 mixture 中的一部分。这样做不需要重新训练，属于经典的 time-frequency mask 思路。

### IRM 辅助微调

IRM 是 Ideal Ratio Mask，训练阶段可以用真实 target 和 residual 算出理想比例：

```text
irm = target_mag^2 / (target_mag^2 + residual_mag^2 + eps)
```

然后让模型预测出来的 mask 更接近这个理想比例：

```text
total_loss = flow_loss + 0.1 * waveform_l1 + 0.1 * irm_l1
```

讲法建议：

> Week6 的第二类改动是继续训练模型。我们没有改主干结构，而是在原来的 flow loss 和 waveform loss 之外，加了一个 IRM mask 监督。它的作用是提醒模型：每个时间-频率位置上，target 应该占 mixture 的多少比例。

### Oracle IRM 的作用

Oracle IRM 用真实 target/residual 计算 mask，因此测试时不能用。它只回答一个问题：

> 如果幅度 mask 做得非常理想，在仍然使用 mixture phase 的条件下，最多大概能提升到什么程度？

需要强调：

- Oracle IRM 不是模型结果。
- Oracle IRM 也不能完美复现 target，因为它仍然没有真实 target phase。
- 如果 Oracle IRM 很高，说明 mask/magnitude 方向还有空间。
- 如果 Oracle IRM 也不高，说明主要瓶颈可能是相位、数据或任务本身。

## 5. 不要讲过头的地方

- 不要说我们的模型达到音频分离 SOTA。
- 不要把 Oracle IRM 当成最终模型表现。
- 不要说 residual 已经被完整建模。当前 residual 是 `mixture - prediction_target`。
- 不要说 DiT 在频谱图上一定不好。更准确的说法是：在小规模 ESC-50 2-mix 和轻量 Mini-DiT 设置下，直接在 STFT 幅度谱上建模效果有限。
- 不要说 Mamba 绝对优于 self-attention。更准确的说法是：在我们当前参数量、数据规模和训练设置下，Mamba-style SSM 更适合这个 baseline。

## 6. 包内重点文件

先看这些文件即可：

```text
项目总README.md
README_包内说明.md
week6/final_results_summary.md
week6/final_results_summary.csv
week6/Week6实验总结.md
week6/展示ppt生成/deep_learning_week6_show.Rmd
week6/展示ppt生成/deep_learning_week6_show.pdf
week6/展示ppt生成/week6汇报文字稿.docx
week5/展示ppt生成/deep_learning_week5_show.Rmd
week4/展示ppt生成/deep_learning_week4_show.Rmd
week3/展示ppt生成/deep_learning_week3_show.Rmd
week2/展示ppt生成/deep_learning_week2_show.Rmd
```

主要 checkpoint：

```text
week3/runs/minidit_self_attention_v3/checkpoints/best.pt
week4/runs/minidit_self_attention_recheck_bs48/checkpoints/best.pt
week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt
week6/runs/mamba_irm_ft_v1/checkpoints/best.pt
```

## 7. 如果结课后继续做，建议方向

### 方向 A：频谱图路线改用 U-Net 或 Open-Unmix 类模型

如果继续使用 STFT 幅度谱，把任务看成 `mixture spectrogram -> source spectrogram/mask`，U-Net 类 encoder-decoder 很自然。Jansson 等人的 Deep U-Net 声音分离工作直接把混合谱图映射到目标 source mask，并使用 mixture phase 重建音频；Open-Unmix 也是成熟的频域音乐源分离参考实现。

适合原因：

- 频谱图有强局部结构，卷积和跳连更容易保留时间-频率细节。
- 在小数据条件下，U-Net 往往比纯 Transformer/DiT 更容易训练稳定。
- 可以作为更强的非生成式 baseline。

可做实验：

- U-Net 预测 target mask。
- U-Net 预测 target magnitude。
- Mamba/DiT 与 U-Net 在同一数据集和同一评估脚本下对比。

参考：

- Jansson et al., 2017, Singing Voice Separation with Deep U-Net Convolutional Networks: https://archives.ismir.net/ismir2017/paper/000171.pdf
- Open-Unmix, JOSS 2019: https://joss.theoj.org/papers/10.21105/joss.01667

### 方向 B：如果坚持 DiT，把建模空间移到 VAE/latent space

DiT 的优势通常在大规模生成建模里更明显。直接在 `[frequency, time]` 的高维频谱图上做小数据训练，容易受数据量、序列长度和算力限制。更合理的路线是先训练或使用一个 audio autoencoder/VAE，把音频压缩到低维 latent，再在 latent 上做 diffusion/flow matching。

适合原因：

- latent 表示更短，DiT 的序列建模压力更小。
- LDM 论文已经证明 latent diffusion 能明显降低高维生成的计算压力。
- AudioLDM 说明音频生成也可以在连续音频 latent 上做扩散建模。

可做实验：

- 使用预训练 audio autoencoder，把 mixture/target 编码到 latent。
- 在 latent 上训练 conditional flow matching 或 diffusion transformer。
- 解码回 waveform 后用 SI-SDR、MR-STFT、人工试听评估。

参考：

- Rombach et al., 2022, Latent Diffusion Models: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
- Peebles & Xie, 2023, Scalable Diffusion Models with Transformers: https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html
- AudioLDM, ICML 2023: https://proceedings.mlr.press/v202/liu23f.html

### 方向 C：加入 phase-aware 或 complex mask

当前模型主要学习幅度谱，重建时使用 mixture phase。这个设计简单，但会带来上限。后续可以预测 complex spectrogram 或 phase-sensitive mask，减少相位误差。

适合原因：

- Week6 的 Oracle IRM 已经说明，幅度 mask 本身还有上限，但 mixture phase 会限制完美重建。
- phase-sensitive mask 和 complex ratio mask 是经典的音频增强/分离方向。
- mixture consistency 可以约束多个输出相加后等于 mixture。

可做实验：

- 预测 real/imag 两通道复数谱。
- 预测 complex ratio mask。
- 在输出层加入 mixture consistency projection。

参考：

- Erdogan et al., 2015, Phase-sensitive speech separation: https://www.merl.com/publications/TR2015-031
- Wisdom et al., 2019, Differentiable Consistency Constraints: https://research.google/pubs/differentiable-consistency-constraints-for-improved-deep-speech-enhancement/

### 方向 D：尝试时域分离模型

如果目标是更高音频质量，可以绕开固定 STFT 幅度谱表示，直接在 waveform 或可学习 encoder 表示上做分离。Wave-U-Net 和 Conv-TasNet 都是代表方向。

适合原因：

- 不需要手工固定 STFT 前端。
- 可以学习相位相关信息。
- Conv-TasNet 明确指出传统 time-frequency 表示存在 magnitude/phase 解耦问题。

可做实验：

- Wave-U-Net baseline。
- Conv-TasNet/TCN baseline。
- 与当前 STFT + Mamba 模型在同一 ESC-50 2-mix 上对比。

参考：

- Wave-U-Net, ISMIR 2018: https://arxiv.org/abs/1806.03185
- Conv-TasNet, TASLP 2019: https://arxiv.org/abs/1809.07454

### 方向 E：扩大数据和任务设置

目前 ESC-50 规模偏小，而且是人工二源混合。继续做时应扩大数据量和复杂度：

- 更多类别。
- 更真实的混合比例和信噪比。
- 三源或多源混合。
- 使用 AudioSet、FSD50K、MUSDB18 等更接近真实应用的数据。

这个方向不是简单“多训练几轮”，而是解决 DiT/flow 模型最缺的数据规模问题。

## 8. Week7 可以直接使用的结尾

推荐结尾：

> 这个课程项目完成了从数据构造、baseline 搭建、算子替换、后训练优化到最终评估展示的完整闭环。结果说明，在小规模 ESC-50 2-mix 设置下，Mamba-style SSM 比 self-attention Mini-DiT 更适合我们的轻量分离框架；进一步加入 waveform 和 IRM/mask 相关约束后，target waveform 指标和 SI-SDR 继续提升。项目的主要局限是仍然在 STFT 幅度谱和 mixture phase 上工作，数据规模也偏小。后续如果继续推进，应优先考虑更强的频谱 U-Net baseline、latent-space DiT、phase-aware/complex mask 和更大规模数据。

