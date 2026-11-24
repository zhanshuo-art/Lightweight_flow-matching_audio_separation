# Week 3 展示讲解稿：从频谱图到 Mini-DiT Baseline

这份文档给明天负责展示的同学使用。目标不是背代码，而是把 Week 3 的工作讲成一条清楚的故事：我们有了 Week 2 构造好的混合音频数据，本周把它变成频谱图，搭了一个能训练、能验证、能输出样本的 Mini-DiT 音源分离 baseline，并且明确了当前模型的效果和不足。

## 1. 这个项目到底在做什么

我们做的是“音源分离”。简单说，就是给模型一段混合声音，让它把其中一个目标声音分出来。

举例：

```text
输入：狗叫声 + 鸟叫声 的混合音频
条件：我要分离 dog
输出：尽量只保留狗叫声
```

这个问题难在：真实混合录音通常没有每个单独声源的答案。模型训练需要知道“正确答案”是什么，所以 Week 2 先用 ESC-50 的干净单事件声音自己合成 2-mix 数据。这样每条数据都知道：

```text
mixture = source_0 + source_1
target  = 当前要分离的那个 source
residual = 剩下的另一个 source
label = target 的类别
```

Week 3 的任务就是：拿 Week 2 的数据，真正搭出一个基础模型，让它开始学习“如何从 mixture 中恢复 target”。

## 2. Week 2 给 Week 3 留下了什么数据

Week 2 已经完成数据构造：

| split | mixture 数 | 训练记录数 | 说明 |
|---|---:|---:|---|
| train | 10,000 | 20,000 | 每个 mixture 对应两个 target |
| val | 1,000 | 2,000 | 训练中选择最优 checkpoint |
| test | 1,000 | 2,000 | 最终只用于评价 |

为什么一个 mixture 会变成两条记录？

假设一个混合音频是：

```text
dog + rain
```

那它可以产生两条训练任务：

```text
任务 1：输入 dog+rain，目标是 dog
任务 2：输入 dog+rain，目标是 rain
```

所以模型不是只学“把两个声音拆开”，而是学“在给定类别条件下，分离指定的目标声音”。

## 3. 为什么不直接用波形训练

原始音频是波形，也就是一长串数字。我们的音频统一为：

```text
16 kHz, mono, 5 秒
```

这意味着每条音频有：

```text
16,000 × 5 = 80,000 个采样点
```

直接把 80,000 长度的一维序列塞进 Transformer / DiT，计算压力很大，也不直观。

所以 Week 3 先把波形转成频谱图。可以把频谱图理解成“声音的图片”：

| 维度 | 含义 |
|---|---|
| 横轴 | 时间 |
| 纵轴 | 频率 |
| 颜色亮度 | 这个时间和频率上的能量强弱 |

这样声音里的结构就更容易被模型处理。例如，敲门声可能表现为短促的竖条，鸟叫可能表现为较高频的纹理，雨声可能表现为持续的噪声状区域。

## 4. STFT 和 log-magnitude 是什么

STFT 的全称是 Short-Time Fourier Transform，中文一般叫“短时傅里叶变换”。

通俗理解：

1. 把 5 秒音频切成很多很短的小片段。
2. 对每个小片段分析“里面有哪些频率”。
3. 把所有小片段按时间排起来，得到一张时频图。

Week 3 使用的参数是：

```text
n_fft = 1024
hop_length = 256
win_length = 1024
```

最终得到的频谱特征大小是：

```text
[1, 513, 313]
```

含义：

| 数字 | 含义 |
|---:|---|
| 1 | 单通道幅度谱 |
| 513 | 频率 bin 数 |
| 313 | 时间帧数 |

为什么叫 `log-magnitude`？

STFT 本来是复数，里面有两部分信息：

```text
幅度 magnitude：声音能量有多强
相位 phase：频率成分在时间上如何对齐
```

正式 baseline 先只取幅度：

```text
|STFT|
```

然后做：

```text
log(1 + |STFT|)
```

这样可以压缩特别大的能量值，让训练更稳定。声音里经常有少数很强的瞬间，如果不取 log，这些强能量点可能会主导 loss，使模型忽略其他区域。

## 5. 模型到底学什么

Week 3 不是让模型从随机噪声直接生成一个目标声音，而是让它学习：

```text
从 mixture 特征出发，往 target 特征修正
```

设：

```text
x0 = mixture 的频谱特征
x1 = target 的频谱特征
```

训练时随机取一个时间步 `t`，构造中间状态：

```text
xt = (1 - t) * x0 + t * x1
```

如果 `t=0`，那 `xt` 就是 mixture。

如果 `t=1`，那 `xt` 就是 target。

如果 `t=0.5`，那就是 mixture 和 target 中间的一张频谱图。

模型要预测的是从 mixture 到 target 的方向：

```text
v* = x1 - x0
```

可以把它理解成：

```text
模型不是直接背答案，而是在学“现在这张图应该往哪里改，才能更像目标声源”。
```

## 6. Flow Matching 不是玄学

Flow Matching 可以先理解成“学习一条从起点到终点的路”。

在我们的设置里：

```text
起点：mixture 频谱
终点：target 频谱
路径：两者之间的直线插值
模型：预测沿这条路前进的方向
```

训练目标有两部分：

```text
velocity loss = L1(v_pred, v*)
endpoint loss = L1(x1_pred, x1)
```

第一项 `velocity loss` 的意思是：模型预测的方向要接近真实方向。

第二项 `endpoint loss` 的意思是：按模型预测方向走到终点后，结果要接近 target。

所以它不是碰运气，而是一个有明确监督信号的训练任务。

## 7. Mini-DiT 是什么

DiT 可以理解成“用 Transformer 思想做生成建模的主干网络”。我们的版本很轻量，所以叫 Mini-DiT。

正式模型大致流程：

```text
xt + mixture feature
-> patchify 分块
-> 4 层 self-attention Mini-DiT
-> 输出 velocity field
```

它还会接收两个条件：

```text
t：当前处在 mixture 到 target 路径的哪个位置
label：当前要分离哪个声音类别
```

正式 baseline 配置：

| 项目 | 数值 |
|---|---:|
| 输入通道 | 2 |
| 输出通道 | 1 |
| patch size | 16 × 16 |
| hidden size | 192 |
| depth | 4 |
| heads | 6 |
| 参数量 | 3,102,529 |

这里输入通道是 2，是因为模型同时看：

```text
当前中间状态 xt
原始 mixture feature
```

这样模型知道“当前修正到哪里了”，也知道“原始混合声音是什么”。

## 8. 为什么后来改成 1-step 推理

一开始我们沿用了生成模型常见的多步采样，比如 100 steps。

但后来发现：在我们的任务里，flow 是非常明确的线性路径：

```text
mixture -> target
```

这不是从纯噪声慢慢生成图片，而是从 mixture 直接修正到 target。理论上，如果模型已经预测了从起点到终点的方向，就没有必要反复走很多小步。

实际验证也支持这一点：

| 推理步数 | Val SI-SDR |
|---:|---:|
| 100 steps | -1.123 dB |
| 1 step | -0.439 dB |

100 steps 更差，是因为每一步预测都有误差，多步积分会把误差累积起来。正式评估因此统一改成：

```text
1-step inference
```

这是 Week 3 很关键的修正。

## 9. 模型输出是什么

正式 baseline 的模型输出不是直接的 `.wav` 音频，而是：

```text
目标声源的 log-magnitude 频谱修正方向
```

推理后得到预测的目标幅度谱，再重建回波形：

```text
predicted magnitude + mixture phase -> complex STFT -> iSTFT -> prediction waveform
```

这里要注意：正式 baseline 没有自己预测相位，而是复用了 mixture 的 phase。所以它有天然缺陷：幅度谱可以变好，但波形质量和 SI-SDR 可能仍受相位限制。

这也是为什么我们后来又做了复数频谱实验。

## 10. 三个指标怎么理解

### 10.1 Spectrogram L1

比较预测频谱和目标频谱的平均绝对误差。

```text
越低越好
```

它回答的是：模型预测出来的频谱图，像不像 target 的频谱图？

正式测试集结果：

```text
mixture:    0.05708
prediction: 0.03539
```

说明频谱图层面确实改善了。

### 10.2 Waveform L1

把预测频谱还原成音频波形后，再和 target 波形比较平均绝对误差。

```text
越低越好
```

正式测试集结果：

```text
mixture:    0.04200
prediction: 0.03203
```

说明还原成音频后，波形误差也有改善。

### 10.3 SI-SDR

SI-SDR 全称是 Scale-Invariant Signal-to-Distortion Ratio，可以理解成“分离出来的声音里，有多少是目标信号，有多少是失真和干扰”。

```text
越高越好
```

正式测试集结果：

```text
mixture:    0.004 dB
prediction: -0.419 dB
```

这说明我们的模型虽然降低了 L1 误差，但在 SI-SDR 这个更严格的音频分离指标上还没超过原始 mixture。

展示时要如实说：这个模型已经学到了一部分分离规律，但还不是成熟的高质量分离系统。

## 11. 复数频谱实验为什么做，为什么没作为主线

正式 baseline 只用幅度谱，不预测相位。这个缺陷很明显，所以我们实现了复数频谱实验：

```text
complex STFT
-> real / imag 双通道
-> signed-log compression
-> Mini-DiT
-> direct iSTFT
```

理论上，复数谱信息更完整，因为它同时包含幅度和相位。

但是实证结果是：

| 模型 | Test waveform L1 | Test SI-SDR |
|---|---:|---:|
| magnitude baseline | 0.03203 | -0.419 dB |
| complex experiment | 0.03665 | -0.937 dB |

复数版反而更差。

原因不是“复数谱没有价值”，而是：

```text
只把输入从 magnitude 换成 real / imag，不等于模型自动会学好相位。
```

复数谱更难学，当前 loss 仍然主要在特征域，没有加入：

```text
waveform-domain loss
phase-sensitive loss
STFT consistency constraint
```

所以它暂时作为扩展实验和反思材料保留，不作为 Week 4 的默认 baseline。

## 12. 当前 Week 3 的严谨结论

可以这样总结：

1. Week 3 已经完成了从数据到模型再到评估的完整闭环。
2. `log-magnitude STFT + self-attention Mini-DiT + mixture-to-target flow` 可以稳定训练。
3. 1-step 推理比 100-step 更符合当前线性 flow 设定。
4. 模型在 spectrogram L1 和 waveform L1 上有明确改善。
5. SI-SDR 仍未超过 mixture，所以当前 baseline 不是成熟分离器。
6. 复数频谱实验说明：相位信息重要，但需要配套损失和约束，不能只换输入格式。

一句话版：

```text
我们已经搭好了一个可复现的 Week 3 baseline，它能学到部分分离趋势，也暴露了后续改进空间。
```

## 13. 明天每页 PPT 怎么讲

新版 PPT 是 22 页，包括标题页。页数比之前多，是为了把概念讲清楚，不再把所有东西压成关键词。

### 第 1 页：标题页

说清楚本周主题：从频谱图到 Mini-DiT baseline。不要急着讲指标，先让听众知道这是 Week 3 的阶段汇报。

### 第 2 页：本周要讲清楚的事情

这页是总览。重点说：

```text
Week 3 不是只做了频谱图，而是跑通了数据、特征、模型、训练、评估这一整条 baseline 链路。
```

### 第 3 页：先明确任务

用 dog + rain 的例子解释“条件音源分离”。重点讲：

```text
同一个 mixture，在不同 label 条件下，模型应该输出不同目标声源。
```

这页一定要讲慢，因为后面 target、label、condition 都建立在这个概念上。

### 第 4 页：项目路线

讲项目到目前为止的连续性：

```text
Week1 定方案
Week2 造数据
Week3 搭 baseline
Week4 做 self-attention / Mamba / hybrid 对比
```

### 第 5 页：Week2 数据如何进入 Week3

解释 train / val / test 和记录数。重点说：

```text
一个 mixture 有两个声源，所以可以产生两条 target 训练任务。
```

### 第 6 页：一条训练样本包含什么

讲四个字段：mixture、target、residual、label。重点句：

```text
mixture 是输入，target 是答案，label 告诉模型这次要分离谁。
```

### 第 7 页：为什么转成频谱图

把频谱图解释成“声音的图片”。重点句：

```text
波形是一维长序列，频谱图把声音展开成时间 × 频率，更适合轻量模型学习。
```

### 第 8 页：STFT 和 log-magnitude

讲三步：

```text
STFT 把声音转成时频图
取幅度谱表示能量强弱
log 压缩让数值更稳定
```

不要只念公式，要解释 `513` 是频率维，`313` 是时间帧。

### 第 9 页：相位限制

这页解释当前 baseline 的缺陷。重点句：

```text
正式模型只预测幅度谱，重建时复用 mixture phase，所以它不是完整建模了所有音频信息。
```

这能为后面的 SI-SDR 不理想和复数谱实验做铺垫。

### 第 10 页：Flow Matching 直觉

先别讲公式，讲直觉：

```text
模型学的是从 mixture 频谱往 target 频谱修正的方向。
```

把 `x0` 叫起点，`x1` 叫终点，`xt` 叫中间状态。

### 第 11 页：Flow Matching 公式

这页讲公式含义，不需要推导。重点解释：

```text
xt 是人为构造的中间点，v* 是正确方向，Mini-DiT 学的是预测这个方向。
```

### 第 12 页：Mini-DiT 吃什么、输出什么

讲模型输入：

```text
xt、mixture 特征、时间 t、类别 label
```

讲模型输出：

```text
不是直接输出 wav，而是输出 frequency feature 里的修正方向 v。
```

### 第 13 页：Mini-DiT 结构配置

讲结构不要太细，只说：

```text
先把频谱图切成 patch，再经过 4 层 self-attention，最后预测方向。
```

强调参数量是 `3,102,529`，这是轻量 baseline。

### 第 14 页：Loss 如何计算

讲两个 loss：

```text
velocity L1 保证方向对
endpoint L1 保证走完后像 target
```

这页可以帮助老师理解我们不是只随便套了一个 loss。

### 第 15 页：训练结果

看曲线。重点句：

```text
训练 loss 和验证 loss 都下降，说明模型链路能学习，不是随机输出。
```

补充最优 checkpoint 在 epoch 39。

### 第 16 页：为什么不用 100-step 推理

这是非常重要的一页。重点说：

```text
我们的 flow 是 mixture 到 target 的线性路径，不是从纯噪声慢慢生成。多步推理会累积误差，所以正式改为 1-step。
```

### 第 17 页：如何还原成音频

讲清楚输出频谱如何变回 wav：

```text
预测 magnitude + mixture phase -> iSTFT -> wav
```

再次强调：相位复用是当前短板。

### 第 18 页：三个评估指标

讲每个指标的方向：

```text
Spectrogram L1 越小越好
Waveform L1 越小越好
SI-SDR 越大越好
```

重点句：

```text
L1 下降说明更接近 target，但 SI-SDR 才更严格地反映分离质量。
```

### 第 19 页：正式测试集结果

如实讲结果：

```text
频谱 L1 和波形 L1 都改善了，但 SI-SDR 没有超过 mixture。
```

不要说模型已经分离得很好。正确说法是：

```text
它已经学到部分分离趋势，但还不是成熟分离器。
```

### 第 20 页：样本输出

让听众看 mixture、target、prediction 三张图。重点句：

```text
Prediction 在向 target 靠近，但仍有非目标成分残留。
```

### 第 21 页：复数频谱实验

解释为什么做：

```text
因为幅度谱不建模相位，所以尝试 real / imag 复数谱。
```

解释为什么暂不作为主线：

```text
复数谱信息更多，但当前 loss 没有专门约束相位，所以结果反而弱于 magnitude baseline。
```

### 第 22 页：结论和 Week4

最后总结：

```text
Week3 完成了可训练、可验证、可测试的 baseline。
当前不足是 SI-SDR 和相位处理。
Week4 固定这套流程，只替换内部 mixer，对比 self-attention、Mamba 和 hybrid。
```

## 14. 老师可能追问的问题

### Q1：为什么 SI-SDR 没有提升，还能说模型有用吗？

可以回答：

```text
它还不是成熟分离器，但不是无效模型。因为 spectrogram L1 和 waveform L1 都有下降，说明模型已经学到了部分从 mixture 到 target 的映射。SI-SDR 没提升说明当前 loss 和相位处理还不够，后续需要改进结构或引入更贴近波形质量的约束。
```

### Q2：为什么不用复数频谱作为主线？

可以回答：

```text
我们已经做了复数频谱实验。复数谱包含相位信息，但当前只用 feature-domain L1 训练，模型没有被显式约束去学好相位一致性。结果显示它弱于 magnitude baseline，所以 Week 4 为了公平做算子对比，先固定表现更稳的 magnitude baseline。
```

### Q3：你们和 SamAudio / DavisFlow 的关系是什么？

可以回答：

```text
我们借鉴 SamAudio 的 DiT + Flow Matching 思路，也借鉴 DavisFlow 直接在频谱图上做轻量建模的想法。但我们的实现更小，数据也更可控，目标是课程项目里先跑通一个轻量可复现 baseline，再比较 self-attention 和 Mamba 等底层算子。
```

### Q4：Week4 为什么只替换 mixer？

可以回答：

```text
如果同时改数据、输入表示、loss 和模型结构，就不知道结果变化来自哪里。Week 4 固定 Week 3 的数据、STFT、flow、loss、1-step 推理和评估脚本，只替换 self-attention / Mamba / hybrid，这样对比才干净。
```

## 15. 展示时的底线

不要说：

```text
我们已经完成了高质量音源分离。
```

应该说：

```text
我们完成了可训练、可评估的基础分离 baseline。它在 L1 指标上有效，但 SI-SDR 仍是短板，因此 Week 4 的结构改进有明确目标。
```

这才是严谨的说法。
