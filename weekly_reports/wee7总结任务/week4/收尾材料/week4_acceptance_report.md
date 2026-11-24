# Week4 收尾验收记录

## 任务定位
Week4 的目标是在 Week3 已经固定的数据、特征、Flow Matching 训练目标和评估协议上，只更换 Mini-DiT block 内部的序列混合算子，验证更高效的序列建模算子是否能改善音频分离结果。

## 已完成内容
- 保留 Week3 的 ESC-50 2-mix 数据划分、log-magnitude STFT 表示、mixture-to-target Flow Matching 训练目标、1-step 推理和同一套 test 指标。
- 将 Mini-DiT 的 block mixer 抽象为可切换模块，支持 `self_attention`、`mamba`、`hybrid`。
- 完成 Mamba-style SSM 实验与同 batch self-attention 对照实验，两次训练均使用 batch size 48、40 epochs、同一 test split。
- 云端正式产物已同步至本地 `week4/runs/`，包含 best checkpoint、训练日志、metrics、test summary 和 Mamba 样例音频/谱图。

## 核心结果
| 模型 | 后端 | 参数量 | best epoch | best val loss | Spec L1 ↓ | Wave L1 ↓ | SI-SDR ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Self-Attn bs48 | self_attention | 3,102,529 | 38 | 0.018675 | 0.03600 | 0.03378 | -1.418 dB |
| Mamba-style SSM bs48 | torch_selective_ssm | 3,014,977 | 40 | 0.019782 | 0.03047 | 0.02815 | 0.642 dB |

Mixture baseline: Spec L1 0.05708, Wave L1 0.04200, SI-SDR 0.004 dB。

## 结论
Mamba-style SSM 在相同 batch size 的对照下明显优于 self-attention：Spec L1 从 0.03600 降到 0.03047，Wave L1 从 0.03378 降到 0.02815，SI-SDR 从 -1.418 dB 提升到 0.642 dB。由于 self-attention 已经按 batch 48 补跑，这个差异不能简单归因于 batch size。

## 需要如实说明的限制
- 当前正式训练使用的是项目内 PyTorch fallback `TorchSelectiveSSMMixer`，不是官方 `mamba-ssm` CUDA kernel。
- 官方 kernel 未能在该无外网云机环境中快速安装：可用镜像只提供 source dist，编译依赖链较重；默认 shell 环境中 CUDA 编译工具链未直接可用，尝试编译耗时过长后中止。
- fallback 版本训练很慢：Mamba 约 13.21 小时，self-attention 对照约 0.28 小时。因此 Week4 的主要结论是“算子替换带来质量提升”，不是“当前实现更快”。
- 模型仍然只预测目标声源的幅度谱，重建阶段沿用 mixture phase；残差声源目前由 mixture - target 的思想间接得到，未显式作为第二个输出头训练。

## 本地验收文件
- 指标表：`week4/收尾材料/week4_operator_comparison_table.csv`
- 训练曲线：`week4/收尾材料/week4_training_curves.png`
- 测试指标图：`week4/收尾材料/week4_test_metrics_comparison.png`
- 运行时间图：`week4/收尾材料/week4_runtime_comparison.png`
- Mamba 样例谱图：`week4/收尾材料/week4_mamba_sample_spectrogram_triplet.png`
