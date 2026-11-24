# 复现清单

本清单用于课程验收和后续 agent 接手。默认目标是复现评估、查看结果和打开试听页，不要求重新训练模型。

## 1. 数据路径

正式数据：

```text
week2/data_esc50_2mix/
week2/data_esc50_2mix/metadata/manifest_train.jsonl
week2/data_esc50_2mix/metadata/manifest_val.jsonl
week2/data_esc50_2mix/metadata/manifest_test.jsonl
```

数据规模：

```text
train: 20,000 target records
val:    2,000 target records
test:   2,000 target records
```

音频设置：

```text
sample_rate: 16000
duration: 5 seconds
num_samples: 80000
mono
```

## 2. 配置与 checkpoint

Week3 baseline：

```text
config:     week3/configs/minidit_self_attention.yaml
checkpoint: week3/runs/minidit_self_attention_v3/checkpoints/best.pt
result:     week3/runs/minidit_self_attention_v3/eval_test_full_1step/summary.json
```

Week4 self-attention recheck：

```text
config:     week4/configs/minidit_self_attention_recheck.yaml
checkpoint: week4/runs/minidit_self_attention_recheck_bs48/checkpoints/best.pt
result:     week4/runs/minidit_self_attention_recheck_bs48/eval_test_1step/summary.json
```

Week4 Mamba：

```text
config:     week4/configs/minidit_mamba.yaml
checkpoint: week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt
result:     week4/runs/minidit_mamba_v1_bs48/eval_test_1step/summary.json
```

Week5 Mamba fine-tuning：

```text
config:     week5/configs/mamba_waveform_finetune.yaml
checkpoint: week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt
result:     week5/analysis/eval_week5_mamba_ft_1step/summary.json
```

## 3. 轻量评估命令

本地当前环境为 CPU 版 PyTorch，完整 test split 会较慢。建议先用小样本确认代码和 checkpoint 可用。

Week5 小样本评估：

```powershell
python week5\scripts\evaluate_week5.py `
  --config week5\configs\mamba_waveform_finetune.yaml `
  --checkpoint week5\runs\mamba_waveform_ft_v1\checkpoints\best.pt `
  --split test `
  --batch-size 2 `
  --num-workers 0 `
  --num-steps 1 `
  --max-samples 4 `
  --output-dir week6\_repro_smoke\week5_eval_4
```

Week5 小样本采样：

```powershell
python week5\scripts\sample_week5.py `
  --config week5\configs\mamba_waveform_finetune.yaml `
  --checkpoint week5\runs\mamba_waveform_ft_v1\checkpoints\best.pt `
  --split test `
  --num-samples 4 `
  --num-steps 1 `
  --output-dir week6\_repro_smoke\week5_samples_4
```

脚本语法检查：

```powershell
python -m py_compile `
  week5\scripts\evaluate_week5.py `
  week5\scripts\sample_week5.py `
  week5\scripts\evaluate_week5_mrstft.py `
  week5\scripts\summarize_week5_step_ablation.py `
  week5\scripts\build_week5_case_study.py
```

## 4. 已保存的完整结果

主结果 full test split：

```text
week3/runs/minidit_self_attention_v3/eval_test_full_1step/summary.json
week4/runs/minidit_self_attention_recheck_bs48/eval_test_1step/summary.json
week4/runs/minidit_mamba_v1_bs48/eval_test_1step/summary.json
week5/analysis/eval_week5_mamba_ft_1step/summary.json
```

Week5 补充分析：

```text
week5/analysis/mrstft_metrics/mrstft_comparison_summary.md      # full test split
week5/analysis/inference_steps/step_ablation_summary.md         # test 前 512 条 target records
week5/analysis/week5_case_study.md                              # 20 个试听样例中选出的案例
```

注意：推理步数对比不是 full test split，而是 CPU 本地运行的 512 条子集，不作为最终主指标。

## 5. 试听页

Week5 试听页：

```text
week5/analysis/week5_listening_demo/index.html
```

试听页包含：

```text
mixture.wav
target.wav
residual_target.wav
week4_prediction_target.wav
week5_prediction_target.wav
week5_prediction_residual.wav
```

## 6. 最终提交入口

```text
README.md
week6/final_results_summary.md
week6/failure_case_analysis.md
week6/final_report.md
week6/展示ppt生成/final_project_show.pdf
```

## 7. 不建议复现的内容

默认不建议在本地 CPU 上重新训练：

```text
week3/scripts/train_minidit.py
week4/scripts/train_minidit.py
week5/scripts/train_mamba_waveform_ft.py
```

训练已在 GPU 服务器上完成，checkpoint 和完整评估结果已经同步到本地。
