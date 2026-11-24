# Week 2 ESC-50 2-mix 数据准备流水线

本目录已补齐 Week 2 所需的数据准备闭环：从已下载的 `ESC-50-master` 读取原始音频，构造 clean source pool，生成语义兼容的 2-mix waveform 数据，写出 target/residual 双向 manifest，并完成验证报告和可视化。

## 目录入口

```text
week2/
  configs/
    esc50_2mix.yaml
  scripts/
    prepare_clean_sources.py
    build_compatibility_matrix.py
    generate_2mix_dataset.py
    validate_2mix_dataset.py
    plot_dataset_examples.py
    run_week2_pipeline.py
  requirements_week2.txt
```

默认输出目录：

```text
week2/data_esc50_2mix/
```

主要产物：

```text
clean_sources/{train,val,test}/
rendered_mixtures/{train,val,test}/mix_000001/{mixture.wav,stem_0.wav,stem_1.wav}
metadata/clean_clips.csv
metadata/compatibility_matrix.csv
metadata/compatibility_pairs.csv
metadata/manifest_train.jsonl
metadata/manifest_val.jsonl
metadata/manifest_test.jsonl
metadata/mixture_index.csv
metadata/preprocessing_config.yaml
metadata/validation_report.md
figures/class_count_bar.png
figures/target_count_bar.png
figures/example_*_spectrogram.png
```

## 环境

当前脚本不依赖 GPU，也不依赖 `librosa`。所需 Python 包见：

```powershell
python -m pip install -r week2/requirements_week2.txt
```

## 一键运行

小规模 smoke test：

```powershell
python week2/scripts/run_week2_pipeline.py `
  --config week2/configs/esc50_2mix.yaml `
  --output-root week2/data_esc50_2mix_smoke `
  --mixtures-train 20 `
  --mixtures-val 6 `
  --mixtures-test 6 `
  --overwrite
```

正式 Week 2 数据：

```powershell
python week2/scripts/run_week2_pipeline.py `
  --config week2/configs/esc50_2mix.yaml `
  --overwrite
```

正式配置会生成：

| Split | Mixtures | Manifest records |
|---|---:|---:|
| train | 10000 | 20000 |
| val | 1000 | 2000 |
| test | 1000 | 2000 |

## 分步运行

```powershell
python week2/scripts/prepare_clean_sources.py --config week2/configs/esc50_2mix.yaml --overwrite
python week2/scripts/build_compatibility_matrix.py --config week2/configs/esc50_2mix.yaml
python week2/scripts/generate_2mix_dataset.py --config week2/configs/esc50_2mix.yaml --overwrite
python week2/scripts/validate_2mix_dataset.py --config week2/configs/esc50_2mix.yaml
```

只重画统计图和示例频谱图：

```powershell
python week2/scripts/plot_dataset_examples.py --config week2/configs/esc50_2mix.yaml
```

## 关键决策

- 数据集名称统一为 `ESC-50`。
- 类别名以 `ESC-50-master/meta/esc50.csv` 为准，敲门类统一写作 `door_wood_knock`。
- 使用官方 fold 避免泄漏：`1,2,3 -> train`，`4 -> val`，`5 -> test`。
- 音频统一为 16 kHz、mono、5 秒、RMS 目标 0.1。
- 混合只做 2-mix；每个 mixture 写两条 manifest 记录。
- 兼容矩阵来自 `week2/configs/esc50_2mix.yaml`，不做完全随机混合。
- WAV 默认保存为 `PCM_16`，所以可加性检查使用 `2e-4` 容差以覆盖量化误差。

## 下周 Dataset 读取接口

`manifest_*.jsonl` 已保留下周需要的字段：

```python
{
    "mixture_path": ".../mixture.wav",
    "target_path": ".../stem_0.wav",
    "residual_path": ".../stem_1.wav",
    "target_label": "dog",
    "target_prompt": "dog barking",
}
```

下周 Dataset 可以按 manifest 读 waveform，再动态计算 STFT/log-magnitude，不需要本周提前固化频谱特征。
