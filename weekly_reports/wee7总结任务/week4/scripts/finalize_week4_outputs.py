from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
WEEK4 = ROOT / "week4"
RUNS = WEEK4 / "runs"
OUT = WEEK4 / "收尾材料"
FIGDIR = WEEK4 / "展示ppt生成" / "figure"

MODELS = {
    "Self-Attn (bs48)": "minidit_self_attention_recheck_bs48",
    "Mamba-style SSM (bs48)": "minidit_mamba_v1_bs48",
}

COLORS = {
    "Self-Attn (bs48)": "#4C78A8",
    "Mamba-style SSM (bs48)": "#F58518",
    "Mixture": "#9A9A9A",
}


def read_metrics(run: str) -> list[dict[str, float | int]]:
    path = RUNS / run / "metrics.csv"
    with path.open("r", encoding="utf-8") as f:
        return [
            {k: (float(v) if k != "epoch" else int(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def read_json(run: str, relpath: str) -> dict:
    with (RUNS / run / relpath).open("r", encoding="utf-8") as f:
        return json.load(f)


def pct_reduction(mixture: float, prediction: float) -> float:
    return 100.0 * (mixture - prediction) / mixture


def save_training_curves(metrics: dict[str, list[dict[str, float | int]]]) -> None:
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    for name, rows in metrics.items():
        epochs = [int(r["epoch"]) for r in rows]
        axes[0].plot(
            epochs,
            [float(r["train_loss"]) for r in rows],
            label=name,
            color=COLORS[name],
            linewidth=2,
        )
        axes[1].plot(
            epochs,
            [float(r["val_loss"]) for r in rows],
            label=name,
            color=COLORS[name],
            linewidth=2,
        )

    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Flow matching loss")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Week4 controlled training curves", y=1.02, fontsize=13)
    fig.tight_layout()
    for path in (FIGDIR / "week4_training_curves.png", OUT / "week4_training_curves.png"):
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_metric_bars(summaries: dict[str, dict]) -> tuple[float, float, float]:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), dpi=160)
    labels = ["Mixture", "Self-Attn (bs48)", "Mamba-style SSM (bs48)"]
    bar_colors = [COLORS[label] for label in labels]

    mixture_spec = summaries["Mamba-style SSM (bs48)"]["spectrogram_l1"]["mixture_mean"]
    mixture_wave = summaries["Mamba-style SSM (bs48)"]["waveform_l1"]["mixture_mean"]
    mixture_sisdr = summaries["Mamba-style SSM (bs48)"]["si_sdr"]["mixture_mean_db"]
    values = [
        (
            "Spectrogram L1",
            [
                mixture_spec,
                summaries["Self-Attn (bs48)"]["spectrogram_l1"]["prediction_mean"],
                summaries["Mamba-style SSM (bs48)"]["spectrogram_l1"]["prediction_mean"],
            ],
            "lower is better",
        ),
        (
            "Waveform L1",
            [
                mixture_wave,
                summaries["Self-Attn (bs48)"]["waveform_l1"]["prediction_mean"],
                summaries["Mamba-style SSM (bs48)"]["waveform_l1"]["prediction_mean"],
            ],
            "lower is better",
        ),
        (
            "SI-SDR (dB)",
            [
                mixture_sisdr,
                summaries["Self-Attn (bs48)"]["si_sdr"]["prediction_mean_db"],
                summaries["Mamba-style SSM (bs48)"]["si_sdr"]["prediction_mean_db"],
            ],
            "higher is better",
        ),
    ]

    for ax, (title, vals, subtitle) in zip(axes, values, strict=True):
        ax.bar(range(3), vals, color=bar_colors, width=0.62)
        ax.set_title(f"{title}\n{subtitle}")
        ax.set_xticks(range(3), ["Mixture", "Self-Attn", "Mamba"], rotation=18, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(vals):
            offset = 0.002 if title != "SI-SDR (dB)" else 0.08
            ax.text(idx, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Week4 test-set comparison (same split, same batch size)", y=1.04, fontsize=13)
    fig.tight_layout()
    for path in (
        FIGDIR / "week4_test_metrics_comparison.png",
        OUT / "week4_test_metrics_comparison.png",
    ):
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return mixture_spec, mixture_wave, mixture_sisdr


def save_runtime_chart(metrics: dict[str, list[dict[str, float | int]]]) -> dict[str, float]:
    runtime_hours = {
        name: sum(float(row["seconds"]) for row in rows) / 3600.0
        for name, rows in metrics.items()
    }

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=160)
    ax.bar(
        runtime_hours.keys(),
        runtime_hours.values(),
        color=[COLORS[name] for name in runtime_hours.keys()],
        width=0.55,
    )
    ax.set_ylabel("Training wall time (hours)")
    ax.set_title("Runtime caveat: torch fallback is much slower")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(runtime_hours.values()):
        ax.text(idx, value + 0.15, f"{value:.2f} h", ha="center", va="bottom")

    fig.tight_layout()
    for path in (FIGDIR / "week4_runtime_comparison.png", OUT / "week4_runtime_comparison.png"):
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return runtime_hours


def build_table_rows(
    summaries: dict[str, dict], run_summaries: dict[str, dict], runtime_hours: dict[str, float]
) -> list[dict[str, float | int | str]]:
    rows = []
    for name in ("Self-Attn (bs48)", "Mamba-style SSM (bs48)"):
        summary = summaries[name]
        run_summary = run_summaries[name]
        rows.append(
            {
                "model": name,
                "mixer_type": run_summary["mixer_type"],
                "backend": run_summary["mixer_summary"][0]["backend"],
                "parameters": run_summary["parameters"],
                "best_epoch": summary["checkpoint_epoch"],
                "best_val_loss": summary["checkpoint_best_val_loss"],
                "spec_l1_pred": summary["spectrogram_l1"]["prediction_mean"],
                "spec_l1_reduction_percent": pct_reduction(
                    summary["spectrogram_l1"]["mixture_mean"],
                    summary["spectrogram_l1"]["prediction_mean"],
                ),
                "wave_l1_pred": summary["waveform_l1"]["prediction_mean"],
                "wave_l1_reduction_percent": pct_reduction(
                    summary["waveform_l1"]["mixture_mean"],
                    summary["waveform_l1"]["prediction_mean"],
                ),
                "si_sdr_pred_db": summary["si_sdr"]["prediction_mean_db"],
                "si_sdr_improvement_db": summary["si_sdr"]["improvement_mean_db"],
                "improved_ratio_wave": summary["waveform_l1"]["improved_ratio"],
                "train_wall_hours": runtime_hours[name],
            }
        )
    return rows


def write_csv(rows: list[dict[str, float | int | str]]) -> None:
    path = OUT / "week4_operator_comparison_table.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def copy_sample_image() -> None:
    sample_src = (
        RUNS
        / "minidit_mamba_v1_bs48"
        / "samples_test_8_1step"
        / "sample_03"
        / "spectrogram_triplet.png"
    )
    if not sample_src.exists():
        return
    for dst in (
        FIGDIR / "week4_mamba_sample_spectrogram_triplet.png",
        OUT / "week4_mamba_sample_spectrogram_triplet.png",
    ):
        shutil.copy2(sample_src, dst)


def write_acceptance_report(
    rows: list[dict[str, float | int | str]],
    mixture_spec: float,
    mixture_wave: float,
    mixture_sisdr: float,
    runtime_hours: dict[str, float],
) -> None:
    attn, mamba = rows
    text = f"""# Week4 收尾验收记录

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
| Self-Attn bs48 | {attn["backend"]} | {int(attn["parameters"]):,} | {int(attn["best_epoch"])} | {float(attn["best_val_loss"]):.6f} | {float(attn["spec_l1_pred"]):.5f} | {float(attn["wave_l1_pred"]):.5f} | {float(attn["si_sdr_pred_db"]):.3f} dB |
| Mamba-style SSM bs48 | {mamba["backend"]} | {int(mamba["parameters"]):,} | {int(mamba["best_epoch"])} | {float(mamba["best_val_loss"]):.6f} | {float(mamba["spec_l1_pred"]):.5f} | {float(mamba["wave_l1_pred"]):.5f} | {float(mamba["si_sdr_pred_db"]):.3f} dB |

Mixture baseline: Spec L1 {mixture_spec:.5f}, Wave L1 {mixture_wave:.5f}, SI-SDR {mixture_sisdr:.3f} dB。

## 结论
Mamba-style SSM 在相同 batch size 的对照下明显优于 self-attention：Spec L1 从 {float(attn["spec_l1_pred"]):.5f} 降到 {float(mamba["spec_l1_pred"]):.5f}，Wave L1 从 {float(attn["wave_l1_pred"]):.5f} 降到 {float(mamba["wave_l1_pred"]):.5f}，SI-SDR 从 {float(attn["si_sdr_pred_db"]):.3f} dB 提升到 {float(mamba["si_sdr_pred_db"]):.3f} dB。由于 self-attention 已经按 batch 48 补跑，这个差异不能简单归因于 batch size。

## 需要如实说明的限制
- 当前正式训练使用的是项目内 PyTorch fallback `TorchSelectiveSSMMixer`，不是官方 `mamba-ssm` CUDA kernel。
- 官方 kernel 未能在该无外网云机环境中快速安装：可用镜像只提供 source dist，编译依赖链较重；默认 shell 环境中 CUDA 编译工具链未直接可用，尝试编译耗时过长后中止。
- fallback 版本训练很慢：Mamba 约 {runtime_hours["Mamba-style SSM (bs48)"]:.2f} 小时，self-attention 对照约 {runtime_hours["Self-Attn (bs48)"]:.2f} 小时。因此 Week4 的主要结论是“算子替换带来质量提升”，不是“当前实现更快”。
- 模型仍然只预测目标声源的幅度谱，重建阶段沿用 mixture phase；残差声源目前由 mixture - target 的思想间接得到，未显式作为第二个输出头训练。

## 本地验收文件
- 指标表：`week4/收尾材料/week4_operator_comparison_table.csv`
- 训练曲线：`week4/收尾材料/week4_training_curves.png`
- 测试指标图：`week4/收尾材料/week4_test_metrics_comparison.png`
- 运行时间图：`week4/收尾材料/week4_runtime_comparison.png`
- Mamba 样例谱图：`week4/收尾材料/week4_mamba_sample_spectrogram_triplet.png`
"""
    (OUT / "week4_acceptance_report.md").write_text(text, encoding="utf-8")


def write_week5_handoff() -> None:
    text = """# Week5 对接说明

## Week4 已封板状态
Week4 已完成“更换底层算子”的核心创新实验。最终可汇报结论是：在 Week3 的 Mini-DiT Flow Matching baseline 上，将 block 内 self-attention 替换为 Mamba-style SSM 后，在相同 batch size 48 的正式对照中取得更好的测试集分离指标。

## Week5 建议目标
Week5 不建议继续无边界地堆新算子。更合理的方向是围绕 Week4 暴露出的真实瓶颈做系统收尾和增强：

1. 让输出更贴近音频分离任务：从“单目标幅度谱 + mixture phase”升级到 phase-aware 或 waveform-aware 训练。
2. 显式建模 residual：增加 target/residual 双输出或加入 mixture consistency，使 `target + residual ≈ mixture` 成为训练约束。
3. 加强评价与展示：补充多样例试听包、失败案例、按类别统计，解释哪些类别更容易/更难分离。
4. 若还有算力，再尝试 hybrid 或官方 CUDA Mamba 环境；这应作为加分项，而不是 Week5 主线。

## 推荐 Week5 实施顺序
- 第一优先级：加入 waveform/STFT consistency loss 或 SI-SDR 辅助 loss，并保持当前 Mamba-style SSM 作为主干。
- 第二优先级：在采样脚本中同时导出 target prediction 和 residual prediction/estimate，形成完整二源分离输出。
- 第三优先级：做按类别和样例级分析，把 Week4 的平均指标变成可解释结论。
- 可选：准备一台能联网或预装 CUDA build toolchain 的服务器，重新安装官方 `mamba-ssm`，检验速度收益。

## Week5 起点文件
- 主模型：`week4/scripts/modeling_minidit.py`
- 训练脚本：`week4/scripts/train_minidit.py`
- Mamba 配置：`week4/configs/minidit_mamba.yaml`
- 最佳 checkpoint：`week4/runs/minidit_mamba_v1_bs48/checkpoints/best.pt`
- Week4 验收记录：`week4/收尾材料/week4_acceptance_report.md`
- Week4 课堂展示稿：`week4/展示ppt生成/deep_learning_week4_show.Rmd`

## 进入 Week5 前的固定口径
- Week3 baseline 的输出不是直接两段音频，而是目标声源的速度场/幅度谱预测，音频由预测幅度谱加 mixture phase 反变换得到。
- Week4 只改变底层序列混合算子，其余数据、目标、推理和指标尽量保持不变，这是实验对照成立的关键。
- 当前 Mamba 结果质量更好，但实现速度慢；速度问题来自 fallback 后端，不代表 Mamba CUDA kernel 的理论速度。
"""
    (OUT / "week5_handoff.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    metrics = {name: read_metrics(run) for name, run in MODELS.items()}
    summaries = {name: read_json(run, "eval_test_1step/summary.json") for name, run in MODELS.items()}
    run_summaries = {name: read_json(run, "run_summary.json") for name, run in MODELS.items()}

    save_training_curves(metrics)
    mixture_spec, mixture_wave, mixture_sisdr = save_metric_bars(summaries)
    runtime_hours = save_runtime_chart(metrics)
    rows = build_table_rows(summaries, run_summaries, runtime_hours)
    write_csv(rows)
    copy_sample_image()
    write_acceptance_report(rows, mixture_spec, mixture_wave, mixture_sisdr, runtime_hours)
    write_week5_handoff()

    for path in (
        OUT / "week4_operator_comparison_table.csv",
        OUT / "week4_acceptance_report.md",
        OUT / "week5_handoff.md",
        FIGDIR / "week4_training_curves.png",
        FIGDIR / "week4_test_metrics_comparison.png",
        FIGDIR / "week4_runtime_comparison.png",
        FIGDIR / "week4_mamba_sample_spectrogram_triplet.png",
    ):
        print(path)


if __name__ == "__main__":
    main()
