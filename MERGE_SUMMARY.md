# 项目文件夹合并总结

## 合并完成

已将原分散的项目文件成功合并到统一文件夹：
**`Lightweight_flow-matching_audio_separation`**

## 合并来源

| 原文件夹 | 内容 | 状态 |
|---------|------|------|
| `代码与模型/` | 核心代码、模型、结果 | ✓ 已合并 |
| `week2/data_esc50_2mix/` | 训练数据 | ✓ 已合并为 `week2_data/` |
| `基于高效序列建模的轻量级流匹配音频分离/wee7总结任务/` | 周报总结 | ✓ 已合并为 `weekly_reports/` |
| 根目录文档 | README、Memory、说明文件 | ✓ 已合并 |

## 去重结果

### 相同文件清理
- week3/week4 部分脚本重复 → **保留所有版本**（体现演进过程）
- 其他重复文件 → **保留完整版本**

### 保留理由
采用保留所有文件的策略，因为：
1. 周期版本间有改进迭代（week3→week4→week5→week6）
2. 重复的脚本反映模型架构演化
3. 便于追溯各阶段的技术决策

## 文件统计

- **总文件数**: 11,000+ 文件
- **总大小**: ~35GB (主要为训练数据)
- **代码文件**: Python脚本、配置文件、训练结果

## 文件夹结构

```
Lightweight_flow-matching_audio_separation/
├── week2/              # 数据集生成脚本
├── week2_data/         # ESC50 2mix 数据集 (~25GB)
├── week3/              # 流匹配模型 v1 (Self-Attention)
├── week4/              # 流匹配模型 v2 (Mamba)
├── week5/              # 波形微调阶段
├── week6/              # 最终模型版本
├── checkpoint/         # 模型检查点 (.pt)
├── results/            # 评估指标 (CSV/JSON)
├── samples_demo/       # 演示样本 (4个案例)
├── report_source/      # 论文源文件和图表
├── weekly_reports/     # 周期性总结
├── README.md           # 项目文档
├── Memory.md           # 研究记录
├── requirements.txt    # 依赖列表
├── PROJECT_STRUCTURE.md    # 项目结构说明
├── team_members.xlsx   # 团队信息
├── 数据来源.txt        # 数据集来源
└── 提交材料说明.md     # 提交指南
```

## 完整性检查

- ✓ 所有代码文件完整
- ✓ 所有数据集完整  
- ✓ 所有配置文件完整
- ✓ 所有训练结果完整
- ✓ 所有演示样本完整
- ✓ 所有文档完整

## 后续建议

1. **访问项目**: 直接使用 `Lightweight_flow-matching_audio_separation/` 文件夹
2. **查看文档**: 阅读 `PROJECT_STRUCTURE.md` 了解详细结构
3. **运行代码**: 参考各 week 文件夹内的脚本
4. **检查依赖**: 安装 `requirements.txt` 中的包

