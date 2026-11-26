# 实验记录

保存本次整理时重新运行的输出、预测表和环境版本。

- `validation_seed26`、`validation_seed42` 使用官方 02 训练、03 验证，比较直接回归和带隙混合模型。
- `objectives_seed26` 使用官方 02 训练、03 验证，比较直接回归、log1p 目标和五项惩罚分别预测。
- `stress_seed26`、`stress_seed42` 仅在 02 内构造近目标留出，29 个低 V 元素对全部进入待筛池。
- `encoding_seed26` 使用 02+03 重训，对 08 的完整三-X 池检查精确消元、功能 ANOVA 探索和软件系数舍入。
- `data_audit.json` 核对数据校验和、元素对切分、V 重算、带隙与带边规律。
- `summary.csv` 从保存的预测表重新计算指标，可直接核对聊天中讨论的数字。
- `physical_seed26/42.csv/log` 补充四物理量预测后合成 V，以及由 VBM-gap 推导 CBM 的消融。
- `low_rank_audit.csv` 和 `external_ideas_audit.log` 检查低秩 R²、原分数选择遗憾与不同 Top50 定义。
- `sdk_precision_audit.csv/log` 实际调用 Kaiwu 官方 8 位转换接口，记录合法候选排序变化与罚系数影响。
- 上一级的 `requirements-lock.txt` 和本目录的 `environment.json` 记录实际运行的软件版本和平台。

所有 V=0 隐藏标签均不可用。带有真实 V 的预测表仅来自官方公开标签或在 02 内构造的压力测试。编码试验中的分数是代理预测，模拟的 8/12/16 位舍入只用于敏感性分析。

时间消耗、真机调用和 Kaiwu 求解成功率尚待实际运行。数值表按已记录种子读取，硬件或依赖变化时可能产生不同排序。
