# 英文论文与提交要求

依据 sources/rules 中的本届格式、纪律与提交说明。官方新增要求优先核对。

| 项目 | 要求 |
|---|---|
| 选题 | 系统为 D，内容和文件名一致 |
| 语言 | 英文摘要、正文、图表、坐标图例和附录说明 |
| 页序 | 第 1 页摘要，含 D 和队号；第 2 页目录；随后正文 |
| 页数 | 正文控制在 25 页内，附录不限 |
| 字体 | 可读的 11/12 pt |
| 页眉 | 队号和页码 |
| 匿名 | 摘要和正文无个人学校信息，签名在单独控制页 |
| 参考 | 真实可查，用 [1] 等编号引用，网络记录访问日期 |
| 源码 | 附录说明软件并给出所用源码，保留至最终名单公布 |
| 文件名 | D<队号>.pdf，D<队号>cns.pdf 或 .jpg |
| 支撑包 | 如提交则 D<队号>fj.zip/rar，排除原始比赛数据 |
| 截止 | 北京时间 2026-11-30 09:00，预留上传时间 |

英文说明对支撑材料有条件提交文字，中文说明称可选。先准备可复现材料，最终上传按系统与最新官方要求核对。

上游 CUMCM 的中文图表、30 页、固定图表公式篇数、国赛 AI 声明来自另一竞赛。专用技能采用本届 APMCM 材料，后续若有适用的 APMCM AI 规定再据其补充。

## 建议英文论文骨架

Summary → Contents → Introduction and Problem Formulation → Data Audit and Evaluation Protocol → Surrogate Modeling for Rare-Target Screening → QUBO Formulations and Exact Partial Minimization → Coefficient Precision, Kaiwu Results and Computational Cost → Sensitivity, Limitations and Conclusions → References → Appendix.

章节按实际方法调整。区分公开真实 V、代理 V_hat、量化能量和最终求解输出。摘要中的数字要有运行记录支持。

## 最后核验

- 三问完整，数据验证、QUBO、Kaiwu 后端与逐项成本明确。
- 输入可用、元素对切分、完整池/blind 成员关系正确。
- 原分数最优、量化后合法最优、实际解及约束违反核对。
- 摘要、正文、图表和代码对应同一模型版本。
- 英文、引用、页面与匿名经过检查。
- 真实队号、真实英文签名，提交前核对系统要求。
