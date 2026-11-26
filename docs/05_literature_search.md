# 文献检索任务

正式英文论文只引用已核验且支持具体论点的来源。

| 优先级 | 方向 | 英文检索词与问题 |
|---|---|---|
| 高 | Castelli 原始数据 | computational screening perovskite solar light capture，核对原数据范围与赛题改编关系 |
| 高 | GLLB-SC、带隙带边 | band gap discontinuity、band edge，解释指标含义与精度 |
| 高 | 组成-only 特征 | composition-based property prediction、elemental descriptors、site-aware descriptors |
| 高 | 稀有目标与预算 | rare target discovery、imbalanced regression、top-k screening、simple regret |
| 高 | 隐藏近目标与外推 | leave-best-out validation、materials discovery extrapolation、selection bias |
| 高 | QUBO 与消元 | one-hot penalty、partial minimization、variable elimination |
| 高 | 系数精度与成本 | Ising coefficient quantization、analog control error、precision adaptation、time-to-solution |
| 方法采用后 | CatBoost、分支、概率校准 | categorical boosting、mixture/hurdle model、probability calibration |
| 工具 | Kaiwu 和 pymatgen | 实际版本的官方 API、授权、系数和后端说明 |

## 已核对的起点

题名与 DOI 可检索，仍需读原文后确定支持哪个具体论点。贡献是否有新意，需要查相近工作。

- Castelli 等，Computational screening of perovskite metal oxides for optimal solar light capture，2012。[10.1039/C1EE02717D](https://doi.org/10.1039/C1EE02717D)
- Kuisma 等，Kohn-Sham potential with discontinuity for band gap materials，2010。[10.1103/PhysRevB.82.115106](https://doi.org/10.1103/PhysRevB.82.115106)
- Ward 等，A general-purpose machine learning framework for predicting properties of inorganic materials，2016。[10.1038/npjcompumats.2016.28](https://doi.org/10.1038/npjcompumats.2016.28)
- Dunn 等，Benchmarking materials property prediction methods: the Matbench test set and Automatminer reference algorithm，2020。[10.1038/s41524-020-00406-3](https://doi.org/10.1038/s41524-020-00406-3)
- Lucas，Ising formulations of many NP problems，2014。[10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005)
- Butler 等，Machine learning for molecular and materials science，2018。[10.1038/s41586-018-0337-2](https://doi.org/10.1038/s41586-018-0337-2)
- [Kaiwu 安装和授权](https://kaiwu-sdk-docs.qboson.com/zh/latest/source/getting_started/sdk_installation_instructions.html)
- [Kaiwu 官方文档](https://kaiwu-sdk-docs.qboson.com/zh/latest/)

## 给另一位 AI 的任务

> 为 2026 APMCM Problem D 检索英文原始论文和官方技术文档。按“组成-only 代理模型、稀有目标排序与单点选择、人工隐藏近目标的外推检验、QUBO 精确消元与 one-hot 罚项、Ising 系数量化和专用求解器基准”组织。每篇提供完整英文题名、作者、年份、期刊、DOI、稳定链接、可公开获取全文的位置、支持本题哪个论点与适用限制。核对 Castelli 原始数据和赛题附件关系，分别标注来源论文的材料范围和比赛额外设置。核实文献存在，提供 BibTeX。赛题的 19 个 V=0 及隐藏集分配属于比赛题面和附件说明。检索分支模型、近目标留出、X 消元和精度检验的相近研究，帮助准确界定贡献。

建议返回文献表、BibTeX 与逐篇可用于正文的证据。AI 概述应与原文核对。
