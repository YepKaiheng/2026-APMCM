# 环境与复现

本机 Python 3.12.14 已核验 CSV 读取、CatBoost 拟合、Matplotlib 绘制、pymatgen 元素查询、Kaiwu QUBO 构造，pip check 通过。TeX Live 2026 和 PDF 工具可用。

Kaiwu 发行包版本 1.4.4，包内 kaiwu.__version__ 曾显示社区版本 1.0.8，版本核对采用 importlib.metadata.version("kaiwu")。实际 SimulatedAnnealingOptimizer 调用停在首次授权，仍需官方用户 ID 和 SDK 授权码，实际求解待完成。

## 新机器

Windows PowerShell，仓库根目录。

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r problem_d/requirements.txt
& .venv/Scripts/python.exe -m pip check
```

启动器找不到已安装的 Python 时，调用实际 Python 3.12 完整路径执行 -m venv .venv。本机原解释器由 uv 管理。直接依赖在 requirements.txt，完整版本在 requirements-lock.txt；跨平台安装先核对轮子兼容性。

## 重现本次实验

```powershell
& .venv/Scripts/python.exe problem_d/exploration/audit_data.py
& .venv/Scripts/python.exe problem_d/exploration/compare_surrogates.py --seed 26 --predictions problem_d/results/validation_seed26.csv
& .venv/Scripts/python.exe problem_d/exploration/compare_surrogates.py --seed 42 --predictions problem_d/results/validation_seed42.csv
& .venv/Scripts/python.exe problem_d/exploration/compare_objectives.py --seed 26 --predictions problem_d/results/objectives_seed26.csv
& .venv/Scripts/python.exe problem_d/exploration/test_innovation_ideas.py stress --seed 26 --predictions problem_d/results/stress_seed26.csv
& .venv/Scripts/python.exe problem_d/exploration/test_innovation_ideas.py stress --seed 42 --predictions problem_d/results/stress_seed42.csv
& .venv/Scripts/python.exe problem_d/exploration/test_innovation_ideas.py encoding --seed 26 --predictions problem_d/results/encoding_seed26.csv
& .venv/Scripts/python.exe problem_d/exploration/summarize_results.py
```

代码固定参数和种子，使用 CPU CatBoost，并保存逐行预测。硬件线程和依赖变化可能引起浮点排序差异，所以记录实际环境。预算指标可独立从预测表复算。

原始 D 题 .venv 只是本机环境，仓库用清单复现。授权文件在用户本机配置。

## LaTeX

复制 sources/latex 模板到论文工作目录，填写 D 和真实队号并替换示例。PDFLaTeX 或现有 XeLaTeX 按模板使用，按引用工具完成多遍编译，检查引用和页面。类文件未注册到系统时与主 tex 同目录即可。
