# C04：短文本意图识别的小样本与带噪标签联合鲁棒学习

本项目研究小样本与标签噪声共同作用下的短文本意图识别，采用 CLINC150 数据集，计划对比交叉熵基线、GCE 鲁棒损失和置信学习清洗三条路线。

当前版本完成数据准备、噪声构造及GloVe-BiLSTM-Attention干净基线，已运行5/10/20/100-shot、三个种子共12组训练与评估。运行方法见[基线运行说明](docs/基线运行说明.md)，结果与问题分析见[阶段2基线结果](docs/阶段2基线结果.md)。

## 当前可运行内容

- 固定官方 CLINC150 `data_full.json` 的版本，下载后保存 SHA256 和下载时间。
- 严格检查 150 个类、六个官方划分及各类样本数，保留官方验证、测试与 OOS 划分。
- 种子 `17 / 42 / 2026` 下，每类 `5 / 10 / 20 / 100` 条的嵌套训练子集。
- 精确控制翻转总数的对称噪声，10% / 20% / 30%，原标签绝不翻回自身。
- 实例相关噪声生成器：接收有明确编码器版本的真实句向量，结合样本与类别原型的相似度构造噪声。当前尚未下载编码器、生成真实实例噪声数据。
- 原标签与翻转记录仅写入独立 `audit/`；供模型使用的 `train.jsonl` 只含 `id / text / label`。
- GloVe初始化BiLSTM-Attention、AdamW交叉熵训练、验证集选模、断点续训和独立测试入口。
- 150类宏F1/准确率、OOS AUROC、逐类F1、混淆矩阵、预测明细与学习曲线。
- 多种子批量实验、均值±样本标准差汇总，以及从逐样本预测独立复核结果的脚本。

## 环境与运行

在本目录执行，推荐Python 3.12；数据阶段唯一第三方依赖为NumPy 2.3.5。以下命令仅运行数据管线。训练环境安装步骤见[基线运行说明](docs/基线运行说明.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m c04.prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m c04.check_stage1
```

无需激活环境。下载失败时，可从官方固定链接获取文件，再运行：

```powershell
.\.venv\Scripts\python.exe -m c04.prepare --source "D:\datasets\data_full.json"
```

本地导入与在线下载均核对固定 SHA256。导入文件路径会记入处理清单，验收脚本直接读取该路径，无需另外复制到默认目录。更换数据版本时需同时修改配置中的提交号与校验值。

官方固定数据地址：
<https://raw.githubusercontent.com/clinc/oos-eval/48a0e1cff8f43dd4d0836ecb4ed5df08733e3d2e/data/data_full.json>

单独生成某组对称噪声：

```powershell
.\.venv\Scripts\python.exe -m c04.noise --input data/processed/fewshot/seed_17/k_10.jsonl --kind symmetric --rate 0.3 --seed 17 --output data/noisy/manual_s17_k10_sym30
```

GCE 使用与 CE 相同的训练入口，只需切换配置：

```powershell
.\.venv\Scripts\python.exe -m c04.train --train data/noisy/manual_s17_k10_sym30/train.jsonl --config configs/gce.json --seed 17 --device cuda --output artifacts/gce_s17_k10_sym30
```

GCE 试跑结果和当前限制记录在 `reports/stage3_gce_pilot.json`。

## 文件与结果位置

| 路径 | 用途 |
| --- | --- |
| `configs/protocol.json` | 固定种子、数据版本、实验与验收约定 |
| `c04/prepare.py` | 官方下载、数量核验、嵌套抽样 |
| `c04/noise.py` | 对称与实例相关噪声构造 |
| `c04/check_stage1.py` | 真实数据验收和 36 组对称噪声数据生成 |
| `c04/prepare_glove.py` | Stanford词向量分段下载、解压和固定SHA256校验 |
| `c04/train.py` / `c04/evaluate.py` | 训练、断点续训与独立闭集/OOS评估 |
| `c04/run_baseline_matrix.py` | 三种子、四个数据规模的干净基线矩阵 |
| `c04/export_results.py` / `c04/check_results.py` | 导出原始预测、日志和图表，并独立重算结果 |
| `reports/stage2/clean_runs/` | 12组逐种子预测、指标、配置、日志与图表 |
| `tests/test_data.py` | 数据隔离、重放一致性与噪声性质测试 |
| `reports/stage1_validation.json` | 数据检查结果与各组数据文件校验值 |
| `data/processed/manifest.json` | 原始文件与处理产物校验值 |
| `data/noisy/symmetric/seed_*/k_*/r_*` | 生成的训练输入、审计真值与清单 |
| [阶段计划与交付检查表](docs/阶段计划与交付检查表.md) | 各阶段验收条件与剩余任务 |
| [实验协议](docs/实验协议.md) | 噪声定义、数据隔离、统计与资源约定 |
| [过程记录](docs/过程记录.md) | 阶段实现、验证记录和已知问题 |

大数据、中间文件不提交 Git；生成脚本、配置、阶段报告进入 Git。重新下载同一固定数据并按相同环境运行即可重建数据。生成时间等元数据随运行变化，训练数据文件的内容与 SHA256 应保持一致。

## 当前验收结果

2026-11-03，在 Python 3.12.14 / NumPy 2.3.5 下完成：13 项自动测试通过，18 个干净/小样本文件通过校验，36 组真实 CLINC150 对称噪声数据生成并重放一致。原数据 SHA256：

`36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`

官方划分的标准化文本存在训练-验证 3 条、训练-测试 2 条重合，已保留原始划分并在检查报告中披露。测试中的合成句向量仅验证代码性质，不代表真实语义噪声实验。

## 下一阶段

完整训练集的三种子测试宏F1为 **0.8945±0.0007**，准确率 **0.8954±0.0007**，OOS AUROC **0.9221±0.0022**。每组最多5轮，检查点仅由验证宏F1选择。

5-shot和10-shot在5轮预算下存在欠拟合，已保留低分结果和单独的30轮预算诊断。后续先固定公平的训练预算并补齐公开基线对照，再接入GCE、五折置信学习和噪声矩阵；当前尚无鲁棒方法对照结论。

数据来源与使用条款以 [CLINC150 官方仓库](https://github.com/clinc/oos-eval) 为准。算法定义与参考资料见[实验协议](docs/实验协议.md)。
