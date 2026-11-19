# C04：短文本意图识别的小样本与带噪标签联合鲁棒学习

本项目研究小样本与标签噪声共同作用下的短文本意图识别，采用 CLINC150 数据集，计划对比交叉熵基线、GCE 鲁棒损失和置信学习清洗三条路线。

当前版本包含数据准备、噪声构造、GloVe-BiLSTM-Attention基线，以及扩展样本量/噪声比例下三种方法的三种子配对对照。10-shot、30%两种噪声的正式结果见[IDN结果](docs/阶段3正式对照结果.md)和[对称噪声结果](docs/阶段3对称噪声正式结果.md)；阶段5矩阵统计与核验见[扩展实验结果](docs/阶段5扩展实验结果.md)，固定协议见[扩展实验协议](docs/扩展实验协议.md)。公开CNN准确率参照和可比限制见[公开CNN核查](docs/公开CNN基线可比性核查.md)。

阶段5共核验10个噪声/样本量条件，每个条件包含CE、GCE和五折置信学习清洗后CE，各三个种子；配对干净CE参考另含5/10/20/100-shot。六个10-shot、30%置信学习结果复用了发布包，统计中没有计入意外重复运行的拷贝。

2026-11-19补充151类CNN重实现，三个种子ID准确率为87.41%±0.56%，OOS召回为23.47%±2.87%；与论文ID准确率相差-3.79个百分点，未达到±2个百分点要求。按论文公开参数和明确的补充选择实现，全部结果、预测、词表和核验脚本见[公开CNN结果](docs/公开CNN重实现结果.md)与[运行说明](docs/公开CNN运行说明.md)。三份本地权重重放全部16,500条预测一致；该项不算完整原代码复现或新环境重训验收。

在相同环境和输入资产准备好后，可重新生成阶段5输入并运行矩阵：

```powershell
python -m c04.prepare_expanded_matrix --output data/stage5_expanded --embeddings-template "artifacts/stage5_embeddings/idn_s{seed}_k{shots}.npz"
python -m c04.run_expanded_matrix --source data/stage5_expanded/matrix.json --output artifacts/stage5_expanded_runs --device cuda
python -m c04.stage5_results
```

已完成本地扩展训练时，可逐条核验预测，并重算汇总和40项配对比较：

```powershell
python -m c04.check_expanded_predictions
```

该命令需要本地 `artifacts/stage5_expanded_runs/`，不只依赖GitHub上的汇总。使用范围和本轮修正见[扩展矩阵代码核验说明](docs/扩展矩阵代码核验说明.md)。

## 当前可运行内容

- 固定官方 CLINC150 `data_full.json` 的版本，下载后保存 SHA256 和下载时间。
- 严格检查 150 个类、六个官方划分及各类样本数，保留官方验证、测试与 OOS 划分。
- 种子 `17 / 42 / 2026` 下，每类 `5 / 10 / 20 / 100` 条的嵌套训练子集。
- 精确控制翻转总数的对称噪声，10% / 20% / 30%，原标签绝不翻回自身。
- 实例相关噪声生成器：固定MiniLM编码器版本，结合真实句向量与类别原型的相似度构造噪声；已生成10-shot三个种子的IDN30%输入。
- 原标签与翻转记录仅写入独立 `audit/`；供模型使用的 `train.jsonl` 只含 `id / text / label`。
- GloVe初始化BiLSTM-Attention、AdamW交叉熵训练、验证集选模、断点续训和独立测试入口。
- 150类宏F1/准确率、OOS AUROC、逐类F1、混淆矩阵、预测明细与学习曲线。
- 多种子批量实验、均值±样本标准差汇总，以及从逐样本预测独立复核结果的脚本。
- cleanlab 2.9.0校准置信联合分布、五折OOF清洗与重训；保存折划分、训练专用词表、概率和清洗决策，支持独立重放。

## 本轮配对对照

每组固定最多30轮、早停耐心5，验证宏F1选模；三方法逐种子复用同一观测噪声文件，另跑相同预算的干净CE作为保持率参考。每组包含12次最终模型训练和15次OOF训练，两组分别运行。协议、依赖和完整命令见[正式对照运行说明](docs/正式对照运行说明.md)。

准备官方数据并安装 `requirements-formal-lock.txt` 后，可核验已发布预测并重建完全相同的观测训练输入：

```powershell
.\.venv\Scripts\python.exe -m c04.formal_results --results reports/stage3/formal_idn10_v2
.\.venv\Scripts\python.exe -m c04.prepare_formal_data --output data/formal_idn10_v2
```

对称噪声使用显式结果目录：

```powershell
.\.venv\Scripts\python.exe -m c04.formal_results --results reports/stage3/formal_sym10_v2
.\.venv\Scripts\python.exe -m c04.prepare_formal_data --results reports/stage3/formal_sym10_v2 --output data/replay_sym10_v2
```

旧清洗试跑使用的是阈值启发式，本轮改为校准置信学习，两个算法版本分别记录。清洗后的负结果保留；不根据测试分数或已知翻转数量调整删样数量。

## 消融实验

IDN30%、10-shot下，固定三个种子比较末状态汇聚、冻结GloVe和GCE q∈{0.2,0.4,0.7,0.9,0.95}。q=0.7完整模型复用既有参考，本轮新增18次训练。协议见[消融实验协议](docs/消融实验协议.md)，结果见[阶段4消融结果](docs/阶段4消融结果.md)，完整重跑见[运行说明](docs/消融实验运行说明.md)。

准备官方数据和依赖后，可从保存的预测、配置、词表与冻结矩阵核验本轮交付：

```powershell
.\.venv\Scripts\python.exe -m c04.ablation_results --results reports/stage4/ablations_idn10_v2
```

本轮只检查一个噪声/样本量条件，不根据测试最优值修改主参数q=0.7。末状态模型改变结构和随机初始化消耗，不能将配对差值当作保持全部权重初值的因果实验。

清洗变体在同一IDN条件下补充OOF硬重标注、概率加权和取消OOF后删除，各三个种子，并重跑普通CE检查权重接口。12次新训练的规则、结果和命令分别见[清洗变体协议](docs/清洗变体实验协议.md)、[阶段4清洗变体结果](docs/阶段4清洗变体结果.md)及[运行说明](docs/清洗变体运行说明.md)。

```powershell
.\.venv\Scripts\python.exe -m c04.cleaning_variant_results --results reports/stage4/cleaning_variants_idn10_v2
```

权重独立按训练ID对齐并纳入检查点指纹。取消OOF使用训练内预测，仅作为消融，不将其当作折外标签质量估计。原始负结果和事后误改/误删诊断均保留。

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

五折 OOF 置信学习入口（需额外安装 `requirements-cleaning.txt`；本轮正式设置使用 `configs/formal_ce.json`）：

```powershell
.\.venv\Scripts\python.exe -m c04.run_confident_learning --train data/noisy/manual_s17_k10_sym30/train.jsonl --config configs/baseline.json --seed 17 --folds 5 --device cuda --output artifacts/confident_learning_s17_k10_sym30
```

该入口只将观测训练标签和折外概率用于清洗，并保存每个样本的折号、概率和清洗报告。`reports/stage3_oof_pilot.json` 为旧阈值算法的历史试跑记录，不能视为当前算法的复现结果。

真实实例相关噪声使用固定的 MiniLM 句向量模型。先安装 `requirements-idn.txt`，再按 `configs/idn_encoder.json` 中的 revision 下载模型并导出训练向量：

```powershell
.\.venv\Scripts\python.exe -m c04.sentence_embeddings --input data/processed/fewshot/seed_17/k_10.jsonl --output artifacts/idn_embeddings_s17_k10.npz --model <local-fixed-model-directory> --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --device cuda
.\.venv\Scripts\python.exe -m c04.noise --input data/processed/fewshot/seed_17/k_10.jsonl --kind instance --rate 0.3 --seed 17 --embeddings artifacts/idn_embeddings_s17_k10.npz --output artifacts/noisy_idn30_s17_k10
```

模型文件和向量校验记录在 `reports/stage3_idn_asset_pilot.json`。

固定真实 IDN30%、10-shot、三个 seed 的 CE 矩阵可用下面的入口重跑：

```powershell
.\.venv\Scripts\python.exe -m c04.run_noise_matrix --seeds 17 42 2026 --shots 10 --noise instance --rate 0.3 --method ce --epochs 5 --device cuda --embeddings-template 'artifacts/stage3_pilot/idn_embeddings_s{seed}_k{shots}.npz' --output artifacts/stage3_idn_matrix_ce_k10
```

汇总结果记录在 `reports/stage3_idn_ce_matrix.json`。

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
| `reports/stage3/formal_idn10_v2/` | 本轮12组预测、日志、图表、OOF概率及清洗证据 |
| `reports/stage3/formal_sym10_v2/` | 对称30%扩展的12组结果及OOF清洗证据 |
| `c04/run_formal_comparison.py` | 固定范围的27次训练与独立评估入口 |
| `c04/formal_results.py` / `c04/prepare_formal_data.py` | 独立验收及精确重建观测标签 |
| `c04/compare_noise_conditions.py` | 两类条件比较及干净参考重跑一致性核对 |
| `configs/ablation_idn10_v2.json` / `c04/run_ablations.py` | 三组消融的固定清单与18次训练入口 |
| `reports/stage4/ablations_idn10_v2/` | 消融预测、配置、词表和冻结嵌入证据 |
| `c04/ablation_results.py` / `c04/analyze_ablations.py` | 消融独立核验、配对差值和图表 |
| `c04/run_cleaning_variants.py` / `configs/cleaning_variants_idn10_v2.json` | 清洗变体固定协议与12次训练入口 |
| `reports/stage4/cleaning_variants_idn10_v2/` | 预测、处理决策、权重和训练内打分证据 |
| `c04/cleaning_variant_results.py` | 重放处理规则及独立核验 |
| `c04/paired_statistics.py` | 固定三种子结果的配对检验、95%区间与Holm校正 |

对固定矩阵执行配对统计（先核对四个结果包审计与文件校验值）：

```powershell
.\.venv\Scripts\python.exe -m c04.paired_statistics
```

统计结果见[配对统计报告](docs/配对统计结果.md)，完整逐种子数据见 `reports/stage4/paired_statistics.json`。
| `tests/test_data.py` | 数据隔离、重放一致性与噪声性质测试 |
| `reports/stage1_validation.json` | 数据检查结果与各组数据文件校验值 |
| `data/processed/manifest.json` | 原始文件与处理产物校验值 |
| `data/noisy/symmetric/seed_*/k_*/r_*` | 生成的训练输入、审计真值与清单 |
| [阶段计划与交付检查表](docs/阶段计划与交付检查表.md) | 各阶段验收条件与剩余任务 |
| [实验协议](docs/实验协议.md) | 噪声定义、数据隔离、统计与资源约定 |
| [过程记录](docs/过程记录.md) | 阶段实现、验证记录和已知问题 |

大数据、中间文件不提交 Git；生成脚本、配置、阶段报告进入 Git。开题报告和课程要求图片仅保留本地，不上传；本轮新增的大批实验产物也不自动上传。重新下载同一固定数据并按相同环境运行即可重建数据。生成时间等元数据随运行变化，训练数据文件的内容与 SHA256 应保持一致。

## 当前验收结果

2026-11-03，在 Python 3.12.14 / NumPy 2.3.5 下完成：13 项自动测试通过，18 个干净/小样本文件通过校验，36 组真实 CLINC150 对称噪声数据生成并重放一致。原数据 SHA256：

`36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`

官方划分的标准化文本存在训练-验证 3 条、训练-测试 2 条重合，已保留原始划分并在检查报告中披露。测试中的合成句向量仅验证代码性质，不代表真实语义噪声实验。

## 已有结果与后续范围

完整训练集的三种子测试宏F1为 **0.8945±0.0007**，准确率 **0.8954±0.0007**，OOS AUROC **0.9221±0.0022**。每组最多5轮，检查点仅由验证宏F1选择。

5-shot和10-shot在5轮预算下存在欠拟合，已保留低分结果和单独的30轮预算诊断。以统一30轮上限完成扩展样本量和噪声比例的10条件矩阵，并在IDN条件下补齐模块、嵌入、q及清洗变体/取消OOF检查；当前矩阵已给出配对统计，但三种子检验能力有限。后续仍需可比公开基线验收、新依赖环境训练验证，以及报告PDF和PPT。

数据来源与使用条款以 [CLINC150 官方仓库](https://github.com/clinc/oos-eval) 为准。算法定义与参考资料见[实验协议](docs/实验协议.md)。
