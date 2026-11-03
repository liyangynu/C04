# C04：短文本意图识别的小样本与带噪标签联合鲁棒学习

本仓库按照开题报告分阶段交付。**当前只完成阶段 1：实验协议与数据管线**，尚未训练模型，也没有声称达到宏 F1 提升 5 个百分点。人员分工由小组自行安排；仓库不生成个人贡献、虚构提交或补写过去日期的记录。

## 当前可运行内容

- 固定官方 CLINC150 `data_full.json` 的版本，下载后保存 SHA256 和下载时间。
- 严格检查 150 个类、六个官方划分及各类样本数，保留官方验证、测试与 OOS 划分。
- 种子 `17 / 42 / 2026` 下，每类 `5 / 10 / 20 / 100` 条的嵌套训练子集。
- 精确控制翻转总数的对称噪声，10% / 20% / 30%，原标签绝不翻回自身。
- 实例相关噪声生成器：接收有明确编码器版本的真实句向量，结合样本与类别原型的相似度构造噪声。当前尚未下载编码器、生成真实实例噪声数据。
- 原标签与翻转记录仅写入独立 `audit/`；供模型使用的 `train.jsonl` 只含 `id / text / label`。

## 环境与运行

在本目录执行，推荐 Python 3.12；阶段 1 唯一第三方依赖为 NumPy 2.3.5。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m c04.prepare
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m c04.check_stage1
```

无需激活环境。下载失败时，可从官方固定链接获取文件，再运行：

```powershell
.\.venv\Scripts\python.exe -m c04.prepare --source "你的data_full.json绝对路径"
```

本地导入会标记为“版本未经下载来源验证”；如需运行完整的阶段验收命令，还应将同一文件保存为 `data/raw/data_full.json`。不应将其他变体改名后冒充完整版本。

官方固定数据地址：
<https://raw.githubusercontent.com/clinc/oos-eval/48a0e1cff8f43dd4d0836ecb4ed5df08733e3d2e/data/data_full.json>

单独生成某组对称噪声：

```powershell
.\.venv\Scripts\python.exe -m c04.noise --input data/processed/fewshot/seed_17/k_10.jsonl --kind symmetric --rate 0.3 --seed 17 --output data/noisy/manual_s17_k10_sym30
```

## 文件与结果位置

| 路径 | 用途 |
| --- | --- |
| `configs/protocol.json` | 固定种子、数据版本、实验与验收约定 |
| `c04/prepare.py` | 官方下载、数量核验、嵌套抽样 |
| `c04/noise.py` | 对称与实例相关噪声构造 |
| `c04/check_stage1.py` | 真实数据验收和 36 组对称噪声数据生成 |
| `tests/test_data.py` | 数据隔离、重放一致性与噪声性质测试 |
| `reports/stage1_validation.json` | 已运行的真实数据检查证据，纳入 Git |
| `data/processed/manifest.json` | 原始文件与处理产物校验值 |
| `data/noisy/symmetric/seed_*/k_*/r_*` | 生成的训练输入、审计真值与清单 |
| `docs/阶段计划与交付检查表.md` | 各阶段验收条件与剩余任务 |
| `docs/实验协议.md` | 噪声定义、数据隔离、统计与资源约定 |
| `docs/过程记录.md` | 本次实际工作与后续记录规则 |

大数据、中间文件不提交 Git；生成脚本、配置、阶段报告进入 Git。重新下载同一固定数据并按相同环境运行即可重建数据。生成时间等元数据随运行变化，训练数据文件的内容与 SHA256 应保持一致。

## 当前验收结果

2026-11-03，在 Python 3.12.14 / NumPy 2.3.5 下完成：10 项自动测试通过，18 个干净/小样本文件通过校验，36 组真实 CLINC150 对称噪声数据生成并重放一致。原数据 SHA256：

`36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`

官方划分的标准化文本存在训练-验证 3 条、训练-测试 2 条重合，已保留原始划分并在检查报告中披露。测试中的合成句向量仅验证代码性质，不代表真实语义噪声实验。

## 下一阶段

阶段 2 实现并验证 GloVe-BiLSTM-Attention 干净基线、训练入口、早停、检查点、闭集与 OOS 指标。后续阶段才增加 GCE、五折置信学习、三种子完整结果与消融；当前仓库没有用空函数冒充这些功能。

阶段 Git 记录是当前真实操作的证据，不等于每位同学的独立工作量证明。后续请由实际完成者提交、注明实际改动和实验产物，不改写本阶段作者或日期。

数据来源与使用条款以 [CLINC150 官方仓库](https://github.com/clinc/oos-eval) 为准。本阶段 Python 代码为本任务新写，没有整块复制第三方实现；论文算法定义与参考资料见实验协议。
