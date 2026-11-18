# CLINC150 公开 CNN 基线可比性核查

## 公开数值

CLINC150 原论文 Larson 等（EMNLP-IJCNLP 2019）表2的 Full 设置中，CNN 闭集意图准确率为91.2%，OOS recall为18.9%。这是论文的 `oos-train` 方案：把 OOS 当作第151类训练，并以该类识别OOS。该数据的官方 Full 划分为每类100条训练、20条验证、30条测试，以及100条OOS训练、100条OOS验证和1000条OOS测试。

论文补充材料仓库的 `hyperparameters.csv` 给出 Full CNN：200个卷积滤波器、卷积激活softmax、隐藏层ReLU、stride=1、valid padding、300维dense、dropout=0.6、softmax输出、batch size=16。材料没有给出卷积窗口宽度、完整网络代码、分词/序列截断规则、优化器和训练轮数；原论文只说明使用非静态GloVe词向量。因此无法从已发布信息逐项确定同一个CNN实现。

## 本项目数值

在相同官方数据的每类100条子集上，30轮上限、验证macro-F1选模的干净CE参考，三个种子的测试准确率为91.69%、92.07%、92.16%，均值91.97%±0.25%；与论文CNN公布的91.2%相差+0.77个百分点，数值落在±2个百分点范围内。项目既有5轮干净基线为89.54%±0.07%，与论文数值相差-1.75个百分点。

这只是准确率数值参照，不能验收为“复现公开CNN”：本项目模型是BiLSTM-Attention，不是CNN；30轮值按项目验证macro-F1选模，原文训练预算未披露；原论文CNN的OOS指标来自151类 `oos-train`，本项目保存的是150类闭集模型上的MSP AUROC。论文的18.9% OOS recall与本项目AUROC不能直接比较，也不能由数值接近推断方法等价。

## 验收结论

课程要求的“公开模型数值±2%”目前只通过未匹配模型下的准确率数值筛查，尚未通过同模型复现验收。由于CNN卷积窗口和训练细节缺失，不应自行补一个窗口后称为论文复现。后续若要完成严格对照，应明确采用一个可获取的公开CNN实现与固定窗口配置，并标成“基于公开论文设置的重实现”；同时将OOS训练为第151类，保持官方训练/验证/测试划分与论文 `oos-train` 指标定义。

参考：Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” EMNLP-IJCNLP 2019, Table 2 and supplementary hyperparameter file. 论文PDF和可检索文本见 `tmp/clinc-reference/paper.pdf`、`tmp/clinc-supp.txt` 与 `tmp/clinc-reference/hyperparameters.csv`；上游数据和补充材料仓库为 <https://github.com/clinc/oos-eval>。
