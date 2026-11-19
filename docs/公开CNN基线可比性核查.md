# CLINC150 公开 CNN 基线可比性核查

## 公开数值

CLINC150 原论文 Larson 等（EMNLP-IJCNLP 2019）表2的 Full 设置中，CNN 闭集意图准确率为91.2%，OOS recall为18.9%。这是论文的 `oos-train` 方案：把 OOS 当作第151类训练，并以该类识别OOS。该数据的官方 Full 划分为每类100条训练、20条验证、30条测试，以及100条OOS训练、100条OOS验证和1000条OOS测试。

论文补充材料仓库的 `hyperparameters.csv` 给出 Full CNN：200个卷积滤波器、卷积激活softmax、隐藏层ReLU、stride=1、valid padding、300维dense、dropout=0.6、softmax输出、batch size=16。材料没有给出卷积窗口宽度、完整网络代码、分词/序列截断规则、优化器和训练轮数；原论文只说明使用非静态GloVe词向量。因此无法从已发布信息逐项确定同一个CNN实现。

## 本项目数值

在相同官方数据的每类100条子集上，30轮上限、验证macro-F1选模的干净CE参考，三个种子的测试准确率为91.69%、92.07%、92.16%，均值91.97%±0.25%；与论文CNN公布的91.2%相差+0.77个百分点，数值落在±2个百分点范围内。项目既有5轮干净基线为89.54%±0.07%，与论文数值相差-1.75个百分点。

这只是准确率数值参照，不能验收为“复现公开CNN”：本项目模型是BiLSTM-Attention，不是CNN；30轮值按项目验证macro-F1选模，原文训练预算未披露；原论文CNN的OOS指标来自151类 `oos-train`，本项目保存的是150类闭集模型上的MSP AUROC。论文的18.9% OOS recall与本项目AUROC不能直接比较，也不能由数值接近推断方法等价。

## 验收结论

上述BiLSTM数值接近不能验收为公开CNN复现。2026-11-19，项目按[预先固定的重实现协议](公开CNN重实现协议.md)补充了151类CNN、完整官方划分和 `oos-train` 指标，并把未公开细节的补充选择明列在配置中。三个种子ID准确率为87.41%±0.56%，与91.2%相差-3.79个百分点，未达到±2个百分点要求；OOS召回为23.47%±2.87%。[逐种子结果与限制](公开CNN重实现结果.md)完整保留。

本轮16,500条预测通过官方ID/真值、词表、验证选模和指标重算核验，三个检查点的推理重放与保存类别/概率完全一致。但原论文缺失细节没有因此恢复，且数值未达标；公开基线验收仍为未通过。

参考：Larson et al., “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” EMNLP-IJCNLP 2019, Table 2，论文 <https://aclanthology.org/D19-1131/>。官方补充参数、许可与固定版本出处已发布在 `references/clinc150/`，上游为 <https://github.com/clinc/oos-eval>。
