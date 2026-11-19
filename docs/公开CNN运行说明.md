# 公开 CNN 重实现的运行与核验

环境沿用 `requirements-formal-lock.txt`（Python 3.12、PyTorch 2.9.1+cu126）。先按README准备官方数据与GloVe。模型配置为 `configs/public_cnn_v1.json`；151类训练和评估入口独立于主BiLSTM实验。

## 三种子训练

```powershell
foreach ($seed in @(17, 42, 2026)) {
    python -m c04.public_cnn --seed $seed --output "artifacts/public_cnn_v1/seed_$seed" --device cuda
    if ($LASTEXITCODE -ne 0) { throw "CNN seed $seed failed" }
}
```

输出目录必须不存在；重跑时用新目录。每个目录包含训练输入、训练专用词表、每轮验证轨迹、最佳检查点和5500条测试预测。训练集含15000条ID与100条OOS，仅验证选模；测试集在选模后读取。

## 发布与独立核验

```powershell
python -m c04.public_cnn_results --source artifacts/public_cnn_v1 --results reports/stage6/public_cnn_v1
python -m c04.public_cnn_results --results reports/stage6/public_cnn_v1
```

发布包只含指标、预测、日志、词表、151类混淆矩阵和逐类表，不含权重。核验同时检查官方划分哈希、全部测试ID/真值、训练词表、最优验证轮数、三种子指标和文件清单。首个命令要求结果目录不存在；第二个命令直接核验已发布文件，可在新Git克隆准备官方数据后执行。

## 本地权重重放

```powershell
python -m c04.public_cnn_results --results reports/stage6/public_cnn_v1 --replay-checkpoints artifacts/public_cnn_v1 --device cuda
```

核对最佳检查点SHA256后，对全部16500条测试记录重新推理；分类结果必须逐条相等，p(OOS)误差上限1e-7。结果记录在 `docs/公开CNN重实现结果_权重重放.json`。这是同一环境下权重推理重放，不是重新安装依赖并重训三次模型。

## 参数来源

原论文：<https://aclanthology.org/D19-1131/>。官方补充参数与CC BY 3.0许可副本存于 `references/clinc150/`，固定上游提交、链接与文件哈希见该目录 `source.json`。CNN结构公开参数与本次补充选择分别见[协议](公开CNN重实现协议.md)，未公布的实现细节不从测试分数反推。
