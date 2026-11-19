"""Export and independently check the three paper-guided CNN runs."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .common import file_sha256, read_json, read_jsonl, write_json
from .metrics import binary_auroc, classification_metrics
from .runtime import load_official_split
from .text import build_vocab


def export(source, output):
    if output.exists():
        raise ValueError("Use a new result directory")
    config_path = Path("configs/public_cnn_v1.json")
    config = read_json(config_path)
    for seed in config["seeds"]:
        if not (source / f"seed_{seed}/metrics.json").is_file():
            raise ValueError(f"Incomplete run: seed={seed}")
    for seed in config["seeds"]:
        run, dest = source / f"seed_{seed}", output / f"seed_{seed}"
        dest.mkdir(parents=True)
        for name in ("run.json", "history.json", "metrics.json", "predictions.jsonl", "confusion_matrix.csv", "per_class.csv"):
            shutil.copyfile(run / name, dest / name)
        shutil.copyfile(run / "assets/vocab.json", dest / "vocab.json")
        log_path = source / f"seed_{seed}.log"
        if log_path.is_file():
            (dest / "train.log").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    shutil.copyfile(config_path, output / "config.json")
    write_json(output / "checksums.json", {p.relative_to(output).as_posix(): file_sha256(p)
                                          for p in sorted(output.rglob("*")) if p.is_file()})


def verify(root):
    checksums = read_json(root / "checksums.json")
    paths = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "checksums.json"}
    if paths != set(checksums):
        raise ValueError("Result file inventory differs from checksums")
    for name, digest in checksums.items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Result checksum mismatch: {name}")
    config = read_json(root / "config.json")
    if file_sha256(root / "config.json") != file_sha256("configs/public_cnn_v1.json"):
        raise ValueError("Published config differs from frozen v1 config")
    official = {}
    for split in ("train", "oos_train", "val", "oos_val", "test", "oos_test"):
        official[split], _ = load_official_split(Path(f"data/processed/clean/{split}.jsonl"), split)
    train_rows = official["train"] + [dict(r, id="train:" + r["id"], label=150) for r in official["oos_train"]]
    expected_train_sha = hashlib.sha256("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in train_rows).encode("utf-8")).hexdigest()
    expected_vocab = build_vocab(train_rows, config["text"]["min_frequency"])
    expected_test = [(r["id"], r["label"]) for r in official["test"]] + [(r["id"], 150) for r in official["oos_test"]]
    records = []
    for seed in config["seeds"]:
        dest = root / f"seed_{seed}"
        run, report = read_json(dest / "run.json"), read_json(dest / "metrics.json")
        if run["seed"] != seed or report["seed"] != seed or run["config"] != config:
            raise ValueError("Run seed/config differs from frozen config")
        if (run["config_sha256"] != file_sha256(root / "config.json") or
                run["train_rows"] != len(train_rows) or run["validation_rows"] != 3100 or
                run["assets"]["train_sha256"] != expected_train_sha or report["train_sha256"] != expected_train_sha):
            raise ValueError("Training configuration/data fingerprint mismatch")
        if read_json(dest / "vocab.json") != expected_vocab:
            raise ValueError("CNN vocabulary differs from training-only vocabulary")
        for split, digest in run["official_input_sha256"].items():
            if file_sha256(f"data/processed/clean/{split}.jsonl") != digest:
                raise ValueError(f"CNN {split} fingerprint mismatch")
        for split, field in (("test", "test_sha256"), ("oos_test", "oos_test_sha256")):
            if file_sha256(f"data/processed/clean/{split}.jsonl") != report[field]:
                raise ValueError("Evaluation split fingerprint mismatch")
        rows = read_jsonl(dest / "predictions.jsonl")
        if [(r["id"], r["true_label"]) for r in rows] != expected_test:
            raise ValueError("Prediction IDs/truth differ from official data")
        if file_sha256(dest / "predictions.jsonl") != report["predictions_sha256"]:
            raise ValueError("Prediction fingerprint mismatch")
        truth = np.asarray([r["true_label"] for r in rows], dtype=np.int64)
        predicted = np.asarray([r["predicted_class"] for r in rows], dtype=np.int64)
        is_id = truth != 150
        id_result = classification_metrics(truth[is_id], predicted[is_id], 151)
        full = classification_metrics(truth, predicted, 151)
        scores = np.asarray([r["p_oos"] for r in rows])
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("Invalid OOS probabilities")
        values = {"id_accuracy": float(np.mean(truth[is_id] == predicted[is_id])),
                  "oos_recall": float(np.mean(predicted[~is_id] == 150)),
                  "id_macro_f1": float(id_result["f1"][:150].mean()),
                  "oos_auroc_p_oos": binary_auroc((~is_id).astype(int), scores)}
        if any(abs(v - report["metrics"][k]) > 1e-12 for k, v in values.items()):
            raise ValueError("Metrics disagree with saved predictions")
        with (dest / "confusion_matrix.csv").open(encoding="utf-8", newline="") as stream:
            table = list(csv.reader(stream))
        if not np.array_equal(np.asarray([r[1:] for r in table[1:]], dtype=int), full["confusion_matrix"]):
            raise ValueError("CNN confusion matrix mismatch")
        with (dest / "per_class.csv").open(encoding="utf-8", newline="") as stream:
            per_class = list(csv.DictReader(stream))
        if len(per_class) != 151 or any(int(r["id"]) != i or any(
                abs(float(r[k]) - full[k][i]) > 1e-12 for k in ("f1", "precision", "recall", "support"))
                for i, r in enumerate(per_class)):
            raise ValueError("CNN per-class F1 mismatch")
        history = read_json(dest / "history.json")
        best = max(history, key=lambda h: h["val_accuracy"])
        if best["epoch"] != report["best_epoch"] or best["val_accuracy"] != report["best_validation_accuracy"]:
            raise ValueError("Checkpoint was not selected by frozen validation criterion")
        records.append({"seed": seed, **values, "best_epoch": report["best_epoch"],
                        "epochs_completed": report["epochs_completed"], "wall_seconds": report["wall_seconds"],
                        "peak_allocated_mb": report["peak_allocated_mb"], "peak_reserved_mb": report["peak_reserved_mb"]})
    summaries = {k: {"mean": float(np.mean([r[k] for r in records])), "std_sample": float(np.std([r[k] for r in records], ddof=1))}
                 for k in ("id_accuracy", "oos_recall", "id_macro_f1", "oos_auroc_p_oos")}
    delta = (summaries["id_accuracy"]["mean"] - config["reference"]["id_accuracy"]) * 100
    return {"all_passed": True, "checksums_sha256": file_sha256(root / "checksums.json"),
            "scope": "saved-prediction, official-split, vocabulary, validation-selection and metric verification; no weight inference replay",
            "records": records, "summary": summaries, "reference": config["reference"],
            "id_accuracy_difference_pp": delta, "within_two_percentage_points": abs(delta) <= 2,
            "implementation_status": config["status"]}


def plot_report(root, report, output):
    cache = Path("tmp/matplotlib").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in report["records"]:
        h = read_json(root / f"seed_{r['seed']}/history.json")
        axes[0].plot([x["epoch"] for x in h], [x["val_accuracy"] for x in h], label=f"seed {r['seed']}")
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (151 classes)", ylim=(0, 1), title="Fixed CNN v1 training")
    axes[0].legend()
    x = np.arange(2)
    means = [report["summary"][k]["mean"] for k in ("id_accuracy", "oos_recall")]
    errors = [report["summary"][k]["std_sample"] for k in ("id_accuracy", "oos_recall")]
    axes[1].bar(x-.18, [report["reference"][k] for k in ("id_accuracy", "oos_recall")], width=.36, label="Paper (Table 2)", color="#8999aa")
    axes[1].bar(x+.18, means, width=.36, yerr=errors, capsize=4, label="Reimplementation (mean +/- SD)", color="#337ab7")
    axes[1].set(xticks=x, xticklabels=["ID accuracy", "OOS recall"], ylim=(0, 1.06), title="Full / oos-train")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_report(report, stem):
    write_json(stem.with_suffix(".json"), report)
    lines = ["# 公开 CNN 重实现结果 v1", "", "固定配置先于本轮训练提交；三个种子未依据测试结果调整配置。参数来源和补充选择见[重实现协议](公开CNN重实现协议.md)。", "",
             "| seed | 最佳/完成轮数 | ID Accuracy | OOS Recall | ID Macro-F1 | p(OOS) AUROC |", "| --- | --- | --- | --- | --- | --- |"]
    for r in report["records"]:
        lines.append(f"| {r['seed']} | {r['best_epoch']}/{r['epochs_completed']} | {r['id_accuracy']:.4f} | {r['oos_recall']:.4f} | {r['id_macro_f1']:.4f} | {r['oos_auroc_p_oos']:.4f} |")
    s = report["summary"]
    lines += ["", f"ID准确率均值±样本标准差为 {s['id_accuracy']['mean']:.2%}±{s['id_accuracy']['std_sample']:.2%}；论文Full/oos-train CNN为91.2%，差 {report['id_accuracy_difference_pp']:+.2f} 个百分点。±2个百分点数值检查：{'达到' if report['within_two_percentage_points'] else '未达到'}。",
              "", f"OOS召回率为 {s['oos_recall']['mean']:.2%}±{s['oos_recall']['std_sample']:.2%}；论文值18.9%。OOS按151类argmax判断，没有使用测试集选择阈值。", "",
              "本轮恢复了论文的CNN模型类别、完整官方划分及oos-train指标口径，但卷积窗口、池化和训练设置等采用协议中明示的补充选择。数值接近不构成原代码或所有超参数一致的证明；结果称为按论文公开参数重实现。", "",
              "ID Macro-F1只在4500条ID测试样本上计算，预测为OOS会造成漏判；151类逐类表/混淆矩阵覆盖全部5500条测试样本，因此逐类表包含OOS误判引起的ID假阳性。p(OOS) AUROC与主实验的MSP AUROC使用不同打分，分别报告。", "",
              "官方测试ID/真值、保存预测、逐类表、混淆矩阵、训练专用词表、验证选模和文件SHA256独立核验通过。结果包保存在reports/stage6/public_cnn_v1；完整模型保留于本地artifacts/public_cnn_v1，本次核验不等于从权重独立重放全部预测。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def replay_weights(root, checkpoints, requested_device):
    """Separately verify that local weights reproduce every published prediction."""
    import torch
    from .public_cnn import PublicCNN, official_rows
    from .runtime import configure_runtime, choose_device, make_loader, predict
    config = read_json(root / "config.json")
    test, _ = official_rows("test")
    oos, _ = official_rows("oos_test")
    rows = test + oos
    records = []
    for seed in config["seeds"]:
        path = checkpoints / f"seed_{seed}/best.pt"
        summary = read_json(root / f"seed_{seed}/metrics.json")
        if file_sha256(path) != summary["checkpoint_sha256"]:
            raise ValueError("Local CNN checkpoint checksum mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["provenance"]["seed"] != seed:
            raise ValueError("Local checkpoint config/seed differs")
        configure_runtime(seed, config["training"])
        device = choose_device(requested_device)
        model = PublicCNN(checkpoint["model"]["embedding.weight"],
                          **{k: v for k, v in config["model"].items() if k != "embedding_dim"}).to(device)
        model.load_state_dict(checkpoint["model"])
        logits, _ = predict(model, make_loader(rows, checkpoint["vocab"], config["text"]["max_length"], config["training"]["batch_size"]), device)
        probability = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 150]
        saved = read_jsonl(root / f"seed_{seed}/predictions.jsonl")
        max_error = float(np.max(np.abs(probability - np.asarray([r["p_oos"] for r in saved]))))
        if not np.array_equal(logits.argmax(axis=1), [r["predicted_class"] for r in saved]) or max_error > 1e-7:
            raise ValueError(f"CNN inference replay differs: seed={seed}, probability error={max_error}")
        records.append({"seed": seed, "predictions": len(saved), "classes_equal": True,
                        "max_probability_error": max_error, "checkpoint_sha256": file_sha256(path)})
    return {"all_passed": True, "device": requested_device, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Export completed local runs before verification")
    parser.add_argument("--results", type=Path, default=Path("reports/stage6/public_cnn_v1"))
    parser.add_argument("--output", type=Path, default=Path("docs/公开CNN重实现结果"))
    parser.add_argument("--replay-checkpoints", type=Path, help="Also replay local model weights")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.source:
        export(args.source, args.results)
    report = verify(args.results)
    write_report(report, args.output)
    plot_report(args.results, report, args.output.with_suffix(".png"))
    if args.replay_checkpoints:
        write_json(args.output.with_name(args.output.name + "_权重重放").with_suffix(".json"),
                   replay_weights(args.results, args.replay_checkpoints, args.device))
    print(f"PASS: three CNN runs, 16500 predictions; ID accuracy delta={report['id_accuracy_difference_pp']:+.2f} pp")


if __name__ == "__main__":
    main()
