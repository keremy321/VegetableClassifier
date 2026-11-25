import argparse
import hashlib
import json
from time import perf_counter

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .audit import load_rgb
from .config import BASELINE_CHECKPOINT, DATASET, EXPERIMENTS, MANIFEST, OUTPUTS, write_json
from .data import VegetableDataset, make_loader
from .predict import Predictor


@torch.inference_mode()
def test_predictions(model, loader, device):
    truth, predictions = [], []
    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        truth.extend(labels.tolist())
    return truth, predictions


@torch.inference_mode()
def benchmark(predictor, path, image_tensor, warmup=20, repetitions=100):
    batch = image_tensor.unsqueeze(0).to(predictor.device)

    def sync():
        if predictor.device.type == "cuda":
            torch.cuda.synchronize(predictor.device)

    result = {"device": str(predictor.device), "batch_size": 1, "warmup": warmup, "repetitions": repetitions}
    methods = {
        "model_forward_only": (lambda: predictor.model(batch), "Model forward on a resident tensor, excluding preprocessing, transfers, softmax, and model load."),
        "file_to_scores": (lambda: predictor.predict_file(path), "File read and decode, EXIF/RGB conversion, resizing, normalization, transfers, model forward, and CPU softmax score output. Excludes model load and Gradio/browser overhead."),
    }
    for name, (operation, scope) in methods.items():
        for _ in range(warmup):
            operation()
        sync()
        durations = []
        for _ in range(repetitions):
            start = perf_counter()
            operation()
            sync()
            durations.append(1000 * (perf_counter() - start))
        result[name] = {"mean_ms": float(np.mean(durations)), "median_ms": float(np.median(durations)),
                        "p95_ms": float(np.percentile(durations, 95)), "scope": scope}
    return result


def plot_history(history, output_dir):
    epochs = history["epochs"]
    for metric, title in (("loss", "Cross-entropy loss"), ("accuracy", "Accuracy")):
        figure, axis = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
        for split, color in (("train", "#126858"), ("validation", "#355f9c")):
            axis.plot([row["epoch"] for row in epochs], [row[f"{split}_{metric}"] for row in epochs],
                      color=color, marker="o", markersize=3, label=split.title())
        axis.axvline(history["best_epoch"], color="#626c77", linestyle="--", linewidth=1, label="Selected epoch")
        if "head_epochs" in history["training"]:
            axis.axvline(history["training"]["head_epochs"] + 0.5, color="#a5822e", linestyle=":",
                         linewidth=1, label="Fine-tuning begins")
        axis.set(xlabel="Epoch", ylabel=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric == "accuracy":
            axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False)
        figure.savefig(output_dir / f"{metric}_curves.png", dpi=180)
        plt.close(figure)


def plot_confusion(matrix, classes, output_dir):
    figure, axis = plt.subplots(figsize=(6.2, 5.1), layout="constrained")
    heatmap = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row, items in enumerate(matrix):
        for col, count in enumerate(items):
            axis.text(col, row, count, ha="center", va="center", color="white" if count > matrix.max() / 2 else "#172b45")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes,
             xlabel="Predicted class", ylabel="True class", title="Held-out test confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    figure.colorbar(heatmap, ax=axis, label="Image count")
    figure.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the validation-selected checkpoint on the held-out test set.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--model", choices=EXPERIMENTS, default="resnet18")
    args = parser.parse_args()
    checkpoint, history_path, output_dir = EXPERIMENTS[args.model]
    predictor = Predictor(checkpoint, args.device)
    metadata = predictor.metadata
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoint_digest = predictor.checkpoint_sha256
    if checkpoint_digest != history["checkpoint_sha256"]:
        raise ValueError("Training history and checkpoint do not match.")
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the one used for training. Restore it or retrain.")

    dataset = VegetableDataset("test", metadata["preprocessing"], predictor.classes)
    loader = make_loader(dataset, metadata["training"]["batch_size"], predictor.device)
    truth, predictions = test_predictions(predictor.model, loader, predictor.device)
    labels = list(range(len(predictor.classes)))
    precision, recall, f1, support = precision_recall_fscore_support(truth, predictions, labels=labels, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=labels)
    latency_path = dataset.samples[0][0]
    latency = benchmark(predictor, latency_path, dataset[0][0])
    if args.model == "resnet18":
        baseline = Predictor(BASELINE_CHECKPOINT, args.device)
        if baseline.metadata["manifest_sha256"] != metadata["manifest_sha256"]:
            raise ValueError("The two checkpoints were trained against different dataset manifests.")
        baseline_latency = benchmark(baseline, latency_path, baseline.transform(load_rgb(latency_path)))
        write_json(OUTPUTS / "latency_comparison.json", {"image": latency_path.relative_to(DATASET).as_posix(),
                   "baseline": baseline_latency, "resnet18": latency})
    metrics = {
        "best_epoch": metadata["best_epoch"], "checkpoint_sha256": checkpoint_digest,
        "class_to_idx": metadata["class_to_idx"], "test_count": len(truth), "correct_count": int(matrix.trace()),
        "accuracy": accuracy_score(truth, predictions),
        "macro": {"precision": float(precision.mean()), "recall": float(recall.mean()), "f1": float(f1.mean())},
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
                             "support": int(support[i])} for i, name in enumerate(predictor.classes)},
        "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
        "training_wall_seconds": history["training_wall_seconds"],
        "latency": latency,
    }
    write_json(output_dir / "test_metrics.json", metrics)
    plot_confusion(matrix, predictor.classes, output_dir)
    plot_history(history, output_dir)
    print(f"Test accuracy: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['test_count']}). "
          f"Macro F1: {metrics['macro']['f1']:.4f}.")


if __name__ == "__main__":
    main()
