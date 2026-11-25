import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .audit import load_rgb, pixel_digest
from .config import EXPERIMENTS, FRESH, MANIFEST, PERSONAL, write_json
from .predict import Predictor


def plot_original(images, results, output_dir, model_name):
    figure, axes = plt.subplots(2, 3, figsize=(9, 6.4), layout="constrained")
    for row, image, axis in zip(results, images, axes.flat):
        axis.imshow(image)
        axis.set_title(f"True: {row['true_class']}\nPredicted: {row['predicted_class']} ({row['predicted_class_score']:.1%})",
                       color="#176152" if row["correct"] else "#a43c26", fontsize=11)
        axis.set_axis_off()
    correct_count = sum(row["correct"] for row in results)
    axes.flat[-1].set_axis_off()
    axes.flat[-1].text(0.04, 0.56, f"{correct_count} of {len(results)} matched\nScores are softmax outputs.",
                      transform=axes.flat[-1].transAxes, fontsize=12, linespacing=1.6, color="#263849")
    figure.suptitle(f"{model_name} on the five diagnostic photographs", fontsize=16, fontweight="bold")
    figure.savefig(output_dir / "personal_predictions.png", dpi=200)
    plt.close(figure)


def read_labels(folder):
    with (folder / "labels.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filename", "true_class"]:
            raise ValueError(f"Use the filename,true_class header in {folder / 'labels.csv'}.")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Each row needs exactly a filename and class label in {folder / 'labels.csv'}.")
        return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate the finalized model on external photographs.")
    parser.add_argument("--model", choices=EXPERIMENTS, default="resnet18")
    parser.add_argument("--set", choices=("original", "fresh"), default="original")
    args = parser.parse_args()
    fresh = args.set == "fresh"
    folder = FRESH if fresh else PERSONAL
    labels = read_labels(folder)
    if not labels:
        parser.error(f"Add labeled photographs to {folder} and fill in labels.csv before evaluation.")

    checkpoint, history_path, output_dir = EXPERIMENTS[args.model]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    predictor = Predictor(checkpoint, expected_sha256=history["checkpoint_sha256"])
    if not fresh and tuple(row["true_class"] for row in labels) != predictor.classes:
        raise ValueError("The original manifest must contain one photograph per class in canonical order.")
    if any(row["true_class"] not in predictor.classes for row in labels):
        raise ValueError("Every photograph must use one of the five canonical class labels.")
    filenames = [row["filename"] for row in labels]
    if len({name.casefold() for name in filenames}) != len(filenames) or any(
        not name or Path(name).name != name or name in {".", ".."} for name in filenames
    ):
        raise ValueError("Use distinct filenames within the selected photo folder.")
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != predictor.metadata["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the one used for training.")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        dataset_hashes = {row["pixel_sha256"] for row in csv.DictReader(stream)}
    images = [load_rgb(folder / name) for name in filenames]
    image_hashes = [pixel_digest(image) for image in images]
    if dataset_hashes.intersection(image_hashes):
        raise ValueError("A submitted photograph exactly matches an image in the audited dataset splits.")
    if len(set(image_hashes)) != len(image_hashes):
        raise ValueError("The manifest contains photographs with identical decoded pixels.")
    if fresh:
        originals = [PERSONAL / row["filename"] for row in read_labels(PERSONAL)]
        if set(image_hashes).intersection(pixel_digest(load_rgb(path)) for path in originals if path.is_file()):
            raise ValueError("A fresh photograph exactly matches an original diagnostic photograph.")
        output_dir = output_dir / "fresh"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, image in zip(labels, images):
        scores = predictor.predict_image(image)
        predicted = max(scores, key=scores.get)
        results.append({**label, "predicted_class": predicted, "predicted_class_score": scores[predicted],
                        "correct": predicted == label["true_class"]})
    if not fresh:
        plot_original(images, results, output_dir, predictor.model_name)

    with (output_dir / "personal_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(output_dir / "personal_predictions.json", {"checkpoint_sha256": predictor.checkpoint_sha256,
               "best_epoch": predictor.metadata["best_epoch"], "photo_set": "fresh_external" if fresh else "previously_inspected_external_diagnostic",
               "count": len(results), "correct": sum(row["correct"] for row in results),
               "exact_matches_in_dataset_splits": 0, "photos": results})
    for row in results:
        print(f"{row['filename']} | true {row['true_class']} | predicted {row['predicted_class']} "
              f"({row['predicted_class_score']:.2%}) | {'correct' if row['correct'] else 'incorrect'}")


if __name__ == "__main__":
    main()
