"""Display the fixed ResNet18 inputs and run a one-crop-per-photo diagnostic."""

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass

import matplotlib
import torch
from PIL import Image, ImageOps

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.transforms.functional import to_pil_image

from .audit import load_rgb
from .config import CLASSES, DATASET, MANIFEST, OUTPUTS, PERSONAL, RESNET_CHECKPOINT, RESNET_OUTPUTS, write_json
from .predict import Predictor


DIAGNOSTICS = OUTPUTS / "generalization_diagnostics"
BOXES = DIAGNOSTICS / "crop_boxes.csv"
COORDINATES = ("left", "top", "right", "bottom")
SEED = 42


@dataclass
class Photo:
    name: str
    image: Image.Image
    box: tuple[int, int, int, int]
    crop: Image.Image


def read_crops():
    with BOXES.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    photos = []
    actual = {path.name for path in PERSONAL.glob("*.jpeg")}
    if len(rows) != len(actual) or {row["filename"] for row in rows} != actual:
        raise ValueError("The crop manifest must contain one box for each original personal photo.")
    for row in rows:
        image = load_rgb(PERSONAL / row["filename"])
        left, top, right, bottom = (int(row[key]) for key in COORDINATES)
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise ValueError(f"Crop coordinates are out of bounds for {row['filename']}.")
        box = left, top, right, bottom
        photos.append(Photo(row["filename"], image, box, image.crop(box)))
    return photos


def show(axis, image):
    axis.imshow(image)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#d7e0db")


def model_view(predictor, image):
    config = predictor.metadata["preprocessing"]
    tensor = predictor.transform(image)
    mean = torch.tensor(config["mean"])[:, None, None]
    std = torch.tensor(config["std"])[:, None, None]
    return to_pil_image((tensor * std + mean).clamp(0, 1).mul(255).round().to(torch.uint8))


def training_sheets(predictor, photos):
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        train_paths = {label: [] for label in CLASSES}
        for row in csv.DictReader(stream):
            if row["split"] == "train":
                train_paths[row["label"]].append(row["path"])
    rng = random.Random(SEED)
    selected = {label: rng.sample(sorted(paths), 10) for label, paths in train_paths.items()}
    samples = []
    for label, paths in selected.items():
        figure, axes = plt.subplots(4, 5, figsize=(13.5, 10.3), layout="constrained")
        for index, path in enumerate(paths):
            image = load_rgb(DATASET / path)
            pair_row, col = (index // 5) * 2, index % 5
            show(axes[pair_row, col], image)
            axes[pair_row, col].set_title(f"{index + 1:02d}  {path.rsplit('/', 1)[-1]}  ·  {image.width} × {image.height}", fontsize=10)
            show(axes[pair_row + 1, col], model_view(predictor, image))
            samples.append({"seed": SEED, "class": label, "sample": index + 1, "path": path,
                            "oriented_width": image.width, "oriented_height": image.height})
        for index, caption in enumerate(("Original RGB", "ResNet18 input", "Original RGB", "ResNet18 input")):
            axes[index, 0].set_ylabel(caption, fontsize=10, fontweight="bold")
        figure.suptitle(f"{label} training contact sheet  ·  Random seed {SEED}", fontsize=17, fontweight="bold")
        figure.savefig(DIAGNOSTICS / f"training_{label.lower()}.png", dpi=150)
        plt.close(figure)

    with (DIAGNOSTICS / "training_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    figure, axes = plt.subplots(4, 4, figsize=(12.5, 12), layout="constrained")
    by_name = {photo.name: photo for photo in photos}
    for col, name in enumerate(("brinjal.jpeg", "potato.jpeg")):
        photo = by_name[name]
        show(axes[0, col * 2], photo.image)
        axes[0, col * 2].set_title(f"Personal  ·  {name}", fontsize=10)
        show(axes[0, col * 2 + 1], model_view(predictor, photo.image))
        axes[0, col * 2 + 1].set_title("ResNet18 full-frame input", fontsize=10)
    for row, label in enumerate(("Brinjal", "Potato", "Carrot"), start=1):
        axes[row, 0].set_ylabel(f"Train  ·  {label}", fontsize=11, fontweight="bold")
        for col, path in enumerate(selected[label][:4]):
            show(axes[row, col], model_view(predictor, load_rgb(DATASET / path)))
            axes[row, col].set_title(path.rsplit("/", 1)[-1], fontsize=10)
    figure.suptitle("Personal brinjal and potato versus three training classes", fontsize=16, fontweight="bold")
    figure.savefig(DIAGNOSTICS / "personal_vs_training.png", dpi=160)
    plt.close(figure)


def framing_sheet(predictor, photos, results=None):
    figure, axes = plt.subplots(3, len(photos), figsize=(15, 9.7), layout="constrained")
    for column, photo in enumerate(photos):
        left, top, right, bottom = photo.box
        full, cropped, input_view = axes[:, column]
        show(full, photo.image)
        full.add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top,
                                 fill=False, edgecolor="#30e8d4", linewidth=2.4))
        show(cropped, photo.crop)
        show(input_view, model_view(predictor, photo.crop))
        full_title = photo.name
        cropped_title = f"({left}, {top}, {right}, {bottom})"
        if results is not None:
            entry = results[column]
            full_title += f"\n{entry['full_prediction']}  {entry['full_score']:.2%}"
            cropped_title += f"\n{entry['crop_prediction']}  {entry['crop_score']:.2%}"
        full.set_title(full_title, fontsize=10)
        cropped.set_title(cropped_title, fontsize=10)
    for index, label in enumerate(("Original with crop box", "Cropped RGB", "ResNet18 crop input")):
        axes[index, 0].set_ylabel(label, fontsize=11, fontweight="bold")
    figure.suptitle("One manual framing crop per personal photograph", fontsize=17, fontweight="bold")
    name = "framing_comparison.png" if results is not None else "crop_preview.png"
    figure.savefig(DIAGNOSTICS / name, dpi=160)
    plt.close(figure)


def infer(predictor, photos, original):
    with (PERSONAL / "labels.csv").open(newline="", encoding="utf-8") as stream:
        ground_truth = {row["filename"]: row["true_class"] for row in csv.DictReader(stream)}
    originals = {row["filename"]: row for row in original["photos"]}
    results = []
    extra_scores = []
    for photo in photos:
        full = predictor.predict_image(photo.image)
        crop = predictor.predict_image(photo.crop)
        full_label = max(full, key=full.get)
        crop_label = max(crop, key=crop.get)
        truth = ground_truth[photo.name]
        old = originals[photo.name]
        if full_label != old["predicted_class"] or abs(full[full_label] - old["predicted_class_score"]) > 5e-5:
            raise ValueError(f"Full-frame prediction differs from the saved diagnostic for {photo.name}.")
        canvas = tuple(predictor.metadata["preprocessing"]["size"])
        contain = ImageOps.contain(photo.crop, canvas)
        results.append({"filename": photo.name, "true_class": truth, **dict(zip(COORDINATES, photo.box)),
                        "crop_width": photo.crop.width, "crop_height": photo.crop.height,
                        "crop_input_padding_fraction": 1 - (contain.width * contain.height) / (canvas[0] * canvas[1]),
                        "full_prediction": full_label, "full_score": full[full_label], "full_correct": full_label == truth,
                        "crop_prediction": crop_label, "crop_score": crop[crop_label], "crop_correct": crop_label == truth})
        extra_scores.append({"filename": photo.name, "full_scores": full, "crop_scores": crop})
    with (DIAGNOSTICS / "framing_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(DIAGNOSTICS / "framing_results.json", {
        "model": predictor.model_name, "checkpoint_sha256": predictor.checkpoint_sha256, "device": str(predictor.device),
        "preprocessing": predictor.metadata["preprocessing"], "crop_manifest_sha256": hashlib.sha256(BOXES.read_bytes()).hexdigest(),
        "coordinates": "EXIF-oriented RGB pixel coordinates; left/top inclusive and right/bottom exclusive",
        "full_correct": sum(row["full_correct"] for row in results), "crop_correct": sum(row["crop_correct"] for row in results),
        "results": results, "all_class_scores": extra_scores,
    })
    framing_sheet(predictor, photos, results)
    for row in results:
        print(f"{row['filename']}: {row['full_prediction']} ({row['full_score']:.2%}) -> "
              f"{row['crop_prediction']} ({row['crop_score']:.2%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("assets", "infer"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    photos = read_crops()
    previous = json.loads((RESNET_OUTPUTS / "personal_predictions.json").read_text(encoding="utf-8"))
    predictor = Predictor(RESNET_CHECKPOINT, "cpu" if args.mode == "assets" else "auto",
                          expected_sha256=previous["checkpoint_sha256"])
    if args.mode == "assets":
        training_sheets(predictor, photos)
        framing_sheet(predictor, photos)
        print("Saved the training contact sheets, personal comparison, and fixed crop preview.")
    else:
        infer(predictor, photos, previous)


if __name__ == "__main__":
    main()
