"""Evaluate the saved ResNet18 on unique photos in the folder-labeled collection."""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib
import torch
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .audit import EXTENSIONS, load_rgb, pixel_digest
from .config import CLASSES, OUTPUTS, PERSONAL, RESNET_CHECKPOINT, RESNET_OUTPUTS, ROOT, write_json
from .evaluate_personal import read_labels
from .predict import Predictor


SOURCE = ROOT / "data" / "personal_new"
DESTINATION = OUTPUTS / "personal_new_evaluation"
FOLDER_LABELS = {label.casefold(): label for label in CLASSES} | {"patato": "Potato"}
ORIGINAL = "matches_original"
ADDITIONAL = "additional_unique"


@dataclass
class Photograph:
    path: Path
    label: str | None
    image: Image.Image
    digest: str

    @property
    def relative_path(self):
        return self.path.relative_to(ROOT).as_posix()


def read_photo(path, label):
    image = load_rgb(path)
    return Photograph(path, label, image, pixel_digest(image))


def read_collection():
    photographs = []
    ignored = []
    unreadable = []
    paths = sorted((path for path in SOURCE.rglob("*") if path.is_file()), key=lambda path: path.as_posix().casefold())
    for path in paths:
        relative = path.relative_to(SOURCE)
        if path.suffix.casefold() not in EXTENSIONS:
            ignored.append(relative.as_posix())
            continue
        folder = relative.parts[0].casefold() if len(relative.parts) > 1 else ""
        label = FOLDER_LABELS.get(folder)
        if label is None:
            raise ValueError(f"The image needs a canonical class folder: {relative}")
        try:
            photographs.append(read_photo(path, label))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            unreadable.append({"path": path.relative_to(ROOT).as_posix(), "true_class": label, "error": str(error)})
    if not photographs:
        raise ValueError(f"No readable photographs found in {SOURCE}.")
    return photographs, ignored, unreadable


def save_csv(name, rows):
    with (DESTINATION / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    scored = [row for row in rows if row["correct"] is not None]
    return {"correct": sum(row["correct"] for row in scored), "total": len(scored), "label_conflicts": len(rows) - len(scored)}


def save_contact_sheet(name, title, items):
    if not items:
        return
    rows = ceil(len(items) / 3)
    figure, axes = plt.subplots(rows, 3, figsize=(12, 3.75 * rows + 0.55), squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_axis_off()
    for axis, (photo, result) in zip(axes.flat, items):
        axis.imshow(photo.image)
        label = result["true_class"] or "Conflicting labels"
        outcome = {True: "Correct", False: "Incorrect", None: "Unscored"}[result["correct"]]
        color = {True: "#176152", False: "#a33a22", None: "#526172"}[result["correct"]]
        axis.set_title(f"{photo.path.name}\nTrue {label}  ·  {outcome}\n"
                       f"Predicted {result['predicted_class']} ({result['predicted_score']:.2%})",
                       fontsize=10.5, color=color)
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.savefig(DESTINATION / name, dpi=175)
    plt.close(figure)


def main():
    torch.set_num_threads(2)
    collection, ignored, unreadable = read_collection()
    original_labels = {row["filename"]: row["true_class"] for row in read_labels(PERSONAL)}
    original_paths = sorted(path for path in PERSONAL.rglob("*") if path.is_file() and path.suffix.casefold() in EXTENSIONS)
    if set(original_labels) - {path.relative_to(PERSONAL).as_posix() for path in original_paths}:
        raise ValueError("At least one original photograph in labels.csv is missing.")
    originals = defaultdict(list)
    for path in original_paths:
        original = read_photo(path, original_labels.get(path.relative_to(PERSONAL).as_posix()))
        originals[original.digest].append(original)

    groups = defaultdict(list)
    ordered = sorted(collection, key=lambda photo: (CLASSES.index(photo.label), photo.relative_path.casefold()))
    for photo in ordered:
        groups[photo.digest].append(photo)

    history = json.loads((RESNET_OUTPUTS / "training_history.json").read_text(encoding="utf-8"))
    predictor = Predictor(RESNET_CHECKPOINT, expected_sha256=history["checkpoint_sha256"])
    if predictor.model_name != "ResNet18":
        raise ValueError("The saved checkpoint must be ResNet18.")
    inventory = []
    predictions = []
    sheet_items = {ORIGINAL: [], ADDITIONAL: []}
    for digest, members in groups.items():
        representative = members[0]
        prior = originals.get(digest, [])
        cohort = ORIGINAL if prior else ADDITIONAL
        labels = {photo.label for photo in members + prior if photo.label is not None}
        true_class = representative.label if len(labels) == 1 else None
        matching_paths = " | ".join(photo.relative_path for photo in prior)
        for photo in members:
            inventory.append({"filename": photo.path.name, "path": photo.relative_path, "true_class": photo.label,
                              "width": photo.image.width, "height": photo.image.height, "pixel_sha256": digest,
                              "cohort": cohort, "original_match_paths": matching_paths, "files_in_collection_group": len(members),
                              "representative_path": representative.relative_path, "counted_representative": photo is representative,
                              "label_conflict": true_class is None})
        scores = predictor.predict_image(representative.image)
        predicted = max(scores, key=scores.get)
        result = {"filename": representative.path.name, "path": representative.relative_path, "true_class": true_class,
                  "predicted_class": predicted, "correct": predicted == true_class if true_class is not None else None,
                  "predicted_score": scores[predicted], **{f"score_{label.lower()}": scores[label] for label in CLASSES},
                  "cohort": cohort, "original_match_paths": matching_paths, "files_in_collection_group": len(members),
                  "pixel_sha256": digest}
        predictions.append(result)
        sheet_items[cohort].append((representative, result))

    DESTINATION.mkdir(parents=True, exist_ok=True)
    save_csv("inventory.csv", inventory)
    save_csv("predictions.csv", predictions)
    for cohort, name, title in ((ORIGINAL, "original_contact_sheet.png", "Original photographs represented in the collection"),
                                (ADDITIONAL, "additional_contact_sheet.png", "Additional unique photographs")):
        save_contact_sheet(name, title, sheet_items[cohort])
    cohorts = {name: [row for row in predictions if row["cohort"] == name] for name in sheet_items}
    summary = {
        "model": predictor.model_name, "best_epoch": predictor.metadata["best_epoch"],
        "checkpoint_sha256": predictor.checkpoint_sha256, "preprocessing": predictor.metadata["preprocessing"],
        "device": str(predictor.device), "match_rule": "SHA-256 of dimensions and EXIF-oriented decoded RGB pixels",
        "raw_image_files": len(collection) + len(unreadable), "readable_image_files": len(collection),
        "ignored_files": ignored, "unreadable_images": unreadable,
        "original_reference_image_files": len(original_paths),
        "collection_files_matching_originals": sum(row["cohort"] == ORIGINAL for row in inventory),
        "duplicate_copies_within_collection": len(collection) - len(groups),
        "duplicate_groups_within_collection": sum(len(members) > 1 for members in groups.values()),
        "unique_images_evaluated": len(predictions), "overall": summarize(predictions),
        "cohorts": {name: summarize(rows) for name, rows in cohorts.items()},
        "per_class": {label: {name: summarize([row for row in rows if row["true_class"] == label])
                              for name, rows in (cohorts | {"combined": predictions}).items()} for label in CLASSES},
    }
    write_json(DESTINATION / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
