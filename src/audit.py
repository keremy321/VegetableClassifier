"""Audit selected dataset folders and write the split manifest before training."""

import csv
import hashlib
import struct
from collections import defaultdict

from PIL import Image, ImageOps

from .config import CLASSES, DATASET, MANIFEST, OUTPUTS, SPLITS, write_json


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def pixel_digest(image):
    return hashlib.sha256(struct.pack(">II", *image.size) + image.tobytes()).hexdigest()


def empty_counts():
    return {split: dict.fromkeys(CLASSES, 0) for split in SPLITS}


def main():
    counts = empty_counts()
    kept = empty_counts()
    groups = defaultdict(list)
    unreadable = []
    other_files = []
    source_classes = {}
    for split in SPLITS:
        source_classes[split] = sorted(folder.name for folder in (DATASET / split).iterdir() if folder.is_dir())
        for label in CLASSES:
            folder = DATASET / split / label
            if not folder.is_dir():
                raise FileNotFoundError(f"Required class folder is missing: {folder}")
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(DATASET).as_posix()
                if path.suffix.lower() not in EXTENSIONS:
                    other_files.append(relative)
                    continue
                counts[split][label] += 1
                try:
                    image = load_rgb(path)
                except (OSError, ValueError, Image.DecompressionBombError) as error:
                    unreadable.append({"path": relative, "error": str(error)})
                    continue
                digest = pixel_digest(image)
                groups[digest].append({"split": split, "label": label, "path": relative, "pixel_sha256": digest})
        print(f"Scanned {split}: {sum(counts[split].values())} image files", flush=True)

    retained = []
    duplicates = []
    for digest, members in groups.items():
        splits = {member["split"] for member in members}
        conflicting = len({member["label"] for member in members}) > 1
        preferred = next(split for split in reversed(SPLITS) if split in splits)
        keep = [] if conflicting else [member for member in members if member["split"] == preferred]
        retained.extend(keep)
        if len(members) > 1:
            duplicates.append({"pixel_sha256": digest, "cross_split": len(splits) > 1,
                               "conflicting_labels": conflicting, "members": [item["path"] for item in members],
                               "retained": [item["path"] for item in keep]})

    for item in retained:
        kept[item["split"]][item["label"]] += 1
    if any(not count for labels in kept.values() for count in labels.values()):
        raise ValueError("At least one split/class combination has no readable retained images.")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "label", "path", "pixel_sha256"))
        writer.writeheader()
        writer.writerows(sorted(retained, key=lambda row: (SPLITS.index(row["split"]), CLASSES.index(row["label"]), row["path"])))
    temporary.replace(MANIFEST)
    audit = {
        "classes": CLASSES,
        "source_class_folders": source_classes,
        "source_counts": counts,
        "retained_counts": kept,
        "unreadable_images": unreadable,
        "ignored_non_image_files": other_files,
        "duplicate_group_count": len(duplicates),
        "cross_split_duplicate_group_count": sum(group["cross_split"] for group in duplicates),
        "conflicting_label_group_count": sum(group["conflicting_labels"] for group in duplicates),
        "removed_duplicate_images": sum(len(group["members"]) - len(group["retained"]) for group in duplicates),
        "duplicate_groups": duplicates,
        "rule": "SHA-256 of width, height, and decoded EXIF-oriented RGB pixels. Drop all members of conflicting-label groups. "
                "Otherwise retain members in the highest-priority split (test, then validation, then train). "
                "Same-label duplicates within the retained split are kept. Exact matching does not detect near duplicates.",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    }
    write_json(OUTPUTS / "data_audit.json", audit)
    print(f"Retained {len(retained)} images. Exact groups: {len(duplicates)}. Cross-split: "
          f"{audit['cross_split_duplicate_group_count']}. Conflicts: {audit['conflicting_label_group_count']}.")


if __name__ == "__main__":
    main()
