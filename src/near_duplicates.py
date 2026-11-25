"""Screen the existing manifest for visually reviewable cross-split candidates."""

import argparse
import csv
import hashlib
from collections import Counter
from html import escape
from itertools import combinations
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from .audit import load_rgb
from .config import DATASET, MANIFEST, OUTPUTS, SPLITS, write_json


DIAGNOSTICS = OUTPUTS / "generalization_diagnostics"
FIELDS = ("id", "split_a", "label_a", "path_a", "split_b", "label_b", "path_b", "distance",
          "best_alignment", "direct_distance", "flip_b_distance", "flip_a_distance", "exact_rgb_match")


def perceptual_hash(image):
    thumbnail = image.resize((32, 32), Image.Resampling.LANCZOS)
    frequencies = dctn(np.asarray(thumbnail, dtype=np.float32), norm="ortho")[:8, :8].ravel()[1:]
    return int.from_bytes(np.packbits(frequencies > np.median(frequencies)).tobytes(), "big")


def build_hashes(rows):
    codes = []
    for row in rows:
        image = load_rgb(DATASET / row["path"]).convert("L")
        codes.append((perceptual_hash(image), perceptual_hash(ImageOps.mirror(image))))
    return np.asarray(codes, dtype=np.uint64).T


def find_candidates(split_rows, split_hashes, threshold):
    alignments = ("direct", "flip_b", "flip_a")
    candidates = []
    for a, b in combinations(SPLITS, 2):
        rows_a, rows_b = split_rows[a], split_rows[b]
        orig_a, mirror_a = split_hashes[a]
        orig_b, mirror_b = split_hashes[b]
        for start in range(0, len(rows_a), 256):
            block = slice(start, start + 256)
            distances = np.stack([
                np.bitwise_count(orig_a[block, None] ^ orig_b),
                np.bitwise_count(orig_a[block, None] ^ mirror_b),
                np.bitwise_count(mirror_a[block, None] ^ orig_b),
            ])
            minimum = distances.min(axis=0)
            for i, j in np.argwhere(minimum <= threshold):
                left, right = rows_a[start + i], rows_b[j]
                direct, flip_b, flip_a = map(int, distances[:, i, j])
                winner = alignments[int(distances[:, i, j].argmin())]
                candidates.append({
                    "split_a": a, "label_a": left["label"], "path_a": left["path"],
                    "split_b": b, "label_b": right["label"], "path_b": right["path"],
                    "distance": int(minimum[i, j]), "best_alignment": winner, "direct_distance": direct,
                    "flip_b_distance": flip_b, "flip_a_distance": flip_a,
                    "exact_rgb_match": left["pixel_sha256"] == right["pixel_sha256"],
                })
    candidates.sort(key=lambda row: (row["distance"], row["path_a"], row["path_b"]))
    for index, candidate in enumerate(candidates, start=1):
        candidate["id"] = f"P{index:05d}"
    return candidates


def write_gallery(candidates, threshold):
    template = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-split perceptual-hash candidates</title>
<style>
body { font: 16px/1.4 system-ui, sans-serif; color: #192b26; background: #f3f7f4; max-width: 1110px; margin: 28px auto; padding: 0 16px; }
article { background: white; border: 1px solid #d3e1d9; border-radius: 8px; padding: 14px; margin: 14px 0; }
h2 { font-size: 1rem; margin: 0 0 8px; } .pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
figure { margin: 0; } img { width: 100%; height: clamp(115px, 30vw, 300px); object-fit: contain; background: #edf3ef; }
figcaption { overflow-wrap: anywhere; } a { color: #155142; }
</style><h1>Cross-split perceptual-hash candidates</h1>"""
    with (DIAGNOSTICS / "near_duplicate_candidates.html").open("w", encoding="utf-8") as stream:
        stream.write(template)
        stream.write(f"<p>{len(candidates):,} pairs at a Hamming distance of {threshold} or less on a 63-bit DCT hash. "
                     "Images are shown in their original orientation. The closest alignment names any horizontal mirror that improved the match. "
                     "Review the scene and object details before assigning a relationship. Select an image to open its original file.</p><main>\n")
        for row in candidates:
            split_labels = "Different folder labels" if row["label_a"] != row["label_b"] else "Same folder label"
            stream.write(f'<article id="{row["id"]}"><h2>{row["id"]} · distance {row["distance"]} · '
                         f'{row["best_alignment"].replace("_", " ")} · {split_labels}</h2><div class="pair">')
            for side in ("a", "b"):
                path = row[f"path_{side}"]
                url = "../../data/vegetables/" + quote(path)
                stream.write(f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{escape(path)}"></a>'
                             f'<figcaption>{escape(path)}</figcaption></figure>')
            stream.write("</div></article>\n")
        stream.write("</main></html>\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=8, choices=range(0, 17), metavar="0..16")
    args = parser.parse_args()
    split_rows = {split: [] for split in SPLITS}
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            split_rows[row["split"]].append(row)
    split_hashes = {split: build_hashes(rows) for split, rows in split_rows.items()}
    candidates = find_candidates(split_rows, split_hashes, args.threshold)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    with (DIAGNOSTICS / "cross_split_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(candidates)
    write_gallery(candidates, args.threshold)

    split_pairs = {}
    for a, b in combinations(SPLITS, 2):
        group = [row for row in candidates if (row["split_a"], row["split_b"]) == (a, b)]
        split_pairs[f"{a} / {b}"] = {"comparisons": len(split_rows[a]) * len(split_rows[b]), "candidates": len(group),
                                    "same_label": sum(row["label_a"] == row["label_b"] for row in group),
                                    "unique_a": len({row["path_a"] for row in group}),
                                    "unique_b": len({row["path_b"] for row in group})}
    write_json(DIAGNOSTICS / "near_duplicate_summary.json", {
        "method": "EXIF-oriented luma, 32x32 Lanczos resize, 8x8 low-frequency orthonormal DCT, DC excluded, median-thresholded 63 bits",
        "distance": "minimum Hamming distance over direct alignment and a horizontal mirror of either image",
        "threshold": args.threshold, "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "images_by_split": {split: len(rows) for split, rows in split_rows.items()}, "split_pairs": split_pairs,
        "total_comparisons": sum(group["comparisons"] for group in split_pairs.values()), "candidates": len(candidates),
        "different_folder_label_candidates": sum(row["label_a"] != row["label_b"] for row in candidates),
        "manifest_exact_rgb_matches": sum(row["exact_rgb_match"] for row in candidates),
        "distance_counts": dict(sorted(Counter(row["distance"] for row in candidates).items())),
    })
    print(f"Saved {len(candidates):,} cross-split candidate pairs for visual review.")


if __name__ == "__main__":
    main()
