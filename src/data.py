import csv

from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .audit import load_rgb
from .config import AUGMENTATION, CLASSES, DATASET, MANIFEST, PREPROCESS


def make_transform(config, augmentation=None):
    if config["mode"] != "RGB" or not config["exif_transpose"] or config["resize"] != "letterbox_bilinear":
        raise ValueError("Unsupported input preprocessing in the checkpoint.")
    size = tuple(config["size"])
    fill = tuple(config["fill"])

    def letterbox(image):
        return ImageOps.pad(image, size, method=Image.Resampling.BILINEAR, color=fill)

    steps = [letterbox]
    if augmentation is not None:
        if "translation_fraction" in augmentation:
            geometry = transforms.RandomAffine(
                degrees=augmentation["rotation_degrees"], translate=augmentation["translation_fraction"],
                scale=augmentation["scale_range"], interpolation=InterpolationMode.BILINEAR, fill=fill,
            )
        else:
            geometry = transforms.RandomRotation(augmentation["rotation_degrees"], InterpolationMode.BILINEAR, fill=fill)
        steps.extend([
            geometry,
            transforms.RandomHorizontalFlip(augmentation["horizontal_flip_p"]),
            transforms.ColorJitter(brightness=augmentation["brightness"], contrast=augmentation.get("contrast", 0),
                                   saturation=augmentation["saturation"]),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"])])
    return transforms.Compose(steps)


class VegetableDataset(Dataset):
    def __init__(self, split, preprocess=PREPROCESS, classes=CLASSES, limit_per_class=None, *, augmentation=AUGMENTATION):
        if not MANIFEST.is_file():
            raise FileNotFoundError("Run python -m src.audit to generate outputs/dataset_manifest.csv first.")
        self.transform = make_transform(preprocess, augmentation if split == "train" else None)
        self.samples = []
        class_to_idx = {label: index for index, label in enumerate(classes)}
        counts = dict.fromkeys(classes, 0)
        with MANIFEST.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["split"] != split:
                    continue
                label = row["label"]
                if limit_per_class is None or counts[label] < limit_per_class:
                    self.samples.append((DATASET / row["path"], class_to_idx[label]))
                    counts[label] += 1
        if not all(counts.values()):
            raise ValueError(f"The manifest must contain every class in {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(load_rgb(path)), label


def make_loader(dataset, batch_size, device, *, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda", generator=generator)
