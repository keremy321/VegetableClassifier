import csv
import hashlib
import json
import random
from io import BytesIO

import matplotlib
import torch
from PIL import Image, ImageOps

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from app import build_app

from .audit import load_rgb
from .config import BASELINE_CHECKPOINT, CLASSES, EXPERIMENTS, MANIFEST, OUTPUTS, PERSONAL, write_json
from .data import VegetableDataset, make_transform
from .predict import Predictor
from .train_resnet import preprocessing


DIAGNOSTICS = OUTPUTS / "input_diagnostics"


def check_decoder():
    for mode, value, rgb in (("L", 37, (37, 37, 37)), ("RGBA", (3, 17, 190, 32), (3, 17, 190))):
        buffer = BytesIO()
        Image.new(mode, (3, 2), value).save(buffer, "PNG")
        buffer.seek(0)
        output = load_rgb(buffer)
        assert output.mode == "RGB" and output.getpixel((0, 0)) == rgb

    original = Image.new("RGB", (12, 7), "red")
    original.paste("blue", (0, 0, 5, 4))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    original.save(buffer, "JPEG", exif=exif)
    with Image.open(BytesIO(buffer.getvalue())) as decoded:
        expected = decoded.transpose(Image.Transpose.ROTATE_270).tobytes()
    buffer.seek(0)
    transformed = load_rgb(buffer)
    assert transformed.size == (7, 12) and transformed.mode == "RGB" and transformed.tobytes() == expected


def verify_baseline(predictor):
    digest = predictor.checkpoint_sha256
    archived_run = json.loads(EXPERIMENTS["baseline"][1].read_text(encoding="utf-8"))
    assert digest == archived_run["checkpoint_sha256"]
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == predictor.metadata["manifest_sha256"]
    assert tuple(predictor.classes) == CLASSES and not any(module.training for module in predictor.model.modules())
    data = VegetableDataset("validation", predictor.metadata["preprocessing"], predictor.classes)
    for path, index in data.samples:
        assert path.parent.name == predictor.classes[index]

    with build_app(predictor) as demo:
        classify = next(function.fn for function in demo.fns.values() if function.api_name == "classify")
        headings = [component["props"].get("value", "") for component in demo.config["components"] if component["type"] == "markdown"]
        assert any(" · ".join(CLASSES) in heading for heading in headings)
        for index in (0, len(data) - 1):
            path, _ = data.samples[index]
            reference, _ = data[index]
            app_inputs = []
            handle = predictor.model.register_forward_pre_hook(lambda module, args: app_inputs.append(args[0].detach().clone()))
            try:
                _, predicted, scores = classify(str(path))
            finally:
                handle.remove()
            assert len(app_inputs) == 1 and torch.equal(app_inputs[0][0], reference)
            assert torch.equal(reference, data[index][0])
            with torch.inference_mode():
                logits = predictor.model(reference.unsqueeze(0))[0]
                expected = logits.softmax(0)
            assert tuple(scores) == CLASSES and predicted == CLASSES[logits.argmax().item()]
            torch.testing.assert_close(torch.tensor(list(scores.values())), expected)
    check_decoder()
    return {"checkpoint_sha256": digest, "matches_original_app_checkpoint": True, "manifest_matches_checkpoint": True,
            "class_order_matches_dataset_and_ui": True, "all_modules_in_eval_mode": True,
            "rgb_conversion_and_exif_6_verified": True, "app_and_validation_input_tensors_identical": True,
            "softmax_index_mapping_verified": True, "verification_inputs": [data.samples[i][0].name for i in (0, len(data) - 1)]}


def show_tensor(axis, tensor, config):
    mean = torch.tensor(config["mean"])[:, None, None]
    std = torch.tensor(config["std"])[:, None, None]
    axis.imshow((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#d3ddd7")


def dataset_sheet(config):
    figure, axes = plt.subplots(len(CLASSES), 6, figsize=(10.5, 9), layout="constrained")
    source = {split: VegetableDataset(split, config, augmentation=None) for split in ("train", "validation")}
    generator = random.Random(42)
    files = []
    for row, label in enumerate(CLASSES):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")
        for part, (split, data) in enumerate(source.items()):
            candidates = [i for i, (_, target) in enumerate(data.samples) if target == row]
            for column, index in enumerate(generator.sample(candidates, 3), start=part * 3):
                path, _ = data.samples[index]
                show_tensor(axes[row, column], data[index][0], config)
                axes[row, column].set_title(f"{split.title()}  {path.stem}", fontsize=8.5)
                files.append({"split": split, "label": label, "file": path.name})
    figure.suptitle("Dataset inputs after deterministic CNN preprocessing", fontsize=15, fontweight="bold")
    figure.savefig(DIAGNOSTICS / "train_validation_contact_sheet.png", dpi=175)
    plt.close(figure)
    return files


def personal_sheet(configs):
    with (PERSONAL / "labels.csv").open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    figure, axes = plt.subplots(len(configs), len(labels), figsize=(12, 5.2), layout="constrained")
    inputs = []
    for row, (name, config) in enumerate(configs.items()):
        transform = make_transform(config)
        axes[row, 0].set_ylabel(name, fontsize=11, fontweight="bold")
        for column, label in enumerate(labels):
            path = PERSONAL / label["filename"]
            image = load_rgb(path)
            width, height = ImageOps.contain(image, tuple(config["size"])).size
            padding = 1 - width * height / (config["size"][0] * config["size"][1])
            axis = axes[row, column]
            show_tensor(axis, transform(image), config)
            axis.set_title(f"{label['true_class']}\n{width} × {height} content, {padding:.0%} pad", fontsize=9)
            inputs.append({"model": name, "filename": path.name, "oriented_image_size": image.size,
                           "resized_content_size": [width, height], "padding_fraction": padding})
    figure.suptitle("Personal diagnostic photographs after model preprocessing", fontsize=15, fontweight="bold")
    figure.savefig(DIAGNOSTICS / "personal_model_inputs.png", dpi=180)
    plt.close(figure)
    return inputs


def main():
    torch.set_num_threads(2)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(BASELINE_CHECKPOINT, "cpu")
    checks = verify_baseline(predictor)
    baseline_config = predictor.metadata["preprocessing"]
    dataset = dataset_sheet(baseline_config)
    personal = personal_sheet({"Small CNN (128)": baseline_config, "ResNet18 (224)": preprocessing()})
    write_json(DIAGNOSTICS / "input_checks.json", {"baseline_checks": checks, "dataset_sample": dataset, "personal_inputs": personal})
    print("Baseline checkpoint, UI indexing, RGB/EXIF, and model-input checks passed. Contact sheets saved.")


if __name__ == "__main__":
    main()
