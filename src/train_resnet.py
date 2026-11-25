import argparse
import hashlib
import math
import os
from time import perf_counter

import torch
from torch import nn
from torchvision.models import ResNet18_Weights

from .config import CLASSES, MANIFEST, OUTPUTS, PREPROCESS, RESNET_CHECKPOINT, RESNET_OUTPUTS, ROOT, write_json
from .data import VegetableDataset, make_loader
from .model import build_model
from .train import environment, run_epoch, seed_training


WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
MODEL = {"architecture": "resnet18", "classes": len(CLASSES)}
AUGMENTATION = {"horizontal_flip_p": 0.5, "rotation_degrees": 10, "translation_fraction": [0.05, 0.05],
                "scale_range": [0.94, 1.06], "brightness": 0.20, "contrast": 0.20, "saturation": 0.18}
TRAINING = {"optimizer": "AdamW", "weight_decay": 1e-4, "loss": "CrossEntropyLoss", "batch_size": 32,
            "seed": 42, "workers": 0, "head_epochs": 3, "head_only_lr": 1e-3, "fine_tune_max_epochs": 20,
            "backbone_lr": 1e-4, "fine_tune_head_lr": 3e-4, "fine_tune_patience": 5}


def preprocessing():
    preset = WEIGHTS.transforms()
    return {**PREPROCESS, "size": list(preset.crop_size) * 2, "mean": list(preset.mean), "std": list(preset.std),
            "fill": [round(channel * 255) for channel in preset.mean]}


def batch_norm_buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()
            if name.endswith(("running_mean", "running_var", "num_batches_tracked"))}


def train(smoke):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment. Check the NVIDIA driver and PyTorch device.")
    device = torch.device("cuda")
    settings = dict(TRAINING)
    settings["training_precision"] = "bfloat16 autocast" if torch.cuda.is_bf16_supported() else "float32"
    if smoke:
        settings.update(head_epochs=1, fine_tune_max_epochs=1, smoke_limit_per_class=8)
    checkpoint = OUTPUTS / "smoke" / "resnet18" / "best.pt" if smoke else RESNET_CHECKPOINT
    history_path = checkpoint.parent / "training_history.json" if smoke else RESNET_OUTPUTS / "training_history.json"

    seed_training(settings["seed"])
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".work" / "torch"))
    preprocess = preprocessing()
    limit = settings.get("smoke_limit_per_class")
    training = VegetableDataset("train", preprocess, limit_per_class=limit, augmentation=AUGMENTATION)
    validation = VegetableDataset("validation", preprocess, limit_per_class=limit)
    train_loader = make_loader(training, settings["batch_size"], device, shuffle=True,
                               generator=torch.Generator().manual_seed(settings["seed"]))
    val_loader = make_loader(validation, settings["batch_size"], device)
    model = build_model(MODEL, weights=WEIGHTS).to(device)
    model.requires_grad_(False)
    model.fc.requires_grad_(True)
    initial_bn = batch_norm_buffers(model)
    loss_function = nn.CrossEntropyLoss()
    metadata = {
        "model": MODEL, "pretrained_weights": f"ResNet18_Weights.{WEIGHTS.name}", "weights_url": WEIGHTS.url,
        "preprocessing": preprocess, "augmentation": AUGMENTATION, "training": settings,
        "class_to_idx": {name: index for index, name in enumerate(CLASSES)}, "environment": environment(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "stage_1_trainable_parameter_count": sum(parameter.numel() for parameter in model.fc.parameters()),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "selection_criterion": "minimum validation loss across both stages (strict improvement)",
        "sample_counts": {"train": len(training), "validation": len(validation)},
    }
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=settings["head_only_lr"], weight_decay=settings["weight_decay"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    stopped = False
    best_fine_tune_loss = math.inf
    stale = 0
    torch.cuda.synchronize()
    start = perf_counter()
    for stage, epochs in (("head", settings["head_epochs"]), ("fine_tune", settings["fine_tune_max_epochs"])):
        frozen = stage == "head"
        if not frozen:
            model.requires_grad_(True)
            backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("fc.")]
            optimizer = torch.optim.AdamW([
                {"params": backbone, "lr": settings["backbone_lr"]},
                {"params": model.fc.parameters(), "lr": settings["fine_tune_head_lr"]},
            ], weight_decay=settings["weight_decay"])
        for stage_epoch in range(1, epochs + 1):
            epoch_start = perf_counter()
            train_loss, train_accuracy = run_epoch(model, train_loader, loss_function, device, optimizer,
                                                   frozen_backbone=frozen, mixed_precision=settings["training_precision"] != "float32")
            val_loss, val_accuracy = run_epoch(model, val_loader, loss_function, device)
            if not all(map(math.isfinite, (train_loss, val_loss))):
                raise RuntimeError(f"Non-finite loss in {stage} epoch {stage_epoch}.")
            row = {"stage": stage, "stage_epoch": stage_epoch, "epoch": len(history) + 1, "train_loss": train_loss,
                   "train_accuracy": train_accuracy, "validation_loss": val_loss, "validation_accuracy": val_accuracy,
                   "epoch_seconds": perf_counter() - epoch_start}
            history.append(row)
            if best is None or val_loss < best["validation_loss"]:
                best = row
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save({**metadata, "best_epoch": row["epoch"], "best_stage": stage,
                            "best_validation_loss": val_loss, "best_validation_accuracy": val_accuracy,
                            "state_dict": model.state_dict()}, temporary)
                temporary.replace(checkpoint)
            print(f"Epoch {row['epoch']:02d} ({stage} {stage_epoch:02d}) | train loss {train_loss:.4f}, "
                  f"accuracy {train_accuracy:.2%} | val loss {val_loss:.4f}, accuracy {val_accuracy:.2%} | "
                  f"{row['epoch_seconds']:.1f} s", flush=True)
            if not frozen:
                if val_loss < best_fine_tune_loss:
                    best_fine_tune_loss, stale = val_loss, 0
                else:
                    stale += 1
                if stale >= settings["fine_tune_patience"]:
                    stopped = True
                    break
        if frozen and any(not torch.equal(before, after) for before, after in zip(initial_bn.values(), batch_norm_buffers(model).values())):
            raise RuntimeError("BatchNorm running statistics changed during the frozen-backbone stage.")

    result = {**metadata, "best_epoch": best["epoch"], "best_stage": best["stage"],
              "best_validation_loss": best["validation_loss"], "best_validation_accuracy": best["validation_accuracy"],
              "early_stopping_triggered": stopped, "stage_1_batch_norm_unchanged": True,
              "training_wall_seconds": perf_counter() - start, "epochs": history,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    write_json(history_path, result)
    print(f"Saved epoch {best['epoch']} to {checkpoint}. Training time: {result['training_wall_seconds']:.2f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train the ImageNet-pretrained ResNet18 experiment on CUDA.")
    parser.add_argument("--smoke", action="store_true", help="Exercise both stages on a small train/validation subset.")
    train(parser.parse_args().smoke)


if __name__ == "__main__":
    main()
