import argparse
import hashlib
import os
import platform
import random
import subprocess
from contextlib import nullcontext
from time import perf_counter

import numpy as np
import torch
import torchvision
from torch import nn

from .config import AUGMENTATION, CLASSES, MANIFEST, MODEL, OUTPUTS, PREPROCESS, TRAINING, write_json
from .data import VegetableDataset, make_loader
from .model import VegetableCNN


def seed_training(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the NVIDIA driver and PyTorch build before training.")
    device = torch.device("cuda")
    value = (torch.ones(32, device=device) * 3).sum().item()
    model = VegetableCNN(**MODEL).to(device)
    images = torch.rand(2, 3, *PREPROCESS["size"], device=device)
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1], device=device))
    loss.backward()
    torch.cuda.synchronize()
    gradient = model.classifier.weight.grad
    if value != 96 or logits.shape != (2, len(CLASSES)) or not torch.isfinite(gradient).all().item():
        raise RuntimeError("The CUDA tensor or CNN forward/backward check returned an invalid result.")
    return {"tensor_sum": value, "logit_shape": list(logits.shape), "loss": loss.item(), "finite_gradients": True}


def environment(device):
    info = {"python": platform.python_version(), "os": platform.platform(), "windows_edition": platform.win32_edition(),
            "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
            "cuda_runtime": torch.version.cuda, "device": str(device)}
    if device.type == "cuda":
        info.update(gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
                    cuda_architectures=torch.cuda.get_arch_list(), cudnn=torch.backends.cudnn.version())
        try:
            info["nvidia_driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).splitlines()[0].strip()
        except (OSError, subprocess.CalledProcessError, IndexError):
            info["nvidia_driver"] = None
    return info


def run_epoch(model, loader, criterion, device, optimizer=None, *, frozen_backbone=False, mixed_precision=False):
    training = optimizer is not None
    model.train(training and not frozen_backbone)
    if training and frozen_backbone:
        model.fc.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    with torch.enable_grad() if training else torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            precision = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if mixed_precision else nullcontext()
            with precision:
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            running_loss += loss.detach() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum()
    return running_loss.item() / len(loader.dataset), correct.item() / len(loader.dataset)


def train(device, smoke):
    settings = dict(TRAINING)
    output_dir = OUTPUTS / ("smoke" if smoke else "baseline_reproduction")
    checkpoint = output_dir / "best.pt"
    history_path = output_dir / "training_history.json"
    if smoke:
        settings["max_epochs"] = 1
        settings["smoke_limit_per_class"] = 8
    limit = settings.get("smoke_limit_per_class")

    seed_training(settings["seed"])
    train_data = VegetableDataset("train", limit_per_class=limit)
    val_data = VegetableDataset("validation", limit_per_class=limit)
    train_loader = make_loader(train_data, settings["batch_size"], device, shuffle=True,
                               generator=torch.Generator().manual_seed(settings["seed"]))
    val_loader = make_loader(val_data, settings["batch_size"], device)
    model = VegetableCNN(**MODEL).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    metadata = {
        "model": MODEL, "class_to_idx": {label: index for index, label in enumerate(CLASSES)},
        "preprocessing": PREPROCESS, "augmentation": AUGMENTATION, "training": settings,
        "environment": environment(device), "parameter_count": sum(p.numel() for p in model.parameters()),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "selection_criterion": "minimum validation loss (strict improvement)",
        "sample_counts": {"train": len(train_data), "validation": len(val_data)},
    }

    history = []
    best_loss = float("inf")
    best_epoch = stale_epochs = 0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    for epoch in range(1, settings["max_epochs"] + 1):
        epoch_start = perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError(f"Non-finite training or validation loss at epoch {epoch}.")
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "validation_loss": val_loss, "validation_accuracy": val_accuracy,
               "epoch_seconds": perf_counter() - epoch_start}
        history.append(row)
        if val_loss < best_loss:
            best_loss, best_epoch, stale_epochs = val_loss, epoch, 0
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save({**metadata, "best_epoch": best_epoch, "best_validation_loss": best_loss,
                        "best_validation_accuracy": val_accuracy, "state_dict": model.state_dict()}, temporary)
            temporary.replace(checkpoint)
        else:
            stale_epochs += 1
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f}, accuracy {train_accuracy:.2%} | "
              f"val loss {val_loss:.4f}, accuracy {val_accuracy:.2%} | {row['epoch_seconds']:.1f} s", flush=True)
        if stale_epochs >= settings["patience"]:
            break

    result = {**metadata, "best_epoch": best_epoch, "best_validation_loss": best_loss,
              "best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
              "training_wall_seconds": perf_counter() - start, "epochs": history,
              "early_stopping_triggered": stale_epochs >= settings["patience"],
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    write_json(history_path, result)
    print(f"Saved epoch {best_epoch} to {checkpoint}. Training time: {result['training_wall_seconds']:.2f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train the vegetable CNN from a fresh initialization.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--check-device", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        seed_training(TRAINING["seed"])
        probe = verify_cuda()
        print(f"CUDA tensor and CNN forward/backward checks passed on {torch.cuda.get_device_name()}.", flush=True)
        write_json(OUTPUTS / "baseline_reproduction" / "cuda_check.json", {**environment(device), **probe})
    elif args.check_device:
        parser.error("--check-device requires CUDA.")
    if not args.check_device:
        train(device, args.smoke)


if __name__ == "__main__":
    main()
