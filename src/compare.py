import argparse
import hashlib
import json

from .config import BASELINE, DEPLOYMENT, EXPERIMENTS, MANIFEST, OUTPUTS, RESNET_OUTPUTS, ROOT, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def deployment_from_validation(histories):
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    losses = {}
    for name, history in histories.items():
        checkpoint = EXPERIMENTS[name][0]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != history["checkpoint_sha256"]:
            raise ValueError(f"The {name} checkpoint and completed training history differ.")
        if history["manifest_sha256"] != manifest_digest:
            raise ValueError(f"The {name} experiment used a different dataset manifest.")
        best = min(history["epochs"], key=lambda row: row["validation_loss"])
        if (best["epoch"], best["validation_loss"]) != (history["best_epoch"], history["best_validation_loss"]):
            raise ValueError(f"The selected {name} epoch is inconsistent with its validation history.")
        losses[name] = best["validation_loss"]

    winner = min(losses, key=losses.get)
    return {"model": winner, "criterion": "lowest selected validation loss, baseline on tie", "validation_loss": losses,
            "checkpoint": EXPERIMENTS[winner][0].relative_to(ROOT).as_posix(),
            "checkpoint_sha256": histories[winner]["checkpoint_sha256"], "best_epoch": histories[winner]["best_epoch"]}


def main():
    parser = argparse.ArgumentParser(description="Choose deployment by validation alone and compare finalized models.")
    parser.add_argument("--select-only", action="store_true", help="Record the deployment before test and photo evaluation.")
    args = parser.parse_args()
    histories = {name: read(paths[1]) for name, paths in EXPERIMENTS.items()}
    selection = deployment_from_validation(histories)
    if args.select_only:
        write_json(DEPLOYMENT, selection)
        print(f"Selected {selection['model']} by validation loss at epoch {selection['best_epoch']}.")
        return
    if read(DEPLOYMENT) != selection:
        raise ValueError("The deployment must be recorded from the current training histories before evaluating.")
    latency = read(OUTPUTS / "latency_comparison.json")
    comparison = {"deployment": selection, "benchmark_image": latency["image"], "models": {}}
    for name, directory in (("baseline", BASELINE / "outputs"), ("resnet18", RESNET_OUTPUTS)):
        history = histories[name]
        test = read(directory / "test_metrics.json")
        photos = read(directory / "personal_predictions.json")
        if not history["checkpoint_sha256"] == test["checkpoint_sha256"] == photos["checkpoint_sha256"]:
            raise ValueError(f"The {name} evaluation and checkpoint hashes differ.")
        comparison["models"][name] = {
            "best_epoch": history["best_epoch"], "best_validation_loss": history["best_validation_loss"],
            "best_validation_accuracy": history["best_validation_accuracy"],
            "parameter_count": history["parameter_count"], "training_wall_seconds": history["training_wall_seconds"],
            "test": {key: test[key] for key in ("accuracy", "macro", "per_class", "test_count", "correct_count")},
            "latency": latency[name],
            "original_diagnostic": {"correct": photos["correct"], "count": photos["count"], "photos": photos["photos"]},
        }
    write_json(OUTPUTS / "model_comparison.json", comparison)
    print("Saved validation selection, test and timing comparison, and the five diagnostic predictions.")


if __name__ == "__main__":
    main()
