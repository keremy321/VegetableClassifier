import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "vegetables"
PERSONAL = ROOT / "data" / "personal"
OUTPUTS = ROOT / "outputs"
MANIFEST = OUTPUTS / "dataset_manifest.csv"
BASELINE = ROOT / "baseline"
BASELINE_CHECKPOINT = BASELINE / "checkpoints" / "best.pt"
RESNET_CHECKPOINT = ROOT / "checkpoints" / "resnet18" / "best.pt"
RESNET_OUTPUTS = OUTPUTS / "resnet18"
DEPLOYMENT = ROOT / "checkpoints" / "deployment.json"
FRESH = ROOT / "data" / "personal_fresh"
EXPERIMENTS = {
    "baseline": (BASELINE_CHECKPOINT, BASELINE / "outputs" / "training_history.json", OUTPUTS / "baseline_rerun"),
    "resnet18": (RESNET_CHECKPOINT, RESNET_OUTPUTS / "training_history.json", RESNET_OUTPUTS),
}
CLASSES = ("Brinjal", "Broccoli", "Carrot", "Potato", "Tomato")
SPLITS = ("train", "validation", "test")
MODEL = {"channels": [32, 64, 128], "classes": len(CLASSES)}
PREPROCESS = {
    "size": [128, 128],
    "mode": "RGB",
    "exif_transpose": True,
    "resize": "letterbox_bilinear",
    "fill": [128, 128, 128],
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
}
AUGMENTATION = {"rotation_degrees": 12, "horizontal_flip_p": 0.5, "brightness": 0.15, "saturation": 0.15}
TRAINING = {"optimizer": "Adam", "loss": "CrossEntropyLoss", "learning_rate": 0.001,
            "batch_size": 32, "max_epochs": 20, "patience": 4, "seed": 42, "workers": 0}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
