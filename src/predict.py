import argparse
import hashlib
import json
from pathlib import Path

import torch

from .audit import load_rgb
from .config import BASELINE_CHECKPOINT, CLASSES, DEPLOYMENT, EXPERIMENTS, RESNET_CHECKPOINT, ROOT
from .data import make_transform
from .model import build_model


def select_device(requested="auto"):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)


def deployed_checkpoint():
    if not DEPLOYMENT.is_file():
        return BASELINE_CHECKPOINT, None
    selection = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    return EXPERIMENTS[selection["model"]][0], selection["checkpoint_sha256"]


class Predictor:
    def __init__(self, checkpoint=BASELINE_CHECKPOINT, device="auto", *, expected_sha256=None):
        self.device = select_device(device)
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            destination = self.checkpoint.relative_to(ROOT).as_posix() if self.checkpoint.is_relative_to(ROOT) else str(self.checkpoint)
            release_assets = {RESNET_CHECKPOINT: "vegetable-resnet18.pt", BASELINE_CHECKPOINT: "vegetable-small-cnn.pt"}
            asset = release_assets.get(self.checkpoint)
            if asset is not None:
                raise FileNotFoundError(
                    f"Model file is missing at {destination}. Obtain {asset} from the project release once available, "
                    f"then place it at {destination}. See MODEL_WEIGHTS.md for the SHA-256 and placement instructions."
                )
            raise FileNotFoundError(f"Model file is missing at {destination}. Use --checkpoint with an existing file. See MODEL_WEIGHTS.md.")
        with self.checkpoint.open("rb") as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
            if expected_sha256 is not None and self.checkpoint_sha256 != expected_sha256:
                raise ValueError("The checkpoint differs from the recorded deployment selection. Check the SHA-256 in MODEL_WEIGHTS.md.")
            stream.seek(0)
            metadata = torch.load(stream, map_location="cpu", weights_only=True)
        mapping = metadata["class_to_idx"]
        if mapping != {name: index for index, name in enumerate(CLASSES)}:
            raise ValueError("Checkpoint class mapping differs from the project's five-class mapping.")
        self.classes = tuple(sorted(mapping, key=mapping.get))
        self.transform = make_transform(metadata["preprocessing"])
        self.model = build_model(metadata["model"])
        self.model.load_state_dict(metadata.pop("state_dict"))
        self.model.to(self.device).eval()
        self.metadata = metadata
        self.model_name = "ResNet18" if metadata["model"].get("architecture") == "resnet18" else "Small CNN baseline"

    @torch.inference_mode()
    def predict_image(self, image):
        batch = self.transform(image).unsqueeze(0).to(self.device)
        scores = self.model(batch).softmax(dim=1)[0].cpu().tolist()
        return dict(zip(self.classes, scores))

    def predict_file(self, path):
        return self.predict_image(load_rgb(path))


def main():
    parser = argparse.ArgumentParser(description="Classify a photograph using the saved vegetable CNN.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=BASELINE_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    predictor = Predictor(args.checkpoint, args.device)
    scores = predictor.predict_file(args.image)
    print(json.dumps({"prediction": max(scores, key=scores.get), "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
