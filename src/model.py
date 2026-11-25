from torch import nn
from torchvision.models import resnet18


class VegetableCNN(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        blocks = []
        input_channels = 3
        for output_channels in channels:
            blocks.extend([
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            input_channels = output_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(channels[-1], classes)

    def forward(self, images):
        return self.classifier(self.features(images).flatten(1))


def build_model(config, *, weights=None):
    architecture = config.get("architecture", "small_cnn")
    if architecture == "small_cnn":
        return VegetableCNN(config["channels"], config["classes"])
    if architecture == "resnet18":
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config["classes"])
        return model
    raise ValueError(f"Unsupported model architecture: {architecture}")
