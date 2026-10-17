from __future__ import annotations

import torch.nn as nn
from torchvision import models

CNN_ARCHITECTURES = {
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_small": "MobileNetV3 Small",
    "resnet18": "ResNet-18",
}


def build_ann(input_shape: tuple[int, ...], hidden_sizes: list[int], num_classes: int) -> nn.Module:
    """ANN. Obrazové vstupy sa sploštia na vektor; tabuľkové vstupy sú už vektor."""
    input_features = 1
    for dimension in input_shape:
        input_features *= int(dimension)
    layers: list[nn.Module] = [nn.Flatten()]
    previous = input_features
    for size in hidden_sizes:
        layers.extend([nn.Linear(previous, int(size)), nn.ReLU()])
        previous = int(size)
    layers.append(nn.Linear(previous, num_classes))
    return nn.Sequential(*layers)


def _replace_input_conv(module: nn.Module, channels: int) -> None:
    """Upraví prvú konvolúciu pre sivý MNIST alebo RGB CIFAR."""
    if isinstance(module, models.ResNet):
        old = module.conv1
        module.conv1 = nn.Conv2d(
            channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        return

    old = module.features[0][0]
    module.features[0][0] = nn.Conv2d(
        channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )


def build_cnn(architecture: str, input_shape: tuple[int, ...], num_classes: int) -> nn.Module:
    """CNN bez predtrénovaných váh; upraví vstupný kanál aj poslednú klasifikačnú vrstvu."""
    if architecture not in CNN_ARCHITECTURES:
        raise ValueError("Vyberte podporovanú CNN architektúru.")
    channels = int(input_shape[0])

    # Zadanie výslovne požaduje weights=None.
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        _replace_input_conv(model, channels)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        _replace_input_conv(model, channels)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        model = models.resnet18(weights=None)
        _replace_input_conv(model, channels)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


def build_model(
    model_type: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    *,
    hidden_sizes: list[int] | None = None,
    architecture: str = "resnet18",
) -> nn.Module:
    if model_type == "ANN":
        return build_ann(input_shape, hidden_sizes or [128, 64], num_classes)
    if model_type == "CNN":
        if len(input_shape) != 3:
            raise ValueError("CNN možno použiť iba na obrazové dáta.")
        return build_cnn(architecture, input_shape, num_classes)
    raise ValueError("Neznámy typ modelu. Vyberte ANN alebo CNN.")
