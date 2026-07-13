import torch
import torch.nn as nn


class SDFHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def forward(self, x):
        return torch.tanh(self.head(x))


if __name__ == "__main__":
    head = SDFHead(48, 9)
    x = torch.randn(2, 48, 224, 224)
    y = head(x)
    print("sdf", tuple(y.shape), "range", float(y.min()), float(y.max()),
          "params", sum(p.numel() for p in head.parameters()))
