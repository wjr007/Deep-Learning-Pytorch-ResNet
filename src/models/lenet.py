import torch
import torch.nn as nn

class LeNet5Block(nn.Module):
    """
    Bloco clássico LeNet-5 (LeCun et al., 1998).
    Composto por:
      1. Convolução 2D: in_channels -> out_channels=6, Kernel 5x5, Stride 1, Padding 0
      2. Subsampling (Average Pooling): Kernel 2x2, Stride 2
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            stride=1,
            padding=0
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> ex: [1, 1, 32, 32]
        x = self.conv1(x)  # -> [1, 6, 28, 28]
        x = self.pool1(x)  # -> [1, 6, 14, 14]
        return x
