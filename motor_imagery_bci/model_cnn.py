import torch
import torch.nn as nn
import torch.nn.functional as F

class MotorImageryCNN(nn.Module):
    def __init__(self, n_channels=3, n_samples=1000, n_classes=2):
        super().__init__()

        self.conv_time = nn.Conv2d(
            1, 16,
            kernel_size=(1, 25),
            padding=(0, 12)
        )

        self.conv_space = nn.Conv2d(
            16, 32,
            kernel_size=(n_channels, 1)
        )

        self.bn = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
     
        x = F.elu(self.conv_time(x))
        x = F.elu(self.bn(self.conv_space(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
