import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        reduce_dim = out_channels // 4

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        ) if stride==2 or in_channels != out_channels else nn.Identity()

        self.stage = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=reduce_dim, out_channels=reduce_dim, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=reduce_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=reduce_dim, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        return self.relu(self.stage(x) + shortcut)

class ResNet50(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self._make_residual(64, 256, 3, 1)
        self.stage2 = self._make_residual(256, 512, 4, 2)
        self.stage3 = self._make_residual(512, 1024, 6, 2)
        self.stage4 = self._make_residual(1024, 2048, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_class)
        )

    def _make_residual(self, in_channels, out_channels, num_block, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        layers += [ResidualBlock(out_channels, out_channels, 1) for _ in range(num_block - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224))
    model = ResNet50()
    model.train()
    x = model(x)

    print(x.shape)