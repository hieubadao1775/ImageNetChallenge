# This is root version

import torch.nn as nn
import torch

class InceptionModule(nn.Module):
    def __init__(self, in_channels, conv11, reduce33, conv33, reduce55, conv55, pool_proj):
        super().__init__()
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=conv11, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce33, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce33, out_channels=conv33, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce55, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce55, out_channels=conv55, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        x4 = self.path4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)

class MyGoogLeNet(nn.Module):
    def __init__(self, num_feature=10):
        super().__init__()
        self.auxiliar1 = self._make_auxiliar(in_channels=512, num_feature=num_feature)
        self.auxiliar2 = self._make_auxiliar(in_channels=528, num_feature=num_feature)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(in_channels=192, conv11=64, reduce33=96, conv33=128, reduce55=16, conv55=32, pool_proj=32)
        self.inception3b = InceptionModule(in_channels=256, conv11=128, reduce33=128, conv33=192, reduce55=32, conv55=96, pool_proj=64)

        self.inception4a = InceptionModule(in_channels=480, conv11=192, reduce33=96, conv33=208, reduce55=16, conv55=48, pool_proj=64)
        self.inception4b = InceptionModule(in_channels=512, conv11=160, reduce33=112, conv33=224, reduce55=24, conv55=64, pool_proj=64)
        self.inception4c = InceptionModule(in_channels=512, conv11=128, reduce33=128, conv33=256, reduce55=24, conv55=64, pool_proj=64)
        self.inception4d = InceptionModule(in_channels=512, conv11=112, reduce33=144, conv33=288, reduce55=32, conv55=64, pool_proj=64)
        self.inception4e = InceptionModule(in_channels=528, conv11=256, reduce33=160, conv33=320, reduce55=32, conv55=128, pool_proj=128)

        self.inception5a = InceptionModule(in_channels=832, conv11=256, reduce33=160, conv33=320, reduce55=32, conv55=128, pool_proj=128)
        self.inception5b= InceptionModule(in_channels=832, conv11=384, reduce33=192, conv33=384, reduce55=48, conv55=128, pool_proj=128)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024, out_features=num_feature)
        )

    def _make_auxiliar(self, in_channels, num_feature):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=num_feature)
        )

    def forward(self, x):
        x = self.stage1(x)
        # print("After stage1:", x.shape)

        x = self.max_pool(x)
        x = self.inception3a(x)
        # print("Inception 3a:", x.shape)
        x = self.inception3b(x)
        # print("Inception 3b:", x.shape)

        x = self.max_pool(x)
        x = self.inception4a(x)
        # print("Inception 4a:", x.shape)
        auxiliar1 = self.auxiliar1(x) if self.training else None
        x = self.inception4b(x)
        # print("Inception 4b:", x.shape)
        x = self.inception4c(x)
        # print("Inception 4c:", x.shape)
        x = self.inception4d(x)
        # print("Inception 4d:", x.shape)
        auxiliar2 = self.auxiliar2(x) if self.training else None
        x = self.inception4e(x)
        # print("Inception 4e:", x.shape)

        x = self.max_pool(x)
        x = self.inception5a(x)
        # print("Inception 5a:", x.shape)
        x = self.inception5b(x)
        # print("Inception 5b:", x.shape)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x, auxiliar1, auxiliar2

if __name__ == '__main__':
    model = MyGoogLeNet(10)
    x = torch.rand((1, 3, 224, 224))

    output, auxiliar1, auxiliar2 = model(x)
    print(output.shape)
    print(auxiliar1.shape)
    print(auxiliar2.shape)
