import torch
import torch.nn as nn

class VGG16D(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = self._make_conv_layer(3, 64, 2)
        self.conv2 = self._make_conv_layer(64, 128, 2)
        self.conv3 = self._make_conv_layer(128, 256, 3)
        self.conv4 = self._make_conv_layer(256, 512, 3)
        self.conv5 = self._make_conv_layer(512, 512, 3)
        self.adapt_max_pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_class, bias=True)
        )

    def _make_conv_layer(self, in_channels, out_channels, conv_num):
        layer = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)]
        layer.extend([nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1) for _ in range(conv_num - 1)])
        return nn.Sequential(*layer, nn.BatchNorm2d(num_features=out_channels), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print("After conv:", x.shape)
        x = self.adapt_max_pool(x)
        # print("After adapt", x.shape)
        # x = x.view(-1)
        x = self.flatten(x)
        # print("After flatten:", x.shape)
        x = self.fc(x)
        return x

# class VGG16C(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         ...
#     def __forward__(self, x):
#         ...

if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = VGG16D(10)
    x = model(x)
    print(x.shape)