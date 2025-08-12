import torch
import torch.nn as nn

class MyAlex(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96, kernel_size=11, stride=4, padding=2, bias=True),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ) # 27
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ) # 13
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_feature)
        )

    def forward(self, x):
        x = self.conv1(x)
        # print("Conv1:", x.shape)
        x = self.conv2(x)
        # print("Conv2:", x.shape)
        x = self.conv3(x)
        # print("Conv3:", x.shape)
        x = self.conv4(x)
        # print("Conv4:", x.shape)
        x = self.conv5(x)
        # print("Conv5:", x.shape)
        x = self.flatten(x)
        # print("Flatten:", x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x

if __name__ == '__main__':
    print("Hello")
    x = torch.rand((1, 3, 224, 224))
    model = MyAlex(1000)

    x = model(x)
    print(x.shape)
