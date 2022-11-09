import torch
from torch import nn


class ShouZheng(nn.Module):
    def __init__(self):
        super(ShouZheng, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2, stride=1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2, stride=1),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2, stride=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 64),
            nn.Dropout(p=0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    sz = ShouZheng()
    test_input = torch.ones(64, 3, 32, 32)
    test_output = sz(test_input)
    print(test_output.shape)
