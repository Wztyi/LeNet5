import torch
from torch import nn
from torchsummary import summary


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # super().__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 池化不改变其通道大小
        self.c3 = nn.Conv2d(in_channels=6, kernel_size=5, stride=1, padding=0, out_channels=16)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(in_features=5 * 5 * 16, out_features=120)
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.f7 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.c1(x)
        x = self.sig(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.sig(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动判断是cpu还是gpu 进行判断
    model = LeNet().to(device)
    print(summary(model, input_size=(1, 28, 28)))
