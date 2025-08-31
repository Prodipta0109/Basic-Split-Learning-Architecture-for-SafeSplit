import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Head (EXACT: 520 params) --------
# Conv2d(1 -> 52, k=3, padding=1, bias=True):
# params = 52 * (1*3*3 + 1) = 52 * 10 = 520
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 52, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return F.relu(self.conv1(x))


# -------- Backbone (EXACT: 435,162 params) --------
# Design:
#   Conv2d(52 -> 54, k=3, p=1, bias=True)
#   BatchNorm2d(54)
#   ReLU
#   MaxPool2d(2)           # 28 -> 14
#   Conv2d(54 -> 16, k=3, p=1, bias=False)
#   BatchNorm2d(16)
#   ReLU
#   MaxPool2d(2)           # 14 -> 7
#   Flatten
#   Linear(16*7*7 -> 512, bias=True)
#
# Param counts:
#   conv2: 52*54*3*3 + 54(bias) = 468*54 + 54 = 25,272 + 54 = 25,326
#   bn2:   2*54 = 108
#   conv3: 54*16*3*3 + 0(bias) = 9*54*16 = 7,776
#   bn3:   2*16 = 32
#   fc1:   (16*7*7)*512 + 512 = 784*512 + 512 = 401,408 + 512 = 401,920
#   TOTAL BACKBONE = 25,326 + 108 + 7,776 + 32 + 401,920 = 435,162  
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = nn.Conv2d(52, 54, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(54)
        self.pool = nn.MaxPool2d(2, 2)   # 28->14 then 14->7
        self.conv3 = nn.Conv2d(54, 16, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 7 * 7, 512, bias=True)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 54x14x14
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # -> 16x7x7
        x = x.view(x.size(0), -1)                        # -> 784
        x = F.relu(self.fc1(x))                          # -> 512
        return x


# -------- Tail (EXACT: 5,130 params) --------
# Linear(512 -> 10, bias=True): 512*10 + 10 = 5,120 + 10 = 5,130
class Tail(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(512, 10, bias=True)

    def forward(self, x):
        return self.fc2(x)
