import torch.nn as nn
from .blocks import Conv2d
from hebb import Competitive
import torch

class ConvModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=(3,3), activation=self.relu)

    def forward(self, x):
        x = self.conv1(x)
 
        return x
    
    def update_weights(self, lr, rule):
        self.conv1.update_weights(lr=lr, rule=rule)