import torch.nn as nn
from .blocks import Linear
from hebb import Competitive
import torch

class SimpleModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.relu = nn.ReLU()

        self.linear1 = Linear(in_size, 256, activation=self.relu)
        self.competitive1 = Competitive(self.linear1)
        
        self.linear2 = Linear(256, 64, activation=self.relu)
        self.competitive2 = Competitive(self.linear2)

        self.linear3 = Linear(64, out_size, activation=self.relu)
        self.competitive3 = Competitive(self.linear3)

        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        out = self.softmax(x)
        return out
    
    def update_weights(self, lr, rule):
        self.competitive1.apply_competition()
        self.linear1.update_weights(lr=lr, rule=rule)

        self.competitive2.apply_competition()
        self.linear2.update_weights(lr=lr, rule=rule)

        self.competitive3.apply_competition()
        self.linear3.update_weights(lr=lr, rule=rule)