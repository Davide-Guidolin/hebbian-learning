from torch import Tensor
import torch.nn as nn
import torch
import logging

class Conv2d(nn.Conv2d):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation= 1,
            group=1,
            bias=True,
            padding_mode='zeros',
            activation=None,
            device=None, 
            dtype=None
    ):
        super().__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride,
            padding,
            dilation,
            group,
            bias,
            padding_mode,
            device,
            dtype)
        
        self.weight.requires_grad = False
        self.activation = activation
    
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        self.out = super().forward(input)

        if self.activation:
            self.out = self.activation(self.out)
        
        return self.out
    
    def compute_update(self, rule='simple'):
        # TODO
        # w: [3, 1, 3, 3] [out_ch, in_ch, k_size_0, k_size_1]
        # in : [64, 1, 28, 28]
        # out:  [64, 3, 26, 26]
        match rule:
            case 'simple':
                print(self.out.shape)
                print(self.input.shape)
                print(self.weight.shape)
                exit(0)
                pass
        return torch.zeros(self.weight.shape)

    def update_weights(self, lr=0.001, rule='simple'):
        delta_w = self.compute_update(rule=rule)
        self.weight = nn.Parameter(self.weight + lr * delta_w, requires_grad=False)