from torch import Tensor
import torch.nn as nn
import torch
import logging

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, activation=None, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight.requires_grad = False
        self.activation = activation
    
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        self.out = super().forward(input)

        if self.activation:
            self.out = self.activation(self.out)
        
        return self.out
    
    def compute_update(self, rule='simple'):

        match rule:
            case 'simple':
                delta_w = self.out.t() @ self.input
            case 'weight_decay':
                # repeat input, out, weight to match sizes
                x = self.input  \
                        .repeat(1,1,self.out.shape[-1]) \
                        .view(self.input.shape[0], self.input.shape[1], self.out.shape[-1]) \
                        .permute(0, 2, 1)
                w = self.weight.repeat(self.input.shape[0], 1, 1)
                y = self.out.repeat(1, 1, self.input.shape[1])  \
                        .view(self.out.shape[0], self.out.shape[1], self.input.shape[1])
                
                # weight decay
                delta_w = y * (x - w)
                # sum over batch
                delta_w = delta_w.sum(dim=0)
                
            case 'hpca':
                delta_w = torch.empty(self.weight.shape, device=self.weight.device)
                partials = torch.empty((self.weight.shape[0], self.out.shape[0], self.weight.shape[1]), device=self.weight.device)
                for i in range(self.weight.shape[0]):
                    partials[i] = self.out[:, i].unsqueeze(1) @ self.weight[i, :].unsqueeze(0)
                    delta_w[i] = (self.out[:, i].unsqueeze(1).t() @ (self.input - partials[:i+1].sum(dim=0))).squeeze(0)
                
                return delta_w
            case _:
                logging.error(f"{rule} not implemented for Linear layer")
                exit(1)
        

        if self.weight.max().isnan():
            logging.error("NaN in weights")
            print(f"Input max: {self.input.max()}")
            print(f"Output max: {self.out.max()}")
            exit(1)
        
        del self.input, self.out
        
        return delta_w

    def update_weights(self, lr=0.001, rule='simple'):
        delta_w = self.compute_update(rule=rule)
        self.weight = nn.Parameter(self.weight + lr * delta_w, requires_grad=False)