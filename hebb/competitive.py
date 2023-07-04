import torch
import torch.nn as nn
import logging
from models.blocks import Linear

class Competitive(nn.Module):
    def __init__(self, layer, strategy='hwta', k=2):
        super().__init__()
        self.layer = layer
        self.strategy = strategy
        self.k = min(k, layer.out_features)

    def forward(self, x):
        return self.layer(x)

    def apply_competition(self):
        
        match self.strategy:
            case 'hwta':
                if type(self.layer) == Linear:
                    self.linear_wta()
            case _:
                logging.error(f"{self.strategy} not implemented")
                exit(1)

    def linear_wta(self):
        # calculate distance between input and weights
        input_weight_distance = torch.cdist(self.layer.input, self.layer.weight, p=2)
        # get indices of highest distances
        max_activities = torch.topk(input_weight_distance, self.k, dim=1).indices
        # set y to 0 or 1 depending on previously calculated indices
        self.layer.out.zero_()
        self.layer.out[torch.arange(self.layer.out.shape[0]).unsqueeze(1), max_activities] = 1
        