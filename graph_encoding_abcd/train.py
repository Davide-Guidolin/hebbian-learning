import torch
from model import *
from data import DataManager
from unrolled_model import UnrolledModel

def init_ABCD_parameters(net: nn.Sequential, start_idx: int = 0, end_idx: int = -1, device: str = 'cpu') -> dict:
    if end_idx == -1 or end_idx > len(net):
        end_idx = len(net)
    
    params = {}
    l_index = start_idx
    i = start_idx
    while i < end_idx:
        if type(net[i]) == nn.Linear:
            params[l_index] = {'A':None, 'B': None, 'C':None, 'D': None}
            l_index += 1
        i += 1
    params[l_index] = {'A':None, 'B': None, 'C':None, 'D': None}
    
    l_index = start_idx
    i = start_idx
    while i < end_idx:
        if type(net[i]) == nn.Linear:
            layer = net[i]
            # A
            params[l_index]['A'] = torch.randn(layer.weight.shape[1], device=device)
            
            # B
            params[l_index + 1]['B'] = torch.randn(layer.weight.shape[0], device=device)
            
            # C, D
            params[l_index]['C'] = torch.randn((layer.weight.shape[0], layer.weight.shape[1]), device=device)
            params[l_index]['D'] = torch.randn((layer.weight.shape[0], layer.weight.shape[1]), device=device)
            l_index += 1
        i += 1
        
    return params


def main():
    
    model = OneConv()
    in_size = 5

    unrolled_model = UnrolledModel(model, in_size)
    
    ABCD_params = init_ABCD_parameters(unrolled_model.unrolled_model)
    
    for layer in unrolled_model.unrolled_model:
        if type(layer) == nn.Linear:
            x = torch.randn(1, 2*5*5)
            unrolled_model.update_weights(layer, x, layer(x), ABCD_params, shared_w=True)

if __name__ == "__main__":
    main()