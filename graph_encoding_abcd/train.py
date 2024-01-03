import torch
from model import *
from data import DataManager
from utils import model_to_unrolled, init_ABCD_parameters, update_weights


def train_step(model, train_loader, abcd_params):
    t = model[0].weight.dtype
    device = t = model[0].weight.device
    
    for i, (x, true_y) in enumerate(train_loader):
        print(f"Batch {i}")
        x = x.view(x.shape[0], -1).to(device).to(t)
        for layer in model:
            print(f"Train layer: {layer}  input shape: {x.shape}")
            if type(layer) == nn.Linear:
                y = layer(x)
                
                shared_w = False
                if hasattr(layer, 'shared_weights'):
                    shared_w = True
                update_weights(layer, x, y, abcd_params, shared_w=shared_w)
                x = y
            else:
                x = layer(x)
                
        acc = torch.sum(torch.argmax(y, dim=-1) == true_y) / true_y.shape[0]
        print(f"Accuracy: {acc}")

def main():
    device = 'cpu'
    model = CNNModel()
    
    data = DataManager("CIFAR10")
    train_loader = data.train_loader
    test_loader = data.test_loader
    
    x, _ = next(iter(train_loader))
    input_size = x.shape[-1]

    unrolled_model = model_to_unrolled(model, input_size, device=device)
    
    print(unrolled_model)
    ABCD_params = init_ABCD_parameters(unrolled_model, device=unrolled_model[0].weight.device)
    
    train_step(unrolled_model, train_loader, ABCD_params)

if __name__ == "__main__":
    main()