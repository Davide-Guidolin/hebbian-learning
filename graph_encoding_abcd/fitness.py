import os
import torch.nn as nn
from torch.optim import SGD
import torch
from hebbian import update_weights
from torch.profiler import profile, record_function, ProfilerActivity

def evaluate(model, data_loader, abcd_params, pop_index=-1, shared_dict=None, abcd_learning_rate=0.0001, bp_last_layer=False, bp_lr=0.00001, bp_loss=nn.MSELoss):
    print(f"[{os.getpid()}] Starting evaluation")
    t = model[0].weight.dtype
    device = t = model[0].weight.device

    if bp_last_layer:
        model[-1].weight.requires_grad = True
        optim = SGD(model[-1].parameters(), lr=bp_lr)
        criterion = bp_loss()
    
    correct = 0
    total = 0
    for i, (x, true_y) in enumerate(data_loader):
        if i%50 == 0:
            print(f"[{os.getpid()}] Batch {i}/{len(data_loader)}")
        x = x.view(x.shape[0], -1).to(device).to(t)
        
        for l, layer in enumerate(model):
            if bp_last_layer and l == len(model)-1:
                optim.zero_grad()
                y = layer(x)
                out = nn.functional.softmax(y, dim=-1)
                true = torch.zeros(out.shape, dtype=out.dtype)
                true[torch.arange(true_y.shape[0]), true_y] = 1.0
                
                loss = criterion(out, true)
                loss.backward()
                optim.step()
                
                x = y
            else:
                # print(f"[{os.getpid()}] Running layer: {layer}  input shape: {x.shape}")
                if type(layer) == nn.Linear:
                    if layer.weight.data.isnan().sum():
                        print(layer.weight.data)
                        print(f"Layer {l} has NAN !!!")
                        exit(0)
                    y = layer(x)
                    
                    shared_w = False
                    if hasattr(layer, 'shared_weights'):
                        shared_w = True
                    
                    # print(f"[{os.getpid()}] Update weights")
                    update_weights(layer, x, y, abcd_params, shared_w=shared_w, lr=abcd_learning_rate)
                    
                    x = y
                else:
                    x = layer(x)
        
        # print(f"[{os.getpid()}]", x[0])
        # print(torch.argmax(x, dim=-1))
        correct += torch.sum(torch.argmax(x, dim=-1) == true_y)
        total += true_y.shape[0]

    print(f"[{os.getpid()}] {correct} {total}")
    acc = correct/total
    print(f"[{os.getpid()}]Accuracy: {acc}")
    
    if shared_dict is not None:
        shared_dict[pop_index] = acc
    
    return acc
        
    
# def evaluate_batch(unrolled_model, batch_x, true_y, abcd_params):
#     print(f"[{os.getpid()}] Starting ...")
#     seq_model = unrolled_model.unrolled_model
#     t = seq_model[0].weight.dtype
#     device = t = seq_model[0].weight.device
    
#     x = batch_x.view(batch_x.shape[0], -1).to(device).to(t)
#     for layer in seq_model:
#         print(f"[{os.getpid()}] Running layer: {layer}  input shape: {x.shape}")
#         if type(layer) == nn.Linear:
#             y = layer(x)
            
#             shared_w = False
#             if hasattr(layer, 'shared_weights'):
#                 shared_w = True
                
#             print(f"[{os.getpid()}] Updating layer weights")
#             update_weights(layer, x, y, abcd_params, shared_w=shared_w)
#             x = y
#         else:
#             x = layer(x)
            
#     correct = torch.sum(torch.argmax(y, dim=-1) == true_y)
#     total = true_y.shape[0]
    
#     del unrolled_model

#     return correct, total