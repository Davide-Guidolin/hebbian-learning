import torch
import torch.nn as nn
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity
import os, psutil

# Linear (12288 -> 23064)
# ~ 741 ms no compiled
# ~ 57 ms compiled
@torch.compile()
def abcd(pre, post, a, b, c0, c1, d0, d1):
    
    s1 = pre.shape[0] # batch size
    s2 = post.shape[-1] # out_shape
    s3 = pre.shape[-1] # in_shape

    pre = pre.unsqueeze(1).expand(-1, s2, -1)
    c0 = c0.unsqueeze(0).unsqueeze(1).expand(-1, s2, -1)
    d0 = d0.unsqueeze(0).expand(s2, -1)

    post = post.unsqueeze(-1).expand(-1, -1, s3)
    c1 = c1.unsqueeze(0).unsqueeze(-1).expand(-1, -1, s3)
    d1 = d1.unsqueeze(-1).expand(-1, s3)
    
    d = (d0 * d1).unsqueeze(0).expand(s1, -1, -1)
    result = torch.zeros(s1, s2, s3, device=pre.device).add_(d)
    
    result.addcmul_(a.expand(s1, s2, -1), pre)

    result.addcmul_(b.unsqueeze(-1).expand(s1, -1, s3), post)

    result.addcmul_(c0 * c1, pre * post)

    return torch.mean(result, dim=0)


def shared_weights(layer, w_matrix):
    in_ch = len(list(layer.shared_weights.keys()))
    out_ch = len(list(layer.shared_weights[0].keys()))
    k_size = len(list(layer.shared_weights[0][0].keys()))
    
    w_copy = w_matrix.clone()
    for i in range(in_ch):
        for o in range(out_ch):
            for k in range(k_size):
                sw = layer.shared_weights[i][o][k]
                
                w_in, w_out = zip(*sw)
                
                w_values = w_matrix[w_out, w_in]
                m = torch.mean(w_values)
                w_copy[w_out, w_in] = m
                
    return w_copy


def update_weights(layer, input, output, ABCD_params, lr=0.0001, shared_w=False):

    if ABCD_params[layer.idx]['in']['C'] == None: # ABCD schared with previous layer
        A = ABCD_params[layer.idx - 1]['out']['A']
        B = ABCD_params[layer.idx]['out']['B']
        C0 = ABCD_params[layer.idx - 1]['out']['C']
        C1 = ABCD_params[layer.idx]['out']['C']
        D0 = ABCD_params[layer.idx - 1]['out']['D']
        D1 = ABCD_params[layer.idx]['out']['D']
    else:
        A = ABCD_params[layer.idx]['in']['A']
        B = ABCD_params[layer.idx]['out']['B']
        C0 = ABCD_params[layer.idx]['in']['C']
        C1 = ABCD_params[layer.idx]['out']['C']
        D0 = ABCD_params[layer.idx]['in']['D']
        D1 = ABCD_params[layer.idx]['out']['D']

    w_matrix = abcd(input, output, A, B, C0, C1, D0, D1)
    
    if hasattr(layer, 'mask_tensor'):
        w_matrix.mul_(layer.mask_tensor.t())
    
    if shared_w:
        w_matrix = shared_weights(layer, w_matrix)

    w_matrix = w_matrix / w_matrix.abs().max()
    
    layer.weight = nn.Parameter(lr * w_matrix, requires_grad=False)
    

@torch.compile()
def softhebb(x, pre_act, w):
    soft = - torch.softmax(pre_act, dim=-1)
    max_neuron = torch.argmax(soft, dim=-1)
    # anti-hebbian
    soft[:, max_neuron] = -soft[:, max_neuron]
    
    dw = torch.matmul(soft.t(), x)
    yu = torch.multiply(soft, pre_act)
    yu = torch.sum(yu.t(), dim=1).view(-1, 1) * w
    dw -= yu
    
    del soft, yu
    
    return dw, max_neuron


def softhebb_update(layer, x, pre_act, lr=0.0001, shared_w=False):
    
    w_update, max_neuron = softhebb(x, pre_act, layer.weight)
    
    # normalize
    w_update = w_update / torch.abs(w_update).amax()

    w_matrix = layer.weight + lr * w_update
    
    if hasattr(layer, 'mask_tensor'):
        w_matrix.mul_(layer.mask_tensor.t())
    
    if shared_w:
        w_matrix = shared_weights(layer, w_matrix)

    layer.weight = nn.Parameter(w_matrix, requires_grad=False)