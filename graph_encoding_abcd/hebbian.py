import torch
import torch.nn as nn
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity
import os, psutil
from numba import jit
import numpy as np 

# Linear (12288 -> 23064)
# ~ 741 ms no compiled
# ~ 57 ms compiled
@torch.compile()
def abcd(pre, post, a, b, c0, c1, d0, d1, lr_in, lr_out):
    
    s1 = pre.shape[0] # batch size
    s2 = post.shape[-1] # out_shape
    s3 = pre.shape[-1] # in_shape

    # expand variables to have size batch_size x out_shape x in_shape
    pre = pre.unsqueeze(1).expand(-1, s2, -1)
    c0 = c0.unsqueeze(0).unsqueeze(1).expand(-1, s2, -1)
    d0 = d0.unsqueeze(0).expand(s2, -1)
    lr_in = lr_in.unsqueeze(0).expand(s2, -1)

    post = post.unsqueeze(-1).expand(-1, -1, s3)
    c1 = c1.unsqueeze(0).unsqueeze(-1).expand(-1, -1, s3)
    d1 = d1.unsqueeze(-1).expand(-1, s3)
    lr_out = lr_out.unsqueeze(-1).expand(-1, s3)
    
    lr = (lr_in + lr_out).div_(2)
    
    # calculate d = d0 * d1
    d = (d0 * d1).unsqueeze(0).expand(s1, -1, -1)
    
    # start from zeros and add d
    result = torch.zeros(s1, s2, s3, device=pre.device, dtype=pre.dtype).add_(d)
    
    # add a * pre
    result.addcmul_(a.expand(s1, s2, -1), pre)

    # add b * post
    result.addcmul_(b.unsqueeze(-1).expand(s1, -1, s3), post)

    # add c0 * c1 * pre * post
    result.addcmul_(c0 * c1, pre * post)

    return torch.mean(result, dim=0).mul_(lr)


@torch.compile()
def dw_fast(w_in, w_out, A_pre, B_post, C_pre_post, D, lr, agg_func):
    return (
            agg_func(A_pre[:, w_in], dim=-1).values + 
            agg_func(B_post[:, w_out], dim=-1).values + 
            agg_func(C_pre_post[:, w_out, w_in], dim=-1).values +
            agg_func(D[w_out, w_in], dim=-1).values
        ) * agg_func(lr[w_out, w_in], dim=-1).values


def shared_weights_abcd_fast(layer, pre, post, a, b, c0, c1, d0, d1, lr_in, lr_out, agg_func=torch.max):

    s1 = pre.shape[0] # batch size
    s2 = post.shape[-1] # out_shape
    s3 = pre.shape[-1] # in_shape

    c0 = c0.unsqueeze(0).unsqueeze(1).expand(s1, s2, -1)
    d0 = d0.unsqueeze(0).expand(s2, -1)
    lr_in = lr_in.unsqueeze(0).expand(s2, -1)

    c1 = c1.unsqueeze(0).unsqueeze(-1).expand(s1, -1, s3)
    d1 = d1.unsqueeze(-1).expand(-1, s3)
    lr_out = lr_out.unsqueeze(-1).expand(-1, s3)
    
    result = torch.zeros(s2, s3, device=pre.device, dtype=pre.dtype)

    in_ch = layer.shared_weights.shape[0]
    out_ch = layer.shared_weights.shape[1]
    k_size = layer.shared_weights.shape[2]

    A_pre = a.unsqueeze(0).expand(s1, -1)*pre
    B_post = b*post
    C_pre_post = c0 * c1 * pre.unsqueeze(1).expand(-1, s2, -1) * post.unsqueeze(-1).expand(-1, -1, s3)
    D = d0 * d1
    lr = (lr_in + lr_out).div_(2)

    for i in range(in_ch):
        for o in range(out_ch):
            w_in = layer.shared_weights[i, o, :, 0]
            w_out = layer.shared_weights[i, o, :, 1]
            
            delta_w = dw_fast(w_in, w_out, A_pre, B_post, C_pre_post, D, lr, agg_func)
            
            result[w_out[:], w_in[:]] = delta_w.mean(axis=0)[:].unsqueeze(-1).expand(-1, w_in.shape[-1])

    return result


def update_weights(layer, pre, post, ABCD_params, lr=0.0001, shared_w=False, agg_func=torch.max):

    if ABCD_params[layer.idx]['in']['C'] == None: # ABCD schared with previous layer
        A = ABCD_params[layer.idx - 1]['out']['A']
        B = ABCD_params[layer.idx]['out']['B']
        C0 = ABCD_params[layer.idx - 1]['out']['C']
        C1 = ABCD_params[layer.idx]['out']['C']
        D0 = ABCD_params[layer.idx - 1]['out']['D']
        D1 = ABCD_params[layer.idx]['out']['D']
        lr_in = ABCD_params[layer.idx - 1]['out']['lr']
        lr_out = ABCD_params[layer.idx]['out']['lr']
    else:
        A = ABCD_params[layer.idx]['in']['A']
        B = ABCD_params[layer.idx]['out']['B']
        C0 = ABCD_params[layer.idx]['in']['C']
        C1 = ABCD_params[layer.idx]['out']['C']
        D0 = ABCD_params[layer.idx]['in']['D']
        D1 = ABCD_params[layer.idx]['out']['D']
        lr_in = ABCD_params[layer.idx]['in']['lr']
        lr_out = ABCD_params[layer.idx]['out']['lr']
    
    if shared_w:
        w_matrix = shared_weights_abcd_fast(layer, pre, post, A, B, C0, C1, D0, D1, lr_in, lr_out, agg_func=agg_func)
    else:
        w_matrix = abcd(pre, post, A, B, C0, C1, D0, D1, lr_in, lr_out)

    if hasattr(layer, 'mask_tensor'):
        w_matrix.mul_(layer.mask_tensor.t())

    w_matrix = layer.weight + w_matrix

    w_matrix = w_matrix / w_matrix.abs().max()

    layer.weight = nn.Parameter(w_matrix, requires_grad=False)
    

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
    
    # if shared_w:
    #     w_matrix = shared_weights(layer, w_matrix)

    layer.weight = nn.Parameter(w_matrix, requires_grad=False)