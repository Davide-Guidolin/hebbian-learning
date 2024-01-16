import torch
import torch.nn as nn
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity

def abcd(pre, post, a, b, c, d):
    # return (a*pre)[:, None, :] + (b*post)[:, :, None] + c*pre[:, None, :]*post[:, :, None] + d
    s1 = pre.shape[0]
    s2 = post.shape[-1]
    s3 = pre.shape[-1]

    pre = pre.unsqueeze(1).expand(-1, s2, -1)
    

    post = post.unsqueeze(-1).expand(-1, -1, s3)
    
    # result = d.unsqueeze(0).repeat(s1, 1, 1)
    result = torch.zeros(s1, s2, s3)
    
    result.addcmul_(a.expand(s1, s2, -1), pre)

    result.addcmul_(b.unsqueeze(-1).expand(s1, -1, s3), post)

    result.addcmul_(c.unsqueeze(0).expand(s1, -1, -1), pre * post)

    return torch.mean(result, dim=0) #.add_(d)
    

def update_weights(layer, input, output, ABCD_params, lr=0.0001, shared_w=False):
    # print(f"[{os.getpid()}] Update Weights")
    
    w_matrix = layer.weight

    A = ABCD_params[layer.idx]['A']
    B = ABCD_params[layer.idx + 1]['B']
    C = ABCD_params[layer.idx]['C']
    D = ABCD_params[layer.idx]['D']

    w_matrix = abcd(input, output, A, B, C, D)
    # w_matrix = torch.mean(w_matrix, dim=0)

    if shared_w:
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
                    
        w_matrix = w_copy

    w_matrix = lr * w_matrix.clamp(-1, 1)
    
    layer.weight = nn.Parameter(w_matrix, requires_grad=False)