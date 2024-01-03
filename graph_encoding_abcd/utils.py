from graph_encoding import Unrolled_GE
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csc_matrix
import scipy
from numba_utils import igraph_edges_to_sparse_matrix
import os, psutil

def conv_to_linear(adj_m: scipy.sparse._csc.csc_matrix, k_size: int, in_ch: int, out_ch: int, device: str) -> nn.Module:
    in_feat = np.count_nonzero(adj_m.getnnz(axis=1))
    out_feat = np.count_nonzero(adj_m.getnnz(axis=0))
    
    print("Reduce adj matrix")
    adj_m = adj_m[:in_feat, in_feat:]
    print("Adj matrix shape: ", adj_m.shape)
    
    # print(adj_m.todense())
    print("Save shared weights indexes")
    shared_w = {}
    for i in range(in_ch):
        shared_w[i] = {}
        for o in range(out_ch):
            shared_w[i][o] = {}
            for k in range(k_size):
                shared_w[i][o][k] = []
                
    indices = np.split(adj_m.indices, adj_m.indptr[1:-1])
    
    # in_ch_step = int(adj_m.shape[0]/in_ch)
    out_ch_step = int(adj_m.shape[1]/out_ch)
    
    out_ch_count = 0
    for out_n in range(adj_m.shape[1]):
        if out_n == out_ch_step*(out_ch_count+1):
            out_ch_count += 1
            
        in_ch_count = 0
        k = 0
        for in_n in indices[out_n]:
            shared_w[in_ch_count][out_ch_count][k].append((in_n, out_n))
            k += 1
            if k == k_size:
                k = 0
                in_ch_count += 1
    
    print("MEM BEFORE LINEAR:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    print(f"Linear ({in_feat} -> {out_feat})")
    if 'cuda' in device:
        t = torch.float16
    else:
        t = torch.float32
        
    layer = nn.Linear(in_feat, out_feat).to(t)
    for p in layer.parameters():
        p.requires_grad = False
    print("Init weight")
    torch.nn.init.xavier_normal_(layer.weight)

    print("create coo_m")
    
    print("MEM BEFORE coo:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    coo_m = adj_m.tocoo()
    
    del adj_m
    # Create a sparse tensor from the COO matrix with values as 1
    
    print("MEM BEFORE mask_t:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    print("create float tensor")
    mask_tensor = torch.sparse_coo_tensor(
        torch.LongTensor([coo_m.row.tolist(), coo_m.col.tolist()]),
        torch.ones_like(torch.FloatTensor(coo_m.data), dtype=torch.bool),
        torch.Size([in_feat, out_feat]),
        dtype=torch.bool
    )
    
    
    print("MEM after mask_t:", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
    print("Multiply weights")
    layer.weight.data *= mask_tensor.t()
    del mask_tensor
    
    layer.shared_weights = shared_w
    
    return layer
    

def model_to_unrolled(model: nn.Module, input_size: int|tuple|list, device: str = 'cpu') -> nn.Sequential:
    if type(input_size) in [tuple, list]:
        if len(input_size) > 2:
            print("Cannot handle more than 2 dimensions")
            exit(1)
            
        if input_size[0] == input_size[1]:
            input_size = input_size[0]
        else:
            print("Input image should be squared")
            exit(1)
            
    graph = Unrolled_GE(model, input_size)

    layers = []
    last_feat = 0
    last_ch = 0
    layer_idx = 0
    for idx_layer in range(0, len(graph.layers)):
        if graph.layers[idx_layer][1] == 'residual':
            raise NotImplementedError
        elif graph.layers[idx_layer][1] == 'conv':
            G = graph.get_graph(idx_layer, mode='igraph')
            print(G.ecount(), G.vcount())

            row_indices, col_indices, data = igraph_edges_to_sparse_matrix(np.array(G.get_edgelist()),  int(G.vcount()), mode='ALL')
            adj_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(int(G.vcount()), int(G.vcount())))
            
            k_size = graph.layers[idx_layer][0].kernel_size
            in_ch = graph.layers[idx_layer][0].in_channels
            out_ch = graph.layers[idx_layer][0].out_channels
            print("k_size, in_ch, out_ch: ", k_size, in_ch, out_ch)
            
            layer = conv_to_linear(adj_matrix, k_size[0]*k_size[1], in_ch, out_ch, device)
            layer.idx = layer_idx
            layer_idx += 1
            
            layers.append(layer)
            
            last_ch = out_ch
            last_feat = layers[-1].out_features
            
        elif graph.layers[idx_layer][1] == 'linear':
            layer = nn.Linear(last_feat, graph.layers[idx_layer][0].out_features)
            if 'cuda' in device:
                layer = layer.to(torch.float16)
            else:
                layer = layer.to(torch.float32)
                
            layer.idx = layer_idx
            layer_idx += 1
            
            layers.append(layer)
            last_feat = layers[-1].out_features
            
        elif graph.layers[idx_layer][1] == 'pooling':
            in_size = graph.fmap_sizes[idx_layer][0]
            layers.append(nn.Unflatten(dim=-1, unflattened_size=(last_ch, in_size, in_size)))
            layers.append(graph.layers[idx_layer][0])
            layers.append(nn.Flatten())
            out_size = graph.fmap_sizes[idx_layer][1]
            last_feat = out_size*out_size*last_ch
            
        elif graph.layers[idx_layer][1] == 'module':
            t = type(graph.layers[idx_layer][0])
            if t == nn.BatchNorm2d:
                layers.append(t(last_feat))
            else:
                layers.append(t())
            
    return nn.Sequential(*layers).to(device)


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
            params[l_index]['C'] = torch.randn(torch.numel(layer.weight), device=device)
            params[l_index]['D'] = torch.randn(torch.numel(layer.weight), device=device)
            l_index += 1
        i += 1
        
    return params


def abcd(pre, post, a, b, c, d):    
    return (a*pre)[:, None, :] + (b*post)[:, :, None] + c*pre[:, None, :]*post[:, :, None] + d


def update_weights(layer, input, output, ABCD_params, shared_w=False):
    device = layer.weight.device
    w_matrix = layer.weight
    
    A = ABCD_params[layer.idx]['A']
    B = ABCD_params[layer.idx + 1]['B']
    C = ABCD_params[layer.idx]['C']
    D = ABCD_params[layer.idx]['D']

    w_matrix = abcd(input, output, A, B, C.reshape(output.shape[-1], input.shape[-1]), D.reshape(output.shape[-1], input.shape[-1]))
    # average over batch
    w_matrix = torch.mean(w_matrix, dim=0)
    
    if shared_w:
        in_ch = len(list(layer.shared_weights.keys()))
        out_ch = len(list(layer.shared_weights[0].keys()))
        k_size = len(list(layer.shared_weights[0][0].keys()))
        
        wl = []
        for i in range(in_ch):
            for o in range(out_ch):                
                for k in range(k_size):
                    w_list = []
                    sw = layer.shared_weights[i][o][k]
                    for (w_in, w_out) in sw:
                        w_list.append(w_matrix[w_out][w_in])
                    
                    m = torch.mean(torch.tensor(w_list))
                    for (w_in, w_out) in sw:
                        if (w_in, w_out) in wl:
                            print("ERROR")
                            exit(0)
                        wl.append((w_in, w_out))
                        w_matrix[w_out][w_in] = m
                        
        pass
    
    layer.weight = nn.Parameter(w_matrix, requires_grad=False)
    