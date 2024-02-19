from graph_encoding import Unrolled_GE
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import csc_matrix
import scipy
from numba_utils import igraph_edges_to_sparse_matrix
import os, psutil
from hebbian import update_weights
from copy import deepcopy

class UnrolledModel:
    def __init__(self, model: nn.Module, input_size: int|tuple|list, device: str = 'cpu'):
        self.input_size = input_size
        if type(self.input_size) in [tuple, list]:
            if len(self.input_size) > 2:
                print("Cannot handle more than 2 dimensions")
                exit(1)
                
            if self.input_size[0] == self.input_size[1]:
                self.input_size = self.input_size[0]
            else:
                print("Input image should be squared")
                exit(1)
                
        self.device = device
        
        self.layers = self.model_to_unrolled(model)

    def conv_to_linear(self, adj_m: scipy.sparse._csc.csc_matrix, k_size: int, in_ch: int, out_ch: int) -> nn.Module:
        in_feat = np.count_nonzero(adj_m.getnnz(axis=1))
        out_feat = np.count_nonzero(adj_m.getnnz(axis=0))
        
        print("Reduce adj matrix")
        adj_m = adj_m[:in_feat, in_feat:]
        print("Adj matrix shape: ", adj_m.shape)
                
        print(f"MEM BEFORE LINEAR: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")

        print(f"Linear ({in_feat} -> {out_feat})")
        if 'cuda' in self.device:
            t = torch.float16
        else:
            t = torch.float32
            
        layer = nn.Linear(in_feat, out_feat).to(t)
        for p in layer.parameters():
            p.requires_grad = False
        print("Init weight")
        torch.nn.init.xavier_normal_(layer.weight)

        print("create coo_m")
        
        print(f"MEM BEFORE coo: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
        coo_m = adj_m.tocoo()
        
        # Create a sparse tensor from the COO matrix with values as 1
        
        print(f"MEM BEFORE mask_t: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
        print("create float tensor")
        mask_tensor = torch.sparse_coo_tensor(
            torch.LongTensor([coo_m.row.tolist(), coo_m.col.tolist()]),
            torch.ones_like(torch.FloatTensor(coo_m.data), dtype=torch.bool),
            torch.Size([in_feat, out_feat]),
            dtype=torch.bool
        )
        
        del coo_m
        
        print(f"MEM after mask_t: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
        print("Multiply weights")
        non_zero_params = torch.count_nonzero(mask_tensor.to_dense())
        print("NON ZERO params", non_zero_params)
        layer.weight.data.mul_(mask_tensor.t())
        
        layer.mask_tensor = mask_tensor.to_dense()
        
        print("Save shared weights indexes")
        n_shared = int(non_zero_params.item()/(in_ch * out_ch * k_size))

        shared_w = torch.zeros(in_ch, out_ch, k_size, 2, n_shared, dtype=torch.int32)
        # counter used to track the number of shared weights
        counters = torch.zeros(in_ch, out_ch, k_size, dtype=torch.int16)
                    
        indices = np.split(adj_m.indices, adj_m.indptr[1:-1])
        
        out_ch_step = int(adj_m.shape[1]/out_ch)
        
        out_ch_count = 0
        # iterate through columns
        for out_n in range(adj_m.shape[1]):
            # every out_ch_step change out channel
            if out_n == out_ch_step*(out_ch_count+1):
                out_ch_count += 1
                
            in_ch_count = 0
            k = 0
            # iterate through rows
            for in_n in indices[out_n]:
                c = counters[in_ch_count][out_ch_count][k].item()
                shared_w[in_ch_count][out_ch_count][k][0][c] = in_n
                shared_w[in_ch_count][out_ch_count][k][1][c] = out_n
                counters[in_ch_count][out_ch_count][k].add_(1)
                
                k += 1
                # every k_size change in channel and reset kernel to 0
                if k == k_size:
                    k = 0
                    in_ch_count += 1

        layer.shared_weights = shared_w
        
        return layer
        

    def model_to_unrolled(self, model: nn.Module) -> nn.Sequential:
                
        graph = Unrolled_GE(model, self.input_size)

        layers = []
        last_feat = 0
        # Save first dimension in case first layer is BatchNorm
        for d in graph.bdim:
            if d[0] != None:
                last_feat = d[0]
                break
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
                
                layer = self.conv_to_linear(adj_matrix, k_size[0]*k_size[1], in_ch, out_ch)
                layer.idx = layer_idx
                layer_idx += 1
                
                layers.append(layer)
                
                last_ch = out_ch
                last_feat = layers[-1].out_features
                
            elif graph.layers[idx_layer][1] == 'linear':
                layer = nn.Linear(last_feat, graph.layers[idx_layer][0].out_features)
                if 'cuda' in self.device:
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
                    layers.append(nn.BatchNorm1d(last_feat))
                else:
                    layers.append(t())
                    
        return layers
    
    def get_new_model(self):
        copy_layers = [deepcopy(l) for l in self.layers]
        return nn.Sequential(*copy_layers)
