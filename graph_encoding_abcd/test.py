from graph_encoding import Unrolled_GE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#generate random Cnn model in pytorch

from scipy.sparse import csc_matrix
from numba_utils import igraph_edges_to_sparse_matrix
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define your CNN layers here
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 3, 5)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(3, 3, 5)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(768, 10)
        self.fc2 = nn.Linear(10, 2)
        

    def forward(self, x):
        # Implement the forward pass of your model here
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 768)
        x = self.fc1(x)

        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(25, 5)
        self.linear2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = x.view(-1, 25)
        x = self.linear1(x)
        x = self.linear2(x)
        
        return x
    
class OneConv(nn.Module):
    def __init__(self):
        super(OneConv, self).__init__()
        self.c1 = nn.Conv2d(1, 3, 3)
        
    def forward(self, x):
        return self.c1(x)


def to_linear(adj_m):
    connections = {}
    m = adj_m.toarray()
    # zero rows
    m = m[np.any(m != 0, axis=1)]
    # zero cols
    m = m[:, np.any(m != 0, axis=0)]
    
    in_feat = m.shape[0]
    out_feat = m.shape[1]
    
    # print(m)
            
    print(f"Linear ({in_feat} -> {out_feat})")
    layer = nn.Linear(in_feat, out_feat)
    print("Layer shape", layer.weight.shape)
    print("Adj matrix shape", np.transpose(m).shape)
    print("Kernel size:", np.sqrt(np.sum(m[:, 0])))

    # for i in range(m.shape[0]):
    #     for j in range(m.shape[1]):
    #         if m[i, j] != 0:
    #             connections[i].append(j)
                
    print("Inverse dict: ", connections)
            
            
model = OneConv() #CNNModel()
for param in model.parameters():
    param.data = torch.ones_like(param.data)
#generate random input


# Generate random input
x = np.random.randn(1, 1, 5, 5)
# x = np.random.randn(1, 1, 25, 25)
model.forward(torch.from_numpy(x).float())

graph = Unrolled_GE(model, 5)

for idx_layer in range(0, len(graph.layers)):
    if graph.layers[idx_layer][1] == 'residual' or graph.layers[idx_layer][1] == 'conv' or graph.layers[idx_layer][1] == 'linear':
        G = graph.get_graph(idx_layer, mode='igraph')
        print(G.ecount(), G.vcount())

        row_indices, col_indices, data = igraph_edges_to_sparse_matrix(np.array(G.get_edgelist()),  int(G.vcount()), mode='ALL')
        adj_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(int(G.vcount()), int(G.vcount())))
        print(adj_matrix.shape)
        to_linear(adj_matrix)
        # print(adj_matrix)
        # if adj_matrix.shape[0] < 20:
        #     print(adj_matrix)
        #     print(adj_matrix.toarray())
            
        #count how many ones in adj_matrix
