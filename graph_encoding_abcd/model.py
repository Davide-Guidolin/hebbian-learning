import torch.nn as nn

class OneConv(nn.Module):
    def __init__(self):
        super(OneConv, self).__init__()
        self.c1 = nn.Conv2d(2, 3, 3)
        
    def forward(self, x):
        return self.c1(x)


class TwoConv(nn.Module):
    def __init__(self):
        super(TwoConv, self).__init__()
        self.c1 = nn.Conv2d(1, 3, 3)
        self.c2 = nn.Conv2d(3, 6, 3)
        
    def forward(self, x):
        x = self.c1(x)
        return self.c2(x)


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
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define your CNN layers here
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(768, 10)
        

    def forward(self, x):
        # Implement the forward pass of your model here
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(-1, 768)
        x = self.fc1(x)

        return x
    
    
def get_out_size(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(4, 2, 1)
        self.bn1 = nn.BatchNorm2d(96)
        out_size = get_out_size(32, 5, pooling_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=384, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(4, 2, 1)
        self.bn2 = nn.BatchNorm2d(384)
        out_size = get_out_size(out_size, 3, pooling_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=1536, kernel_size=3)
        self.avg_pool3 = nn.AvgPool2d(2, 2, 0)
        self.bn3 = nn.BatchNorm2d(1536)
        out_size = get_out_size(out_size, 3, pooling_size=2)

        self.fc4 = nn.Linear(out_size*out_size*1536, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.avg_pool3(x)
        x = self.bn3(x)

        x = self.fc4(x.view(x.shape[0], -1))

        return x
    

# 48 - 96 - 192 ok
# 96 - 96 - 192 ok creation - no training 2 process
# 96 - 384 - 192 core dumped creation 384-192 
#                                       in 96-384
#                                       in 96-384 res = GraphBase.add_edges(graph, es) OverflowError: int too big to convert 359, in get_graph_conv_unrolled
class BaseNet2(nn.Module):
    def __init__(self):
        super(BaseNet2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(4, 2, 1)
        self.bn1 = nn.BatchNorm2d(96)
        out_size = get_out_size(32, 5, pooling_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=384, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(4, 2, 1)
        self.bn2 = nn.BatchNorm2d(384)
        out_size = get_out_size(out_size, 3, pooling_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=192, kernel_size=3)
        self.avg_pool3 = nn.AvgPool2d(2, 2, 0)
        self.bn3 = nn.BatchNorm2d(192)
        out_size = get_out_size(out_size, 3, pooling_size=2)

        self.fc4 = nn.Linear(out_size*out_size*192, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.bn2(x)
        
        x = self.conv3(x)
        x = self.avg_pool3(x)
        x = self.bn3(x)

        x = self.fc4(x.view(x.shape[0], -1))

        return x