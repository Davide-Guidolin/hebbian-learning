import torch.nn as nn
import torch.nn.functional as F
import torch

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
    
# Triangle activation used in SoftHebb    
class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power

    
def get_out_size(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)

#https://www.quora.com/What-percentage-has-been-reached-on-CIFAR-10-using-only-a-multi-layer-perceptron
class BaseNet(nn.Module): #) ABCD Params: 270888
    def __init__(self):
        super(BaseNet, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.activ1 = Triangle(power=0.7)
        self.max_pool1 = nn.MaxPool2d(4, 2, 1)
        out_size = get_out_size(32, 5, pooling_size=2)
        
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3)
        self.activ2 = Triangle(power=1.4)
        self.max_pool2 = nn.MaxPool2d(4, 2, 1)
        out_size = get_out_size(out_size, 3, pooling_size=2)
        
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3)
        self.activ3 = Triangle(power=1.)
        self.avg_pool3 = nn.AvgPool2d(2, 2, 0)
        out_size = get_out_size(out_size, 3, pooling_size=2)

        self.fc4 = nn.Linear(out_size*out_size*256, 10)        
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.activ1(x)
        x = self.max_pool1(x)
        
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.activ2(x)
        x = self.max_pool2(x)
        
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.activ3(x)
        x = self.avg_pool3(x)

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
        
        self.conv3 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3)
        self.avg_pool3 = nn.AvgPool2d(2, 2, 0)
        self.bn3 = nn.BatchNorm2d(384)
        out_size = get_out_size(out_size, 3, pooling_size=2)

        self.fc4 = nn.Linear(out_size*out_size*384, 10)
        
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
    
    
class CNN_CarRacing(nn.Module):
    def __init__(self):
        super(CNN_CarRacing, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, bias=False)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=8, kernel_size=5, stride=2, bias=False)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(392, 128, bias=False)
        self.tanh3 = nn.Tanh()
        self.linear2 = nn.Linear(128, 64, bias=False)
        self.tanh4 = nn.Tanh()
        self.out = nn.Linear(64, 3, bias=False)
    
    
    def forward(self, ob):
        
        state = torch.as_tensor(np.array(ob.copy()))
        state = state.float()
        
        x1 = self.pool(torch.tanh(self.conv1(state)))
        x2 = self.pool(torch.tanh(self.conv2(x1)))
        
        x3 = x2.view(-1)
        
        x4 = torch.tanh(self.linear1(x3))   
        x5 = torch.tanh(self.linear2(x4))
        
        o = self.out(x5)

        return x3, x4, x5, o
