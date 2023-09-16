import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from typing import Tuple
import gymnasium as gym
import random
import matplotlib.pyplot as plt

def get_out_shape(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

class ClassicNet(nn.Sequential):
    def __init__(self, n_obs, n_actions):
        super().__init__(
            nn.Linear(n_obs, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, n_actions),
            nn.Softmax()
        )
    
    
class ConvNet(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        
        c, w, h = in_shape
        
        out_shape = [w, h]
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=16, kernel_size=5)
        out_shape = list(map(lambda x: get_out_shape(x, 5, pooling_size=2), out_shape))
        
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        out_shape = list(map(lambda x: get_out_shape(x, 3, pooling_size=2), out_shape))
        
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        out_shape = list(map(lambda x: get_out_shape(x, 3, pooling_size=2), out_shape))
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        out_size = out_shape[0] * out_shape[1] * 64
        
        self.head = nn.Linear(out_size, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.pool(x)
        
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.dropout(self.head(x))
        
        out = self.softmax(x)
        
        return out


def discount_rewards(rewards, gamma=0.95):
    disc_return = torch.pow(gamma, torch.arange(len(rewards)).float()) * rewards
    # disc_return = disc_return - disc_return.mean() - disc_return.min()                     
    return disc_return

def loss_fn(preds, r):
    loss = -1 * torch.sum(r * preds)
    
    if loss.isnan():
        print(r, preds)
        exit(0)
        
    return loss

def train_step(model, env, optimizer, loss_fn, transform, state_type='image', device='cuda'):
    
    model.train()
    
    state, _  = env.reset()
    done = False
    
    states = []
    actions = []
    cum_rewards = [] 
    
    total_rew = 0
    while not done:
        if state_type == 'image':
            s = env.render()
            s = transform(s).cuda()
            states.append(s.tolist())
        else:
            states.append(state)
            s = torch.tensor(state, device='cuda')
        
        if random.random() > 0.8:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = torch.argmax(model(s.unsqueeze(0)), dim=-1).item()
        
        state, rew, done, _, _ = env.step(a)
        total_rew += rew
        
        actions.append(a)
        cum_rewards.append(total_rew)
    
    states = torch.tensor(states, device=device)
    pred_batch = model(states).cpu()
    actions_batch = torch.tensor(actions)
    probs_batch = pred_batch.gather(dim=1, index=actions_batch.long().view(-1,1)).squeeze()
    
    disc_rewards = discount_rewards(torch.tensor(cum_rewards))
    loss = loss_fn(probs_batch, disc_rewards)
    
    optimizer.zero_grad()
    loss.backward()
    
    for name, p in model.named_parameters():
        if p.grad.sum().isnan():
            print(f"NaN grad in {name}")
            exit(1)
            
    optimizer.step()
    
    return loss, total_rew


def eval_step(model, env, transform, state_type='image', device='cuda'):
    model.eval()
    
    state, _  = env.reset()
    done = False
    
    total_rew = 0
    while not done:
        if state_type == 'image':
            s = env.render()
            s = transform(s).cuda()
        else:
            s = torch.tensor(state, device='cuda')
        
        with torch.no_grad():
            a = torch.argmax(model(s.unsqueeze(0)), dim=-1).item()
        
        state, rew, done, _, _ = env.step(a)
        total_rew += rew
    
    return total_rew
    
def main():
    N_EPOCHS = 2000
    EVAL_INTERVAL = 10
    img_size = (128, 128)
    state_type = 'numbers' # image
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    n_actions = env.action_space.n
    env.reset()
    
    transform = Compose([ToTensor(), Resize(img_size)])
    
    if state_type == 'image':
        model = ConvNet((3, img_size[0], img_size[1]), n_actions)
    else:
        model = ClassicNet(n_obs=4, n_actions=2)    
    
    model.apply(init_weights)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-1)
    
    losses = []
    rewards = []
    
    for e in range(N_EPOCHS):
        loss, total_rew = train_step(model, env, optim, loss_fn, transform, state_type=state_type)
        losses.append(loss.detach().numpy())
        
        print(f"[{e+1}/{N_EPOCHS}")
        print(f"loss: {loss} Rew: {total_rew}")
        
        if (e+1) % EVAL_INTERVAL == 0:
            total_rew = eval_step(model, env, transform, state_type=state_type)
            rewards.append(total_rew)
            print(f"Eval reward = {total_rew}")
    
    plt.plot(losses)
    plt.savefig("./loss.jpg")
    plt.clf()
    plt.plot(range(1, N_EPOCHS, EVAL_INTERVAL), rewards)
    plt.savefig("./eval_reward.jpg")
    
if __name__ == "__main__":
    main()