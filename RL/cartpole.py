# https://nandakishorej8.medium.com/part-2-policy-based-reinforcement-learning-openais-cartpole-with-reinforce-algorithm-18de8cb5efa4

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from typing import Tuple
import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt

def get_out_shape(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)

class ClassicNet(nn.Sequential):
    def __init__(self, n_obs, n_actions):
        super().__init__(
            torch.nn.Linear(n_obs, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, n_actions),
            torch.nn.Softmax()
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


def discount_rewards(rewards, gamma=0.99):
    disc_return = torch.pow(gamma, torch.arange(len(rewards)).float()) * rewards
    disc_return = disc_return / (disc_return.max() - disc_return.min())                     
    return disc_return

def loss_fn(preds, r):
    loss = -1 * torch.sum(r * torch.log(preds))
    
    if loss.isnan():
        print(r, preds)
        exit(0)
        
    return loss

def train_step(model, env, optimizer, loss_fn, transform, state_type='image', device='cuda'):
    
    model.train()
    
    state, _  = env.reset()
    done = False
    truncated = False
    
    transitions = []
    
    total_rew = 0
    while not (done or truncated):
        if state_type == 'image':
            s = env.render()
            s = transform(s).cuda()
            state = s.tolist()
        else:
            s = torch.tensor(state, device='cuda')
        
        with torch.no_grad():
            act_prob = model(s.unsqueeze(0)).squeeze()
            a = np.random.choice(np.array([0,1]), p=act_prob.cpu().numpy())
        
        total_rew += 1
        transitions.append((state, a, total_rew))
        
        state, rew, done, truncated, _ = env.step(a)
    
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))

    disc_rewards = discount_rewards(torch.tensor(reward_batch))
    
    state_batch = torch.Tensor([s for (s,a,r) in transitions]).cuda()
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    pred_batch = model(state_batch).cpu()  
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
    
    loss = loss_fn(prob_batch, disc_rewards)
    
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
    truncated = False
    
    total_rew = 0
    while not (done or truncated):
        if state_type == 'image':
            s = env.render()
            s = transform(s).cuda()
        else:
            s = torch.tensor(state, device='cuda')
        
        with torch.no_grad():
            a = torch.argmax(model(s.unsqueeze(0)), dim=-1).item()
        
        state, rew, done, truncated, _ = env.step(a)
        total_rew += rew
    
    return total_rew
    
def main():
    N_EPOCHS = 2000
    EVAL_INTERVAL = 50
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
    
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-1)
    
    losses = []
    rewards = []
    
    for e in range(N_EPOCHS):
        loss, total_rew = train_step(model, env, optim, loss_fn, transform, state_type=state_type)
        losses.append(loss.detach().numpy())
        
        if (e+1) % EVAL_INTERVAL == 0:
            print(f"[{e+1}/{N_EPOCHS}")
            print(f"Train loss: {loss} Train Rew: {total_rew}")
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