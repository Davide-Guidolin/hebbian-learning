# https://nandakishorej8.medium.com/part-2-policy-based-reinforcement-learning-openais-cartpole-with-reinforce-algorithm-18de8cb5efa4

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
import torchvision.transforms.functional as F
from typing import Tuple
import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

def get_out_shape(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)

class ConvNet(nn.Module):
    def __init__(self, in_shape, n_actions):
        super().__init__()
        
        c, w, h = in_shape
        
        out_shape = [w, h]
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        out_shape = list(map(lambda x: get_out_shape(x, 5, pooling_size=1, stride=2), out_shape))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        out_shape = list(map(lambda x: get_out_shape(x, 5, pooling_size=1, stride=2), out_shape))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        out_shape = list(map(lambda x: get_out_shape(x, 5, pooling_size=1, stride=2), out_shape))

        
        self.flatten = nn.Flatten()
        out_size = out_shape[0] * out_shape[1] * 32
        
        self.head = nn.Linear(out_size, n_actions)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
    
        x = self.flatten(x)
        x = self.head(x)
        
        out = self.softmax(x)
        
        return out


def discount_rewards(rewards, gamma=0.95):
    disc_return = torch.pow(gamma, torch.arange(len(rewards)).float()) * rewards
    disc_return = disc_return / (disc_return.max() - disc_return.min())                     
    return disc_return


def loss_fn(preds, r):
    loss = -1 * torch.sum(r * torch.log(preds))
    
    if loss.isnan():
        print("Loss is NaN")
        print(r, preds)
        exit(0)
        
    return loss


def get_transitions(model, env, eps, transform, memory_size=1024, device='cuda'):
    model.eval()
    
    transitions = []
    while len(transitions) < memory_size:
        
        states = []
        actions = []
        rewards = []
        
        state, _  = env.reset()
        done = False
        truncated = False
        
        total_rew = 0
        
        while not (done or truncated):
            s = env.render()
            s = transform(s).to(device)
            state = s.cpu().numpy()
            
            with torch.no_grad():
                act_prob = model(s.unsqueeze(0)).squeeze()
                if random.random() < eps:
                    probs = act_prob.cpu().numpy()
                else:
                    probs = [0.5, 0.5]

                a = np.random.choice(np.array([0, 1]), p=probs)
                
            total_rew += 1
            states.append(state)
            actions.append(a)
            rewards.append(total_rew)
            
            state, rew, done, truncated, _ = env.step(a)
        
        disc_rewards = discount_rewards(torch.tensor(rewards).flip(dims=(0,)))
        for s, a, r in zip(states, actions, disc_rewards):
            transitions.append([s, a, r] )
    
    return transitions[:memory_size]


def train_step(model, memory, optimizer, loss_fn, batch_size=128, device='cuda'):
    
    model.train()
    
    samples = np.random.choice(len(memory), batch_size)
    reward_batch = torch.Tensor([r for (s,a,r) in memory])[samples]
    state_batch = torch.Tensor([s for (s,a,r) in memory])[samples].to(device)
    action_batch = torch.Tensor([a for (s,a,r) in memory])[samples]
    
    pred_batch = model(state_batch).cpu()
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1,1)).squeeze()
    
    loss = loss_fn(prob_batch, reward_batch)
    
    optimizer.zero_grad()
    loss.backward()
    
    for name, p in model.named_parameters():
        if p.grad.sum().isnan():
            print(f"NaN grad in {name}")
            exit(1)
            
    optimizer.step()
    
    torch.cuda.empty_cache()
    
    return loss


def eval_step(model, env, transform, device='cuda'):
    model.eval()
    
    state, _  = env.reset()
    done = False
    truncated = False
    
    total_rew = 0
    while not (done or truncated):
        s = env.render()
        s = transform(s).to(device)
        
        with torch.no_grad():
            a = torch.argmax(model(s.unsqueeze(0)), dim=-1).item()
        
        state, rew, done, truncated, _ = env.step(a)
        total_rew += rew
    
    return total_rew


def main():
    N_EPOCHS = 2000
    EVAL_INTERVAL = 10
    BATCH_SIZE = 64
    MEMORY_SIZE = 64
    eps = 0.9
    img_size = 160
    
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    n_actions = env.action_space.n
    env.reset()
    img = env.render()
    h, w = img.shape[:2]
    if h < w:
        img_size = [img_size, int(w * img_size/h)]
    else:
        img_size = [int(h * img_size/w), img_size]
    
    transform = Compose([ToTensor(), Grayscale(), Resize(img_size)])
    # cv2.imwrite("./res.jpg", cv2.cvtColor(transform(img).cpu().numpy().transpose(1, 2, 0) * 255., cv2.COLOR_RGB2BGR))
    # # print(img.shape)
    # exit(0)
    
    model = ConvNet((1, img_size[0], img_size[1]), n_actions)
    
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-2)
    
    losses = []
    rewards = []
    for e in range(N_EPOCHS):
        print(f"[{e+1}/{N_EPOCHS}")
        eps = 0.6 * min(1, e/(N_EPOCHS*3/4)) + 0.4
        
        memory_buffer = get_transitions(model, env, eps, transform, MEMORY_SIZE)
        loss = train_step(model, memory_buffer, optim, loss_fn, BATCH_SIZE)
        
        losses.append(loss.detach().numpy())
        
        if (e+1) % EVAL_INTERVAL == 0:
            print(f"Train loss: {loss}")
            total_rew = eval_step(model, env, transform)
            rewards.append(total_rew)
            print(f"Eval reward = {total_rew}")
    
    plt.plot(losses)
    plt.savefig("./loss.jpg")
    plt.clf()
    plt.plot(range(1, N_EPOCHS, EVAL_INTERVAL), rewards)
    plt.savefig("./eval_reward.jpg")
    

if __name__ == "__main__":
    main()