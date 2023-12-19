# https://github.com/lmarza/CartPole-CNN

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from torch.utils.tensorboard import SummaryWriter
import gc

writer = SummaryWriter('runs/DQN_car_racing')
tb_step = 0

env = gym.make('CarRacing-v2', continuous=False)

is_python = 'inline' in matplotlib.get_backend()
if is_python:
    from IPython import display
    
plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_out_shape(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)


class DQN(nn.Module):

    def __init__(self, in_shape, n_actions):
        super().__init__()
        
        c, w, h = in_shape
        
        out_shape = [w, h]
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        out_shape = list(map(lambda x: get_out_shape(x, 8, pooling_size=1, stride=4), out_shape))
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        out_shape = list(map(lambda x: get_out_shape(x, 4, pooling_size=1, stride=2), out_shape))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        out_shape = list(map(lambda x: get_out_shape(x, 3, pooling_size=1, stride=1), out_shape))

        
        self.flatten = nn.Flatten()
        out_size = out_shape[0] * out_shape[1] * 64
        
        self.head = nn.Linear(out_size, n_actions)
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
    
        x = self.flatten(x)
        x = self.head(x)
        
        return x
    

MEMORY_SIZE = 2000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.05
LR = 0.0003
N_STATES = 2
img_size = 84

n_actions = env.action_space.n
state, info = env.reset()

h, w = state.shape[:2]
if h < w:
    img_size = [img_size, int(w * img_size/h)]
else:
    img_size = [int(h * img_size/w), img_size]

transform = Compose([ToTensor(), Grayscale(), Resize(img_size)])

policy_net = DQN((N_STATES, img_size[0], img_size[1]), n_actions).to(device)
target_net = DQN((N_STATES, img_size[0], img_size[1]), n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to(device)).max(1)[1].view(1, 1).cpu()
    else:
        return torch.tensor([[env.action_space.sample()]], device='cpu')
        

episode_durations = []
losses = []

def plot_durations(show_results=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_results:
        plt.title('Results')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.plot(durations_t.numpy())
    
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        
    plt.pause(0.001)
    if is_python:
        if not show_results:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            

def plot_loss(show_results=False):
    plt.figure(2)
    loss = torch.tensor(losses, dtype=torch.float)
    if show_results:
        plt.title('Losses')
    else:
        plt.clf()
        plt.title('Training losses')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(loss.numpy())
        
    plt.pause(0.001)
    if is_python:
        if not show_results:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    global tb_step
    
    if len(memory) < BATCH_SIZE:
        return 0
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # mask of non final states
    non_final_mask = torch.tensor(tuple(map(lambda s: None not in s, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.cat(list(s), dim=1) for s in batch.next_state if None not in s])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch.to(device)).cpu().gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device='cpu')
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).cpu().max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    writer.add_scalar('loss', loss, tb_step)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
    
    writer.add_scalar('actor/cnn1_grad', policy_net.conv1.weight.grad.mean().item(), tb_step)
    writer.add_scalar('actor/cnn2_grad', policy_net.conv2.weight.grad.mean().item(), tb_step)
    writer.add_scalar('actor/cnn3_grad', policy_net.conv3.weight.grad.mean().item(), tb_step)
    writer.add_scalar('actor/head_grad', policy_net.head.weight.grad.mean().item(), tb_step)
    
    tb_step += 1
    
    optimizer.step()
    torch.cuda.empty_cache()
    # gc.collect()
    
    return loss.cpu()

num_episodes = 5000

states_2 = deque([], maxlen=N_STATES)
next_states_2 = deque([], maxlen=N_STATES)
states_2_t = torch.empty((1, N_STATES, img_size[0], img_size[1]))
next_states_2_t = torch.empty((1, N_STATES, img_size[0], img_size[1]))

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = transform(state).unsqueeze(0)
    states_2.append(state)
    total_rew = 0
        
    for t in count():
        if t <= N_STATES-1:
            action = torch.tensor([[env.action_space.sample()]], device='cpu', dtype=torch.long)
        else:
            action = select_action(states_2_t)
            
        state, reward, terminated, truncated, _ = env.step(action.item())
        state = transform(state).unsqueeze(0)
        done = terminated or truncated
            
        reward = torch.tensor([reward], device='cpu')
        total_rew += reward.item()
        
        if terminated:
            next_state = None
        else:
            next_state = state
        
        if t==0:
            next_states_2.append(next_state)
        elif t < N_STATES-1:
            states_2.append(state)
            next_states_2.append(next_state)
        elif t >= N_STATES:
            states_2.append(state)
            next_states_2.append(next_state)
            
            states_2_t = torch.cat(list(states_2), dim=1)
            memory.push(states_2_t, action, list(next_states_2), reward)
        
        state = next_state
        
        loss = optimize_model()
        # losses.append(loss)        
        
        torch.cuda.empty_cache()
        
        if done:
            episode_durations.append(t+1)
            print(f"[{i_episode}/{num_episodes}] {total_rew}")
            # plot_durations()
            # plot_loss()
            target_net.cpu()
            policy_net.cpu()
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
            
            target_net.load_state_dict(target_net_state_dict)
            
            torch.cuda.empty_cache()
            target_net.cuda()
            policy_net.cuda()
            
            writer.add_scalar('reward/train_reward', total_rew, i_episode)  
            break
    
        
    torch.cuda.empty_cache()

print('Complete')
plot_durations(show_results=True)
plt.ioff()
plt.show()