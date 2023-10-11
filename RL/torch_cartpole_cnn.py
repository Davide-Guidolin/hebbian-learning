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

env = gym.make('CartPole-v1', render_mode='rgb_array')

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
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=64, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm2d(64)
        out_shape = list(map(lambda x: get_out_shape(x, 7, pooling_size=1, stride=3), out_shape))
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=3)
        self.bn2 = nn.BatchNorm2d(64)
        out_shape = list(map(lambda x: get_out_shape(x, 7, pooling_size=1, stride=3), out_shape))

        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(16)
        # out_shape = list(map(lambda x: get_out_shape(x, 3, pooling_size=1, stride=1), out_shape))

        
        self.flatten = nn.Flatten()
        out_size = out_shape[0] * out_shape[1] * 64
        
        self.head = nn.Sequential(
            nn.Linear(out_size, 128),
            nn.Linear(128, n_actions)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)
    
        x = self.flatten(x)
        x = self.head(x)
        
        return x
    

MEMORY_SIZE = 5000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.01
EPS_DECAY = 1000
TAU = 0.05
LR = 0.00001
N_STATES = 2
img_size = 160

n_actions = env.action_space.n
state, info = env.reset()
img = env.render()

h, w = img.shape[:2]
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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

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
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()
    torch.cuda.empty_cache()
    
    return loss.cpu()

num_episodes = 600

states_2 = deque([], maxlen=N_STATES)
next_states_2 = deque([], maxlen=N_STATES)
states_2_t = torch.empty((1, N_STATES, img_size[0], img_size[1]))
next_states_2_t = torch.empty((1, N_STATES, img_size[0], img_size[1]))

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = transform(env.render()).to(device).unsqueeze(0)
    states_2.append(state)
        
    for t in count():
        if t <= N_STATES-1:
            action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            action = select_action(states_2_t)
            
        _, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        
        if terminated:
            next_state = None
        else:
            next_state = transform(env.render()).to(device).unsqueeze(0)
        
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
        losses.append(loss)        
        
        if done:
            episode_durations.append(t+1)
            plot_durations()
            plot_loss()
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
            
            break
        
        
    torch.cuda.empty_cache()

print('Complete')
plot_durations(show_results=True)
plt.ioff()
plt.show()