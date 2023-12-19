import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import cv2

train_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = gym.make('CartPole-v1', render_mode='rgb_array')

SEED = 1234

np.random.seed(SEED)
torch.manual_seed(SEED)

writer = SummaryWriter()
tb_step = 0

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()

        self.actor = actor
        self.critic = critic

    def forward(self, state):

        action_pred = self.actor(state)
        value_pred = self.critic(state)

        return action_pred, value_pred

INPUT_DIM = train_env.observation_space.shape[0]
HIDDEN_DIM = 128
OUTPUT_DIM = train_env.action_space.n

actor = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
critic = MLP(INPUT_DIM, HIDDEN_DIM, 1)

policy = ActorCritic(actor, critic)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

policy.apply(init_weights)

LEARNING_RATE = 0.01

optimizer = optim.Adam(policy.parameters(), lr = LEARNING_RATE)

def train(env, policy, optimizer, discount_factor, ppo_steps, ppo_clip):

    policy.train()

    states = []
    actions = []
    log_prob_actions = []
    values = []
    rewards = []
    done = False
    episode_reward = 0

    state, _ = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        #append state here, not after we get the next state from env.step()
        states.append(state)

        action_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred, dim = -1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)

        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)

        episode_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values)

    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward

def calculate_returns(rewards, discount_factor, normalize = True):

    returns = []
    R = 0

    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)

    returns = torch.tensor(returns)

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns

def calculate_advantages(returns, values, normalize = True):

    advantages = returns - values

    if normalize:

        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages

def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):
    
    global tb_step
    
    total_policy_loss = 0
    total_value_loss = 0

    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()

    for _ in range(ppo_steps):

        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)

        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)

        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()

        writer.add_scalar('policy_loss/policy_ratio', policy_ratio.mean(), tb_step)

        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        writer.add_scalar('policy_loss/policy_loss_1', policy_loss_1.mean(), tb_step)
        writer.add_scalar('policy_loss/policy_loss_2', policy_loss_2.mean(), tb_step)

        policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

        value_loss = F.smooth_l1_loss(returns, value_pred)

        optimizer.zero_grad()

        policy_loss.backward()
        value_loss.backward()

        writer.add_scalar('actor/fc_1_grad', policy.actor.fc_1.weight.grad.mean().item(), tb_step)
        writer.add_scalar('actor/fc_2_grad', policy.actor.fc_2.weight.grad.mean().item(), tb_step)
        
        writer.add_scalar('critic/fc_1_grad', policy.critic.fc_1.weight.grad.mean().item(), tb_step)
        writer.add_scalar('critic/fc_2_grad', policy.critic.fc_2.weight.grad.mean().item(), tb_step)
        
        tb_step += 1
        
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps

def evaluate(env, policy):

    policy.eval()

    rewards = []
    done = False
    episode_reward = 0

    state, _ = env.reset()

    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():

            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)

        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        episode_reward += reward

    return episode_reward


def visualize(env, policy):
     
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE) 
    
    policy.eval()

    done = False

    state, _ = env.reset()
    
    img = env.render()
    cv2.imshow("Display", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    while not done:

        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():

            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)

        state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated # or truncated
        
        img = env.render()
        cv2.waitKey(10)
        cv2.imshow("Display", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
    cv2.destroyAllWindows() 


def collect_data(env, policy, n_states=1024):
    buffer = []
    
    policy.eval()

    while len(buffer) < n_states:
        done = False

        state, _ = env.reset()
        
        img = env.render()
            
        while not done:

            state = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():

                action_pred, _ = policy(state)

                action_prob = F.softmax(action_pred, dim = -1)
                
            buffer.append((img, action_prob))

            action = torch.argmax(action_prob, dim = -1)

            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated # or truncated
            
            img = env.render()
    
    return buffer


def get_out_shape(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)


class ClassificationModel(nn.Module):
    def __init__(self, in_shape, out_dim):
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
        
        self.head = nn.Sequential(
            nn.Linear(out_size, out_dim),
        )
        
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
    

class RLDataset(Dataset):
    def __init__(self, buffer, img_size=None):
        super().__init__()
        
        self.imgs = list(map(lambda x: x[0], buffer))
        self.action_probs = list(map(lambda x: x[1], buffer))
        
        self.transform = None
        if img_size:
            self.transform = Compose([ToTensor(), Resize(img_size)])
            
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        act = self.action_probs[idx]
        
        if self.transform:
            img = self.transform[img]
            
        return img, act
        

def train_classification_model(model, dataloader, optim, criterion):
    
    total_loss = 0
    for img, true_act in dataloader:
        out = model(img)
        
        loss = criterion(out, true_act)
        total_loss += loss.item()
        
        loss.backward()
        optim.step()
    
    return total_loss/len(dataloader)
        

MAX_EPISODES = 500
DISCOUNT_FACTOR = 0.99
N_TRIALS = 25
REWARD_THRESHOLD = 480
PRINT_EVERY = 10
PPO_STEPS = 5
PPO_CLIP = 0.2

train_rewards = []
test_rewards = []

for episode in range(1, MAX_EPISODES+1):

    policy_loss, value_loss, train_reward = train(train_env, policy, optimizer, DISCOUNT_FACTOR, PPO_STEPS, PPO_CLIP)

    writer.add_scalar('loss/policy', policy_loss, tb_step)
    writer.add_scalar('loss/value', value_loss, tb_step)
    
    test_reward = evaluate(test_env, policy)

    train_rewards.append(train_reward)
    test_rewards.append(test_reward)

    writer.add_scalar('reward/train_reward', train_reward, tb_step)
    writer.add_scalar('reward/test_reward', test_reward, tb_step)
    
    mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
    mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])

    if episode % PRINT_EVERY == 0:

        print(f'| Episode: {episode:3} | Mean Train Rewards: {mean_train_rewards:5.1f} | Mean Test Rewards: {mean_test_rewards:5.1f} |')

    if mean_test_rewards >= REWARD_THRESHOLD:

        print(f'Reached reward threshold in {episode} episodes')

        break    


img_size = 128
train_env.reset()
img = train_env.render()

h, w = img.shape[:2]
if h < w:
    img_size = [img_size, int(w * img_size/h)]
else:
    img_size = [int(h * img_size/w), img_size]
    
dataset = RLDataset(collect_data(train_env, policy), img_size=img_size)
dataloader = DataLoader(dataset, batch_size=128)

visualize(test_env, policy)

plt.figure(figsize=(12,8))
plt.plot(test_rewards, label='Test Reward')
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Reward', fontsize=20)
plt.legend(loc='lower right')
plt.grid()
plt.show()