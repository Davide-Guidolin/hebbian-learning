import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import cv2


SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

train_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = gym.make('CartPole-v1', render_mode='rgb_array')

DEVICE = 'cuda'
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
MAX_EPISODES = 20000
DISCOUNT_FACTOR = 0.95
N_TRIALS = 50
REWARD_THRESHOLD = 475
PRINT_EVERY = 10
PPO_STEPS = 5
PPO_CLIP = 0.2
VAL_W = 1.0
ENTROPY_W = 0.01
HIDDEN_DIM = 64
OUTPUT_DIM = train_env.action_space.n
img_size = 160
accum_step = 0
tb_step = 0
writer = SummaryWriter(f'runs/ppo_5cnn_gelu_{BATCH_SIZE}bs_ppo{PPO_STEPS}_{PPO_CLIP}_{VAL_W}val_{ENTROPY_W}entropy_{LEARNING_RATE}lr_no_norm_rew_{DISCOUNT_FACTOR}disc_1linear_5negative_final_rew_batch_norm')

def get_out_shape(dim, k_size, stride = 1, padding = 0, pooling_size = 1):
    return int(((dim - k_size + 2*padding)/stride + 1)/pooling_size)


class CNN(nn.Module):

    def __init__(self, in_shape, hidden_dim, out_dim):
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
        
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        # self.bn4 = nn.BatchNorm2d(128)
        # out_shape = list(map(lambda x: get_out_shape(x, 3, pooling_size=1, stride=1), out_shape))
        
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        # self.bn5 = nn.BatchNorm2d(128)
        # out_shape = list(map(lambda x: get_out_shape(x, 3, pooling_size=1, stride=1), out_shape))

        
        self.flatten = nn.Flatten()
        out_size = out_shape[0] * out_shape[1] * 64
        
        self.head = nn.Sequential(
            nn.Linear(out_size, out_dim),
            # nn.Dropout(),
            # nn.GELU(),
            # nn.Linear(hidden_dim, out_dim)
        )
        
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.l_relu = nn.LeakyReLU()
        
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
        
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.gelu(x)
        
        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.gelu(x)
        
        x = self.flatten(x)
        x = self.head(x)
        
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


# Create networks

state, info = train_env.reset()
img = train_env.render()

h, w = img.shape[:2]
if h < w:
    img_size = [img_size, int(w * img_size/h)]
else:
    img_size = [int(h * img_size/w), img_size]

transform = Compose([ToTensor(), Resize(img_size)])

actor = CNN((3, *img_size), HIDDEN_DIM, OUTPUT_DIM)
critic = CNN((3, *img_size), HIDDEN_DIM, 1)

policy = ActorCritic(actor, critic).to(DEVICE)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

policy.apply(init_weights)

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

    env.reset()
    state = env.render()

    while not done:
        
        state = transform(state).to(DEVICE).unsqueeze(0)

        #append state here, not after we get the next state from env.step()
        states.append(state)

        action_pred, value_pred = policy(state)

        action_prob = F.softmax(action_pred.cpu(), dim = -1)

        dist = distributions.Categorical(action_prob)

        action = dist.sample()

        log_prob_action = dist.log_prob(action)

        _, reward, terminated, truncated, _ = env.step(action.item())
        state = env.render()
        done = terminated or truncated
        if done and episode_reward < 450:
            reward = -5

        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred.cpu())
        rewards.append(reward)

        episode_reward += reward

    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions)
    values = torch.cat(values).squeeze(-1)

    returns = calculate_returns(rewards, discount_factor)
    advantages = calculate_advantages(returns, values.detach())

    policy_loss, value_loss = update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip)

    return policy_loss, value_loss, episode_reward


def calculate_returns(rewards, discount_factor, normalize = False):

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

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    return advantages


def update_policy(policy, states, actions, log_prob_actions, advantages, returns, optimizer, ppo_steps, ppo_clip):

    global accum_step
    global tb_step
    
    total_policy_loss = 0
    total_value_loss = 0

    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()

    for _ in range(ppo_steps):

        #get new log prob of actions for all input states
        action_pred, value_pred = policy(states)
        value_pred = value_pred.cpu().squeeze(-1)
        action_prob = F.softmax(action_pred.cpu(), dim = -1)
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

        value_loss = F.mse_loss(value_pred, returns)
        
        entropy = dist.entropy().mean()
        
        loss = policy_loss + VAL_W * value_loss  + ENTROPY_W * entropy
        
        loss = loss/BATCH_SIZE
        loss.backward()
        
        accum_step += 1
        
        writer.add_scalar('policy_loss/entropy', entropy, tb_step)
        
        writer.add_scalar('actor/cnn1_grad', policy.actor.conv1.weight.grad.mean().item(), tb_step)
        writer.add_scalar('actor/cnn2_grad', policy.actor.conv2.weight.grad.mean().item(), tb_step)
        writer.add_scalar('actor/cnn3_grad', policy.actor.conv3.weight.grad.mean().item(), tb_step)
        writer.add_scalar('actor/head_grad', policy.actor.head[0].weight.grad.mean().item(), tb_step)
        
        writer.add_scalar('critic/cnn1_grad', policy.critic.conv1.weight.grad.mean().item(), tb_step)
        writer.add_scalar('critic/cnn2_grad', policy.critic.conv2.weight.grad.mean().item(), tb_step)
        writer.add_scalar('critic/cnn3_grad', policy.critic.conv3.weight.grad.mean().item(), tb_step)
        writer.add_scalar('critic/head_grad', policy.critic.head[0].weight.grad.mean().item(), tb_step)
        
        tb_step += 1
        
        if accum_step % BATCH_SIZE == 0:
            nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def evaluate(env, policy):

    policy.eval()

    done = False
    episode_reward = 0

    env.reset()
    state = env.render()

    while not done:

        state = transform(state).to(DEVICE).unsqueeze(0)

        with torch.no_grad():

            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)

        _, reward, terminated, truncated, _ = env.step(action.item())
        state = env.render()
        done = terminated or truncated

        episode_reward += reward

    return episode_reward


def visualize(env, policy):
     
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE) 
    
    policy.eval()

    done = False

    env.reset()
    
    state = env.render()
    cv2.imshow("Display", cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        
    while not done:

        state = transform(state).to(DEVICE).unsqueeze(0)

        with torch.no_grad():

            action_pred, _ = policy(state)

            action_prob = F.softmax(action_pred, dim = -1)

        action = torch.argmax(action_prob, dim = -1)

        _, reward, terminated, truncated, _ = env.step(action.item())
        state = env.render()
        done = terminated or truncated
        
        cv2.waitKey(10)
        cv2.imshow("Display", cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        
    cv2.destroyAllWindows() 


train_rewards = []
test_rewards = []

for episode in range(1, MAX_EPISODES+1):
    tb_episode = episode

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
    
writer.close()

plt.figure(1)
plt.plot(test_rewards, label='Test Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Test Reward', fontsize=20)
plt.grid()

plt.figure(2)
plt.plot(train_rewards, label='Train Reward')
plt.xlabel('Episode', fontsize=20)
plt.ylabel('Train Reward', fontsize=20)
plt.grid()

plt.show()

visualize(test_env, policy)