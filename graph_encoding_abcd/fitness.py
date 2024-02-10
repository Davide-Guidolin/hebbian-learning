import os, psutil
import torch.nn as nn
from torch.optim import SGD
import torch
import numpy as np
import gymnasium as gym
from gymnasium import wrappers as w
from hebbian import update_weights, softhebb_update
from torch.profiler import profile, record_function, ProfilerActivity

ACTIVATIONS_LIST = [nn.ReLU, nn.Tanh, nn.ELU, nn.LeakyReLU, nn.GELU, nn.Sigmoid] # add others if used

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # This undoes the memory optimization, use with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0
    

class CropFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return np.array(observation)[:84][:, :84]


def evaluate_classification(model, data_loader, abcd_params=None, pop_index=-1, shared_dict=None, abcd_learning_rate=0.1, bp_last_layer=False, bp_lr=0.00001, bp_loss=nn.CrossEntropyLoss, softhebb_train=False, softhebb_lr=0.001):
    print(f"[{os.getpid()}] Starting evaluation of population {pop_index}")
    
    t = model[0].weight.dtype
    device = model[0].weight.device

    if bp_last_layer:
        model[-1].weight.requires_grad = True
        optim = SGD(model[-1].parameters(), lr=bp_lr)
        criterion = bp_loss()
    
    correct = 0
    total = 0
    for i, (x, true_y) in enumerate(data_loader):
                
        if i%75 == 0:
            print(f"[{os.getpid()}] Batch {i}/{len(data_loader)} Partial accuracy {correct/max(1,total):.5f}  ({correct}/{total})")
        x = x.view(x.shape[0], -1).to(device)
        
        activation = False
        for l, layer in enumerate(model):
            if activation:
                activation = False
                continue
            
            if bp_last_layer and l == len(model)-1:
                optim.zero_grad()
                y = layer(x)
                out = nn.functional.softmax(y, dim=-1)
                true = torch.zeros(out.shape, dtype=out.dtype)
                true[torch.arange(true_y.shape[0]), true_y] = 1.0
                
                loss = criterion(out, true)
                loss.backward()
                optim.step()
                
                x = y
            else:
                if type(layer) == nn.Linear:
                    
                    y = layer(x)
                    
                    pre_act = y
                    if l < len(model)-1 and type(model[l+1]) in ACTIVATIONS_LIST:
                        activation = True
                        y = model[l+1](pre_act)
                    
                    if y.isnan().any():
                        print(f"[{os.getpid()}] Layer {l} produced NAN output!!! {layer}")
                        exit(1)
                    
                    shared_w = False
                    if hasattr(layer, 'shared_weights'):
                        shared_w = True
                    
                    if softhebb_train:
                        softhebb_update(layer, x, pre_act, shared_w=shared_w, lr=softhebb_lr)
                    else:
                        update_weights(layer, x, y, abcd_params, shared_w=shared_w, lr=abcd_learning_rate)
                    
                    x = y
                else:
                    x = layer(x)

        correct += torch.sum(torch.argmax(x, dim=-1) == true_y)
        total += true_y.shape[0]

    print(f"[{os.getpid()}] {correct}/{total}")
    acc = correct/total
    print(f"[{os.getpid()}]Accuracy: {acc}")
    
    if shared_dict is not None:
        shared_dict[pop_index] = acc
    
    return acc


def evaluate_car_racing(model, env_type, abcd_params, pop_index=-1, shared_dict=None, abcd_learning_rate=0.1, bp_last_layer=False, bp_lr=0.00001, in_size=64, device='cuda'):
    print(f"[{os.getpid()}] Starting evaluation of population {pop_index}")
    
    if device != 'cpu':
        for layer in model:
            layer.to(device)
            if hasattr(layer, 'mask_tensor'):
                layer.mask_tensor = layer.mask_tensor.to(device)
                
    env = gym.make(env_type)
    env = w.ResizeObservation(env, in_size)        # Resize and normalize input
    env = CropFrame(env)
    env = ScaledFloatFrame(env)
    
    state, _ = env.reset()
    state = np.swapaxes(state,0,2)
    
    total_rew = 0
    neg_count = 0
    with torch.no_grad():
        while True:
            x = torch.from_numpy(state.reshape(-1)).unsqueeze(0).to(device)
            
            activation = False
            for l, layer in enumerate(model):
                if activation:
                    activation = False
                    continue
                
                if type(layer) == nn.Linear:
                    y = layer(x)
                    
                    if l < len(model)-1 and type(model[l+1]) in ACTIVATIONS_LIST:
                        activation = True
                        y = model[l+1](y)
                    
                    if y.isnan().any():
                        print(f"[{os.getpid()}] Layer {l} produced NAN output!!! {layer}")
                        exit(1)
                    
                    shared_w = False
                    if hasattr(layer, 'shared_weights'):
                        shared_w = True
                    
                    update_weights(layer, x, y, abcd_params, shared_w=shared_w, lr=abcd_learning_rate)
                    
                    x = y
                else:
                    x = layer(x)
            
            action = np.array([torch.tanh(x[:, 0]).squeeze().cpu(), torch.sigmoid(x[:, 1]).squeeze().cpu(), torch.sigmoid(x[:, 2]).squeeze().cpu()])
            next_state, reward, done, truncated, info = env.step(action)
            total_rew += reward
            
            neg_count = neg_count+1 if reward < 0.0 else 0
            if (done or truncated or neg_count > 20):
                break
            
            state = np.swapaxes(next_state, 0, 2)
        
    if shared_dict is not None:
        shared_dict[pop_index] = total_rew
    
    print(f"[{os.getpid()}] Reward {total_rew}")
    env.close()
    
    del model
    del env
    
    return total_rew