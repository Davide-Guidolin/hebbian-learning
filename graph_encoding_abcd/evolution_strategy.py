import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy
import os, psutil

from unrolled_model import UnrolledModel
from fitness import evaluate_classification, evaluate_car_racing
from data import DataManager

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="CarRacing_conv_unrolling_abcd",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.2,
    "population_size": 10,
    "architecture": "CNN_CarRacing",
    "dataset": "CarRacing-v2",
    "epochs": 30,
    }
)

def compute_ranks(x):
  """
  Returns rank as a vector of len(x) with integers from 0 to len(x)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  Maps x to [-0.5, 0.5] and returns the rank
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

class EvolutionStrategy:
    def __init__(self, rolled_model, dataset_type="CIFAR10", population_size=100, num_threads=1, sigma=0.1, abcd_perturbation_std=1, abcd_learning_rate=0.2, abcd_lr_decay=0.995, bp_last_layer=True, bp_learning_rate=0.01):
        self.model = rolled_model
        
        self.population_size = population_size
        self.sigma = sigma
        self.decay = abcd_lr_decay
        self.abcd_lr = abcd_learning_rate
        self.perturbation_factor = abcd_perturbation_std
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        
        self.dataset_type = dataset_type
        if self.dataset_type != "CarRacing-v2":
            self.data = DataManager(self.dataset_type)
            self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        else:
            self.input_size = 64
        
        self.unrolled_model = UnrolledModel(self.model, self.input_size)
        self.params = self.init_ABCD_parameters(self.unrolled_model.get_new_model())
        
        self.bp_last_layer = bp_last_layer
        self.bp_lr = bp_learning_rate
        
        
    def normalize_params(self, params):
        for layer in params:
            for p in params[layer]:
                if layer == 0 and p == 'B':
                    continue
                if layer == list(params.keys())[-1] and p != 'B':
                    continue
                
                params[layer][p] = params[layer][p] / params[layer][p].abs().max()
                
        return params
        
        
    def init_ABCD_parameters(self, net: nn.Sequential, start_idx: int = 0, end_idx: int = -1, device: str = 'cpu') -> dict:
        if end_idx == -1 or end_idx > len(net):
            end_idx = len(net)
        
        params = {}
        l_index = start_idx
        i = start_idx
        while i < end_idx:
            if type(net[i]) == nn.Linear:
                params[l_index] = {'A':None, 'B': None, 'C':None, 'D': None}
                l_index += 1
            i += 1
        params[l_index] = {'A':None, 'B': None, 'C':None, 'D': None}
        
        l_index = start_idx
        i = start_idx
        while i < end_idx:
            if type(net[i]) == nn.Linear:
                layer = net[i]

                params[l_index]['A'] = torch.randn(layer.weight.shape[1], device=device)
                params[l_index + 1]['B'] = torch.randn(layer.weight.shape[0], device=device)
                params[l_index]['C'] = torch.randn((layer.weight.shape[0], layer.weight.shape[1]), device=device)
                params[l_index]['D'] = torch.randn((layer.weight.shape[0], layer.weight.shape[1]), device=device)
                l_index += 1
            i += 1
        
        # params = self.normalize_params(params)
        
        return params
    
    
    def perturbate(self, params):
        new_p = deepcopy(params)
        for layer in new_p:
            for p in new_p[layer]:
                if layer == 0 and p == 'B':
                    continue
                if layer == list(new_p.keys())[-1] and p != 'B':
                    continue
                
                noise = torch.randn_like(new_p[layer][p])
                new_p[layer][p].add_(self.perturbation_factor * noise)

        # new_p = self.normalize_params(new_p)
        
        return new_p
    
    
    def get_scores(self, parallel=None):
        print("IN get_scores")
        population = []

        if parallel:
            print("Parallelizing")
            processes = []
            
            manager = mp.Manager()
            shared_dict = manager.dict()
            
            pop_evaluated = 0
            thread_used = 0
            processes_joined = 0
            while pop_evaluated < self.population_size:
                if thread_used < self.num_threads:
                    population.append(self.perturbate(self.params))
                    model = self.unrolled_model.get_new_model()
                    if self.dataset_type != "CarRacing-v2":
                        loader = self.data.get_new_loader(train=False)
                        args = (model, loader, population[pop_evaluated], pop_evaluated, shared_dict, self.abcd_lr, self.bp_last_layer, self.bp_lr)
                        target_fn = evaluate_classification
                    else:
                        args = (model, self.dataset_type, population[pop_evaluated], pop_evaluated, shared_dict, self.abcd_lr, self.bp_last_layer, self.bp_lr, self.input_size)
                        target_fn = evaluate_car_racing
                    proc = mp.Process(target=target_fn, args=args)
                    proc.start()
                    
                    processes.append(proc)
                    thread_used += 1
                    pop_evaluated += 1
                    print(f"Processes spawned: {len(processes)} - Processes running {thread_used}")
                else:
                    processes[processes_joined].join()
                    processes_joined += 1
                    thread_used -= 1
            
            for proc in processes:
                proc.join()
                proc.close()
            
            scores = list(dict(sorted(shared_dict.items(), key=lambda x: x[0])).values())
            
        else:
            print("No Parallel")
            scores = []
            for pop_idx in range(self.population_size):
                print(f"MEM BEFORE population {pop_idx} creation: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
                p = self.perturbate(self.params)
                print(f"MEM AFTER population {pop_idx} creation: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
                population.append(p)
                if self.dataset_type != "CarRacing-v2":
                    scores.append(evaluate(self.unrolled_model.get_new_model(), self.data.test_loader, p, pop_idx, abcd_learning_rate=self.abcd_lr, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr))
                else:
                    scores.append(evaluate_car_racing(self.unrolled_model.get_new_model(), self.dataset_type, p, pop_idx, abcd_learning_rate=self.abcd_lr, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, in_size=self.input_size))
        
        scores = np.array(scores)

        return population, scores    
    
    
    def update_params(self, population, scores):
        ranks = compute_centered_ranks(scores)
        
        ranks = (ranks - ranks.mean()) / ranks.std()
        
        self.update_factor = self.abcd_lr / (self.population_size * self.sigma)
        for layer in population[0].keys():
            for key in population[0][layer].keys():
                if layer == 0 and key == 'B':
                    continue
                if layer == list(population[0].keys())[-1] and key != 'B':
                    continue
                
                param_pop = np.array([p[layer][key] for p in population])
                
                self.params[layer][key].add_(torch.from_numpy(self.update_factor * np.dot(param_pop.T, ranks).T))
        
        # self.params = self.normalize_params(self.params)
        
        if self.abcd_lr > 0.001:
            self.abcd_lr *= self.decay

        if self.sigma > 0.01:
            self.sigma *= 0.999
        
        del population
        print("Parameters update done")
    
    
    def run(self, iterations):
        
        # export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1
        parallel = self.num_threads > 1
        
        if parallel:
            if (not 'MKL_NUM_THREADS' in os.environ) or (not 'OMP_NUM_THREADS' in os.environ) or (os.environ['MKL_NUM_THREADS'] != "1") or (os.environ['OMP_NUM_THREADS'] != "1"):
                print(f"For parallel execution run this command: export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1")
                exit(0)
        
        best_accuracies = []
        for iteration in range(iterations):
            print(f"Iter [{iteration}/{iterations}]")
            population, scores = self.get_scores(parallel=parallel)
            print(f"Best accuracy: {np.amax(scores)}")
            wandb.log({"best reward": np.amax(scores), "scores": scores}, step=iteration)
            best_accuracies.append(np.amax(scores))
            self.update_params(population, scores)
            del population
        
        for iteration in range(iterations):
            print(f"Best accuracy [{iteration}/{iterations}]: {best_accuracies[iteration]}")
    

