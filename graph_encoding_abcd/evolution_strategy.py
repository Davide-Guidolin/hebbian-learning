import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy
import os

from unrolled_model import UnrolledModel
from fitness import evaluate
from data import DataManager


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
    def __init__(self, rolled_model, population_size=100, sigma=0.1, abcd_perturbation_std=1, abcd_learning_rate=0.2, abcd_lr_decay=0.995, num_threads=1, bp_last_layer=True, bp_learning_rate=0.01):
        self.model = rolled_model
        
        self.population_size = population_size
        self.sigma = sigma
        self.decay = abcd_lr_decay
        self.abcd_lr = abcd_learning_rate
        self.perturbation_factor = abcd_perturbation_std
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        
        self.data = DataManager("CIFAR10")
        self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        
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
        
        params = self.normalize_params(params)
        
        return params
    
    
    def perturbate(self, params):
        print("Perturbating")
        new_p = deepcopy(params)
        for layer in new_p:
            for p in new_p[layer]:
                if layer == 0 and p == 'B':
                    continue
                if layer == list(new_p.keys())[-1] and p != 'B':
                    continue
                
                noise = torch.randn_like(new_p[layer][p])
                new_p[layer][p].add_(self.perturbation_factor * noise)

        new_p = self.normalize_params(new_p)
        
        return new_p
    
    
    def init_population(self):
        pop = []
        for _ in range(self.population_size):
            pop.append(self.perturbate(self.params))
            
        return pop
        
    
    def get_scores(self, population, parallel=None):
        print("IN get_scores")
        # Take population and run parallel evaluations
        if parallel:
            print("Parallelizing")
            processes = []
            
            manager = mp.Manager()
            shared_dict = manager.dict()
            
            pop_evaluated = 0
            processes_used = 0
            processes_joined = 0
            while pop_evaluated < len(population):
                print(f"Processes spawned: {len(processes)}")
                if processes_used < self.num_threads:
                    loader = self.data.get_new_loader(train=False)
                    model = self.unrolled_model.get_new_model()
                    score = mp.Process(target=evaluate, args=(model, loader, population[pop_evaluated], pop_evaluated, shared_dict, self.abcd_lr, self.bp_last_layer, self.bp_lr))
                    print(f"Evaluating population {pop_evaluated}")
                    score.start()
                    
                    processes.append(score)
                    processes_used += 1
                    pop_evaluated += 1
                else:
                    processes[processes_joined].join()
                    processes_joined += 1
                    processes_used -= 1
            
            for score in processes:
                score.join()
                score.close()
            
            scores = list(dict(sorted(shared_dict.items(), key=lambda x: x[0])).values())
            
        else:
            print("No Parallel")
            scores = []
            for p in population:
                scores.append(evaluate(self.unrolled_model.get_new_model(), self.data.test_loader, p, abcd_learning_rate=self.abcd_lr, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr))
        
        scores = np.array(scores)
        print(f"Best accuracy: {np.amax(scores)}")

        return scores    
    
    
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
                
        if self.abcd_lr > 0.001:
            self.abcd_lr *= self.decay

        if self.sigma > 0.01:
            self.sigma *= 0.999
        
        print("Parameters update done")
    
    
    def run(self, iterations):
        
        # export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1
        parallel = self.num_threads > 1
        
        if parallel:
            if (not 'MKL_NUM_THREADS' in os.environ) or (not 'OMP_NUM_THREADS' in os.environ) or (os.environ['MKL_NUM_THREADS'] != "1") or (os.environ['OMP_NUM_THREADS'] != "1"):
                print(f"For parallel execution run this command: export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1")
                exit(0)
        
        for iteration in range(iterations):
            print(f"Iter [{iteration}/{iterations}]")
            population = self.init_population()
            scores = self.get_scores(population, parallel=parallel)
            self.update_params(population, scores)
    

