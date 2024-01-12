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
    def __init__(self, rolled_model, population_size=100, sigma=0.1, learning_rate=0.2, num_threads=1, distribution='normal'):
        self.model = rolled_model
        
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.update_factor = self.learning_rate / (self.population_size * self.sigma)
        self.perturbation_factor = 1
        self.num_threads = num_threads
        
        self.data = DataManager("CIFAR10")
        self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        
        self.unrolled_model = UnrolledModel(self.model, self.input_size)
        self.params = self.init_ABCD_parameters(self.unrolled_model.get_new_model())
        
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
                # A
                params[l_index]['A'] = torch.randn(layer.weight.shape[1], device=device)
                
                # B
                params[l_index + 1]['B'] = torch.randn(layer.weight.shape[0], device=device)
                
                # C, D
                params[l_index]['C'] = torch.randn((layer.weight.shape[0], layer.weight.shape[1]), device=device)
                params[l_index]['D'] = torch.randn((layer.weight.shape[0], layer.weight.shape[1]), device=device)
                l_index += 1
            i += 1
            
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
                new_p[layer][p].add_(self.perturbation_factor * torch.randn_like(new_p[layer][p]))
    
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
                    score = mp.Process(target=evaluate, args=(model, loader, population[pop_evaluated], pop_evaluated, shared_dict))
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
            
            scores = list(dict(sorted(shared_dict.items(), key=lambda x: x[0])).values())
            
        else:
            print("No Parallel")
            scores = []
            for p in population:
                scores.append(evaluate(self.unrolled_model.get_new_model(), self.data.test_loader, p))
        
        scores = np.array(scores)
        print(f"Best accuracy: {np.amax(scores)}")

        return scores    
    
    
    def update_params(self, population, scores):
        ranks = compute_centered_ranks(scores)
                    
        for layer in population[0].keys():
            for key in population[0][layer].keys():
                if layer == 0 and key == 'B':
                    continue
                if layer == list(population[0].keys())[-1] and key != 'B':
                    continue
                
                param_pop = np.array([p[layer][key] for p in population])
                
                self.params[layer][key].add_(torch.from_numpy(self.update_factor * np.dot(param_pop.T, ranks).T))
        
        print("Parameters update done")
    
    
    def run(self, iterations):
        
        # export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1
        parallel = self.num_threads > 1
        
        if parallel and os.environ['MKL_NUM_THREADS'] != "1":
            print(f"Setting MKL_NUM_THREADS to 1 for parallel execution")
            os.environ['MKL_NUM_THREADS'] = "1"
        if parallel and os.environ['OMP_NUM_THREADS'] != "1":
            print(f"Setting OMP_NUM_THREADS to 1 for parallel execution")
            os.environ['OMP_NUM_THREADS'] = "1"       
        
        for iteration in range(iterations):
            print(f"Iter [{iteration}/{iterations}]")
            population = self.init_population()
            scores = self.get_scores(population, parallel=parallel)
            self.update_params(population, scores)
            
            
        # if pool is not None:
        #     pool.close()
        #     pool.join()
    

