import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy
import os, psutil, sys
import pickle

from unrolled_model import UnrolledModel
from fitness import evaluate_classification, evaluate_car_racing
from data import DataManager

import wandb

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
    def __init__(self, 
                 rolled_model, 
                 dataset_type="CIFAR10", 
                 population_size=100,
                 num_threads=1,
                 sigma=0.1,
                 abcd_perturbation_std=1,
                 abcd_learning_rate=0.2,
                 abcd_lr_decay=0.995,
                 aggregation_function='max',
                 bp_last_layer=True,
                 bp_learning_rate=0.01,
                 saving_path=None,
                 resume_file=None
                ):
        
        self.model = rolled_model
        
        self.start_iteration = 0
        self.population_size = population_size
        self.sigma = sigma
        self.decay = abcd_lr_decay
        self.abcd_lr = abcd_learning_rate
        self.perturbation_factor = abcd_perturbation_std
        self.num_threads = mp.cpu_count() if num_threads == -1 else num_threads
        
        n_workers = 8
        if num_threads > 1:
            if (not 'MKL_NUM_THREADS' in os.environ) or (not 'OMP_NUM_THREADS' in os.environ) or (os.environ['MKL_NUM_THREADS'] != "1") or (os.environ['OMP_NUM_THREADS'] != "1"):
                print(f"For parallel execution run this command: export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1")
                exit(0)
                
            mp.set_start_method('spawn')
            mp.set_sharing_strategy('file_system')
            n_workers = 0
        
        self.bp_last_layer = bp_last_layer
        self.bp_lr = bp_learning_rate
        
        self.aggregation_function_str = aggregation_function
        if aggregation_function == 'min':
            self.aggregation_function = torch.min
        elif aggregation_function == 'max':
            self.aggregation_function = torch.max
        elif aggregation_function == 'median':
            self.aggregation_function = torch.median
        elif aggregation_function == 'mean':
            self.aggregation_function = torch.mean
        else:
            print(f'Invalid aggregation function {aggregation_function}')
            exit(1)
        
        self.dataset_type = dataset_type
        if self.dataset_type != "CarRacing-v2":
            print(f"Using {n_workers} workers")
            self.data = DataManager(self.dataset_type, num_workers=n_workers, batch_size=64)
            self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        else:
            self.input_size = 64
        
        self.unrolled_model = UnrolledModel(self.model, self.input_size)
        
        self.unrolled_model_type = self.unrolled_model.get_new_model(self.aggregation_function)
        print(self.unrolled_model_type)
        if resume_file:
            self.params = self.load_params(resume_file)
        else:
            self.params = self.init_ABCD_parameters(self.unrolled_model_type)
            
        print(f"ABCD Params: {self.get_ABCD_params_number()}")
        
        self.saving_path = saving_path
        os.makedirs(self.saving_path, exist_ok=True)
        
        
    def get_ABCD_params_number(self, print_layers=False):
        pms = 0
        l = 0
        for layer in self.params:
            for side in ['in', 'out']:
                for p in self.params[layer][side]:
                    if self.params[layer][side][p] != None:
                        if print_layers:
                            print(f"{l}_{side} - {p}: {self.params[layer][side][p].shape}")
                        pms += self.params[layer][side][p].shape[0]
            l += 1
        
        return pms        

    def init_ABCD_parameters(self, net: nn.Sequential, start_idx: int = 0, end_idx: int = -1, device: str = 'cpu') -> dict:
        if end_idx == -1 or end_idx > len(net):
            end_idx = len(net)
        
        params = {}
        l_index = start_idx
        i = start_idx
        while i < end_idx:
            if type(net[i]) == nn.Linear:
                params[l_index] = {'in': None, 'out': None}
                params[l_index]['in'] = {'A':None, 'B': None, 'C':None, 'D': None, 'lr':None}
                params[l_index]['out'] = {'A':None, 'B': None, 'C':None, 'D': None, 'lr':None}
                l_index += 1
            i += 1
        
        last_l_index = l_index - 1
        l_index = start_idx
        i = start_idx
        after_pooling = False
        while i < end_idx:
            if type(net[i]) in [nn.MaxPool2d, nn.AvgPool2d]:
                after_pooling = True
            if type(net[i]) == nn.Linear:
                layer = net[i]

                in_shape = layer.weight.shape[1]
                out_shape = layer.weight.shape[0]
                
                # first linear no B in
                if l_index == 0:
                    params[l_index]['in']['A'] = torch.randn(in_shape, device=device)
                    params[l_index]['in']['C'] = torch.randn(in_shape, device=device)
                    params[l_index]['in']['D'] = torch.randn(in_shape, device=device)
                    
                    params[l_index]['out']['A'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['B'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['C'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['D'] = torch.randn(out_shape, device=device)
                    
                    params[l_index]['in']['lr'] = torch.randn(in_shape, device=device)
                    params[l_index]['out']['lr'] = torch.randn(out_shape, device=device)
                # last linear no A out
                elif l_index == last_l_index:
                    if after_pooling:
                        params[l_index]['in']['A'] = torch.randn(in_shape, device=device)
                        params[l_index]['in']['C'] = torch.randn(in_shape, device=device)
                        params[l_index]['in']['D'] = torch.randn(in_shape, device=device)
                        params[l_index]['in']['lr'] = torch.randn(in_shape, device=device)
                        
                        params[l_index - 1]['out']['A'] = None # not needed A before pooling as layers do not share neurons
                        
                        after_pooling = False
                        
                    params[l_index]['out']['B'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['C'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['D'] = torch.randn(out_shape, device=device)
                    
                    params[l_index]['out']['lr'] = torch.randn(out_shape, device=device)
                    
                else:
                    if after_pooling:
                        params[l_index]['in']['A'] = torch.randn(in_shape, device=device)
                        params[l_index]['in']['C'] = torch.randn(in_shape, device=device)
                        params[l_index]['in']['D'] = torch.randn(in_shape, device=device)
                        
                        params[l_index]['in']['lr'] = torch.randn(in_shape, device=device)
                        
                        params[l_index - 1]['out']['A'] = None # not needed A before pooling as layers do not share neurons
                        
                        after_pooling = False
                    
                    params[l_index]['out']['A'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['B'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['C'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['D'] = torch.randn(out_shape, device=device)
                    params[l_index]['out']['lr'] = torch.randn(out_shape, device=device)
                    
                
                l_index += 1
            i += 1
        
        return params
    
    
    def init_population(self):
        pop = []
        for i in range(self.population_size):
            pop.append(self.init_ABCD_parameters(self.unrolled_model_type))
            
        return pop
    
    
    def perturbate(self, params, noise):
        new_p = deepcopy(params)
        for layer in new_p:
            for side in ['in', 'out']:
                for p in new_p[layer][side]:
                    if new_p[layer][side][p] != None:
                        jittered = self.sigma * noise[layer][side][p]
                        new_p[layer][side][p].add_(jittered)
        
        return new_p
    
    
    def pop_to_device(self, p, device):
        if device == 'cuda':
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        for layer in p:
            for side in p[layer]:
                for k in p[layer][side]:
                    if p[layer][side][k] != None:
                        p[layer][side][k] = p[layer][side][k].to(device).to(dtype)
        
        return p
    
    
    def get_scores(self, pool, population, parallel=False, device='cpu'):        
        if parallel:
            print("Parallelizing")
            processes = []
            
            manager = mp.Manager()
            shared_dict = manager.dict()
            
            pop_evaluated = 0
            while pop_evaluated < self.population_size:
                pop = self.perturbate(self.params, population[pop_evaluated])
        
                if device != 'cpu':
                    pop = self.pop_to_device(pop, device)

                model = self.unrolled_model.get_new_model(self.aggregation_function)
                shared_dict[pop_evaluated] = None
                if self.dataset_type != "CarRacing-v2":
                    loader = self.data.get_new_loader(train=True)
                    args = (model, loader, pop, pop_evaluated, shared_dict, device, self.abcd_lr, self.aggregation_function, self.bp_last_layer, self.bp_lr)
                    target_fn = evaluate_classification
                else:
                    args = (model, self.dataset_type, pop, pop_evaluated, shared_dict, self.abcd_lr, self.bp_last_layer, self.bp_lr, self.input_size, self.aggregation_function, device)
                    target_fn = evaluate_car_racing
                    
                processes.append(pool.apply_async(target_fn, args))
                pop_evaluated += 1
                
            for p in processes:
                p.wait()   
            
            scores = list(dict(sorted(shared_dict.items(), key=lambda x: x[0])).values())
            
        else:
            print("No Parallel")
            scores = []
            for pop_idx, p in enumerate(population):
                pop = self.perturbate(self.params, p)
                
                if device != 'cpu':
                    pop = self.pop_to_device(pop, device)
                
                if self.dataset_type != "CarRacing-v2":
                    scores.append(evaluate_classification(self.unrolled_model.get_new_model(self.aggregation_function), self.data.train_loader, pop, pop_idx, abcd_learning_rate=self.abcd_lr, agg_func=self.aggregation_function, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, device=device))
                else:
                    scores.append(evaluate_car_racing(self.unrolled_model.get_new_model(self.aggregation_function), self.dataset_type, pop, pop_idx, abcd_learning_rate=self.abcd_lr, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, in_size=self.input_size, agg_func=self.aggregation_function, device=device))
        
        idx_to_remove = []
        for i, s in enumerate(scores):
            if s is None:
                idx_to_remove.append(i)
        
        c = 0
        for i in idx_to_remove:
            del population[i - c]
            del scores[i - c]
            c += 1
        
        scores = np.array(scores)
        
        return scores    
    
    
    def update_params(self, population, scores):
        
        ranks = compute_centered_ranks(scores)
        
        ranks = (ranks - ranks.mean()) / ranks.std()
        
        self.update_factor = self.abcd_lr / (self.population_size * self.sigma)
        for layer in population[0].keys():
            for side in population[0][layer].keys():
                for key in population[0][layer][side].keys():
                    if self.params[layer][side][key] != None:
                
                        param_pop = np.array([p[layer][side][key] for p in population])
                
                        self.params[layer][side][key].add_(torch.from_numpy(self.update_factor * np.dot(param_pop.T, ranks).T))
            
        if self.abcd_lr > 0.001:
            self.abcd_lr *= self.decay

        if self.sigma > 0.01:
            self.sigma *= 0.999
        
        del population
        print("Parameters update done")
        
    
    def load_params(self, file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        
        dataset_type, population_size, abcd_lr, decay, perturbation_factor, sigma, agg_func, bp_last_layer, bp_lr, iteration, best_score, avg_score = file_path.split('.pickle')[0].split('_')
        
        population_size = int(population_size)
        abcd_lr = float(abcd_lr)
        decay = float(decay)
        perturbation_factor = float(perturbation_factor)
        sigma = float(sigma)
        bp_lr = float(bp_lr)
        iteration = int(iteration)
        best_score = float(best_score)
        avg_score = float(avg_score)
        
        print(f"Loading {dataset_type} ABCD parameters at iteration {iteration} (best score: {best_score:.2f}, avg score: {avg_score:.2f})")
        
        self.start_iteration = iteration
        self.abcd_lr = abcd_lr
        self.decay = decay
        self.sigma = sigma
        self.bp_lr = bp_lr
        
        return params
    
    
    def save_params(self, iteration=0, best_score=0, avg_score=0):
        filename = os.path.join(self.saving_path, f'{self.dataset_type}_{self.population_size}_{self.abcd_lr}_{self.decay}_{self.perturbation_factor}_{self.sigma}_{self.aggregation_function_str}_{self.bp_last_layer}_{self.bp_lr}_{iteration}_{best_score:.2f}_{avg_score:.2f}.pickle')
        
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
    
    def run(self, iterations, device='cpu'):
        
        # export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1
        parallel = self.num_threads > 1
        
        pool = mp.Pool(self.num_threads) if parallel else None
        
        best_scores = []
        for iteration in range(self.start_iteration, iterations):
            print(f"Iter [{iteration}/{iterations}]")
            
            population = self.init_population()
            scores = self.get_scores(pool=pool, population=population, parallel=parallel, device=device)
            
            best_score = np.amax(scores)
            avg_score = np.mean(scores)
            print(f"Best score: {best_score}")
            print(f"Avg score: {avg_score}")
            wandb.log({"best reward": best_score, "avg reward": avg_score, "scores": scores}, step=iteration)
            best_scores.append(best_score)
            
            self.update_params(population, scores)
            if self.saving_path:
                self.save_params(iteration, best_score, avg_score)

        for iteration in range(iterations):
            print(f"Best accuracy [{iteration}/{iterations}]: {best_scores[iteration]}")
    
        if pool is not None:
            pool.close()
            pool.join()
