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
                 bp_last_layer=True,
                 bp_learning_rate=0.01,
                 saving_path=None,
                 resume_file=None
                ):
        
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
        
        new_m = self.unrolled_model.get_new_model()
        print(new_m)
        self.params = self.init_ABCD_parameters(new_m)
        print(f"ABCD Params: {self.get_ABCD_params_number()}")
        
        self.bp_last_layer = bp_last_layer
        self.bp_lr = bp_learning_rate
        
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
            pop.append(self.init_ABCD_parameters(self.unrolled_model.get_new_model()))
            
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
        for layer in p:
            for side in p[layer]:
                for k in p[layer][side]:
                    if p[layer][side][k] != None:
                        p[layer][side][k] = p[layer][side][k].to(device)
        
        return p
    
    
    def start_new_eval(self, p, processes, thread_used, pop_evaluated, shared_dict, device='cpu'):
        
        pop = self.perturbate(self.params, p)
        
        if device != 'cpu':
            pop = self.pop_to_device(pop, device)

        model = self.unrolled_model.get_new_model()
        shared_dict[pop_evaluated] = None
        if self.dataset_type != "CarRacing-v2":
            loader = self.data.get_new_loader(train=True)
            args = (model, loader, pop, pop_evaluated, shared_dict, device, self.abcd_lr, self.bp_last_layer, self.bp_lr)
            target_fn = evaluate_classification
        else:
            args = (model, self.dataset_type, pop, pop_evaluated, shared_dict, self.abcd_lr, self.bp_last_layer, self.bp_lr, self.input_size, device)
            target_fn = evaluate_car_racing
        proc = mp.Process(target=target_fn, args=args)
        proc.start()
        
        processes.append(proc)
        thread_used += 1
        pop_evaluated += 1
        print(f"Processes spawned: {len(processes)} - Processes running {thread_used}")
        
        return processes, thread_used, pop_evaluated
    
    
    def get_scores(self, population, parallel=False, keep_best_only=False, device='cpu'):        
        if keep_best_only:
            best_score = 0
            best_pop = None        

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
                    processes, thread_used, pop_evaluated = self.start_new_eval(population[pop_evaluated], processes, thread_used, pop_evaluated, shared_dict, device=device)
                else:
                    processes[processes_joined].join()
                    processes_joined += 1
                    thread_used -= 1
                    
                    # if keep_best_only:
                    #     print(f"MEM Before pop selection: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
                    #     for pop, score in shared_dict.items():
                    #         if score != None:
                    #             if score > best_score:
                    #                 print(f"New best score found: pop {pop} score {score}")
                    #                 best_score = score
                    #                 best_pop = population[pop]
                    #                 del shared_dict[pop]
                                
                    #             population[pop] = None
                                
                        # print(f"MEM After pop selection: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
                        
            
            for proc in processes:
                proc.join()
                proc.close()     
            
            # if keep_best_only:
            #     print(f"MEM Before pop selection: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")
            #     for pop, score in shared_dict.items():
            #         if score != None:
            #             if score > best_score:
            #                 print(f"New best score found: pop {pop} score {score}")
            #                 best_score = score
            #                 best_pop = population[pop]
            #                 del shared_dict[pop]
                            
            #             population[pop] = None
            #     print(f"MEM After pop selection: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} MB")       
            # else:
            scores = list(dict(sorted(shared_dict.items(), key=lambda x: x[0])).values())
            
        else:
            print("No Parallel")
            scores = []
            for pop_idx, p in enumerate(population):
                pop = self.perturbate(self.params, p)
                
                if device != 'cpu':
                    pop = self.pop_to_device(pop, device)
                
                if self.dataset_type != "CarRacing-v2":
                    scores.append(evaluate_classification(self.unrolled_model.get_new_model(), self.data.test_loader, pop, pop_idx, abcd_learning_rate=self.abcd_lr, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, device=device))
                else:
                    scores.append(evaluate_car_racing(self.unrolled_model.get_new_model(), self.dataset_type, pop, pop_idx, abcd_learning_rate=self.abcd_lr, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, in_size=self.input_size, device=device))
                
                # if keep_best_only:
                #     if scores[-1] > best_score:
                #         best_score = scores[-1]
                #         best_pop = population[-1]
                #         population[-1] = None
                        
        # if device != 'cpu':
        #     for i, p in enumerate(population):
        #         population[i] = self.pop_to_device(p, 'cpu')               
        
        # if keep_best_only:
        #     return best_pop, best_score
        # else:
        scores = np.array(scores)
        return scores    
    
    
    def update_params(self, population, scores, keep_best_only=False):
        
        # if keep_best_only:
        #     if scores > self.best_total_score:
        #         up_pct = 0.8
        #     else:
        #         up_pct = 0.6
                
        #     for layer in population.keys():
        #         for key in population[layer].keys():
        #             if layer == 0 and key == 'B':
        #                 continue
        #             if layer == list(population.keys())[-1] and key != 'B':
        #                 continue
                    
        #             self.params[layer][key] = self.params[layer][key] * (1-up_pct) + population[layer][key] * up_pct
        # else:
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
    
    
    def save_params(self, iteration=0, best_score=0, avg_score=0):
        filename = os.path.join(self.saving_path, f'{self.dataset_type}_{self.population_size}_{self.abcd_lr}_{self.decay}_{self.perturbation_factor}_{self.sigma}_{self.bp_last_layer}_{self.bp_lr}_{iteration}_{best_score:.2f}_{avg_score:.2f}.pickle')
        
        with open(filename, 'wb') as f:
            pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
    
    def run(self, iterations, device='cpu'):
        keep_best_only = False
        if keep_best_only:
            self.best_total_score = 0
        
        # export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1
        parallel = self.num_threads > 1
        
        if parallel:
            if (not 'MKL_NUM_THREADS' in os.environ) or (not 'OMP_NUM_THREADS' in os.environ) or (os.environ['MKL_NUM_THREADS'] != "1") or (os.environ['OMP_NUM_THREADS'] != "1"):
                print(f"For parallel execution run this command: export MKL_NUM_THREADS=1; export OMP_NUM_THREADS=1")
                exit(0)
            mp.set_start_method('spawn')
            mp.set_sharing_strategy('file_system')
        
        best_scores = []
        for iteration in range(iterations):
            print(f"Iter [{iteration}/{iterations}]")
            population = self.init_population()
            scores = self.get_scores(population=population, parallel=parallel, keep_best_only=keep_best_only, device=device)
            # if not keep_best_only:
            best_score = np.amax(scores)
            avg_score = np.mean(scores)
            print(f"Best score: {best_score}")
            print(f"Avg score: {avg_score}")
            wandb.log({"best reward": best_score, "avg reward": avg_score, "scores": scores}, step=iteration)
            best_scores.append(best_score)
            self.update_params(population, scores, keep_best_only=keep_best_only)
            if self.saving_path:
                self.save_params(iteration, best_score, avg_score)
                
            # else:
            #     best_pop = population
            #     best_score = scores
            #     print(f"Best score: {best_score}")
            #     wandb.log({"best reward": best_score}, step=iteration)
            #     best_scores.append(best_score)
            #     self.update_params(best_pop, best_score, keep_best_only=keep_best_only)
            #     del best_pop
        
        for iteration in range(iterations):
            print(f"Best accuracy [{iteration}/{iterations}]: {best_scores[iteration]}")
    

