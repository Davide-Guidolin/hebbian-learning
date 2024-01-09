import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from copy import deepcopy

from unrolled_model import UnrolledModel, evaluate
from data import DataManager

class EvolutionStrategy:
    def __init__(self, rolled_model, population_size=100, sigma=0.1, learning_rate=0.2, decay=0.995, num_threads=1, distribution='normal'):
        self.model = rolled_model
        
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.update_factor = self.learning_rate / (self.population_size * self.sigma)
        self.decay = decay
        self.num_threads = num_threads
        self.distribution = distribution
        
        self.data = DataManager("CIFAR10")
        self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        
        self.unrolled_model = UnrolledModel(self.model, self.input_size)
        
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
    
    def init_population(self):
        pop = []
        for _ in range(self.population_size):
            pop.append(self.init_ABCD_parameters(self.unrolled_model.get_new_model()))
            
        return pop
        
    
    def get_scores(self, population, parallel=None):
        print("IN get_scores")
        # Take population and run parallel evaluations
        if parallel:
            print("Parallelizing")
            processes = []
            
            correct = []
            total = []

            pop_evaluated = 0
            processes_used = 0
            processes_joined = 0
            while pop_evaluated < len(population):
                print(f"Processes spawned: {len(processes)}")
                if processes_used < self.num_threads:
                    loader = self.data.get_new_loader(train=True)
                    # model = self.unrolled_model.get_new_model()
                    score = mp.Process(target=evaluate, args=(deepcopy(self.unrolled_model.layers), loader, population[pop_evaluated]))
                    print(f"Evaluating population {pop_evaluated}")
                    score.start()
                    
                    processes.append(score)
                    processes_used += 1
                    pop_evaluated += 1
                else:
                    processes[processes_joined].join()
                    processes_joined += 1
                    processes_used -= 1
            
            print("Wait for join")
            for score in processes:
                score.join()
                
            # for j, p in enumerate(population):
            #     print(f"Creating model {j}")
            #     worker_args.append((self.data.train_set, p))

            # corr_total = pool.starmap(deepcopy(self.unrolled_model).evaluate, worker_args)
            # corr_total.wait()
            # for j, (corr, tot) in enumerate(corr_total):
            #     correct[j] += corr
            #     total[j] += tot

            scores = correct/total
            
        else:
            print("No Parallel")
            scores = []
            for p in population:
                scores.append(evaluate(self.unrolled_model.get_new_model(), self.data.test_loader, p))
        
        scores = np.array(scores)
        print(scores)
        exit(0)

        return scores    
    
    
    def update_params(self, population, scores):
        # Take best population and update parameters
        pass
    
    
    def run(self, iterations):
        
        parallel = self.num_threads > 1
        
        for iteration in range(iterations):
            population = self.init_population()
            scores = self.get_scores(population, parallel=parallel)
            exit(0)
            self.update_params(population, scores)
            
            print(f"Iter [{iteration}/{iterations}]: Reward {scores.mean()}")
            
        # if pool is not None:
        #     pool.close()
        #     pool.join()
    

