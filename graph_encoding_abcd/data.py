from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
import torch
import os
from copy import deepcopy

class DataManager():
    
    dataset_classes = {
        "CIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
    }
    
    def __init__(self, dataset_name: str = "CIFAR10", batch_size: int = 64, num_workers: int = 0, save_path: str = "./data"):
        
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        os.makedirs(save_path, exist_ok=True)
        
        t = Compose([ToTensor(), Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        
        self.train_set = self.dataset_classes[self.dataset_name](self.save_path, transform=t, train=True, download=True)
        self.test_set = self.dataset_classes[self.dataset_name](self.save_path, transform=t, train=False, download=True)

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def get_new_loader(self, train=True):
        torch.manual_seed(53)
        if train:
            return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)