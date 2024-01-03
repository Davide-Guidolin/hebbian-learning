from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os

class DataManager():
    
    dataset_classes = {
        "CIFAR10": CIFAR10,
        "CIFAR100": CIFAR100,
    }
    
    def __init__(self, dataset_name: str = "CIFAR10", batch_size: int = 64, num_workers: int = 4, save_path: str = "./data"):
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        os.makedirs(save_path, exist_ok=True)
        
        self.train_loader = self.load_dataset(train=True)
        self.test_loader = self.load_dataset(train=False)
        
    def load_dataset(self, train=True):
        ds = self.dataset_classes[self.dataset_name](self.save_path, transform=ToTensor(), train=train, download=True)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True if train else False, num_workers=self.num_workers)