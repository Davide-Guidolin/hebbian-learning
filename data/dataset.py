from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import logging

def get_loaders(cfg):
    match cfg.name:
        case "mnist":
            train_set = MNIST(root=cfg.root_dir, train=True, download=True, transform=ToTensor())
            test_set = MNIST(root=cfg.root_dir, train=False, transform=ToTensor())
        case "cifar10":
            train_set = CIFAR10(root=cfg.root_dir, train=True, download=True, transform=ToTensor())
            test_set = CIFAR10(root=cfg.root_dir, train=False, transform=ToTensor())
        case _:
            logging.error(f"{cfg.name} dataset not available")
            exit(1)
    
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True)
    
    return train_loader, test_loader
