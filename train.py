import hydra
from omegaconf import DictConfig
import logging
import torch

from data import get_loaders
from models.simple_model import SimpleModel

@hydra.main(version_base=None, config_path="configs", config_name="base_config.yaml")
def train(cfg: DictConfig):
    train_loader, test_loader = get_loaders(cfg.dataset)

    logging.info(f"Length train loader: {len(train_loader)} batches")
    logging.info(f"Length test loader: {len(test_loader)} batches")
    
    model = SimpleModel(28*28, 10).cuda()
    for e in range(cfg.epochs):
        print(f"Epoch {e}")
        correct = 0
        total = 0
        for img, label in train_loader:
        
            img = img.cuda()
            label = label.cuda()

            out = model(img)

            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.shape[0]

            model.update_weights(lr=0.000001, rule='hpca')

        print(f"Train Acc: {correct/total :.10f}")

        correct = 0
        total = 0
        for img, label in test_loader:
            img = img.cuda()
            label = label.cuda()

            out = model(img)

            pred = torch.argmax(out, dim=1)
            correct += (pred == label).sum().item()
            total += label.shape[0]
        
        print(f"Test Acc: {correct/total :.10f}")
            



if __name__ == "__main__":
    train()
    