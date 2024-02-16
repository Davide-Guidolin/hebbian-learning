from data import DataManager
import wandb
import torch
import torch.nn as nn

from model import BaseNet

n_epochs = 150
bp_lr = 1e-5
device = 'cuda'

data = DataManager('CIFAR10', batch_size=128, num_workers=8)
m = BaseNet()

wandb.init(
        project="CIFAR-Softhebb",
        config={
            'bp_lr': bp_lr,
        }
)

def evaluate(model, data_loader, device='cpu'):
        model.eval()
        if device != 'cpu':
            model.to(device)
                    
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for i, (x, true_y) in enumerate(data_loader):
                x = x.to(device)
                true_y = true_y.to(device)
                
                y = model(x)
                test_loss = criterion(y, true_y)
                total_loss += test_loss
                correct += torch.sum(torch.argmax(y, dim=-1) == true_y)
                total += true_y.shape[0]
            

        acc = correct/total
        
        return acc, total_loss/len(data_loader)

print("\nStarting backprop training:")
print(f"\tbp_lr = {bp_lr}")

train_loader = data.get_new_loader(train=True)
test_loader = data.get_new_loader(train=False)

if device != 'cpu':
    m.to(device)
        
for p in m.parameters():
    p.requires_grad = True
    
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(m.parameters(), lr=bp_lr)

for e in range(n_epochs):
    m.train()
    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0
    correct = 0
    total = 0
    for i, (x, true_y) in enumerate(train_loader):
        optim.zero_grad()
        
        x = x.to(device)
        true_y = true_y.to(device)
        
        y = m(x)
        
        loss = criterion(y, true_y)
        train_loss += loss.item()
        loss.backward()
        optim.step()
        
        correct += torch.sum(torch.argmax(y, dim=-1) == true_y)
        total += true_y.shape[0]
    
    train_loss = train_loss / len(train_loader)
    train_acc = correct/total
    
    test_acc, test_loss = evaluate(m, test_loader, device)
    
    print(f"[{e+1}/{n_epochs}] Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f} Train Loss:  {train_loss:.5f} Test Loss: {test_loss:.5f}")
    wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Train Loss": train_loss, "Test Loss": test_loss}, step=e)