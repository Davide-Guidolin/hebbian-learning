from fitness import evaluate_classification
from data import DataManager
from unrolled_model import UnrolledModel
import wandb
import torch
import torch.nn as nn

def apply_prune_mask(net):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = []
    for layer in net.modules():
        if hasattr(layer, 'shared_weights'):
            prunable_layers.append(layer)
    
    masks = []
    for l in prunable_layers:
        if hasattr(l, 'mask_tensor'):
            masks.append(l.mask_tensor.t())

    for layer, keep_mask in zip(prunable_layers, masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))

class SoftHebbTrain:
    def __init__(self, rolled_model, dataset_type="CIFAR10", bp_last_layer=True, bp_learning_rate=0.001, softhebb_lr=0.005, aggregation_function='max'):
        self.model = rolled_model
        
        self.dataset_type = dataset_type
        
        self.data = DataManager(self.dataset_type, batch_size=128, num_workers=8)
        self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        
        self.unrolled_model = UnrolledModel(self.model, self.input_size)
        
        self.bp_last_layer = bp_last_layer
        self.bp_lr = bp_learning_rate
        self.softhebb_lr = softhebb_lr
        
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
        
    
    def train_backprop(self, n_epochs=160, device='cpu'):
        print("\nStarting backprop training:")
        print(f"\tbp_lr = {self.bp_lr}")
        
        m = self.unrolled_model.get_new_model()
        print(m)
        train_loader = self.data.get_new_loader(train=True)
        test_loader = self.data.get_new_loader(train=False)
        
        if device != 'cpu':
            for layer in m:
                layer.to(device)
                if hasattr(layer, 'mask_tensor'):
                    layer.mask_tensor = layer.mask_tensor.to(device)
                
        for p in m.parameters():
            p.requires_grad = True
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(m.parameters(), lr=self.bp_lr, momentum=0.9, weight_decay=1e-4) # 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,120], gamma=0.1)
        apply_prune_mask(m)
        
        for e in range(n_epochs):
            m.train()
            train_loss = 0
            train_acc = 0
            test_loss = 0
            test_acc = 0
            correct = 0
            total = 0
            for i, (x, true_y) in enumerate(train_loader):
                optimizer.zero_grad()
                
                x = x.view(x.shape[0], -1).to(device)
                true_y = true_y.to(device)
                
                y = m(x)
                
                loss = criterion(y, true_y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                correct += torch.sum(torch.argmax(y, dim=-1) == true_y)
                total += true_y.shape[0]
            
            scheduler.step()
            
            train_loss = train_loss / len(train_loader)
            train_acc = correct/total
            
            test_acc, test_loss = self.evaluate(m, test_loader, device)
            
            print(f"[{e+1}/{n_epochs}] Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f} Train Loss:  {train_loss:.5f} Test Loss: {test_loss:.5f}")
            wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Train Loss": train_loss, "Test Loss": test_loss}, step=e)
            
    
    def train(self, n_epochs=10, device='cpu'):
        m = self.unrolled_model.get_new_model()
        print(m)
        train_loader = self.data.get_new_loader(train=True)
        test_loader = self.data.get_new_loader(train=False)
        
        if device != 'cpu':
            for layer in m:
                layer.to(device)
                if hasattr(layer, 'mask_tensor'):
                    layer.mask_tensor = layer.mask_tensor.to(device)
        
        print("\nStarting softhebb training:")
        print(f"\tbp_last_layer = {self.bp_last_layer}")
        print(f"\tbp_lr = {self.bp_lr}")
        print(f"\tsofthebb_lr = {self.softhebb_lr}\n\n")
        
        for i in range(n_epochs):
            train_acc, train_loss = evaluate_classification(m, train_loader, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, softhebb_train=True, softhebb_lr=self.softhebb_lr, agg_func=self.aggregation_function, device=device)
            test_acc, test_loss = self.evaluate(m, test_loader, device=device)
            
            print(f"[{i+1}/{n_epochs}] Train Acc: {train_acc:.4f}  Test Acc: {test_acc:.4f} Train Loss:  {train_loss:.5f} Test Loss: {test_loss:.5f}")
            # wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Train Loss": train_loss, "Test Loss": test_loss}, step=i)
            
    
    def evaluate(self, model, data_loader, device='cpu'):
        t = model[0].weight.dtype
        model.eval()
                    
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for i, (x, true_y) in enumerate(data_loader):
                x = x.view(x.shape[0], -1).to(device)
                true_y = true_y.to(device)
                
                y = model(x)
                test_loss = criterion(y, true_y)
                total_loss += test_loss
                correct += torch.sum(torch.argmax(y, dim=-1) == true_y)
                total += true_y.shape[0]
            

        acc = correct/total
        
        return acc, total_loss/len(data_loader)