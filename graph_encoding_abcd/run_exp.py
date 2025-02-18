from model import *
from evolution_strategy import EvolutionStrategy
from softhebb_train import SoftHebbTrain
import argparse
import torch

import wandb

def main():
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'CarRacing', 'CarRacing-v2'], help='Dataset to use', default='CIFAR10')
    parser.add_argument('--population_size', type=int, default=10, help='Size of the population for the es algorithm')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use, -1 to use all the cpus')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    parser.add_argument('--sigma', default=0.1, type=float, help='Sigma used in the ES mutation')
    parser.add_argument('--abcd_learning_rate', default=0.2, type=float, help='Learning rate of the ABCD parameters')
    parser.add_argument('--abcd_lr_decay', default=0.995, type=float, help='Decay factor for abcd_learning_rate')
    parser.add_argument('--aggregation_function', default='max', type=str, help='Aggregation function used in the ABCD update')
    
    parser.add_argument('--bp_last_layer', action='store_true', help='Use backpropagation in the last layer')
    parser.add_argument('--bp_lr', default=0.001, type=float, help='Learning rate to use for backpropagation')
    parser.add_argument('--bp', action='store_true', help='Train the full network using backpropagation. Only implemented for training with --softhebb for now')
    
    parser.add_argument('--softhebb', action='store_true', help='Train using Softhebb learning rule. To use only with classification tasks')
    parser.add_argument('--softhebb_lr', default=0.005, type=float, help='Softhebb learning rate if training using Softhebb')
    
    parser.add_argument('--saving_path', default='./params', type=str, help='Path where ABCD parameters will be saved')
    
    parser.add_argument('--resume_file', default=None, type=str, help='Path to the file containing the ABCD parameters')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    if dataset == 'CarRacing':
        dataset = 'CarRacing-v2'
    
    if dataset == 'CarRacing-v2':
        model = CNN_CarRacing()
    elif dataset == 'CIFAR10':
        model = BaseNet()
    elif dataset == 'CIFAR100':
        model = BaseNet100()
    else:
        print(f"Invalid dataset {dataset}")
        exit(1)
        
    if dataset == "CarRacing-v2":
        project = "CarRacing_abcd_ws"
    elif args.softhebb:
        if args.bp:
            project = "Backprop"
        else:
            project = "CIFAR-softhebb-final"
    else:
        project = "CIFAR-ABCD"
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        
        # track hyperparameters and run metadata
        config={
        'dataset_type': dataset,
        'population_size': args.population_size,
        'num_threads': args.num_threads,
        'sigma':args.sigma,
        'abcd_learning_rate': args.abcd_learning_rate,
        'abcd_lr_decay': args.abcd_lr_decay,
        'aggregation_function': args.aggregation_function,
        'architecture': model,
        'dataset': dataset,
        'bp_last_layer': args.bp_last_layer,
        'bp_lr': args.bp_lr,
        'softhebb': args.softhebb,
        'softhebb_lr': args.softhebb_lr,
        'epochs': args.epochs,
        'args': args,
        }
    )
    
    if not args.softhebb:
        es = EvolutionStrategy(model,
                               dataset_type=dataset,
                               population_size=args.population_size,
                               num_threads=args.num_threads,
                               sigma=args.sigma,
                               abcd_learning_rate=args.abcd_learning_rate,
                               abcd_lr_decay=args.abcd_lr_decay,
                               aggregation_function=args.aggregation_function,
                               bp_last_layer=args.bp_last_layer,
                               bp_learning_rate=args.bp_lr,
                               saving_path=args.saving_path,
                               resume_file=args.resume_file)
        
        es.run(iterations=args.epochs, device=args.device)
    else:
        if dataset == "CarRacing-v2":
            print("Softhebb training not implemented for CarRacing")
            exit(0)
        sh = SoftHebbTrain(model,
                           dataset_type=dataset,
                           bp_last_layer=args.bp_last_layer,
                           bp_learning_rate=args.bp_lr,
                           softhebb_lr=args.softhebb_lr,
                           aggregation_function=args.aggregation_function)
        
        if args.bp:
            sh.train_backprop(n_epochs=args.epochs, device=args.device)
        else:
            sh.train(n_epochs=args.epochs, device=args.device)
    
    
if __name__ == "__main__":
    main()
    