from model import *
from evolution_strategy import EvolutionStrategy
import argparse

import wandb
# python3 run_exp.py --dataset CarRacing --population_size 4 --num_threads 1 --epochs 30

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', choices=['CIFAR10', 'CIFAR100', 'CarRacing', 'CarRacing-v2'], help='Dataset to use', default='CIFAR10')
    parser.add_argument('--population_size', type=int, default=10, help='Size of the population for the es algorithm')
    parser.add_argument('--num_threads', type=int, default=1, help='Number of threads to use, -1 to use all the cpus')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    
    parser.add_argument('--sigma', default=0.1, type=float, help='Sigma used in the ABCD parameters update factor')
    parser.add_argument('--abcd_perturbation_std', default=1, type=float, help='Standard deviation of the gaussian noise used to perturbate the parameters')
    parser.add_argument('--abcd_learning_rate', default=0.2, type=float, help='Learning rate of the ABCD parameters')
    parser.add_argument('--abcd_lr_decay', default=0.995, type=float, help='Decay factor for abcd_learning_rate')
    
    parser.add_argument('--bp_last_layer', action='store_true', help='Use backpropagation in the last layer')
    parser.add_argument('--bp_learning_rate', default=0.01, type=float, help='Learning rate to use for backpropagation')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    if dataset == 'CarRacing':
        dataset = 'CarRacing-v2'
    
    device = 'cpu'
    if dataset == 'CarRacing-v2':
        model = CNN_CarRacing()
    else:
        model = BaseNet2()
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="CarRacing_conv_unrolling_abcd",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": args.abcd_learning_rate,
        "population_size": args.population_size,
        "architecture": model,
        "dataset": dataset,
        "epochs": args.epochs,
        }
    )
    
    es = EvolutionStrategy(model,
                           dataset_type=dataset,
                           population_size=args.population_size,
                           num_threads=args.num_threads,
                           sigma=args.sigma,
                           abcd_perturbation_std=args.abcd_perturbation_std,
                           abcd_learning_rate=args.abcd_learning_rate,
                           abcd_lr_decay=args.abcd_lr_decay,
                           bp_last_layer=args.bp_last_layer,
                           bp_learning_rate=args.bp_learning_rate)
    
    es.run(iterations=args.epochs)
    
    
if __name__ == "__main__":
    main()
    