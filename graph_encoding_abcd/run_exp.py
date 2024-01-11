from model import *
from evolution_strategy import EvolutionStrategy

def main():
    device = 'cpu'
    model = CNNModel()
    
    # TODO add args
    es = EvolutionStrategy(model, population_size=3, num_threads=3)
    
    es.run(iterations=10)
    
    
if __name__ == "__main__":
    main()
    