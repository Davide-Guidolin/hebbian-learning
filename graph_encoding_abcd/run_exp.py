from model import *
from evolution_strategy import EvolutionStrategy

def main():
    device = 'cpu'
    model = CNNModel()# BaseNet2()
    
    # TODO add args
    es = EvolutionStrategy(model, population_size=5, num_threads=3)
    
    es.run(iterations=20)
    
    
if __name__ == "__main__":
    main()
    