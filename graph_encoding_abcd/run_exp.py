from model import *
from evolution_strategy import EvolutionStrategy

def main():
    device = 'cpu'
    model = BaseNet2()# BaseNet2()
    
    # TODO add args
    es = EvolutionStrategy(model, population_size=1, num_threads=1)
    
    es.run(iterations=30)
    
    
if __name__ == "__main__":
    main()
    