from model import *
from evolution_strategy import EvolutionStrategy

def main():
    device = 'cpu'
    model = CNN_CarRacing()
    
    # TODO add args
    es = EvolutionStrategy(model, dataset_type="CarRacing-v2", population_size=4, num_threads=2)
    
    es.run(iterations=30)
    
    
if __name__ == "__main__":
    main()
    