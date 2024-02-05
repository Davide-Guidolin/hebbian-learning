from fitness import evaluate_classification
from data import DataManager
from unrolled_model import UnrolledModel

class SoftHebbTrain:
    def __init__(self, rolled_model, dataset_type="CIFAR10", bp_last_layer=True, bp_learning_rate=0.01, softhebb_lr=0.001):
        self.model = rolled_model
        
        self.dataset_type = dataset_type
        
        self.data = DataManager(self.dataset_type, batch_size=32)
        self.input_size = next(iter(self.data.train_loader))[0].shape[-1]
        
        self.unrolled_model = UnrolledModel(self.model, self.input_size)
        
        self.bp_last_layer = bp_last_layer
        self.bp_lr = bp_learning_rate
        self.softhebb_lr = softhebb_lr
        
    def train(self, n_epochs=10):
        m = self.unrolled_model.get_new_model()
        loader = self.data.get_new_loader()
        
        for i in range(n_epochs):
            epoch_acc = evaluate_classification(m, loader, bp_last_layer=self.bp_last_layer, bp_lr=self.bp_lr, softhebb_train=True, softhebb_lr=self.softhebb_lr)
            print(f"[{i}/{n_epochs}] Acc: {epoch_acc:.2f}")
        