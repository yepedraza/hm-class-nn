from dataset import Dataset
import numpy as np

class NeuralNetwork(Dataset):
    layers_dim = []
    params = {}
    errors = []
    epochs = int
    lr = float
    training = True

    def __init__(self, x_train, y_train, layers_dim):
        super().__init__(x_train, y_train)
        self.layers_dim = layers_dim

    def initialize_params(self):
        L = len(self.layers_dim)
        for l in range(0, L-1):
            self.params['W' + str(l+1)] = (np.random.rand(self.layers_dim[l],self.layers_dim[l+1]) * 2 ) - 1
            self.params['b' + str(l+1)] =  (np.random.rand(1,self.layers_dim[l+1]) * 2 ) - 1