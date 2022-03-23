from dataset import Dataset
import numpy as np

class NeuralNetwork:
    layers_dim = []
    params = {}
    errors = []
    epochs = int 
    alpha = float #Learning rate
    training = True
    train_data = Dataset([],[])

    def __init__(self, train_data, layers_dim):
        self.train_data = train_data
        self.layers_dim = layers_dim

    def initialize_params(self):
        L = len(self.layers_dim)
        for l in range(0, L-1):
            self.params['W' + str(l+1)] = (np.random.rand(self.layers_dim[l],self.layers_dim[l+1]) * 2 ) - 1
            self.params['b' + str(l+1)] =  (np.random.rand(1,self.layers_dim[l+1]) * 2 ) - 1

    def tanh(self, x, d=False):
        if d:
            return 4/(np.exp(x)+np.exp(-x))**2
        else:
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def forward_training(self):
        self.params['A0'] = self.train_data.x_train

        self.params['Z1'] = (self.params['A0']@self.params['W1']) + self.params['b1'] 
        self.params['A1'] = self.tanh(self.params['Z1']) 

        self.params['Z2'] = (self.params['A1']@self.params['W2']) + self.params['b2'] 
        self.params['A2'] = self.tanh(self.params['Z2']) 
