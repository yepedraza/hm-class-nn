from turtle import forward
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

    def training(self):
        self._initialize_params()
        self._forward()
        self._back_propagation(d = True)
        self._weight_adjust()

    def _initialize_params(self):
        L = len(self.layers_dim)
        for l in range(0, L-1):
            self.params['W' + str(l+1)] = (np.random.rand(self.layers_dim[l],self.layers_dim[l+1]) * 2 ) - 1
            self.params['b' + str(l+1)] =  (np.random.rand(1,self.layers_dim[l+1]) * 2 ) - 1


    def _tanh(self, x, d=False):
        if d:
            return 4/(np.exp(x)+np.exp(-x))**2
        else:
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def _mse(self, y, y_hat, d = False):
        if d:
            return y_hat-y
        else:
            return np.mean((y_hat - y)**2)

    def _forward(self):
        self.params['A0'] = self.train_data.x_train

        self.params['Z1'] = (self.params['A0']@self.params['W1']) + self.params['b1'] 
        self.params['A1'] = self._tanh(self.params['Z1']) 

        self.params['Z2'] = (self.params['A1']@self.params['W2']) + self.params['b2'] 
        self.params['A2'] = self._tanh(self.params['Z2']) 

    def _back_propagation(self, d):
            self.params['dZ2'] = self._mse(self.train_data.y_train, self.params['A2'], d) * self._tanh(self.params['A2'], d)
            self.params['dW2'] = self.params['A1'].T@self.params['dZ2']

            self.params['dZ1'] = self.params['dZ2']@self.params['W2'].T * self._tanh(self.params['A1'], d)
            self.params['dW1'] = self.params['A0'].T@self.params['dZ1']

    def _weight_adjust(self):
        self.params['W2'] = self.params['W2'] - self.params['dW2'] * self.alpha
        self.params['b2'] = self.params['b2'] - (np.mean(self.params['dZ2'], axis=0, keepdims=True)) * self.alpha

        self.params['W1'] = self.params['W1'] - self.params['dW1'] * self.alpha
        self.params['b1'] = self.params['b1'] - (np.mean(self.params['dZ1'], axis=0, keepdims=True)) * self.alpha