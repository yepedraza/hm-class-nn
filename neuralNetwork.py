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

    def TrainingOFF(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self._Initialize_params()

        i=0
        while i <= self.epochs:
            self._ForwardOFF(self.train_data.x_train)
            self._BackPropagationOFF(d = True)
            self._WeightAdjustOFF()
            self.errors.append(self._mse(self.train_data.y_train, self.params['A2']))
            i += 1

    def ValidateOFF(self):
        self._forward(self.train_data.x_test)
        return self.params['A2']

    def TrainingON(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self._Initialize_params()

        i=0
        while i <= self.epochs:
            A2 = self._ForwardON(self.train_data.x_train)
            self.errors.append(self._mse(self.train_data.y_train, A2))

    def _Initialize_params(self):
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

    ######################################## OFFLINE TRAINING (EACH EPOCH) ###################################################
    def _ForwardOFF(self, x_data):
        self.params['A0'] = x_data

        self.params['Z1'] = (self.params['A0']@self.params['W1']) + self.params['b1'] 
        self.params['A1'] = self._tanh(self.params['Z1']) 

        self.params['Z2'] = (self.params['A1']@self.params['W2']) + self.params['b2'] 
        self.params['A2'] = self._tanh(self.params['Z2']) 

    def _BackPropagationOFF(self, d):
            self.params['dZ2'] = self._mse(self.train_data.y_train, self.params['A2'], d) * self._tanh(self.params['A2'], d)
            self.params['dW2'] = self.params['A1'].T@self.params['dZ2']

            self.params['dZ1'] = self.params['dZ2']@self.params['W2'].T * self._tanh(self.params['A1'], d)
            self.params['dW1'] = self.params['A0'].T@self.params['dZ1']

    def _WeightAdjustOFF(self):
        self.params['W2'] = self.params['W2'] - self.params['dW2'] * self.alpha
        self.params['b2'] = self.params['b2'] - (np.mean(self.params['dZ2'], axis=0, keepdims=True)) * self.alpha

        self.params['W1'] = self.params['W1'] - self.params['dW1'] * self.alpha
        self.params['b1'] = self.params['b1'] - (np.mean(self.params['dZ1'], axis=0, keepdims=True)) * self.alpha

    ######################################## ONLINE TRAINING (EACH ITER) ###################################################
    def _ForwardON(self, x_data):
        A0 = x_data
        L = len(self.params['A0'])
        Z1, Z2, A1, A2= [], [], [], []

        for l in range(0, L-1):
            print(A0[l])
            Z1.append((A0[l]@self.params['W1']) + self.params['b1'])
            A1.append(self._tanh(Z1[l]))
            print(A1[l])
            Z2.append((A1[l]@self.params['W2']) + self.params['b1'])
            A2.append(self._tanh(Z2[l]))

            self._BackPropagationON(l, A1[l], A2[l], d = True)
        
        return A2

    def _BackPropagationON(self, l, A1, A2, d):

        Y = self.train_data.y_train

        dZ2 = self._mse(Y[l], A2, d) * self._tanh(A2,d)
        dW2 = A1@dZ2[l]

        dZ1= dZ2@self.params['W2'] * self._tanh(A1,d)
        dW1 = A1@dZ2[l]

        self._WeightAdjustON(dZ1,dZ2,dW1,dW2)

    def _WeightAdjustON(self,dZ1,dZ2,dW1,dW2):
        self.params['W2'] = self.params['W2'] + dW2 * self.alpha
        self.params['b2'] = self.params['b2'] - (np.mean(dZ2, axis=0, keepdims=True)) * self.alpha

        self.params['W1'] = self.params['W1'] + dW1 * self.alpha
        self.params['b1'] = self.params['b1'] - (np.mean(dZ1, axis=0, keepdims=True)) * self.alpha