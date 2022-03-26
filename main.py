import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles
from dataset import Dataset
from neuralNetwork import NeuralNetwork

def run():
    N  = 1000 
    gaussian_quantiles= make_gaussian_quantiles(mean=None, cov=0.1, n_samples = N, n_features=2,
                                                n_classes = 2, shuffle=True, random_state = None)

    X, Y = gaussian_quantiles
    Y = Y[:, np.newaxis]

    neural_network = NeuralNetwork(Dataset(X,Y), [2, 4, 1])

    neural_network.TrainingOFF(alpha=0.001, epochs=50)
    neural_network.errors.pop()
    plt.plot(neural_network.errors)


if __name__ == '__main__':
    run()