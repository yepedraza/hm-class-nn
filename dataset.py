class Dataset:
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train