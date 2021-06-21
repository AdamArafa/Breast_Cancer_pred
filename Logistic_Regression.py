import numpy as np

class LogisticReg:
    def __init__(self, lr=0.001, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m_samples, n_features = X.shape # m_samples is number of samples in X_test and n_features is number of columns
        self.weights = np.zeros(n_features) # weights is a vector of zeros with 1*n_features dimension
        self.bias = 0

        # using gradient descent to calculate the best weights and bias
        for _ in range(self.num_iters):
            linear_model = np.dot(X, self.weights) + self.bias #approximate y(continues value) with the help of x, weights and bias
            y_pred = self.sigmoid(linear_model) #find the probability for a sample with the help of sigmoid method and liner_model

            dw = (1 / m_samples) * np.dot(X.T, (y_pred - y)) #finding partial derivative with respect to w
            db = (1 / m_samples) * np.sum(y_pred - y) #finding partial derivative with respect to b

            self.weights = self.weights - self.lr * dw #updating weights for every iteration
            self.bias = self.bias - self.lr * db #updating bias for every iteration

    #predict the class(y) of given sample/samples
    def predict(self, X):
        y_pred_classifiction = []
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model) #finding the probability of each sample in X
        for i in y_pred:
            if i >= 0.5: y_pred_classifiction.append(1)
            else: y_pred_classifiction.append(0)
        return np.array(y_pred_classifiction)

    # calculate a value between 0 and 1 (probability)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))




