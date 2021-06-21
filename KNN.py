import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=5):
        self.k = k

    # just feeding the training data to the model. nothing more
    def fit(self, X, y):
        self.X = X
        self.y = y

    # predict the y for every sample in the X_test
    def predict(self, X):
        y_predicted = []
        for x in X:
            y_predicted.append(self.predict_helper(x))
        return np.array(y_predicted)

    def predict_helper(self, x):
        distances = []
        labels = []

        # calculate the distance between a given sample and all the samples in the training dataset
        for sample in self.X:
            dist_to_points = self.distance(x, sample)
            distances.append(dist_to_points)

        # sort the distance list by indexes/locations of the smallest distances
        # and assign just the first "k" elements(distances) of the list to k_indices
        k_indices = np.argsort(distances)[:self.k]

        # get labels from y_train for k neighbours using k_indices as indexes
        for j in k_indices:
            labels.append(self.y[j])

        # we get a list with 1 tuple, first element in the tuple is the most common class
        # and second element is the number of how many times it occurs
        most_common = Counter(labels).most_common(1)
        return most_common[0][0] # just get the first element of the tuple (the class 0 or 1)

    #calculating the distance between 2 samples using the Euclidean formula
    def distance(self, sample1, sample2):
        dist = np.sqrt(np.sum((sample1 - sample2)**2))
        return dist


