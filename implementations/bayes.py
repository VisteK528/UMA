from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}

    def build_classifier(self, train_features, train_classes, discrete_x=False):
        if discrete_x:
            discrete_xtrain = train_features
        else:
            discrete_xtrain = self.data_discretization(train_features)

        labels = np.unique(train_classes)
        labels_count = Counter(train_classes)

        for label in labels:
            self.priors[label] = labels_count[label] / len(train_classes)

            label_dict = {}
            mask = train_classes == label
            for i in range(discrete_xtrain.shape[1]):
                attribute = discrete_xtrain[mask, i]
                attribute_counter = Counter(attribute)
                attribute_dict = {x: 0 for x in np.unique(discrete_xtrain)}
                for unique_value, number in attribute_counter.items():
                    attribute_dict[unique_value] = number / labels_count[label]

                label_dict[i] = attribute_dict
            self.likelihoods[label] = label_dict

    @staticmethod
    def data_discretization(data: np.ndarray) -> np.ndarray:
        intervals = 4

        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[1]):
            min_value = np.min(data[:, i])
            max_value = np.max(data[:, i])
            interval = (max_value - min_value) / intervals
            for j in range(data.shape[0]):
                discrete_array[j, i] = min((data[j, i] - min_value) // interval,
                                           intervals - 1)

        return discrete_array

    def predict(self, sample):
        predictions = {}
        for label, prior in self.priors.items():
            value = 1
            for i, att_value in enumerate(sample):
                try:
                    value *= self.likelihoods[label][i][att_value]
                except KeyError:
                    value *= 0
            predictions[label] = value

        return max(predictions, key=predictions.get)