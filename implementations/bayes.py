from collections import Counter
import numpy as np
from .classifiers import Classifier


class NaiveBayes(Classifier):
    def __init__(self):
        super().__init__()
        self._priors = {}
        self._likelihoods = {}

    def fit(self, data: np.array, classes: np.array, **kwargs) -> None:
        super().fit(data, classes, **kwargs)
        
        discrete_x = kwargs.get('discrete_x', False)
        if discrete_x:
            discrete_xtrain = data
        else:
            discrete_xtrain = self.data_discretization(data)

        labels = np.unique(classes)
        labels_count = Counter(classes)

        for label in labels:
            self._priors[label] = labels_count[label] / len(classes)

            label_dict = {}
            mask = classes == label
            for i in range(discrete_xtrain.shape[1]):
                attribute = discrete_xtrain[mask, i]
                attribute_counter = Counter(attribute)
                attribute_dict = {x: 0 for x in np.unique(discrete_xtrain)}
                for unique_value, number in attribute_counter.items():
                    attribute_dict[unique_value] = number / labels_count[label]

                label_dict[i] = attribute_dict
            self._likelihoods[label] = label_dict
        
        self._trained = True

    @staticmethod
    def data_discretization(data: np.ndarray) -> np.ndarray:
        intervals = 10

        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[1]):
            min_value = np.min(data[:, i])
            max_value = np.max(data[:, i])
            interval = (max_value - min_value) / intervals
            for j in range(data.shape[0]):
                discrete_array[j, i] = min((data[j, i] - min_value) // interval,
                                           intervals - 1)

        return discrete_array

    def _predict_sample(self, data: np.array) -> np.array:
        predictions = {}
        for label, prior in self._priors.items():
            value = 1
            for i, att_value in enumerate(data):
                try:
                    value *= self._likelihoods[label][i][att_value]
                except KeyError:
                    value *= 0
            predictions[label] = value * prior

        # return sorted(predictions.values(), key=predictions.get)
        return max(predictions, key=predictions.get)