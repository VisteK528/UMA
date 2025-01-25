"""
Filename: bayes.py
Authors:
    Piotr Patek, email: piotr.patek.stud@pw.edu.pl
    Jan Potaszyński, email: jan.potaszynski.stud@pw.edu.pl
Date: January 2025
Version: 1.0
Description:
    This file contains implementation of NaiveBayes model class.
Dependencies: collections, numpy
"""

from collections import Counter
import numpy as np
from .classifiers import Classifier


def assign_bin(value, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


class NaiveBayes(Classifier):
    """
    A Naive Bayes classifier.

    Parameters
    ----------
    discrete_x : bool, optional
        Whether the input features (`X`) are discrete.
        - `False` (default): Features are treated as continuous, and discretization will be applied.
        - `True`: All input features are treated as categorical/discrete.
    discretization_type : str or None, optional
        The type of discretization to apply to continuous features if `discrete_x` is `False`.
        - `None` (default): No discretization is applied.
        - `"uniform"`: Uniform discretization is applied.
        - `"percentile"`: Percentile-based discretization is applied.
    """
    def __init__(self, discrete_x=False, discretization_type=None):
        super().__init__()
        self._priors = {}
        self._likelihoods = {}
        self._discrete_intervals = []
        self._discrete_x = discrete_x
        self._discretization_type = discretization_type

    def fit(self, data: np.array, classes: np.array, **kwargs) -> None:
        super().fit(data, classes, **kwargs)

        if not self._discrete_x:
            if self._discretization_type == "percentile":
                data = self._percentile_discretization(data)
            else:
                data = self._uniform_discretization(data)

        labels = np.unique(classes)
        labels_count = Counter(classes)

        for label in labels:
            self._priors[label] = labels_count[label] / len(classes)

            label_dict = {}
            mask = classes == label
            for i in range(data.shape[1]):
                attribute = data[mask, i]
                attribute_counter = Counter(attribute)
                attribute_dict = {x: 0 for x in np.unique(data)}
                for unique_value, number in attribute_counter.items():
                    attribute_dict[unique_value] = number / labels_count[label]

                label_dict[i] = attribute_dict
            self._likelihoods[label] = label_dict

        self._trained = True


    def _uniform_discretization(self, data: np.ndarray, intervals=10) -> np.ndarray:
        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[1]):
            min_value = np.min(data[:, i])
            max_value = np.max(data[:, i])
            interval = (max_value - min_value) / intervals
            self._discrete_intervals.append((min_value, interval, intervals))
            for j in range(data.shape[0]):
                discrete_array[j, i] = min((data[j, i] - min_value) // interval,
                                           intervals - 1)

        return discrete_array

    def _percentile_discretization(self, data: np.ndarray) -> np.ndarray:
        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[1]):
            quantile_breakpoints = np.percentile(data[:, i], [x for x in range(1, 101)])

            # Add min and max to include all data
            bin_edges = np.concatenate(([-np.inf], quantile_breakpoints, [np.inf]))

            self._discrete_intervals.append(bin_edges)
            for j in range(data.shape[0]):
                output = assign_bin(data[j, i], bin_edges)
                discrete_array[j, i] = output

        return discrete_array

    def uniform_discretization_sample(self, data: np.ndarray) -> np.ndarray:
        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[0]):
            discrete_array[i] = min((data[i] - self._discrete_intervals[i][0]) // self._discrete_intervals[i][1],  self._discrete_intervals[i][2] - 1)

        return discrete_array

    def percentile_discretization_sample(self, data: np.ndarray) -> np.ndarray:
        discrete_array = np.zeros(data.shape)
        for i in range(data.shape[0]):
            discrete_array[i] = assign_bin(data[i], self._discrete_intervals[i])

        return discrete_array

    def _predict_sample(self, data: np.array) -> np.array:
        if not self._discrete_x:
            if self._discretization_type == "percentile":
                data = self.percentile_discretization_sample(data)
            else:
                data = self.uniform_discretization_sample(data)

        predictions = {}
        for label, prior in self._priors.items():
            value = 1
            for i, att_value in enumerate(data):
                try:
                    value *= self._likelihoods[label][i][att_value]
                except KeyError:
                    value *= 0
            predictions[label] = value*prior
        return np.array([predictions[k] for k in sorted(predictions)])