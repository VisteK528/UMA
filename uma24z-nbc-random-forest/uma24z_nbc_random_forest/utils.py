"""
Filename: utils.py
Authors:
    Piotr Patek, email: piotr.patek.stud@pw.edu.pl
    Jan Potaszyński, email: jan.potaszynski.stud@pw.edu.pl
Date: January 2025
Version: 1.0
Description:
    This file contains MeasuresOfQuality class which is used to calculate such measures as accuracy, error,
     true positive rate etc. as well as support functions used for import and export of the models.
Dependencies: numpy, pickle
"""

import numpy as np
from .exceptions import ClassNotExistingError, MeasuresOfQualityNotCompiledError
from .classifiers import Classifier
import pickle

def store_model(model: Classifier, filename: str):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def load_model(filename: str) -> 'Classifier':
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model

class MeasuresOfQuality:
    def __init__(self, y_pred: np.array, y_true: np.array):
        self._y_pred = y_pred
        self._y_true = y_true

        self._tp = None
        self._tn = None
        self._fp = None
        self._fn = None

        self._cm = None

        self._compiled = False
        self._unique_classes = None
        self._unique_classes_dict = None

    def compile(self) -> None:
        self._unique_classes = np.sort(np.unique(self._y_true))
        self._unique_classes_dict = {x: i for i, x in enumerate(self._unique_classes)}
        N = len(self._unique_classes)

        self._cm = np.zeros((N, N))
        for y_pred, y_true in zip(self._y_pred, self._y_true):
            y_pred = np.argmax(y_pred)
            i, j = self._unique_classes_dict.get(y_pred), self._unique_classes_dict.get(y_true)
            self._cm[i, j] += 1

        self._tp = np.zeros((N,))
        self._tn = np.zeros((N,))
        self._fp = np.zeros((N,))
        self._fn = np.zeros((N,))
        for i, unique_class in enumerate(self._unique_classes):
            self._tp[i] = self._cm[i, i]
            self._fp[i] = sum([self._cm[i, k] for k in range(N) if k != i])
            self._fn[i] = sum([self._cm[k, i] for k in range(N) if k != i])
            self._tn[i] = np.sum(self._cm) - self._tp[i] - self._fp[i] - self._fn[i]

        self._compiled = True

    def _prepare_for_measures(self, selected_class=None) -> list[int]:
        if not self._compiled:
            raise MeasuresOfQualityNotCompiledError("Desired measure of quality cannot be computed, because the confusion matrix has not been computed!")

        if selected_class is not None:
            i = self._unique_classes_dict.get(selected_class, None)

            if i is None:
                raise ClassNotExistingError("Selected class does not exist in the dataset!")

            return [self._tp[i], self._tn[i], self._fp[i], self._fn[i]]

        return [sum(self._tp), sum(self._tn), sum(self._fp), sum(self._fn)]

    def accuracy(self, selected_class=None) -> float:
        tp, tn, fp, fn = self._prepare_for_measures(selected_class)
        return (tp + tn) / (tp + tn + fp + fn)

    def error(self, selected_class=None):
        tp, tn, fp, fn = self._prepare_for_measures(selected_class)
        return (fp + fn) / (tp + tn + fp + fn)

    def true_positive_rate(self, selected_class=None):
        tp, _, _, fn = self._prepare_for_measures(selected_class)
        return tp / (tp + fn)

    def false_positive_rate(self, selected_class=None):
        _, tn, fp, _ = self._prepare_for_measures(selected_class)
        return fp / (tn + fp)

    def precision(self, selected_class=None):
        tp, _, fp, _ = self._prepare_for_measures(selected_class)
        return tp / (tp + fp)

    def get_confusion_matrix(self, selected_class=None):
        if self._cm is not None:
            if selected_class is None:
                return self._cm
            else:
                elements = self._prepare_for_measures(selected_class)
                return np.array([[elements[0], elements[2]], [elements[3], elements[1]]])
