import numpy as np
from collections import Counter
from .exceptions import ModelAlreadyTrainedError, ModelNotTrainedError


class Classifier:
    def __init__(self):
        self._trained = False
        self._possible_classes = None

    def fit(self, data: np.array, classes: np.array, **kwargs) -> None:
        if self._trained:
            raise ModelAlreadyTrainedError("The model has been already trained and cannot be re-trained!")
        self._possible_classes = sorted(list(Counter(classes).keys()))

    def _predict_sample(self, data: np.array) -> np.array:
        pass

    def predict(self, data: np.ndarray) -> np.array:
        """
        Returns probability
        :param data:
        :return:
        """
        if not self._trained:
            raise ModelNotTrainedError("The prediction cannot be done due to the model not being trained!")

        if data.ndim == 1:
            return self._predict_sample(data)
        else:
            return np.array([self._predict_sample(sample) for sample in data])

    def evaluate(self, data: np.array, classes: np.array, verbose=0) -> float:
        samples = len(classes)
        positively_predicted = 0
        for i, subdataset in enumerate(zip(data, classes), 1):
            sample, actual_class = subdataset
            predicted_class = np.argmax(self.predict(sample))
            if predicted_class == actual_class:
                positively_predicted += 1
            if verbose == 1:
                print(f"Predicting {i:>2}/{samples}\tPrediction: {predicted_class}\tActual class: {actual_class}")
        accuracy = positively_predicted / samples
        if verbose == 1:
            print()
        print(f"Accuracy after predicting {samples} samples: {accuracy * 100:.2f}%")
        return accuracy
