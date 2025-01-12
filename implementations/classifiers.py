import numpy as np
from collections import Counter
from .exceptions import ModelAlreadyTrainedError, ModelNotTrainedError


class Classifier:
    def __init__(self):
        self._trained = False
        self._possible_classes = None

    def _get_probabilities_for_classes(self, classes: np.array) -> np.array:
        c = Counter(classes)
        return np.array([c.get(x, 0) / sum(c.values()) for x in self._possible_classes])

    def fit(self, data: np.array, classes: np.array, **kwargs) -> None:
        """
        Trains the model on the provided dataset and class labels.

        Parameters
        ----------
        data : np.array
            A 2D array of shape (n_samples, n_features) containing the training data.
        classes : np.array
            A 1D array of shape (n_samples,) containing the class labels for the training data.
        **kwargs : dict, optional
            Additional parameters for training.

        Raises
        ------
        ModelAlreadyTrainedError
            If the model has already been trained.
        """
        if self._trained:
            raise ModelAlreadyTrainedError("The model has been already trained and cannot be re-trained!")
        self._possible_classes = sorted(list(Counter(classes).keys()))

    def _predict_sample(self, data: np.array) -> np.array:
        """
        Internal method for predicting the class probabilities for a single sample.
        Parameters
        ----------
        data : np.array
            A 1D array of shape (n_features,) representing a single data sample.

        Returns
        -------
        np.array
            A 1D array of probabilities for each class.
        """
        pass

    def predict(self, data: np.ndarray) -> np.array:
        """
        Predicts the class probabilities for the given input data.

        Parameters
        ----------
        data : np.ndarray
            A 2D array of shape (n_samples, n_features) or a 1D array of shape (n_features,)
            containing the input data.

        Returns
        -------
        np.array
            If `data` is a 2D array, returns a 2D array of shape (n_samples, n_classes)
            with predicted probabilities for each class. If `data` is a 1D array,
            returns a 1D array of predicted probabilities for each class.

        Raises
        ------
        ModelNotTrainedError
            If the model has not been trained.
        """
        if not self._trained:
            raise ModelNotTrainedError("The prediction cannot be done due to the model not being trained!")

        if data.ndim == 1:
            return self._predict_sample(data)
        else:
            return np.array([self._predict_sample(sample) for sample in data])

    def evaluate(self, data: np.array, classes: np.array, verbose=0) -> float:
        """
        Evaluates the model on the provided dataset and class labels.

        Parameters
        ----------
        data : np.array
            A 2D array of shape (n_samples, n_features) containing the input data.
        classes : np.array
            A 1D array of shape (n_samples,) containing the true class labels.
        verbose : int, optional, default=0
            If set to 1, prints detailed prediction information for each sample.

        Returns
        -------
        float
            The accuracy of the model, calculated as the ratio of correctly predicted
            samples to the total number of samples.

        """
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

