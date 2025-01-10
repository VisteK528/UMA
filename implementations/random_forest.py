import numpy as np
from .id3 import DecisionTreeClassifier
from .bayes import NaiveBayes
from typing import Tuple
from collections import Counter
from joblib import Parallel, delayed


class RandomForestClassifier:
    def __init__(self, classifiers_number: int):
        self._classifiers_number = classifiers_number
        self._models = []

    @staticmethod
    def bootstrap(x: np.ndarray, y: np.ndarray) -> Tuple[np.array, np.array]:
        assert len(x) == len(y)
        N = len(x)
        new_x = []
        new_y = []

        for _ in range(N):
            i = np.random.randint(N)
            new_x.append(x[i])
            new_y.append(y[i])

        return np.array(new_x), np.array(new_y)

    def train_random_forest(self, x: np.ndarray, y: np.ndarray, tree_percentage=1.0):
        trees_number = int(tree_percentage * self._classifiers_number)

        self._models.extend(
            Parallel(n_jobs=-1)(delayed(self._train_tree)(x, y) for _ in range(trees_number))
        )

        self._models.extend(
            Parallel(n_jobs=-1)(
                delayed(self._train_naive_bayes)(x, y) for _ in range(self._classifiers_number - trees_number))
        )

    def _train_tree(self, x, y):
        dc = DecisionTreeClassifier(1e10, random_forest_version=False)
        x_i, y_i = self.bootstrap(x, y)
        dc.fit(x_i, y_i)
        return dc

    def _train_naive_bayes(self, x, y):
        bc = NaiveBayes()
        x_i, y_i = self.bootstrap(x, y)
        bc.build_classifier(x_i, y_i, discrete_x=True)
        return bc

    # def train_random_forest(self, x: np.ndarray, y: np.ndarray, tree_percentage=1.0):
    #     trees_number = int(tree_percentage * self._classifiers_number)
    #
    #     for _ in range(trees_number):
    #         dc = DecisionTreeClassifier(1e10, random_forest_version=False)
    #
    #         x_i, y_i = self.bootstrap(x, y)
    #         dc.fit(x_i, y_i)
    #         self._models.append(dc)
    #
    #     for _ in range(self._classifiers_number - trees_number):
    #         bc = NaiveBayes()
    #         x_i, y_i = self.bootstrap(x, y)
    #
    #         bc.build_classifier(x_i, y_i, discrete_x=True)
    #         self._models.append(bc)

    def predict(self, data: np.ndarray) -> int:
        if self._models is not None:
            predictions = [model.predict(data) for model in self._models]
            c = Counter(predictions)
            return c.most_common(1)[0][0]

    def evaluate(self, data: np.ndarray, classes: np.ndarray, verbose=0) -> float:
        if self._models is not None:
            samples = len(classes)
            positively_predicted = 0
            for i, subdataset in enumerate(zip(data, classes), 1):
                sample, actual_class = subdataset
                predicted_class = self.predict(sample)
                if predicted_class == actual_class:
                    positively_predicted += 1
                if verbose == 1:
                    print(f"Predicting {i:>2}/{samples}\tPrediction: {predicted_class}\tActual class: {actual_class}")
            accuracy = positively_predicted / samples
            if verbose == 1:
                print()
            print(f"Accuracy after predicting {samples} samples: {accuracy * 100:.2f}%")
            return accuracy
