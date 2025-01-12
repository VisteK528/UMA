import numpy as np
from .id3 import DecisionTreeClassifier
from .bayes import NaiveBayes
from .classifiers import Classifier
from typing import Tuple
from collections import Counter
from joblib import Parallel, delayed


class RandomForestClassifier(Classifier):
    def __init__(self, classifiers_number: int):
        super().__init__()
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

    def fit(self, data: np.array, classes: np.array, **kwargs) -> None:
        super().fit(data, classes)

        tree_percentage = kwargs.get("tree_percentage", 1.0)
        trees_number = int(tree_percentage * self._classifiers_number)

        self._models.extend(
            Parallel(n_jobs=-1)(delayed(self._train_tree)(data, classes) for _ in range(trees_number))
        )

        self._models.extend(
            Parallel(n_jobs=-1)(
                delayed(self._train_naive_bayes)(data, classes) for _ in range(self._classifiers_number - trees_number))
        )
        self._trained = True

    def _train_tree(self, x, y):
        dc = DecisionTreeClassifier(1e10, random_forest_version=True)
        x_i, y_i = self.bootstrap(x, y)
        dc.fit(x_i, y_i)
        return dc

    def _train_naive_bayes(self, x, y):
        bc = NaiveBayes()
        x_i, y_i = self.bootstrap(x, y)
        bc.build_classifier(x_i, y_i, discrete_x=True)
        return bc

    def _predict_sample(self, data: np.array) -> np.array:
        if self._models is not None:
            predictions = [np.argmax(model.predict(data)) for model in self._models]
            return self._get_probabilities_for_classes(predictions)
