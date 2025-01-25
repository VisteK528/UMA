import pytest
import numpy as np
from uma24z_nbc_random_forest.classifiers import Classifier
from uma24z_nbc_random_forest.exceptions import ModelNotTrainedError


def test_invalid_input_fit_x():
    x = [[], []]
    y = [0, 1]

    model = Classifier()
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_invalid_input_fit_y():
    x = [[1], [2]]
    y = []

    model = Classifier()
    with pytest.raises(ValueError):
        model.fit(x, y)


def test_invalid_input_evaluate_x():
    x = [[], []]
    y = [0, 1]

    model = Classifier()
    with pytest.raises(ValueError):
        model.evaluate(x, y)


def test_invalid_input_evaluate_y():
    x = [[1], [2]]
    y = []

    model = Classifier()
    with pytest.raises(ValueError):
        model.evaluate(x, y)


def test_model_not_trained_error():
    model = Classifier()
    X_test = np.array([[1], [2], [3]])

    with pytest.raises(ModelNotTrainedError):
        model.predict(X_test)