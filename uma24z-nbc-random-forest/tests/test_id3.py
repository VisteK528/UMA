import numpy as np
import pytest
from uma24z_nbc_random_forest.id3 import DecisionTreeClassifier, entropy_func, split
from uma24z_nbc_random_forest.exceptions import ModelNotTrainedError, ModelAlreadyTrainedError


def test_decision_tree_initialization():
    id3 = DecisionTreeClassifier(10)
    assert id3._tree is None
    assert id3._trained == False


def test_model_not_trained_error():
    id3 = DecisionTreeClassifier(10)
    X_test = np.array([[1], [2], [3]])

    with pytest.raises(ModelNotTrainedError):
        id3.predict(X_test)


def test_entropy_func_for_one_class():
    samples = 10
    class_0_samples = 5
    assert np.isclose(entropy_func(class_0_samples, samples), 0.5)


def test_entropy_func_for_one_class2():
    samples = 10
    class_0_samples = 10
    assert np.isclose(entropy_func(class_0_samples, samples), 0)


def test_fit_decision_tree():
    id3 = DecisionTreeClassifier(10)
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    id3.fit(X, y)

    with pytest.raises(ModelAlreadyTrainedError):
        id3.fit(X, y)


data = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

classes = np.array([0, 1, 0, 1])


def test_split_basic_case():
    split_feature = 0
    split_val = 4
    child_a, child_b = split(data, classes, split_feature, split_val)

    expected_a = np.array([[5, 6, 0], [7, 8, 1]])
    expected_b = np.array([[1, 2, 0], [3, 4, 1]])

    assert np.array_equal(child_a, expected_a)
    assert np.array_equal(child_b, expected_b)


def test_split_all_greater():
    split_feature = 1
    split_val = 1
    child_a, child_b = split(data, classes, split_feature, split_val)

    assert len(child_a) == len(data)
    assert len(child_b) == 0


def test_split_all_smaller():
    split_feature = 1
    split_val = 9
    child_a, child_b = split(data, classes, split_feature, split_val)

    assert len(child_a) == 0
    assert len(child_b) == len(data)


def test_split_with_equal_values():
    data_equal = np.array([
        [2, 2],
        [2, 2],
        [2, 2]
    ])
    classes_equal = np.array([0, 1, 0])

    split_feature = 0
    split_val = 2

    child_a, child_b = split(data_equal, classes_equal, split_feature, split_val)

    assert len(child_a) == 3
    assert len(child_b) == 0


def test_split_large_dataset():
    large_data = np.random.rand(1000, 10)
    large_classes = np.random.randint(0, 2, size=1000)
    split_feature = 5
    split_val = 0.5

    child_a, child_b = split(large_data, large_classes, split_feature, split_val)

    assert np.all(child_a[:, split_feature] >= split_val)
    assert np.all(child_b[:, split_feature] < split_val)