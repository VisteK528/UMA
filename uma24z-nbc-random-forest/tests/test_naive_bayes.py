from uma24z_nbc_random_forest.bayes import NaiveBayes
import numpy as np

def test_initialization():
    nb = NaiveBayes(discrete_x=True, discretization_type="uniform")
    assert nb._discrete_x is True
    assert nb._discretization_type == "uniform"
    assert nb._priors == {}
    assert nb._likelihoods == {}
    assert nb._discrete_intervals == []


def test_uniform_discretization():
    data = np.array([[1, 10], [2, 20], [3, 30]])
    nb = NaiveBayes(discrete_x=False, discretization_type="uniform")
    discrete_data = nb._uniform_discretization(data)
    assert discrete_data.shape == data.shape
    assert np.all(discrete_data >= 0)
    assert len(nb._discrete_intervals) == data.shape[1]


def test_percentile_discretization():
    data = np.array([[1, 10], [2, 20], [3, 30]])
    nb = NaiveBayes(discrete_x=False, discretization_type="percentile")
    discrete_data = nb._percentile_discretization(data)
    assert discrete_data.shape == data.shape
    assert len(nb._discrete_intervals) == data.shape[1]
    assert all(len(edges) > 0 for edges in nb._discrete_intervals)


def test_fit():
    data = np.array([[1], [1], [2], [2], [3]])
    classes = np.array([0, 0, 1, 1, 1])
    nb = NaiveBayes(discrete_x=True)
    nb.fit(data, classes)
    assert nb._priors[0] == 0.4
    assert nb._priors[1] == 0.6
    assert nb._likelihoods[1][0][2] == 2 / 3
