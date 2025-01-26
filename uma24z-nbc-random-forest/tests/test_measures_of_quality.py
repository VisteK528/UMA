import numpy as np
from uma24z_nbc_random_forest.utils import MeasuresOfQuality


def test_measures_of_quality_binary():
    y_true = np.array([0, 1, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([[0.67, 0.33], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0],
                       [0.4, 0.6], [0.6, 0.4], [0.6, 0.4]])

    qua = MeasuresOfQuality(y_pred, y_true)
    qua.compile()
    assert np.isclose(qua.accuracy(selected_class=1), 0.625)
    assert np.isclose(qua.true_positive_rate(selected_class=1), 0.6)
    assert np.isclose(qua.false_positive_rate(selected_class=1), 1/3)
    assert np.isclose(qua.precision(selected_class=1), 0.75)
