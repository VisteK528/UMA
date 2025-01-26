import numpy as np
from uma24z_nbc_random_forest.utils import MeasuresOfQuality
from sklearn.metrics import precision_score

y_true = np.array([0, 1, 1, 1, 0, 0, 1, 1])
y_pred = np.array([[0.67, 0.33], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.4, 0.6], [0.6, 0.4], [0.6, 0.4]])

qua = MeasuresOfQuality(y_pred, y_true)
qua.compile()
print(qua._fp[0])
print(qua._fn[0])
print(qua._tn[0])
print(qua._tp[0])
print(qua.get_confusion_matrix(selected_class=0))
print(qua.accuracy(selected_class=0))
print(qua.true_positive_rate(selected_class=0))
print(qua.false_positive_rate(selected_class=0))
print(qua.precision(selected_class=0))

print(precision_score(y_true, np.argmax(y_pred, axis=1)))