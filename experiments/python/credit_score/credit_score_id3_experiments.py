import numpy as np
from uma24z_nbc_random_forest.id3 import DecisionTreeClassifier
from uma24z_nbc_random_forest.experiments_utils import run_tests, evaluate_multiclass_tests

DEPTH = 8
ATTEMPTS = 25
SAVE_IMAGES = True
SAVE_RESULTS = True

if __name__ == "__main__":
    X = np.genfromtxt("../../../data_processed/credit_score/percentiles/X.csv", dtype=float, delimiter=",")
    l = len(X[0])
    xx = np.zeros([X.shape[0], l])
    for i in range(X.shape[0]):
        for j in range(l):
            xx[i, j] = X[i][j]
    X = xx
    y = np.loadtxt("../../../data_processed/credit_score/percentiles/y.csv", dtype=float, delimiter=",")

    model = DecisionTreeClassifier(max_depth=DEPTH)
    y_true_test, y_pred_test, y_true_train, y_pred_train, test_accuracies, train_accuracies, _, _ = run_tests(X, y,
                                                                                                              ATTEMPTS,
                                                                                                              model,
                                                                                                              verbose=1)
    evaluate_multiclass_tests(y_true_test, y_pred_test, test_accuracies, train_accuracies, ATTEMPTS, SAVE_IMAGES, SAVE_RESULTS,
                              "credit_score", "id3", ["poor", "standard", "good"])


