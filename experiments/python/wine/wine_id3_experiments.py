import numpy as np
from uma24z_nbc_random_forest.id3 import DecisionTreeClassifier
from uma24z_nbc_random_forest.experiments_utils import run_tests, evaluate_binary_tests

ATTEMPTS = 25
SAVE_IMAGES = True
DEPTH = 10

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/wine/simple_processing/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/wine/simple_processing/y.csv", dtype=float, delimiter=",")

    model = DecisionTreeClassifier(max_depth=DEPTH)
    y_true_test, y_pred_test, y_true_train, y_pred_train, test_accuracies, train_accuracies, _, _ = run_tests(X, y, ATTEMPTS, model, verbose=1)
    evaluate_binary_tests(y_true_test, y_pred_test, test_accuracies, train_accuracies, ATTEMPTS, SAVE_IMAGES,
                              "wine", "id3", ["bad", "good"])