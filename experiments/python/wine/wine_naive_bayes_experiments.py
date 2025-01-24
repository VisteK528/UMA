import numpy as np
from implementations.bayes import NaiveBayes
from implementations.experiments_utils import run_tests, evaluate_binary_tests

ATTEMPTS = 50
SAVE_IMAGES = True

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/wine/simple_processing/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/wine/simple_processing/y.csv", dtype=float, delimiter=",")

    model = NaiveBayes(discrete_x=False, discretization_type="percentile")
    y_true_test, y_pred_test, y_true_train, y_pred_train, test_accuracies, train_accuracies, _, _ = run_tests(X, y, ATTEMPTS, model, verbose=1)
    evaluate_binary_tests(y_true_test, y_pred_test, test_accuracies, train_accuracies, ATTEMPTS, SAVE_IMAGES,
                              "wine", "naive_bayes", ["bad", "good"])