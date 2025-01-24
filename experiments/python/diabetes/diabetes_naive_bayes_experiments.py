import numpy as np
from uma24z_nbc_random_forest.bayes import NaiveBayes
from uma24z_nbc_random_forest.experiments_utils import run_tests, evaluate_binary_tests

ATTEMPTS = 50
SAVE_IMAGES = True

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/diabetes/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/diabetes/y.csv", dtype=float, delimiter=",")

    model = NaiveBayes()
    y_true_test, y_pred_test, y_true_train, y_pred_train, test_accuracies, train_accuracies, _, _ = run_tests(X, y, ATTEMPTS, model, verbose=1)
    evaluate_binary_tests(y_true_test, y_pred_test, test_accuracies, train_accuracies, ATTEMPTS, SAVE_IMAGES,
                              "diabetes", "naive_bayes", ["non-diabetic", "diabetic"])