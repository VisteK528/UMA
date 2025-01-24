import numpy as np
from implementations.random_forest import RandomForestClassifier
from implementations.experiments_utils import run_tests, evaluate_multiclass_tests

ATTEMPTS = 25
SAVE_IMAGES = True

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/y.csv", dtype=float, delimiter=",")

    model = RandomForestClassifier(classifiers_number=50, tree_percentage=0.5, discrete_x=True)
    y_true_test, y_pred_test, y_true_train, y_pred_train, test_accuracies, train_accuracies, _, _ = run_tests(X, y, ATTEMPTS, model, verbose=1)
    evaluate_multiclass_tests(y_true_test, y_pred_test, test_accuracies, train_accuracies, ATTEMPTS, SAVE_IMAGES,
                              "healthcare", "forest_id3_bayes", ["abnormal", "normal", "inconclusive"])