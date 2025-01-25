import numpy as np
from uma24z_nbc_random_forest.random_forest import RandomForestClassifier
from uma24z_nbc_random_forest.experiments_utils import run_tests, evaluate_multiclass_tests

ATTEMPTS = 25
SAVE_IMAGES = True
SAVE_RESULTS = True

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/y.csv", dtype=float, delimiter=",")

    model = RandomForestClassifier(classifiers_number=50, tree_percentage=1.0)
    y_true_test, y_pred_test, y_true_train, y_pred_train, test_accuracies, train_accuracies, _, _ = run_tests(X, y, ATTEMPTS, model, verbose=1)
    evaluate_multiclass_tests(y_true_test, y_pred_test, test_accuracies, train_accuracies, ATTEMPTS, SAVE_IMAGES, SAVE_RESULTS,
                              "healthcare", "forest_id3", ["abnormal", "normal", "inconclusive"])