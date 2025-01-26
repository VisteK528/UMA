import numpy as np
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from uma24z_nbc_random_forest.id3 import DecisionTreeClassifier


ATTEMPTS = 10
SAVE_IMAGES = True

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/y.csv", dtype=float, delimiter=",")

    best_model = None
    best_model_score = 0.0
    worst_model = None
    worst_model_score = 100.0

    y_true = []
    y_pred_test = []
    test_accuracies = []
    train_accuracies = []

    depths = [0, 2, 4, 6, 8, 10]

    for depth in depths:
        print(f"Depth: {depth}")
        test_depth_accuracies = []
        train_depth_accuracies = []
        for attempt_nb in range(1, ATTEMPTS + 1):
            print(f"Attempt: {attempt_nb}")
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1)
            kFold = KFold(n_splits=5, shuffle=False)

            best_model_k_fold = None
            best_score_k_fold = 0.0
            for train_indices, validate_indices in kFold.split(X_train, Y_train):
                x_train = X_train[train_indices]
                y_train = Y_train[train_indices]
                x_val = X_train[validate_indices]
                y_val = Y_train[validate_indices]

                id3 = DecisionTreeClassifier(max_depth=depth)
                time1 = time.time()
                id3.fit(x_train, y_train, discrete_x=True)
                print(f"Training took: {time.time() - time1:.3f} s")
                acc = id3.evaluate(x_val, y_val)

                if acc > best_score_k_fold:
                    best_model_k_fold = id3
                    best_score_k_fold = acc

            # Evaluate best model on test data
            print(f"Best model accuracy on validation data: {best_score_k_fold * 100:.2f}")
            best_k_fold_accuracy = best_model_k_fold.evaluate(X_test, Y_test)
            best_k_fold_accuracy_train = best_model_k_fold.evaluate(X_train, Y_train)

            test_depth_accuracies.append(best_k_fold_accuracy)
            train_depth_accuracies.append(best_k_fold_accuracy_train)

        test_accuracies.append(test_depth_accuracies)
        train_accuracies.append(train_depth_accuracies)

    test_accuracies = np.array(test_accuracies) * 100
    train_accuracies = np.array(train_accuracies) * 100

    means_accuracies = [np.mean(depth_accuracies) for depth_accuracies in test_accuracies]
    min_accuracies = [np.min(depth_accuracies) for depth_accuracies in test_accuracies]
    max_accuracies = [np.max(depth_accuracies) for depth_accuracies in test_accuracies]

    train_means_accuracies = [np.mean(depth_accuracies) for depth_accuracies in train_accuracies]
    train_min_accuracies = [np.min(depth_accuracies) for depth_accuracies in train_accuracies]
    train_max_accuracies = [np.max(depth_accuracies) for depth_accuracies in train_accuracies]

    plt.plot(depths, train_means_accuracies, marker="o", color="red", label="Train mean accuracy", linewidth=2)
    plt.fill_between(depths, train_min_accuracies, train_max_accuracies, color='red', alpha=0.2)
    for x, y in zip(depths, train_means_accuracies):
        label = f"{y:.2f}%"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 14), ha='center')

    plt.plot(depths, means_accuracies, marker="o", color="blue", label="Test mean accuracy", linewidth=2)
    plt.fill_between(depths, min_accuracies, max_accuracies, color='blue', alpha=0.2)

    for x, y in zip(depths, means_accuracies):
        label = f"{y:.2f}%"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 4), ha='center')

    plt.xlabel("Tree Depth", fontsize=12)
    plt.ylabel("Accuracy [%]", fontsize=12)
    plt.legend()
    plt.grid()

    if SAVE_IMAGES:
        plt.savefig("images/healthcare_id3_select_depth.pdf", dpi=300)


