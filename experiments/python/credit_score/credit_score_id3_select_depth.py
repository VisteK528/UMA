import numpy as np
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from implementations.id3 import DecisionTreeClassifier



ATTEMPTS = 25
SAVE_IMAGES = True

if __name__ == "__main__":
    X = np.genfromtxt("../../../data_processed/credit_score/percentiles/X.csv", dtype=float, delimiter=",")
    l = len(X[0])
    xx = np.zeros([X.shape[0], l])
    for i in range(X.shape[0]):
        for j in range(l):
            xx[i, j] = X[i][j]
    X = xx
    y = np.loadtxt("../../../data_processed/credit_score/percentiles/y.csv", dtype=float, delimiter=",")

    best_model = None
    best_model_score = 0.0
    worst_model = None
    worst_model_score = 100.0

    y_true = []
    y_pred_test = []
    accuracies = []

    depths = [2, 4, 6, 8, 10]

    for depth in depths:
        print(f"Depth: {depth}")
        depth_accuracies = []
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
            depth_accuracies.append(best_k_fold_accuracy)

        accuracies.append(depth_accuracies)

    accuracies = np.array(accuracies) * 100

    means_accuracies = [np.mean(depth_accuracies)for depth_accuracies in accuracies]
    min_accuracies = [np.min(depth_accuracies) for depth_accuracies in accuracies]
    max_accuracies = [np.max(depth_accuracies) for depth_accuracies in accuracies]



    # Overlay the line plot for the means
    plt.plot(depths, means_accuracies, marker="o", color="blue", label="Mean Accuracy", linewidth=2)
    plt.fill_between(depths, min_accuracies, max_accuracies, color='blue', alpha=0.2)

    for x, y in zip(depths, means_accuracies):
        label = f"{y:.2f}%"
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')


    plt.xlabel("Depth", fontsize=12)
    plt.ylabel("Accuracy [%]", fontsize=12)
    plt.ylim(0, 100)
    plt.grid()

    if SAVE_IMAGES:
        plt.savefig("images/credit_score_id3_select_depth.pdf", dpi=300)


