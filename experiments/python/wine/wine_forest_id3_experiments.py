import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from implementations.bayes import NaiveBayes
from implementations.random_forest import RandomForestClassifier
from implementations.utils import MeasuresOfQuality

ATTEMPTS = 50
SAVE_IMAGES = True


def ovr_cm(y_true_ovr, y_pred_ovr, selected_class, selected_class_str, attempts, save_fig, roc_filename=None):
    y_true_ovr_filtered = np.array([x[:, selected_class] for x in y_true_ovr]).flatten()
    y_pred_ovr_filtered = np.array([x[:, selected_class] for x in y_pred_ovr]).flatten()

    cm = confusion_matrix(y_true_ovr_filtered, y_pred_ovr_filtered, labels=[0, 1])
    cm = cm // attempts
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["other", selected_class_str])
    disp.plot(cmap=plt.cm.Blues)

    if save_fig:
        plt.savefig(roc_filename, dpi=300)


def ovr_test(y_true_ovr, y_pred_ovr, selected_class, save_fig, roc_filename=None):
    precision_list = []
    tpr_list = []
    fpr_list = []

    tpr_list_0 = []
    fpr_list_0 = []
    auc_list_0 = []

    for y_true_ovr_batch, y_pred_ovr_batch in zip(y_true_ovr, y_pred_ovr):
        fpr0, tpr0, _ = roc_curve(y_true_ovr_batch[:, selected_class], y_pred_ovr_batch[:, selected_class])
        roc_auc0 = auc(fpr0, tpr0)

        measures = MeasuresOfQuality(y_pred_ovr_batch[:, selected_class], y_true_ovr_batch[:, selected_class])
        measures.compile()
        precision_list.append(measures.precision(1))
        tpr_list.append(measures.true_positive_rate(1))
        fpr_list.append(measures.false_positive_rate(1))

        fpr_list_0.append(fpr0)
        tpr_list_0.append(tpr0)
        auc_list_0.append(roc_auc0)

    # Define a common set of FPR values
    mean_fpr0 = np.linspace(0, 1, 100)

    # Interpolate TPR values
    tpr_interp0 = [np.interp(mean_fpr0, fpr0, tpr0) for fpr0, tpr0 in zip(fpr_list_0, tpr_list_0)]

    # Average the interpolated TPR values
    max_tpr0 = np.max(tpr_interp0, axis=0)
    min_tpr0 = np.min(tpr_interp0, axis=0)
    mean_tpr0 = np.mean(tpr_interp0, axis=0)

    # Compute the mean AUC
    mean_auc0 = np.mean(auc_list_0)
    # Plot ROC curve
    plt.figure()
    plt.fill_between(mean_fpr0, min_tpr0, max_tpr0, color='blue', alpha=0.2)
    plt.plot(mean_fpr0, mean_tpr0, color='blue', lw=2, label=f'ROC curve (AUC = {mean_auc0:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()

    if save_fig:
        plt.savefig(roc_filename, dpi=300)

    return np.array(precision_list), np.array(tpr_list), np.array(fpr_list)


if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/wine/simple_processing/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/wine/simple_processing/y.csv", dtype=float, delimiter=",")

    best_model = None
    best_model_score = 0.0
    worst_model = None
    worst_model_score = 100.0
    accuracies = []
    y_true = []
    y_pred = []

    for attempt_rf in range(1, ATTEMPTS + 1):
        print(f"Attempt: {attempt_rf}")
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1)
        kFold = KFold(n_splits=5, shuffle=False)

        best_model_k_fold = None
        best_score_k_fold = 0.0
        for train_indices, validate_indices in kFold.split(X_train, Y_train):
            x_train = X_train[train_indices]
            y_train = Y_train[train_indices]
            x_val = X_train[validate_indices]
            y_val = Y_train[validate_indices]

            rf = RandomForestClassifier(classifiers_number=50, tree_percentage=1.0)
            time1 = time.time()
            rf.fit(x_train, y_train, discrete_x=True)
            print(f"Training took: {time.time() - time1:.3f} s")
            acc = rf.evaluate(x_val, y_val)

            if acc > best_score_k_fold:
                best_model_k_fold = rf
                best_score_k_fold = acc

        # Evaluate best model on test data
        print(f"Best model accuracy on validation data: {best_score_k_fold * 100:.2f}")
        best_k_fold_accuracy = best_model_k_fold.evaluate(X_test, Y_test)
        accuracies.append(best_k_fold_accuracy)

        if best_k_fold_accuracy > best_model_score:
            best_model = best_model_k_fold
            best_model_score = best_k_fold_accuracy

        if best_k_fold_accuracy < worst_model_score:
            worst_model = best_model_k_fold
            worst_model_score = best_k_fold_accuracy

        # Make predictions
        y_true.append(Y_test)
        y_pred.append(np.argmax(best_model_k_fold.predict(X_test), axis=1))

    print(f"Accuracies")
    accuracies = np.array(accuracies) * 100
    print(
        f"Max: {np.max(accuracies):.2f}\tMin: {np.min(accuracies):.2f}\tMean: {np.mean(accuracies):.2f}\tStd dev: {np.std(accuracies):.2f}")

    cm = confusion_matrix(np.array(y_true).flatten(), np.array(y_pred).flatten(), labels=[0, 1, 2])
    cm = cm // ATTEMPTS
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["abnormal", "normal", "inconclusive"])
    disp.plot(cmap=plt.cm.Blues)

    if SAVE_IMAGES:
        plt.savefig("images/wine_forest_id3_cm.pdf")

    encoder = LabelBinarizer()
    y_true_ovr = [encoder.fit_transform(x) for x in y_true]
    y_pred_ovr = [encoder.fit_transform(x) for x in y_pred]

    # 0 vs rest
    ovr_cm(y_true_ovr, y_pred_ovr, 0, "abnormal", ATTEMPTS, SAVE_IMAGES, "images/wine_forest_id3_cm_class=0.pdf")

    precision_list_0, tpr_list_0, fpr_list_0 = ovr_test(y_true_ovr, y_pred_ovr, 0, SAVE_IMAGES,
                                                        "images/wine_forest_id3_roc_class=0.pdf")

    print("\nPrecision for 0 vs rest")
    print(
        f"Max: {np.max(precision_list_0):.2f}\tMin: {np.min(precision_list_0):.2f}\tMean: {np.mean(precision_list_0):.2f}\tStd dev: {np.std(precision_list_0):.2f}")
    print("True positive rate for 0 vs rest")
    print(
        f"Max: {np.max(tpr_list_0):.2f}\tMin: {np.min(tpr_list_0):.2f}\tMean: {np.mean(tpr_list_0):.2f}\tStd dev: {np.std(tpr_list_0):.2f}")
    print("False positive rate for 0 vs rest")
    print(
        f"Max: {np.max(fpr_list_0):.2f}\tMin: {np.min(fpr_list_0):.2f}\tMean: {np.mean(fpr_list_0):.2f}\tStd dev: {np.std(fpr_list_0):.2f}")

    # 1 vs rest
    ovr_cm(y_true_ovr, y_pred_ovr, 1, "normal", ATTEMPTS, SAVE_IMAGES, "images/wine_forest_id3_cm_class=1.pdf")

    precision_list_1, tpr_list_1, fpr_list_1 = ovr_test(y_true_ovr, y_pred_ovr, 1, SAVE_IMAGES,
                                                        "images/wine_forest_id3_roc_class=1.pdf")

    print("\nPrecision for 1 vs rest")
    print(
        f"Max: {np.max(precision_list_1):.2f}\tMin: {np.min(precision_list_1):.2f}\tMean: {np.mean(precision_list_1):.2f}\tStd dev: {np.std(precision_list_1):.2f}")
    print("True positive rate for 1 vs rest")
    print(
        f"Max: {np.max(tpr_list_1):.2f}\tMin: {np.min(tpr_list_1):.2f}\tMean: {np.mean(tpr_list_1):.2f}\tStd dev: {np.std(tpr_list_1):.2f}")
    print("False positive rate for 1 vs rest")
    print(
        f"Max: {np.max(fpr_list_1):.2f}\tMin: {np.min(fpr_list_1):.2f}\tMean: {np.mean(fpr_list_1):.2f}\tStd dev: {np.std(fpr_list_1):.2f}")

    # 2 vs rest
    ovr_cm(y_true_ovr, y_pred_ovr, 2, "inconclusive", ATTEMPTS, SAVE_IMAGES,
           "images/wine_forest_id3_cm_class=2.pdf")

    precision_list_2, tpr_list_2, fpr_list_2 = ovr_test(y_true_ovr, y_pred_ovr, 2, SAVE_IMAGES,
                                                        "images/wine_forest_id3_roc_class=2.pdf")

    print("\nPrecision for 2 vs rest")
    print(
        f"Max: {np.max(precision_list_2):.2f}\tMin: {np.min(precision_list_2):.2f}\tMean: {np.mean(precision_list_2):.2f}\tStd dev: {np.std(precision_list_2):.2f}")
    print("True positive rate for 2 vs rest")
    print(
        f"Max: {np.max(tpr_list_2):.2f}\tMin: {np.min(tpr_list_2):.2f}\tMean: {np.mean(tpr_list_2):.2f}\tStd dev: {np.std(tpr_list_2):.2f}")
    print("False positive rate for 2 vs rest")
    print(
        f"Max: {np.max(fpr_list_2):.2f}\tMin: {np.min(fpr_list_2):.2f}\tMean: {np.mean(fpr_list_2):.2f}\tStd dev: {np.std(fpr_list_2):.2f}")
