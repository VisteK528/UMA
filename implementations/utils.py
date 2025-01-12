import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MeasuresOfQuality:
    def __init__(self, y_pred: np.array, y_true: np.array):
        self._y_pred = y_pred
        self._y_true = y_true

        self._tp = None
        self._tn = None
        self._fp = None
        self._fn = None

        self._compiled = False
        self._unique_classes = None
        self._unique_classes_dict = None

    def compile(self) -> None:
        self._unique_classes = np.sort(np.unique(self._y_true))
        self._unique_classes_dict = {x: i for i, x in enumerate(self._unique_classes)}
        N = len(self._unique_classes)

        self._tp = np.zeros((N,))
        self._tn = np.zeros((N,))
        self._fp = np.zeros((N,))
        self._fn = np.zeros((N,))
        for i, unique_class in enumerate(self._unique_classes):
            for y_pred, y_true in zip(self._y_pred, self._y_true):
                if y_true == unique_class and y_pred == unique_class:
                    self._tp[i] += 1
                elif y_true == unique_class and y_pred != unique_class:
                    self._fn[i] += 1
                elif y_true != unique_class and y_pred == unique_class:
                    self._fp[i] += 1
                elif y_true != unique_class and y_pred != unique_class:
                    self._tn[i] += 1

        self._compiled = True

    def _prepare_for_measures(self, selected_class=None) -> list[int]:
        if not self._compiled:
            pass

        if selected_class is not None:
            i = self._unique_classes_dict.get(selected_class, None)

            if i is None:
                pass
            return [self._tp[i], self._tn[i], self._fp[i], self._fn[i]]

        return [sum(self._tp), sum(self._tn), sum(self._fp), sum(self._fn)]

    def accuracy(self, selected_class=None) -> float:
        tp, tn, fp, fn = self._prepare_for_measures(selected_class)
        return (tp + tn) / (tp + tn + fp + fn)

    def error(self, selected_class=None):
        tp, tn, fp, fn = self._prepare_for_measures(selected_class)
        return (fp + fn) / (tp + tn + fp + fn)

    def true_positive_rate(self, selected_class=None):
        tp, _, _, fn = self._prepare_for_measures(selected_class)
        return tp / (tp + fn)

    def false_positive_rate(self, selected_class=None):
        _, tn, fp, _ = self._prepare_for_measures(selected_class)
        return fp / (tn + fp)

    def precision(self, selected_class=None):
        tp, _, fp, _ = self._prepare_for_measures(selected_class)
        return tp / (tp + fp)

    def get_confusion_matrix(self):
        if self._unique_classes is not None:
            if len(self._unique_classes) == 2:
                return np.array([[self._tp[0], self._fp[0]], [self._fn[0], self._tn[0]]])
            else:
                cm = np.zeros((len(self._unique_classes), len(self._unique_classes)))
                for y_pred, y_true in zip(self._y_pred, self._y_true):
                    i, j = self._unique_classes_dict.get(y_pred), self._unique_classes_dict.get(y_true)
                    cm[i, j] += 1
                return cm

    # def plot_confusion_matrix(self):
    #
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(confusion_mtx,
    #                 xticklabels=commands,
    #                 yticklabels=commands,
    #                 annot=True, fmt='g')
    #     plt.xlabel('Prediction')
    #     plt.ylabel('Label')
    #     plt.show()
