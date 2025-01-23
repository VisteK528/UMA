from ucimlrepo import fetch_ucirepo
from implementations import bayes
from implementations import id3
from implementations import random_forest
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from typing import Tuple
import pandas as pd
import time
import importlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from implementations.bayes import NaiveBayes

ATTEMPTS = 1

if __name__ == "__main__":
    X = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/X.csv", dtype=float, delimiter=",")
    y = np.loadtxt("../../../data_processed/healthcare/percentile_n_hotone/y.csv", dtype=float, delimiter=",")


    for attempt_nb in range(1, ATTEMPTS+1):
        print(f"Attempt: {attempt_nb}")
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.1)
        kFold = KFold(n_splits=5, shuffle=False)

        #print(X_train.shape)
        best_model_k_fold = None
        best_score_k_fold = 0.0
        for train_indices, validate_indices in kFold.split(X_train, Y_train):
            x_train = X_train[train_indices]
            y_train = Y_train[train_indices]
            x_val = X_train[validate_indices]
            y_val = Y_train[validate_indices]


            rf = random_forest.RandomForestClassifier(classifiers_number=50)
            time1 = time.time()
            rf.fit(x_train, y_train, discrete_x=True, tree_percentage=1.0)
            print(f"Training took: {time.time() - time1:.3f} s")
            acc = rf.evaluate(x_val, y_val)

            if acc > best_score_k_fold:
                best_model_k_fold = rf
                best_score_k_fold = acc

        # Evaluate best model on test data
        print(f"Best model accuracy on validation data: {best_score_k_fold*100:.2f}")
        best_model_k_fold.evaluate(X_test, Y_test)





