from sklearn.ensemble import RandomForestClassifier
import numpy as np

id3 = RandomForestClassifier()

X_test = np.array([[], []])
y = np.array([])


print(f"Found array with 0 feature(s) (shape={X_test.shape}) while a minimum of 1 is required.")

X = np.random.random(size=(1000, 5))
y = np.random.randint(low=0, high=2, size=(1000,))
print(X.shape)
print(y)