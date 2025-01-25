import numpy as np
from uma24z_nbc_random_forest.random_forest import RandomForestClassifier

def test_random_forest_bootstrap():
    X = np.random.random(size=(1000, 5))
    y = np.random.randint(low=0, high=2, size=(1000,))

    X_bootstrap, y_bootstrap = RandomForestClassifier.bootstrap(X, y)

    assert X_bootstrap.shape == (1000, 5)
    assert y_bootstrap.shape == (1000, )

