import numpy as np
import sklearn.model_selection

import sklearn.datasets

X, y = sklearn.datasets.load_svmlight_file("data/colon-cancer")

X = X.toarray()

y = y.astype(np.int64)
y[y == -1] = 0

print("X.shape = ", X.shape)
print("y.shape = ", y.shape)

print(X)
print(y)

dataScalerX = sklearn.preprocessing.RobustScaler().fit(X)
X = dataScalerX.transform(X)

whole_data = {}
whole_data["X"] = X
whole_data["y"] = y
np.save("microarray_data_prepared/" + "colon_whole_data", whole_data)

print("saved all prepared data")

