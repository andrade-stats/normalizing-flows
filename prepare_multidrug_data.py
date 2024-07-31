

import pandas as pd
import numpy as np
import sklearn.model_selection


# data also used in "Empirical Priors for Prediction in Sparse High-dimensional Linear Regression", JMLR, 2020
# from R package mixOmics with:
# write.csv(multidrug$compound,"/Users/danielandrade/ResearchProjects/NormalizingFlows/microarray_data_raw/multidrug_compounds.csv", row.names = FALSE)
# write.csv(multidrug$ABC.trans,"/Users/danielandrade/ResearchProjects/NormalizingFlows/microarray_data_raw/multidrug_transporter_genes.csv", row.names = FALSE)

TARGET_GENE = "ABCB1"

df_compounds = pd.read_csv("microarray_data_raw/" + "multidrug_compounds.csv")

#  expression level of 12 different human ABC transporter genes
df_transporter_genes = pd.read_csv("microarray_data_raw/" + "multidrug_transporter_genes.csv")

# this would give the data used in "Empirical Priors for Prediction in Sparse High-dimensional Linear Regression", JMLR, 2020
# df_compounds = df_compounds.dropna(axis=1)

print("df_compounds = ", df_compounds)

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="median")
X = imp.fit_transform(df_compounds)

y = df_transporter_genes[TARGET_GENE].to_numpy()

print("X = ", X)
print("y = ", y)

dataScalerX = sklearn.preprocessing.RobustScaler().fit(X)
X = dataScalerX.transform(X)

y = y.reshape((-1, 1))
dataScalerY = sklearn.preprocessing.RobustScaler().fit(y)
y = dataScalerY.transform(y)
y = y[:, 0]

print("X = ", X)
print("y = ", y)

whole_data = {}
whole_data["X"] = X
whole_data["y"] = y
# np.save("microarray_data_prepared/" + "multidrug_" + TARGET_GENE +  "_whole_data", whole_data)

print("X.shape = ", X.shape)
print("saved all prepared data")
