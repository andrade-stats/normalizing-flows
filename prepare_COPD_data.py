

import pandas as pd
import numpy as np
import sklearn.model_selection

# data from "Gene expression microarray public dataset reanalysis in chronic obstructive pulmonary disease"
# goal detect Chronic obstructive pulmonary disease (COPD) from microarray data

MICRO_ARRAY_DATA_NAME = "DF2_UpdatedSubsetDataPostBoxCox_Standardized_NONAs_060319.csv"

MICRO_ARRAY_DATA_DETAILS = "DF1_UpdatedTargetsFile_060319_AgeGroups.csv"

df_details = pd.read_csv("microarray_data_raw/" + MICRO_ARRAY_DATA_DETAILS)
df_data = pd.read_csv("microarray_data_raw/" + MICRO_ARRAY_DATA_NAME)

number_of_samples = df_data.shape[1] - 1
number_of_genes = df_data.shape[0]


print("number_of_samples = ", number_of_samples)
print("number_of_genes = ", number_of_genes)

y = np.zeros(number_of_samples, dtype = np.int64)
X = np.zeros((number_of_samples, number_of_genes), dtype = np.double)

df_details.set_index("SampleID", inplace = True)

for i, sample_id in enumerate(df_details.index):
    
    X[i, :] = df_data.loc[:, sample_id]
    
    label = df_details.loc[sample_id, "DiseaseState"]
    if  label == "COPD":
        y[i] = 1
    else:
        assert(label == "control")
        y[i] = 0


kf = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=43432)

for foldId, (train_index, test_index) in enumerate(kf.split(X)):
    X_train = X[train_index, :]
    y_train = y[train_index]

    X_test = X[test_index, :]
    y_test = y[test_index]

    dataScalerX = sklearn.preprocessing.RobustScaler().fit(X_train)
    X_train = dataScalerX.transform(X_train)
    X_test = dataScalerX.transform(X_test)

    all_data_one_fold = {}
    all_data_one_fold["X_train"] = X_train
    all_data_one_fold["X_test"] = X_test
    all_data_one_fold["y_train"] = y_train
    all_data_one_fold["y_test"] = y_test
    np.save("microarray_data_prepared/" + "COPD_fold_" + str(foldId), all_data_one_fold)


dataScalerX = sklearn.preprocessing.RobustScaler().fit(X)
X = dataScalerX.transform(X)

whole_data = {}
whole_data["X"] = X
whole_data["y"] = y
np.save("microarray_data_prepared/" + "COPD_whole_data", whole_data)

print("saved all prepared data")
