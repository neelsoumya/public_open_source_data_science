import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import math

# Functions for random forest classifiers

def train_predict_random_forest_regressor(n_estimators, n_jobs, train, target, test):
    """
    Function to train random forest algorithm
    """

    rf = RandomForestRegressor(n_estimators = n_estimators, n_jobs = n_jobs, oob_score = True)
    print("Training random forest regressor model ...")
    rf.fit(train, target)

    pred_prob_array = rf.predict(test)
    print("Predicting using random forest model (regression)...")
    #[x for x in pred_prob_array]
    #print([x[1] for x in pred_prob_array])

    # Statistics and important features of fit
    print("Statistics and important features of fit\n")
    print(rf.estimators_) # list of DecisionTreeRegressor, The collection of fitted sub-estimators.

    print("Important features\n")
    print(rf.feature_importances_) # : array of shape = [n_features] The feature importances (the higher, the more important the feature).

    print("Number of features\n")
    print(rf.n_features_) #: int The number of features when fit is performed.

    print("The number of outputs when fit is performed\n")
    print(rf.n_outputs_) # : int The number of outputs when fit is performed.

    print("OOB score\n")
    print(rf.oob_score_) # : float Score of the training dataset obtained using an out-of-bag estimate.

    #print(rf.oob_prediction)

    return rf, pred_prob_array


def train_predict_random_forest_classifier(n_estimators, n_jobs, train, target, test):
    """
    Function to train random forest algorithm and also predict
    """

    rf = RandomForestClassifier(n_estimators = n_estimators, n_jobs = n_jobs, oob_score = True)
    print("Training random forest model ...")
    rf.fit(train, target)

    pred_prob_array = rf.predict_proba(test)
    print("Predicting using random forest model ...")
    [x[1] for x in pred_prob_array]
    # print([x[1] for x in pred_prob_array])

    return rf, pred_prob_array




if __name__ == "__main__":
    # load data
    data = pd.read_csv('aggregated_timeseries.csv')
    print(data)

    # get columns
    #data['Activity']
    print(data.describe())
    #data.iloc[0:, 0:]


    # first column is target
    target = data.iloc[0:, 0]

    # Convert target to numeric type
    target = pd.to_numeric(target)


    # all other columns are training set
    train = data.iloc[0:, 1:]

    # Convert target to numeric type
    train = train.apply(lambda x: pd.to_numeric(x) )

    # train random forest model
    print("Training random forest model ...")


    test = np.matrix([[2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1]  ])

    fit_rf_model, pred_prob_array = train_predict_random_forest_regressor(n_estimators=100, n_jobs=2,
                                                                train=train, target=target, test = test)

    print(pred_prob_array)

    
