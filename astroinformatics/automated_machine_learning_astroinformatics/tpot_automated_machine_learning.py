#####################################################
# TPOT automated mL
#
# INSTALLATION:
# pip3 install deap update_checker tqdm stopit
# pip3 install tpot
# pip3 install xgboost
#
# Usage:
# python3 tpot_automated_machine_learning.py
#
# Adapted from
# http://epistasislab.github.io/tpot/examples/
#
#####################################################


###################################################
# Load libraries
###################################################
from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd
import pdb


#####################################################################################
# Example
#####################################################################################
#iris = load_iris()
#X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
#    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

#tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))
#tpot.export('tpot_iris_pipeline.py')

#####################################################################################
#  Load data
#####################################################################################

# read in pandas dataframe
#temp_str_peptide_file = "breast-cancer-wisconsin_MOD.data"
temp_str_peptide_file = "FINAL_intersected_genes_cohort_gse12251_osm_MOD.csv"
temp_peptide_df = pd.read_csv(temp_str_peptide_file)#, header=None)
#            temp_peptide_df.columns = ['peptide', 'abundance']
# TODO: 1) split data into training and cv

# NOTE: this is all data
i_split_train_test_from = 23
x_train = temp_peptide_df.iloc[0:i_split_train_test_from,1:-1] #temp_peptide_df["epithelial_cell_size"]
y_train = temp_peptide_df.iloc[0:i_split_train_test_from,-1] #temp_peptide_df["class"]

x_train_array = np.array(x_train)
y_train_array = np.array(y_train)



X_train, X_test, y_train, y_test = train_test_split(x_train_array,
    y_train_array, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=100, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))



#####################################################################################
# Feature scaling
#####################################################################################
#x_train_array = keras.utils.normalize(x_train_array)
#y_train_array = keras.utils.normalize(y_train_array)


#y_binary = to_categorical(y_int)
#y_train = to_categorical(y_train)#,2)

# TODO: cross validation as in
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, x_train_array, y_train, cv=kfold)
#print(results.mean())
#i_fitting_epochs = 100000
#model.fit(x_train_array, y_train, epochs=i_fitting_epochs, batch_size=320)


#####################################################################################
# Evaluate your performance in one line:
#####################################################################################
# TODO: randperm
#x_test = temp_peptide_df.iloc[i_split_train_test_from:,1:-1] #temp_peptide_df["epithelial_cell_size"]
#y_test = temp_peptide_df.iloc[i_split_train_test_from:,-1] #temp_peptide_df["class"]

#x_test_array = np.array(x_test)
# feature scaling
#x_test_array = keras.utils.normalize(x_test_array)

#y_test = to_categorical(y_test)#,2)


#loss_and_metrics = model.evaluate(x_test_array, y_test, batch_size=1280)
#print("loss_and_metrics",loss_and_metrics)


#####################################################################################
# TODO: 1) cross validate
# TODO: 2) model selection
#####################################################################################


# TODO: look at different layers
# https://keras.io/layers/about-keras-layers/

#####################################################################################
# Generate predictions on new data:
#####################################################################################
#classes = model.predict(x_test, batch_size=128)
#print(y_test)
#print(classes)


#####################################################################################
# Prints summary of a model
#####################################################################################
#keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)


#####################################################################################
# Plot model
#####################################################################################
#keras.utils.plot_model(model, to_file='keras_model.png', show_shapes=False,
#                       show_layer_names=True, rankdir='TB')



# TODO: init="glorot_uniform",
# TODO: dropout_probability=0.0,
# from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/class1_neural_network.py
# from

# Other links
# https://github.com/hammerlab/mhcflurry-icml-compbio-2016/blob/master/notebooks/validation.ipynb
#

#pdb.set_trace()
