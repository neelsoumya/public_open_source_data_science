#####################################################
# Deep learning using keras and tensorflow
#
# INSTALLATION:
# pip install tensorflow 
# https://www.tensorflow.org/install/install_mac
# sudo pip install keras
# https://keras.io/#installation
# pip install graphviz
# pip install pydot
# pip install -U scikit-learn
#
# Usage:
# python3 deep_learning_keras.py
#
#####################################################


###################################################
# Load libraries
###################################################
from keras.models import Sequential
from keras.layers import Dense

from keras.utils import to_categorical
from keras import regularizers

import keras
import pydot
import graphviz

#import sklearn
#import scikit
#from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import pdb

###################################################
# Initialize model
###################################################
model = Sequential()

# TODO: MOSt IMPORTANT MODEL SELECTION
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/

###################################################
# Stacking layers is as easy as .add():
###################################################
#model.add(Dense(units=64, activation='relu', input_dim=9))
#model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=2, activation='relu', input_dim=663,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
# TODO: ADD MORE HIDDEN LAYERS AND OTHER ARCHITECTURE SEE CHENGWEI and LINKS BELOW
# http://cs231n.github.io/convolutional-networks/#architectures
# MHCflurry contains (1) an embedding layer which transforms amino acids to learned vector representations,
# (2) a single hidden layer with tanh nonlinearity, and (3) a sigmoidal scalar output.
# It maps input peptides to a 32-dimensional space, which then feeds into a fully connected layer.
# https://www.biorxiv.org/content/early/2017/08/09/174243
# https://github.com/openvax/mhcflurry/blob/master/mhcflurry/class1_neural_network.py
model.add(Dense(units=2, activation='softmax'))
# model.add(Dense(units=2, activation='sigmoid'))
# tanh

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))

#####################################################################################
# You can now iterate on your training data in batches:
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
#####################################################################################

#####################################################################################
#  Load data
#####################################################################################

# read in pandas dataframe
#temp_str_peptide_file = "breast-cancer-wisconsin_MOD.data"
temp_str_peptide_file = "FINAL_intersected_genes_cohort_gse12251_osm_MOD.csv"
temp_peptide_df = pd.read_csv(temp_str_peptide_file)#, header=None)
#            temp_peptide_df.columns = ['peptide', 'abundance']
# TODO: 1) split data into training and cv
i_split_train_test_from = 16
x_train = temp_peptide_df.iloc[0:i_split_train_test_from,1:-1] #temp_peptide_df["epithelial_cell_size"]
y_train = temp_peptide_df.iloc[0:i_split_train_test_from,-1] #temp_peptide_df["class"]

x_train_array = np.array(x_train)
#y_train_array = np.array(y_train)


#####################################################################################
# Feature scaling
#####################################################################################
x_train_array = keras.utils.normalize(x_train_array)
#y_train_array = keras.utils.normalize(y_train_array)


#y_binary = to_categorical(y_int)
y_train = to_categorical(y_train)#,2)

# TODO: cross validation as in
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, x_train_array, y_train, cv=kfold)
#print(results.mean())
i_fitting_epochs = 100000
model.fit(x_train_array, y_train, epochs=i_fitting_epochs, batch_size=320)


#####################################################################################
# Evaluate your performance in one line:
#####################################################################################
# TODO: randperm
x_test = temp_peptide_df.iloc[i_split_train_test_from:,1:-1] #temp_peptide_df["epithelial_cell_size"]
y_test = temp_peptide_df.iloc[i_split_train_test_from:,-1] #temp_peptide_df["class"]

x_test_array = np.array(x_test)
# feature scaling
x_test_array = keras.utils.normalize(x_test_array)

y_test = to_categorical(y_test)#,2)


loss_and_metrics = model.evaluate(x_test_array, y_test, batch_size=1280)
print("loss_and_metrics",loss_and_metrics)


#####################################################################################
# TODO: 1) cross validate
# TODO: 2) model selection
#####################################################################################


# TODO: look at different layers
# https://keras.io/layers/about-keras-layers/

#####################################################################################
# Generate predictions on new data:
#####################################################################################
classes = model.predict(x_test, batch_size=128)
print(y_test)
print(classes)


#####################################################################################
# Prints summary of a model
#####################################################################################
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)


#####################################################################################
# Plot model
#####################################################################################
keras.utils.plot_model(model, to_file='keras_model.png', show_shapes=False,
                       show_layer_names=True, rankdir='TB')



# TODO: init="glorot_uniform",
# TODO: dropout_probability=0.0,
# from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/class1_neural_network.py
# from

# Other links
# https://github.com/hammerlab/mhcflurry-icml-compbio-2016/blob/master/notebooks/validation.ipynb
#

#pdb.set_trace()