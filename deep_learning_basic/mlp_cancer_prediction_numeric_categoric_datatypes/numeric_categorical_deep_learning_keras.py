#####################################################
# Deep learning using keras and tensorflow
#       applied to a dataset from the UCI machine
#       learning repository
#       Supposes that there is a mix of data types :
#           numeric and categorical
#
# INSTALLATION:
#   pip3 install tensorflow
#   https://www.tensorflow.org/install/install_mac
#   sudo pip3 install keras
#   https://keras.io/#installation
#   pip3 install graphviz
#   pip3 install pydot
#   pip3 install -U scikit-learn
#   pip3 install deap update_checker tqdm stopit
#   pip3 install tpot
#   pip3 install xgboost
#
#
# Usage:
#    python3 numeric_categorical_deep_learning_keras.py
#
# Adapted from:
#   https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#   https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#   https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
#
#####################################################


###################################################
# Load libraries
###################################################
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
from keras import regularizers
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
import keras
import pydot
import graphviz

import sklearn
# import scikit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy import interp

import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
# from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


###################################################
# Model parameters
###################################################
# see https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
f_dropout = 0.20
i_num_neurons_layer_1 = 10
i_num_neurons_layer_2 = 2
str_activation_layer1 = 'relu'  # tanh
str_activation_layer2 = 'sigmoid'
f_learning_rate = 0.01
f_learning_rate_decay = 1e-6
f_momentum = 0.9
i_fitting_epochs = 1000
i_batch_size = 500
f_validation_split = 0.33
# k_kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
# k_kernel_initializer = keras.initializers.glorot_uniform(seed=None) # Xavier Initialization
k_kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)
k_kernel_regularizer = keras.regularizers.l1(0.01)
k_activity_regularizer = keras.regularizers.l1(0.01)
# k_kernel_regularizer   = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
sgd = keras.optimizers.SGD(lr=f_learning_rate, decay=f_learning_rate_decay,
                           momentum=f_momentum, nesterov=True)


#####################################################################################
#  Load data
#####################################################################################
print("\n ********** Data Loading Section ********** \n")
print("Loading dataset: \n")

# read in pandas dataframe
# generated from breast-cancer-wisconsin_MOD.data using data_munging.R
temp_str_peptide_file = "df_drug_data_coded_tobesaved_curated.tsv" # "df_patients_joined_meds_orderedbymeds_curated_fewfields2.csv"
temp_peptide_df = pd.read_csv(temp_str_peptide_file, header=None, sep="\t")
#            temp_peptide_df.columns = ['peptide', 'abundance']

#####################################################################################
# Split data into training and test set
#####################################################################################
print("\n ********** Train Test Split Section ********** \n")
print("Splitting data into training and test set: \n")

x_numeric      = temp_peptide_df.iloc[:, 0]  # temp_peptide_df["epithelial_cell_size"]
x_categorical1 = temp_peptide_df.iloc[:, -3]
x_categorical2 = temp_peptide_df.iloc[:, -2]    # temp_peptide_df["epithelial_cell_size"]
y              = temp_peptide_df.iloc[:, -1]    # temp_peptide_df["class"]

x_categorical1 = keras.utils.to_categorical(x_categorical1)
x_categorical2 = keras.utils.to_categorical(x_categorical2)

# x_concat = np.concatenate([x_numeric, x_categorical1, x_categorical2], axis = 1)
x_concat = np.concatenate([  np.transpose(( np.vstack((x_numeric,x_numeric))) ) , x_categorical1, x_categorical2],
                          axis = 1)


x_train, x_test, y_train, y_test = train_test_split(x_concat,
                                                    y,
                                                    train_size=0.7,
                                                    test_size=0.3)



print(np.shape(x_test))
print(np.shape(x_train))

#####################################################################################
# Feature scaling
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")

# x_train_array = keras.utils.normalize(x_train_array)
# y_train_array = keras.utils.normalize(y_train_array)

# TODO:
# `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes). If your targets are integer classes, you can convert them to the expected format via:
# from keras.utils import to_categorical
# y_binary = to_categorical(y_int)
# Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.
# cross validation as in
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# kfold = StratifiedKFold(n_splits=10, shuffle=True)  # , random_state=seed)
# results = cross_val_score(model, x_train_array, y_train, cv=kfold)
# print(results.mean())
# model.fit(x_train_array, y_train, epochs=i_fitting_epochs, batch_size=320)


###################################################
# Initialize model
###################################################
print("\n ********** Model Creation Section ********** \n")
print("Creating deep learning model: \n")

model = Sequential()

###################################################
# Adding layers
# Stacking layers is as easy as .add():
###################################################
# model.add(Dense(units=64, activation='relu', input_dim=9))
i_input_dimension = np.shape(x_test)[1]
model.add(Dense(units=i_num_neurons_layer_1, activation=str_activation_layer2, input_dim=i_input_dimension,
                kernel_initializer=k_kernel_initializer,
                kernel_regularizer=k_kernel_regularizer,
                activity_regularizer=k_activity_regularizer
                )
          )
# model.add(Dense(units=i_num_neurons_layer_2, activation=str_activation_layer2))

###################################################
# Model regularization
# see https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
###################################################
# model.add(Dropout(f_dropout))
# model.add(Dense(units=i_num_neurons_layer_2, activation=str_activation_layer2))

# model.compile(loss='categorical_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])

# model.compile(loss='sparse_categorical_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

# model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))



#####################################################################################
# Plot model fit over training epochs
#  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#####################################################################################

# Account for imbalanced classes in model fitting
# NOTE: Currently only works for binary classification with labels 1 and 0 (numeric)

i_class_weight_label_1 = sum([int(x) for x in y_train]) / len(y_train) # number of 1s in target (y) of training set
i_class_weight_label_0 = 1 - i_class_weight_label_1
dict_class_weights     = {1:i_class_weight_label_1, 0:i_class_weight_label_0} # dict of class weights

history = model.fit(x=x_train, y=y_train,
                    validation_split=f_validation_split,
                    class_weight=dict_class_weights,
                    epochs=i_fitting_epochs,
                    batch_size=i_batch_size,
                    verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.grid()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig('multipledatatypes_learning_curve_accuracy.png', dpi=300)
# plt.show()
# summarize history for loss
plt.figure()
plt.grid()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig('multipledatatypes_learning_curve_loss.png', dpi=300)
# plt.show()


#####################################################################################
# Evaluate your performance in one line:
#####################################################################################

print("\n ********** Model Evaluation Section ********** \n")
print("Printing model performance details on test set: \n")

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=i_batch_size)
print("loss_and_metrics", loss_and_metrics)

#####################################################################################
# TODO: MODEL SELECTION
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#####################################################################################

# TODO: look at different layers
# https://keras.io/layers/about-keras-layers/

#####################################################################################
# Generate predictions on new data:
#####################################################################################
print("\n ********** Model Prediction Section ********** \n")
print("Printing model prediction details on test set: \n")

classes = model.predict(x_test, batch_size=i_batch_size)
print(y_test)
print(classes)
model.predict_classes(x_test, batch_size=i_batch_size, verbose=0)  # , steps=1)
# pdb.set_trace()

#####################################################################################
# Plot AUPR curves or ROC curves
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
#####################################################################################
# TODO: AUPR curves not ROC
# see https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a

# pdb.set_trace()
# probas = model.predict_proba(x_test_array)
# probas = model.predict(x_test, batch_size=i_batch_size)
# probas = model.predict_classes(x_test, batch_size=i_batch_size)
# probas = model.predict_classes(x_test_array, batch_size=i_batch_size)
probas = model.predict(x_test)  # x_test

# pdb.set_trace()
# fpr, tpr, thresholds = roc_curve(y_test[:,1], probas[:,1], pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)

print("ROC score", roc_auc)

plt.figure(figsize=(7, 5))
plt.grid()
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('multipledatatypes_roc_curve.png', dpi=300)

#####################################################################################
# Visualize balance or imbalance of training data
#####################################################################################
plt.figure(figsize=(8, 4))
sns.countplot(x=y_train)
plt.savefig('multipledatatypes_balance_trainingset.png', dpi=300)

plt.figure(figsize=(8, 4))
sns.countplot(x=y_test)
plt.savefig('multipledatatypes_balance_testset.png', dpi=300)

#####################################################################################
# Print summary of a model and inspect model
#####################################################################################
print("\n ********** Model Summary Section ********** \n")
print("Printing model summary and model details: \n")
print(model.summary())
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
print(model.output_shape)
print(model.input_shape)
print(model.get_config())
#print(model.get_weights())

print("\n ********** Model Parameters ********** \n")
print("\nLearning rate: ", f_learning_rate)
print("Learning rate decay: ", f_learning_rate_decay)
print("Optimizer:", "SGD")
print("Momentum: ", f_momentum)
print("Fitting epochs: ", i_fitting_epochs)
print("Batch size: ", i_batch_size)
print("Validation split of training set: ", f_validation_split)
print("Dropout probability: ", f_dropout)
print("Number of neurons in first hidden layer: ", i_num_neurons_layer_1)
print("Number of neurons in second hidden layer: ", i_num_neurons_layer_2)
print("Activation function for first hidden layer: ", str_activation_layer1)
print("Activation function for first hidden layer: ", str_activation_layer2)
print("Kernel initialization: ", "Orthogonal")  # k_kernel_initializer   = keras.initializers.Orthogonal(gain=1.0, seed=None)
print("Kernel regularization: ", "L1")  # k_kernel_regularizer   = keras.regularizers.l1(0.01)
print("Kernel activity initialization: ", "L1")  # k_activity_regularizer = keras.regularizers.l1(0.01)
print("**************************************")

#####################################################################################
# Plot model
#####################################################################################
# plot_model(model, to_file='model.png')
#keras.utils.plot_model(model, to_file='multiple_datatypes_keras_model.png', show_shapes=True,
#                       show_layer_names=True, rankdir='TB')

#####################################################################################
# Additional bells and whistles
#####################################################################################
# TODO: init="glorot_uniform",
# TODO: dropout_probability=0.0,
# from https://github.com/openvax/mhcflurry/blob/master/mhcflurry/class1_neural_network.py
# Other links
# https://github.com/hammerlab/mhcflurry-icml-compbio-2016/blob/master/notebooks/validation.ipynb
# TODO: convolutional layer
# TODO: max pooling layer
# TODO: dealing with unbalanced data in deep learning % https://medium.com/@jahir.brokenkite/fraud-prevention-in-peer-to-peer-p2p-transaction-networks-using-neural-nets-a-node-embedding-b14a99f8ba30
# code is in % https://github.com/Jhird/TrustKeeper



#####################################################################################
# Save model
#####################################################################################
print("\n ********** Model Save Section ********** \n")
print("       Saving model ......  \n")
str_model_save_filename = "model_file_uci_dataset.h5"
# model.save(str_model_save_filename)
# model_saved = load_model(str_model_save_filename)

##################################################################
# Other baseline models
##################################################################

# Random forest
rf = RandomForestClassifier(max_depth=3, n_estimators=10)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
auc_rf = auc(fpr_rf, tpr_rf)
print("AUC from a baseline random forest algorithm: ", auc_rf)

# Automated machine learning using TPOT
# reformat for TPOT (numpy arrays)
#x_train_array = np.array(x_train)
# y_train_array = np.array(y_train)

# TODO: change this to all data
#X_train, X_test, Y_train, Y_test = train_test_split(x_train_array_concat,
#                                                    y_train_array, train_size=0.75, test_size=0.25)

# tpot = TPOTClassifier(generations=2, population_size=50, verbosity=2)
# tpot.fit(x_train, y_train)
# print("Score from a baseline automated machine learning algorithm: ", tpot.score(x_test, y_test))


print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")
