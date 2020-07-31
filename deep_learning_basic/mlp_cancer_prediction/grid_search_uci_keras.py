########################################################################################
# Deep learning using keras and tensorflow
#       applied to a dataset from the UCI machine
#       learning repository
#       with grid search for hyperparameter tuning
#
# INSTALLATION:
#   pip3 install tensorflow
#   https://www.tensorflow.org/install/install_mac
#   sudo pip3 install keras
#   https://keras.io/#installation
#   pip3 install graphviz
#   pip3 install pydot
#   pip3 install -U scikit-learn
#
# Usage:
#   python3 grid_search_uci_keras.py
#
# Adapted from:
#   https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#
########################################################################################


###############################################################
# Load libraries
###############################################################
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy import interp

import keras
from keras.utils.vis_utils import plot_model

import numpy as np
import pdb
import matplotlib.pyplot as plt


###############################################################
# Function to create model
###############################################################
print("\n ********** Model Creation Section ********** \n")
print("Creating deep learning model: \n")

#########################################
# Model parameters
#########################################
f_dropout = 0.20
i_num_neurons_layer_1 = 12
i_num_neurons_layer_2 = 8
str_activation_layer1 = 'relu'  # tanh
str_activation_layer2 = 'sigmoid'
f_learning_rate = 0.01
f_learning_rate_decay = 1e-6
f_momentum = 0.9
#i_fitting_epochs = 1000
#i_batch_size = 320
f_validation_split = 0.33
# k_kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
# k_kernel_initializer  = keras.initializers.glorot_uniform(seed=None) # Xavier Initialization


def create_model(optimizer='rmsprop', init='glorot_uniform'):
    """
        Create deep learning model

    :param optimizer:
    :param init:
    :return: keras model

    """

    # create model
    model = Sequential()
    model.add(Dense(i_num_neurons_layer_1, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dropout(f_dropout))
    # model.add(Dense(i_num_neurons_layer_2, kernel_initializer=init, activation='relu'))
    # model.add(Dropout(f_dropout))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model




###############################################################
# Load UCI dataset
###############################################################
# generated from breast-cancer-wisconsin_MOD.data using data_munging.R
dataset = np.loadtxt("breast-cancer-wisconsin_MOD_CURATED.data", delimiter=",")

###############################################################
# split into input (X) and target (Y) variables
###############################################################
X = dataset[:,0:8]
Y = dataset[:,8]

###############################################################
# Perform feature scaling
###############################################################
scaler = StandardScaler()
scaler = scaler.fit(X)
X_scaled = scaler.transform(X)

###############################################################
# Split into train and test using train_test_split
###############################################################
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size = 0.30)


###############################################################
# Create deep learning model
###############################################################
model = KerasClassifier(build_fn=create_model, verbose=0)

###############################################################
# grid search for hyperparameter tuning
###############################################################
print("\n ********** Hyperparameter Optimization Section ********** \n")
print("       Selecting best model ......  \n")
i_num_fold_cross_validation = 10
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform']#, 'normal']# 'uniform']
epochs = [50, 100]
batches = [10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=i_num_fold_cross_validation)
grid_result = grid.fit(X_train, Y_train)

###############################################################
# Summarize results
###############################################################
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


###############################################################
# Print summary of best model
###############################################################
# Get best parameters from hyperparameter optimization
optimal_params = grid_result.best_params_
optimal_init   = optimal_params['init']
optimal_optim  = optimal_params['optimizer']
optimal_batch  = optimal_params['batch_size']
optimal_epoch  = optimal_params['epochs']
# create this model
k_model = create_model(optimizer=optimal_optim, init=optimal_init)
# print summary
print("\n ********** Model Summary Section ********** \n")
print("Printing best model summary and model details: \n")
print(k_model.summary())
keras.utils.print_summary(k_model, line_length=None, positions=None, print_fn=None)
print(k_model.output_shape)
print(k_model.input_shape)
print(k_model.get_config())
print(k_model.get_weights())

#####################################################################################
# Plot model
#####################################################################################
keras.utils.plot_model(k_model, to_file='graph_grid_search_uci.png', show_shapes=True,
                       show_layer_names=True, rankdir='TB')

###############################################################
# Save best model
###############################################################
print("\n ********** Model Save Section ********** \n")
print("       Saving model ......  \n")
str_model_save_filename = "optimized_model_file_uci_dataset.h5"
#k_model.save(str_model_save_filename)
#model_saved = load_model(str_model_save_filename)


#####################################################################################
# Plot model fit over training epochs
#####################################################################################
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

history = k_model.fit(X_train, Y_train, validation_split=f_validation_split, epochs=optimal_epoch, batch_size=optimal_batch, verbose=0)
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
plt.savefig('optimalmodel_learning_curve_accuracy.png', dpi=300)
#plt.show()
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
plt.savefig('optimalmodel_learning_curve_loss.png', dpi=300)
#plt.show()


#####################################################################################
# Evaluate your performance in one line:
#####################################################################################
print("\n ********** Model Evaluation Section ********** \n")
print("Printing model performance details on test set: \n")

loss_and_metrics = k_model.evaluate(X_test, Y_test, batch_size=optimal_batch)
print("loss_and_metrics",loss_and_metrics)

#####################################################################################
# Generate predictions on new data:
#####################################################################################
print("\n ********** Model Prediction Section ********** \n")
print("Printing model prediction details on test set: \n")

classes = k_model.predict(X_test, batch_size=optimal_batch)
#print(Y_test)
print(classes)
k_model.predict_classes(X_test, batch_size=optimal_batch, verbose=0)#, steps=1)
#pdb.set_trace()

#####################################################################################
# Plot AUPR curves or ROC curves
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
#####################################################################################
# TODO: AUPR curves not ROC
#pdb.set_trace()
#probas = model.predict_proba(x_test_array)
#probas = model.predict(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test_array, batch_size=i_batch_size)
probas = k_model.predict(X_test) # x_test

#pdb.set_trace()
#fpr, tpr, thresholds = roc_curve(y_test[:,1], probas[:,1], pos_label=1)
fpr, tpr, thresholds = roc_curve(Y_test, probas, pos_label=1)

#mean_tpr = 0.0
#mean_fpr = np.linspace(0, 1, 100)
#all_tpr = []

#mean_tpr += interp(mean_fpr, fpr, tpr)
#mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.grid()
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'% (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('optimalmodel_roc_curve.png', dpi=300)


print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")
