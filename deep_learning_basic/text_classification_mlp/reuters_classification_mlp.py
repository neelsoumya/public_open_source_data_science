
##############################################################################
# Simple example of an MLP for text classification
#
# Usage:
#    python3 reuters_classification_mlp.py
#
# Adapted from
#    https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py
#    https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#    https://keras.io/getting-started/sequential-model-guide/#examples
#    https://github.com/PetarV-/X-CNN/blob/master/models/cifar10_example.py
#    https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#    https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#    https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
#
##############################################################################

###################################################
# Load libraries
###################################################
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

from keras.datasets import imdb, reuters
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy import interp

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

import pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


###################################################
# Get data
###################################################
print("\n ********** Data Loading Section ********** \n")
print("Loading dataset: \n")

max_words = 1000
f_test_split = 0.3

(x_train, y_train),(x_test, y_test) = reuters.load_data(num_words=max_words, test_split=f_test_split)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

i_num_classes = np.max(y_train) + 1


#####################################################################################
# Feature scaling
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")

print('Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Convert class vector to binary class matrix '
     '(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, i_num_classes)
y_test  = keras.utils.to_categorical(y_test, i_num_classes)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


###################################################
# Model parameters
###################################################
f_dropout             = 0.20
i_fitting_epochs      = 50
i_batch_size          = 64
f_validation_split    = 0.33
str_optimizer         = 'sgd'
str_loss_function     = 'categorical_crossentropy'
i_num_neurons_layer_1 = 128
i_num_neurons_layer_2 = i_num_classes
str_activation_layer1 = 'relu' # tanh
str_activation_layer2 = 'softmax'
i_stride_length       = 1
k_kernel_initializer  = keras.initializers.glorot_uniform(seed=None) # Xavier Initialization

###################################################
# Initialize model
###################################################
print("\n ********** Model Creation Section ********** \n")
print("Creating deep learning model: \n")

model = Sequential()
model.add(Dense(i_num_neurons_layer_1, input_shape=(max_words,), kernel_initializer=k_kernel_initializer))
model.add(Activation(str_activation_layer1))
model.add(Dropout(f_dropout))
model.add(Dense(i_num_classes))
model.add(Activation(str_activation_layer2))

model.compile(loss=str_loss_function,
              optimizer=str_optimizer,
                metrics=['accuracy'])

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=str_optimizer, metrics=['accuracy'])


#####################################################################################
# Plot model fit over training epochs
#####################################################################################
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
history = model.fit(x_train, y_train,
          batch_size=i_batch_size,
          epochs=i_fitting_epochs,
          validation_data=(x_test, y_test)
          )

score, acc = model.evaluate(x_test, y_test,
                            batch_size=i_batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# TODO:
# RNNs are tricky. Choice of batch size is important,
# choice of loss and optimizer is critical, etc.
# Some configurations won't converge.
# - LSTM loss decrease patterns during training can be quite different
# from what you see with CNNs/MLPs/etc.


#history = model.fit(trainX, trainY, validation_split=f_validation_split, batch_size=i_batch_size, epochs=i_fitting_epochs)
#score = model.evaluate(testX, testY, batch_size=i_batch_size)
#print(score)

# list all data in history
#print(history.history.keys())
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
plt.savefig('reuters_learning_curve_accuracy.png', dpi=300)
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
plt.savefig('reuters_learning_curve_loss.png', dpi=300)
#plt.show()

#exit()


#####################################################################################
# Generate predictions on new data:
#####################################################################################
print("\n ********** Model Prediction Section ********** \n")
print("Printing model prediction details on test set: \n")

classes = model.predict(x_test, batch_size=i_batch_size)
print(y_test)
print(classes)
model.predict_classes(x_test, batch_size=i_batch_size, verbose=0)#, steps=1)
#pdb.set_trace()

#####################################################################################
# Plot AUPR curves or ROC curves
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
#####################################################################################
# TODO: AUPR curves not ROC
# see https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
#pdb.set_trace()
#probas = model.predict_proba(x_test_array)
probas = model.predict(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test_array, batch_size=i_batch_size)
#probas = model.predict(x_test) # x_test

#pdb.set_trace()
#fpr, tpr, thresholds = roc_curve(y_test, probas, pos_label=1)
#roc_auc = auc(fpr, tpr)

#plt.figure(figsize=(7, 5))
#plt.grid()
#plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'% (roc_auc))
#plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False positive rate')
#plt.ylabel('True positive rate')
#plt.title('Receiver Operator Characteristic')
#plt.legend(loc="lower right")
#plt.tight_layout()
#plt.savefig('reuters_roc_curve.png', dpi=300)


#####################################################################################
# Print summary of a model and inspect model
#####################################################################################
print("\n ********** Model Summary Section ********** \n")
print("Printing model summary and model details: \n")
print(model.summary())
#keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
print(model.output_shape)
print(model.input_shape)
#print(model.get_config())
#print(model.get_weights())

print("\n ********** Model Parameters ********** \n")
#print("\nLearning rate: ", f_learning_rate)
#print("Learning rate decay: ", f_learning_rate_decay)
print("\n Optimizer:", str_optimizer)
print("Loss function:", str_loss_function)
#print("Momentum: ", f_momentum)
print("Fitting epochs: ", i_fitting_epochs)
print("Batch size: ", i_batch_size)
print("Validation split of training set: ", f_validation_split)
print("Dropout probability: ", f_dropout)
print("Number of neurons in first hidden layer: ", i_num_neurons_layer_1)
print("Number of neurons in second hidden layer: ", i_num_neurons_layer_2)
print("Activation function for first hidden layer: ", str_activation_layer1)
print("Activation function for first hidden layer: ", str_activation_layer2)
print("Kernel initialization: ", k_kernel_initializer)#"Orthogonal") # k_kernel_initializer   = keras.initializers.Orthogonal(gain=1.0, seed=None)
#print("Kernel regularization: ", "L1")         # k_kernel_regularizer   = keras.regularizers.l1(0.01)
#print("Kernel activity initialization: ", "L1") # k_activity_regularizer = keras.regularizers.l1(0.01)
print("**************************************")


#####################################################################################
# Save model
#####################################################################################
print("\n ********** Model Save Section ********** \n")
print("       Saving model ......  \n")
str_model_save_filename = "model_file_reuters_mlp.h5"
#model.save(str_model_save_filename)
#model_saved = load_model(str_model_save_filename)

print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")
