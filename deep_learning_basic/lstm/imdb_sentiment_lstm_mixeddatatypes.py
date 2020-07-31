
##############################################################################
# Simple example of an LSTM for sentiment classification
#       dataset used is the IMDB database
#   NOTE: Additional modification; appending other numeric data types to
#           categorical information in words
#
# Usage:
#    python3 imdb_sentiment_lstm_mixeddatatypes.py
#
# Adapted from
#    https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
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
from keras.datasets import imdb

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

import pydot
import graphviz
from keras.utils.vis_utils import plot_model

#############################################################
# Get data
#############################################################
# Dataset of 25,000 movies reviews from IMDB,
# labeled by sentiment (positive/negative). Reviews
# have been preprocessed, and each review is encoded
# as a sequence of word indexes (integers). For
# convenience, words are indexed by overall frequency
# in the dataset, so that for instance the integer
# "3" encodes the 3rd most frequent word in the data.
# This allows for quick filtering operations such as:
# "only consider the top 10,000 most common words,
# but eliminate the top 20 most common words".
# As a convention, "0" does not stand for a specific
# word, but instead is used to encode any unknown word.

print("\n ********** Deep Learning LSTM model for sentiment prediction ********** \n")
print("\n ********** Data Loading Section ********** \n")
print("Loading dataset: \n")

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80

i_num_classes = 2 - 1

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

#####################################################################################
# Adding other datatypes like numeric
#####################################################################################
[x.append(1.4) for x in x_train]
[x.append(1.43) for x in x_test]

#####################################################################################
# Feature scaling and data munging
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test  = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


###################################################
# Model parameters
###################################################
f_learning_rate       = 0.01
f_learning_rate_decay = 1e-6
f_momentum            = 0.9
f_dropout             = 0.20
i_fitting_epochs      = 20
i_batch_size          = 32
f_validation_split    = 0.33
str_optimizer         = 'adam'
i_num_neurons_layer_1 = 128
i_num_neurons_layer_2 = 1
str_activation_layer1 = 'tanh' # 'relu'
str_activation_layer2 = 'sigmoid'
i_stride_length       = 1
k_kernel_initializer  = keras.initializers.glorot_uniform(seed=None) # Xavier Initialization
str_loss_function     = 'binary_crossentropy'

###################################################
# Initialize model
###################################################
print("\n ********** Model Creation Section ********** \n")
print("Creating deep learning model: \n")

# create and fit the LSTM network
model = Sequential()
# Create an Embedding
# Turns positive integers (indexes)
# into dense vectors of fixed size.
# eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
# This layer can only be used as the first layer in a model.
# For more information, see:
#       https://keras.io/layers/embeddings/

# Also coursera course by Andrew Ng for description of embeddings
#    https://www.coursera.org/learn/nlp-sequence-models/lecture/APM5s/learning-word-embeddings
model.add(Embedding(input_dim=max_features, output_dim=i_num_neurons_layer_1))
# Create an LSTM layer
#   Tutorial on LSTM:
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#
# NOTE: the output dimensions of Embedding is same as input dimension for LSTM
model.add(LSTM(units=i_num_neurons_layer_1, activation=str_activation_layer1,
               dropout=f_dropout, recurrent_dropout=f_dropout, kernel_initializer=k_kernel_initializer)
          )
model.add(Dense(units=i_num_neurons_layer_2, activation=str_activation_layer2))

# try using different optimizers and different optimizer configs
model.compile(loss=str_loss_function,
              optimizer=str_optimizer,
              metrics=['accuracy'])
# TODO: visualize the embedding
# np.shape(model.predict(x_test))
# model.layers[-3]


# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=str_optimizer, metrics=['accuracy'])


#####################################################################################
# Plot model fit over training epochs
#####################################################################################
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
history = model.fit(x_train, y_train,
                    batch_size=i_batch_size,
                    epochs=i_fitting_epochs,
                    validation_data=(x_test, y_test),
                    verbose=0
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
plt.figure()
plt.grid()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig('imdb_lstm_learning_curve_accuracy.png', dpi=300)
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
plt.savefig('imdb_lstm_learning_curve_loss.png', dpi=300)


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
fpr, tpr, thresholds = roc_curve(y_test, probas, pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.grid()
plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'% (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='random guessing')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('imdb_lstm_roc_curve.png', dpi=300)


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


#####################################################################################
# Plot model
#####################################################################################
#plot_model(model, to_file='model.png')
keras.utils.plot_model(model, to_file='graph_lstm_imdb_model.png', show_shapes=True,
                       show_layer_names=True, rankdir='TB')


print("\n ********** Model Parameters ********** \n")
#print("\nLearning rate: ", f_learning_rate)
#print("Learning rate decay: ", f_learning_rate_decay)
print("\n Optimizer:", str_optimizer)
#print("Momentum: ", f_momentum)
print("Fitting epochs: ", i_fitting_epochs)
print("Batch size: ", i_batch_size)
print("Validation split of training set: ", f_validation_split)
print("Dropout probability: ", f_dropout)
print("Number of neurons in first hidden layer: ", i_num_neurons_layer_1)
print("Number of neurons in second hidden layer: ", i_num_neurons_layer_2)
print("Activation function for first hidden layer: ", str_activation_layer1)
print("Activation function for first hidden layer: ", str_activation_layer2)
#print("Kernel initialization: ", "Orthogonal") # k_kernel_initializer   = keras.initializers.Orthogonal(gain=1.0, seed=None)
#print("Kernel regularization: ", "L1")         # k_kernel_regularizer   = keras.regularizers.l1(0.01)
#print("Kernel activity initialization: ", "L1") # k_activity_regularizer = keras.regularizers.l1(0.01)
print("**************************************")


#####################################################################################
# Save model
#####################################################################################
print("\n ********** Model Save Section ********** \n")
print("       Saving model ......  \n")
str_model_save_filename = "model_file_imdb_lstm.h5"
#model.save(str_model_save_filename)
#model_saved = load_model(str_model_save_filename)

print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")
