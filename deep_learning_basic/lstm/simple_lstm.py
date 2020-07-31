
##############################################################################
# Simple example of an LSTM for time series prediction
#
# Usage:
#    python3 simple_lstm.py
#
# Adapted from
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
#from keras.datasets import fashion_mnist

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
i_train_samplesize = 100
i_test_samplesize  = 20
i_num_classes      = 10

# Data from https://datamarket.com/data/set/22u3/international-airline-passengers-monthly-totals-in-thousands-jan-49-dec-60#!ds=22u3&display=line
dataset = pd.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()

#ataset = dataframe.values
dataset = dataset.astype('float32')

#####################################################################################
# Feature scaling
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX  = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


###################################################
# Model parameters
###################################################
f_dropout             = 0.70
i_fitting_epochs      = 100
i_batch_size          = 2
f_validation_split    = 0.33
str_optimizer         = 'adam'
i_num_neurons_layer_1 = 256
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

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=4, input_shape=(1, look_back)))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer=str_optimizer)
model.fit(trainX, trainY, epochs=i_fitting_epochs, batch_size=i_batch_size, verbose=2)

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=str_optimizer, metrics=['accuracy'])


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


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
str_model_save_filename = "model_file_convnet.h5"
#model.save(str_model_save_filename)
#model_saved = load_model(str_model_save_filename)

print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")



exit()


# TODO: FIX ERROR



#####################################################################################
# Plot model fit over training epochs
#####################################################################################
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
history = model.fit(trainX, trainY, validation_split=f_validation_split, batch_size=i_batch_size, epochs=i_fitting_epochs)
score = model.evaluate(testX, testY, batch_size=i_batch_size)
print(score)

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
plt.savefig('lstm_learning_curve_accuracy.png', dpi=300)
plt.show()
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
plt.savefig('convnet_learning_curve_loss.png', dpi=300)
plt.show()




exit()




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
#probas = model.predict(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test_array, batch_size=i_batch_size)
probas = model.predict(x_test) # x_test

#pdb.set_trace()
#fpr, tpr, thresholds = roc_curve(y_test, probas[:,1], pos_label=1)
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
#plt.savefig('convnet_roc_curve.png', dpi=300)


