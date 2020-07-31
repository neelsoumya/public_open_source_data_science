
##############################################################################
# Simple example of a convolutional neural net
#
# Usage:
#    python3 simple_convnet.py
#
# Adapted from
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
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
# from keras.datasets import fashion_mnist

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from scipy import interp

import pdb
# import matplotlib.pyplot as plt
import numpy as np


###################################################
# Generate synthetic data
###################################################
print("\n ********** Data Loading Section ********** \n")
print("Loading dataset: \n")
i_train_samplesize = 100
i_test_samplesize  = 20
i_num_classes      = 10

# first modality
x_train = np.random.random((i_train_samplesize, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(i_num_classes, size=(100, 1)), num_classes=i_num_classes)
x_test = np.random.random((i_test_samplesize, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(i_num_classes, size=(i_test_samplesize, 1)), num_classes=i_num_classes)

# second modality
x_train_2 = np.random.random((i_train_samplesize, 100, 100, 3))
x_test_2  = keras.utils.to_categorical(np.random.randint(i_num_classes, size=(i_test_samplesize, 1)), num_classes=i_num_classes)

#####################################################################################
# Feature scaling
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")
x_train = keras.utils.normalize(x_train)
x_test  = keras.utils.normalize(x_test)

###################################################
# Model parameters
###################################################
f_dropout             = 0.70
i_fitting_epochs      = 10
i_batch_size          = 32
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

model = Sequential()

# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(BatchNormalization(axis=1, input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)) )#, kernel_initializer=k_kernel_initializer)
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(f_dropout))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(f_dropout))

model.add(Flatten())

# first input model
visible1 = Input(shape=(100,100,3))
conv11 = Conv2D(32, kernel_size=4, activation='relu')(visible1)
pool11 = MaxPooling2D(pool_size=(2, 2))(conv11)
conv12 = Conv2D(16, kernel_size=4, activation='relu')(pool11)
pool12 = MaxPooling2D(pool_size=(2, 2))(conv12)
flat1 = Flatten()(pool12)

# second input model
visible2 = Input(shape=(100,100,3))
conv21 = Conv2D(32, kernel_size=4, activation='relu')(visible2)
pool21 = MaxPooling2D(pool_size=(2, 2))(conv21)
conv22 = Conv2D(16, kernel_size=4, activation='relu')(pool21)
pool22 = MaxPooling2D(pool_size=(2, 2))(conv22)
flat2 = Flatten()(pool22)

# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
# plot_model(model, to_file='multiple_inputs_deeplearning.png')
# keras.utils.plot_model(model, to_file='graph_multiple_inputs_deeplearning_detailed.png', show_shapes=True,
#                       show_layer_names=True, rankdir='TB')


# model.add(Dense(i_num_neurons_layer_1, activation='relu'))
# model.add(Dropout(f_dropout))
# model.add(Dense(i_num_classes, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=str_optimizer, metrics=['accuracy'])


# exit()

#####################################################################################
# Plot model fit over training epochs
#####################################################################################
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
pdb.set_trace()
history = model.fit( np.array((x_train, x_train_2)), (y_train,y_train), validation_split=f_validation_split, batch_size=i_batch_size, epochs=i_fitting_epochs)
score = model.evaluate(x_test, y_test, batch_size=i_batch_size)
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
plt.savefig('convnet_learning_curve_accuracy.png', dpi=300)
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
plt.savefig('convnet_learning_curve_loss.png', dpi=300)
#plt.show()

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
