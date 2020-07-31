#####################################################
# Autoencoder using keras and tensorflow
#       applied to a dataset from the UCI machine
#       learning repository
#
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
#   python3 basic_autoencoder_keras.py
#
# Adapted from:
#   https://blog.keras.io/building-autoencoders-in-keras.html
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
from keras.models import Model
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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.datasets import mnist


def function_tsne_generic(f_matrix, perplexities):

    """
    Function for tSNE
    Returns:

    Code adapted from:
    https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html

    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    from sklearn import manifold, datasets

    # n_samples = 300
    n_components = 2
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))

    # X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # str_file_name = "breast-cancer-wisconsin_MOD.data"
    ## X_withcode = pd.read_csv(str_file_name)#, header=None)

    # modifications here and data munging
    # X = X_withcode.iloc[:,1:] # ignore code column
    X = f_matrix
    # pdb.set_trace()

    for i, perplexity in enumerate(perplexities):
        ax = subplots[1][i + 1]

        # t0 = time()
        tsne = manifold.TSNE(n_components=n_components, init='random',
                             random_state=0, perplexity=perplexity)
        Y = tsne.fit_transform(X)
        # t1 = time()
        # print("S-curve, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1],  cmap=plt.cm.viridis) # c=color,
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')



    # plt.show()
    plt.savefig('autoencoder_tsne_visualization.png')

    # pdb.set_trace()



###################################################
# Model parameters
###################################################
# see https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
f_dropout = 0.20
i_num_neurons_layer_1 = 10
i_num_neurons_layer_2 = 4
str_activation_layer1 = 'relu'  # tanh
str_activation_layer2 = 'sigmoid'
f_learning_rate = 0.01
f_learning_rate_decay = 1e-6
f_momentum = 0.9
i_fitting_epochs = 100
i_batch_size = 320
f_validation_split = 0.33
# k_kernel_initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
# k_kernel_initializer = keras.initializers.glorot_uniform(seed=None) # Xavier Initialization
k_kernel_initializer = keras.initializers.Orthogonal(gain=1.0, seed=None)
k_kernel_regularizer = keras.regularizers.l1(0.00001)
k_activity_regularizer = keras.regularizers.l1(0.00001)
# k_kernel_regularizer   = keras.regularizers.l1_l2(l1=0.01, l2=0.01)
sgd = keras.optimizers.SGD(lr=f_learning_rate, decay=f_learning_rate_decay,
                           momentum=f_momentum, nesterov=True)


###################################################
# Initialize model
###################################################
print("\n ********** Model Creation Section ********** \n")
print("Creating deep learning model: \n")

i_input_shape_x = 28
i_input_shape_y = 28
i_original_dim = 784
i_encoding_dim = 32

input_img = Input(shape=(i_original_dim,))

# goes from 784 to 32
encoded = Dense(units=i_encoding_dim, activation=str_activation_layer1 ) #,
                # kernel_regularizer=k_kernel_regularizer,
                # activity_regularizer=k_activity_regularizer
                # )

# encoded object will take the input of input_img
encoded(inputs=input_img)

# back to number of dimensions you had originally
decoded = Dense(units=i_original_dim, activation=str_activation_layer2)

# decoded will take the output of encoder which is
decoded(inputs=encoded(inputs=input_img))

######################################################################################################
# these are all inputs

# we need to create the model
#   first is input of class Input
#   second is output Tensor (not of class Input)
#
#   CONCEPT: when you create the model, you define the path that the data will follow,
#       which in this case is from the input to the output:
#   See:
#       https://stackoverflow.com/questions/43765381/anyone-some-api-about-the-keras-layers-input
######################################################################################################
model_autoencoder = Model( input_img, decoded(inputs=encoded(inputs=input_img)) )

model_autoencoder.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model_autoencoder.summary()

###################################################
# Now create decoder model
# Before you create that recall that the input to decoder model
#       needs to be of class Input

# so create an Input class for decoder
###################################################

input_to_decoder = Input(shape=(i_encoding_dim,))


model_autoencoder.layers

# Last layer of autoencoder Dense object
dense_decoder = model_autoencoder.layers[-1]

# NOTE: dense_decoder is an object of type Dense()
# hence you need to give it an object of type Input()
# In this case this will be the output of encoder
#       also called input_to_decoder

dense_decoder(input_to_decoder)

# Putting it altogether

# Now decoder model has
#   input:  input_to_decoder
#   output:
model_decoder = Model(input_to_decoder, dense_decoder(input_to_decoder)  )



#####################################################################################
#  Load data
#####################################################################################
print("\n ********** Data Loading Section ********** \n")
print("Loading dataset: \n")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# rescale to be between 0 and 1
x_train = x_train.astype('float32')/255
y_train = y_train.astype('float32')/255
x_test  = x_test.astype('float32')/255
y_test  = y_test.astype('float32')/255

x_train.shape

x_train.shape[1:] # 28 x 28
np.prod(x_train.shape[1:])

#####################################################################################
# Feature scaling
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")

# flatten to vector for each image (60000 x 784)
tpl_flatten_new_dimensions = (  len(x_train),  np.prod(x_train.shape[1:])  )
x_train = np.reshape( x_train,  tpl_flatten_new_dimensions )

tpl_flatten_new_dimensions = (  len(x_test),  np.prod(x_test.shape[1:])  )
x_test = np.reshape( x_test,  tpl_flatten_new_dimensions )



#####################################################################################
# Plot model fit over training epochs
#  https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
#####################################################################################

# Fit model
history = model_autoencoder.fit(x_train, x_train, validation_split=f_validation_split, epochs=i_fitting_epochs,
                    batch_size=i_batch_size, verbose=1)

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
plt.savefig('autoencoder_learning_curve_accuracy.png', dpi=300)
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
plt.savefig('autoencoder_learning_curve_loss.png', dpi=300)
# plt.show()


#####################################################################################
# Visualize encoder and decoder model output
#       predict on test sets
#   With a difference for autoencoders
#   since we are not interested in the final model output
#       just in the intermediate model output
#####################################################################################

#####################################################################################
# This is also Generate predictions on new data:
#####################################################################################
print("\n ********** Model Prediction and Visualization Section ********** \n")
print("Printing model prediction details on test set: \n")
print("Visualize encoder and decoder model output on test set: \n")

# predict on test set using encoder
#  NOT autoencoder
# the following is wrong
# predictions_test_encoded = model_autoencoder.predict(x_test)

# CONCEPT: the autoencoder model is the complete model consisting of
#           BOTH encoder and decoder

# we will now build an encoder Model
#   input:  input_img
#   output: encoded Dense object
model_encoder = Model(input_img, encoded(inputs=input_img))

# model_encoder.compile()
predictions_testset_model_encoder = model_encoder.predict(x_test)

# plot and visualize encoded matrix in tSNE space
perplexities = [5, 30, 50, 100]
function_tsne_generic(f_matrix=predictions_testset_model_encoder, perplexities=perplexities)


# feed these predictions to decoder
predictions_testset_model_decoder = model_decoder.predict(predictions_testset_model_encoder)


###################################################
# Now plot images
###################################################

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))

for i_temp_counter in np.arange(0, n):

    # show original figure
    ax = plt.subplot(2, n, i_temp_counter+1)
    plt.imshow(x_test[i_temp_counter].reshape( (i_input_shape_x, i_input_shape_y) )) # 28 x 28
    # plt.gray()


    # show encoded figure
    ax = plt.subplot(2, n, i_temp_counter + 1 + n)
    plt.imshow(predictions_testset_model_decoder[i_temp_counter].reshape( (i_input_shape_x, i_input_shape_y) )) # 28 x 28


plt.savefig('autoencoder_original_and_encoded_images.png')



#####################################################################################
# Visualize balance or imbalance of training data
#####################################################################################
plt.figure(figsize=(8, 4))
sns.countplot(x=y_train)
plt.savefig('autoencoder_balance_trainingset.png', dpi=300)

plt.figure(figsize=(8, 4))
sns.countplot(x=y_test)
plt.savefig('autoencoder_balance_testset.png', dpi=300)

#####################################################################################
# Print summary of a model and inspect model
#####################################################################################
print("\n ********** Model Summary Section ********** \n")
print("Printing model summary and model details: \n")
print(model_autoencoder.summary())
keras.utils.print_summary(model_autoencoder, line_length=None, positions=None, print_fn=None)
print(model_autoencoder.output_shape)
print(model_autoencoder.input_shape)
print(model_autoencoder.get_config())
print(model_autoencoder.get_weights())

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
print(
"Kernel initialization: ", "Orthogonal")  # k_kernel_initializer   = keras.initializers.Orthogonal(gain=1.0, seed=None)
print("Kernel regularization: ", "L1")  # k_kernel_regularizer   = keras.regularizers.l1(0.01)
print("Kernel activity initialization: ", "L1")  # k_activity_regularizer = keras.regularizers.l1(0.01)
print("**************************************")

#####################################################################################
# Plot model
#####################################################################################
# plot_model(model, to_file='model.png')
keras.utils.plot_model(model_autoencoder, to_file='autoencoder_keras_model.png', show_shapes=True,
                       show_layer_names=True, rankdir='TB')

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
# Penalize cost function for rare events
#####################################################################################
# TODO: penalize cost function
#
# More negative examples than positive;
#       https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras


#####################################################################################
# Save model
#####################################################################################
print("\n ********** Model Save Section ********** \n")
print("       Saving model ......  \n")
str_model_save_filename = "model_file_uci_dataset.h5"
# model.save(str_model_save_filename)
# model_saved = load_model(str_model_save_filename)


print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")


exit()






# TODO: clean up later















###################################################
# Adding layers
# Stacking layers is as easy as .add():
###################################################
# model.add(Dense(units=64, activation='relu', input_dim=9))
model.add(Dense(units=i_num_neurons_layer_1, activation=str_activation_layer1, input_dim=8,
                kernel_initializer=k_kernel_initializer,
                kernel_regularizer=k_kernel_regularizer,
                activity_regularizer=k_activity_regularizer
                )
          )
# model.add(Dense(units=2, activation='softmax'))

###################################################
# Model regularization
# see https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
###################################################
model.add(Dropout(f_dropout))
model.add(Dense(units=i_num_neurons_layer_2, activation=str_activation_layer2))

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
#  Load data
#####################################################################################
print("\n ********** Data Loading Section ********** \n")
print("Loading dataset: \n")

# read in pandas dataframe
# generated from breast-cancer-wisconsin_MOD.data using data_munging.R
temp_str_peptide_file = "breast-cancer-wisconsin_MOD_CURATED.data"
temp_peptide_df = pd.read_csv(temp_str_peptide_file, header=None)
#            temp_peptide_df.columns = ['peptide', 'abundance']

#####################################################################################
# Split data into training and test set
#####################################################################################
print("\n ********** Train Test Split Section ********** \n")
print("Splitting data into training and test set: \n")

i_split_train_test_from = 500
x_train = temp_peptide_df.iloc[0:i_split_train_test_from, 1:-1]  # temp_peptide_df["epithelial_cell_size"]
y_train = temp_peptide_df.iloc[0:i_split_train_test_from, -1]  # temp_peptide_df["class"]

x_train_array = np.array(x_train)
# y_train_array = np.array(y_train)


#####################################################################################
# Data munging
#####################################################################################


#####################################################################################
# Feature scaling
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing feature scaling: \n")

x_train_array = keras.utils.normalize(x_train_array)
# y_train_array = keras.utils.normalize(y_train_array)

# y_binary = to_categorical(y_int)
# y_train = to_categorical(y_train)#,2)


# TODO:
# `categorical_crossentropy` expects targets to be binary matrices (1s and 0s) of shape (samples, classes). If your targets are integer classes, you can convert them to the expected format via:
# from keras.utils import to_categorical
# y_binary = to_categorical(y_int)
# Alternatively, you can use the loss function `sparse_categorical_crossentropy` instead, which does expect integer targets.

#####################################################################################
# Cross validation
#       1. determine hyperparameters
#       2. Determine regularization coeffcients
#####################################################################################

# cross validation as in
# https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
kfold = StratifiedKFold(n_splits=10, shuffle=True)  # , random_state=seed)
# results = cross_val_score(model, x_train_array, y_train, cv=kfold)
# print(results.mean())
# model.fit(x_train_array, y_train, epochs=i_fitting_epochs, batch_size=320)



#####################################################################################
# Evaluate your performance in one line:
#####################################################################################
x_test = temp_peptide_df.iloc[i_split_train_test_from:, 1:-1]  # temp_peptide_df["epithelial_cell_size"]
y_test = temp_peptide_df.iloc[i_split_train_test_from:, -1]  # temp_peptide_df["class"]

x_test_array = np.array(x_test)
# feature scaling
x_test_array = keras.utils.normalize(x_test_array)

# y_test = to_categorical(y_test)#,2)

print("\n ********** Model Evaluation Section ********** \n")
print("Printing model performance details on test set: \n")

loss_and_metrics = model.evaluate(x_test_array, y_test, batch_size=i_batch_size)
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
probas = model.predict(x_test_array)  # x_test

# pdb.set_trace()
# fpr, tpr, thresholds = roc_curve(y_test[:,1], probas[:,1], pos_label=1)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1], pos_label=1)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

mean_tpr += interp(mean_fpr, fpr, tpr)
mean_tpr[0] = 0.0
roc_auc = auc(fpr, tpr)

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
plt.savefig('roc_curve.png', dpi=300)


