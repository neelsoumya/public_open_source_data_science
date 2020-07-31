
##############################################################################
# Simple example of an LSTM for generic sequence classification
#
# Usage:
#    python3 generic_sequence_lstm.py
#
# Adapted from
#    https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb
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

import csv
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def function_tsne_generic(f_matrix, perplexities, str_save_file_tsne):

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
    # from yellowbrick.text import TSNEVisualizer

    # n_samples = 300
    n_components = 2
    i_num_panels_per_row = 5
    i_num_rows = int( np.ceil(len(perplexities)/i_num_panels_per_row) )

    (fig, subplots) = plt.subplots(i_num_rows, i_num_panels_per_row, figsize=(15, 8))

    # X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # str_file_name = "breast-cancer-wisconsin_MOD.data"
    ## X_withcode = pd.read_csv(str_file_name)#, header=None)

    # modifications here and data munging
    # X = X_withcode.iloc[:,1:] # ignore code column
    X = f_matrix
    # pdb.set_trace()

    for i, perplexity in enumerate(perplexities):

        i_row_number = i%i_num_panels_per_row
        j_col_number = int( np.floor(i/i_num_panels_per_row) )

        #print(i)
        #print(j_col_number, i_row_number)

        ax = subplots[j_col_number][i_row_number]

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
    plt.savefig(str_save_file_tsne)

    # pdb.set_trace()

    return (Y)



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

max_features     = 1000
# cut texts after this number of words (among top max_features most common words)
vocab_size       = 1000 # how many unique words
maxlen           = 120
i_num_classes    = 2 - 1
trunc_type       = "post"
padding_type     = "post"
oov_token        = "<oov>"
training_portion = 0.8


# TODO: load BBC data
#   https://storage.googleapis.com/laurencemoroney-blog.appspot.com/bbc-text.csv
# TODO: use tokenizer
#   https://github.com/neelsoumya/nlp_resources/blob/master/NLP_Course_bbc_data_munging_Week_2_Exercise_Answer.ipynb
# https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb

sentences = []
labels    = []
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
             "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
             "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
             "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
             "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
             "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
             "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
             "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves" ]

# open file and read and do data munging
# str_input_filename = 'bbc-text.csv'
# modified file with binary 0 1 target labels
str_input_filename = 'bbc-text_mod.csv'
with open(str_input_filename, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)

    # for every row
    for row in reader:
        labels.append(row[0])
        sentence = row[1]

        # in this sentence, replace stopwords with ''
        for word in stopwords:
            sentence.replace(" " + word + " ", " ")

        # append this sentence to sentences
        sentences.append(sentence)

    # next row


# print(sentences[1])


# split into training and test
train_size = int(len(sentences) * training_portion)

train_sentences = sentences[:train_size]
train_labels    = labels[:train_size]

test_sentences  = sentences[train_size:]
test_labels     = labels[train_size:]


#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#print(len(x_train), 'train sequences')
#print(len(x_test), 'test sequences')


#####################################################################################
# Feature scaling and data munging
#####################################################################################
print("\n ********** Data Munging Section ********** \n")
print("Performing tokenizing, token to sequence and padding: \n")

# tokenize
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

# convert token to sequence
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded    = pad_sequences(train_sequences, padding=padding_type, maxlen=maxlen)

test_sequences  = tokenizer.texts_to_sequences(test_sentences)
test_padded     = pad_sequences(test_sequences, padding=padding_type, maxlen=maxlen)

print(train_padded[1])

# tokenize labels
# label_tokenizer = Tokenizer()
# label_tokenizer.fit_on_texts(labels)

# training_label_seq = label_tokenizer.texts_to_sequences(train_labels)
# training_label_seq = np.array(training_label_seq)

# test_label_seq     = label_tokenizer.texts_to_sequences(test_labels)
# test_label_seq     = np.array(test_label_seq)

# print(training_label_seq)

# '1' and '0' convert to 1 and 0
# TODO: remove if multiple categories
train_labels = [int(x) for x in train_labels]
test_labels  = [int(x) for x in test_labels]

# TODO: remove if multiple categories
training_label_seq = train_labels
test_label_seq     = test_labels
# pdb.set_trace()


# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test  = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)


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
str_activation_layer2 = 'sigmoid' # 'softmax'
i_stride_length       = 1
k_kernel_initializer  = keras.initializers.glorot_uniform(seed=None) # Xavier Initialization
str_loss_function     = 'binary_crossentropy' # 'categorical_crossentropy' #

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
model.add(Embedding(input_dim=vocab_size, output_dim=i_num_neurons_layer_1))
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
history = model.fit(train_padded, training_label_seq,
                    batch_size=i_batch_size,
                    epochs=i_fitting_epochs,
                    # validation_data=(x_test, y_test),
                    validation_split = f_validation_split,
                    verbose=0
                    )

score, acc = model.evaluate(test_padded, test_label_seq,
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

classes = model.predict(test_padded, batch_size=i_batch_size)
print(test_labels)
print(classes)
model.predict_classes(test_padded, batch_size=i_batch_size, verbose=0)#, steps=1)
#pdb.set_trace()

#####################################################################################
# Plot AUPR curves or ROC curves
# https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb
#####################################################################################
# TODO: AUPR curves not ROC
# see https://hackernoon.com/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier-2ecc6c73115a
#pdb.set_trace()
#probas = model.predict_proba(x_test_array)
probas = model.predict(test_padded, batch_size=i_batch_size)
#probas = model.predict_classes(x_test, batch_size=i_batch_size)
#probas = model.predict_classes(x_test_array, batch_size=i_batch_size)
#probas = model.predict(x_test) # x_test

#pdb.set_trace()
fpr, tpr, thresholds = roc_curve(test_labels, probas, pos_label=1)
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
# Visualize embedding
#####################################################################################
embedding_layer_model = model.layers[0]
embedding_weights = embedding_layer_model.get_weights()[0]

# plot and visualize encoded matrix in tSNE space
perplexities = [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
str_output_filename_prefix = 'imdb_sentiment_lstm_'
tsne_reduced_matrix = function_tsne_generic(f_matrix=embedding_weights,
                                            perplexities=perplexities,
                                            str_save_file_tsne=str_output_filename_prefix + 'autoencoder_tsne_visualization.png')

# save this in a file also
f_file_save = open('tsne_save.csv', 'w')
for item in embedding_weights:
    f_file_save.writelines(str(item))
    f_file_save.writelines('\n')

f_file_save.close()

f_file_save = open('tsne_reduced_matrix_save.csv', 'w')
for item in tsne_reduced_matrix:
    f_file_save.writelines(str(item))
    f_file_save.writelines('\n')

f_file_save.close()

# reverse word index and visualize on
# http://projector.tensorflow.org/
reverse_word_index = dict( [(value,key) for (key,value) in word_index.items() ] )

def decode_sentence(text):
    return (  ' '.join( reverse_word_index.get(i, '?') for i in text  ) )


out_embedding = io.open('vecs.tsv', 'w', encoding='utf-8')
out_words     = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):
    word = reverse_word_index[word_num]
    embedding = embedding_weights[word_num]

    out_words.write(word + "\n")
    out_embedding.write("\t".join( [str(x) for x in embedding] ) + "\n" )

out_embedding.close()
out_words.close()


# TODO: plot scatter plot of these tsne latent dimensions or encoder weights on test set labels
# plt.figure(figsize=(7,5))
# plt.scatter(tsne_reduced_matrix[:,0], tsne_reduced_matrix[:,1], c=y_test)
# plt.colorbar()
# plt.savefig('tsne_reduced_matrix_colored_by_label.png', dpi=300)

# TODO: plot scatter plot of these latent dimensions or encoder weights on test set
# plt.figure(figsize=(7, 5))
# plt.scatter(embedding_weights[:,0], embedding_weights[:,1], c=y_test)
# plt.colorbar()
# plt.savefig('reduced_dimensions_plot.png', dpi=300)


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
model.save(str_model_save_filename)
#model_saved = load_model(str_model_save_filename)

print("\n ***************************************** \n")
print("   All tasks successfully completed \n")
print(" ***************************************** \n")
