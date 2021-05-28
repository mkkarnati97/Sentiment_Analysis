import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten
from keras.preprocessing.text import Tokenizer


from keras.preprocessing.sequence import pad_sequences

import pandas as pd
from termcolor import colored

# Load data
print(colored("Loading train and test data", "yellow"))
train_data = pd.read_csv('data_clean_train.csv')
test_data = pd.read_csv('data_clean_test.csv')
print(colored("Data loaded", "yellow"))

# Tokenization
print(colored("Tokenizing and padding data", "yellow"))
tokenizer = Tokenizer(num_words = 15000, split = ' ')
tokenizer.fit_on_texts(train_data['Clean_tweet'].astype(str).values)
train_tweets = tokenizer.texts_to_sequences(train_data['Clean_tweet'].astype(str).values)
max_len = max([len(i) for i in train_tweets])
train_tweets = pad_sequences(train_tweets, maxlen = max_len)
test_tweets = tokenizer.texts_to_sequences(test_data['Clean_tweet'].astype(str).values)
test_tweets = pad_sequences(test_tweets, maxlen = max_len)
vocab_size = len(tokenizer.word_index) + 1
print(colored("Tokenizing and padding complete", "yellow"))

# Building the model
print(colored("Creating the DNN model", "yellow"))
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length = train_tweets.shape[1]))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Training the model
print(colored("Training the DNN model", "green"))
history = model.fit(train_tweets, pd.get_dummies(train_data['Sentiment']).values, epochs = 5, batch_size = 128, verbose=1, validation_split = 0.2)
print(colored(history, "green"))

# Testing the model
print(colored("Testing the DNN model", "green"))
score, accuracy = model.evaluate(test_tweets, pd.get_dummies(test_data['Sentiment']).values, batch_size = 128, verbose=1)
print("Test accuracy: {}".format(accuracy))

from sklearn import metrics
test_prediction = model.predict_classes(test_tweets)
print(metrics.classification_report(test_data['Sentiment'], test_prediction))

test_pred_prob = model.predict(test_tweets)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_data['Sentiment'], test_pred_prob[:, 1], average = 'macro')
precision, recall, fscore, support = score(test_data['Sentiment'], test_prediction, average = 'macro')


#importing the csv module 
import csv

#evaluation data as dictionary object
my_dict1 = [{'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F_score': fscore, 'AUC': auc, 'Model': 'DNN', 'Textmining_model':'Word_embedding'}]
#field names
fields = ['Textmining_model', 'Model', 'Accuracy', 'Precision', 'Recall', 'F_score', 'AUC']

#name of csv file

filename = "IMDB_Tweets.csv"

#writing data to csv file

with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    writer.writerows(my_dict1)







































