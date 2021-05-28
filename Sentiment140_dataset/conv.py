import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D, MaxPooling1D, Flatten, Conv1D
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
print(colored("Tokenizing and padding complete", "yellow"))

# Building the model
print(colored("Creating the CONVOLUTION model", "yellow"))
model = Sequential()
model.add(Embedding(15001, 300, input_length = train_tweets.shape[1], trainable = False))
model.add(Conv1D(filters=64, kernel_size=3, padding = 'valid', activation='relu', strides = 1))
model.add(Conv1D(filters=32, kernel_size=3, padding = 'valid', activation='relu', strides = 1))
model.add(MaxPooling1D(3))
model.add(Conv1D(filters=16, kernel_size=3, padding = 'valid', activation='relu', strides = 1))
model.add(Conv1D(filters=8, kernel_size=3, padding = 'valid', activation='relu', strides = 1))
model.add(GlobalAveragePooling1D())
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# Training the model
print(colored("Training the CONV model", "green"))
history = model.fit(train_tweets, pd.get_dummies(train_data['Sentiment']).values, epochs = 10, batch_size = 4096, validation_split = 0.2)
print(colored(history, "green"))

# Testing the model
print(colored("Testing the CONV model", "green"))
score, accuracy = model.evaluate(test_tweets, pd.get_dummies(test_data['Sentiment']).values, batch_size = 4096)
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
my_dict2 = [{'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F_score': fscore, 'AUC': auc, 'Model': 'CNN', 'Textmining_model':'Word_embedding'}]
fields = ['Textmining_model', 'Model', 'Accuracy', 'Precision', 'Recall', 'F_score', 'AUC']
filename = "Sentiment1.6M_Tweets.csv"
#writing data to csv file

with open(filename, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writerows(my_dict2)
    
    csvfile.close()







