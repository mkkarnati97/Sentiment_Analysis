import numpy as np
import pandas as pd
from termcolor import colored
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, GlobalAveragePooling1D, MaxPooling1D, Flatten, Conv1D, Dropout

print(colored("Loading train and test data", "yellow"))
train_data = pd.read_csv('data_clean_train.csv')
test_data = pd.read_csv('data_clean_test.csv')
print(colored("Data loaded", "yellow"))

# Tf-IDF
print(colored("Applying TF-IDF transformation", "yellow"))
tfidfVectorizer = TfidfVectorizer(min_df = 5, max_features = 1000)
tfidfVectorizer.fit(train_data['Clean_tweet'].apply(lambda x: np.str_(x)))


train_tweet_vector = tfidfVectorizer.transform(train_data['Clean_tweet'].apply(lambda x: np.str_(x)))
test_tweet_vector = tfidfVectorizer.transform(test_data['Clean_tweet'].apply(lambda x: np.str_(x)))
train_tweet_vector = train_tweet_vector.toarray()
test_tweet_vector = test_tweet_vector.toarray()

print(train_tweet_vector.shape)
print(test_tweet_vector.shape)


# Building a covn model

print(colored("Creating the DNN model", "yellow"))
model = Sequential()
model.add(Dense(1000,input_shape= (1000,), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

# Training the model
print(colored("Training the DNN model", "green"))
history = model.fit(train_tweet_vector, pd.get_dummies(train_data['Sentiment']).values, epochs = 10, batch_size = 64, verbose = 1, validation_split = 0.2)
print(colored(history, "green"))

# Testing the model
print(colored("Testing the DNN model", "green"))
score, accuracy = model.evaluate(test_tweet_vector, pd.get_dummies(test_data['Sentiment']).values, batch_size = 64)
print("Test accuracy: {}".format(accuracy))


from sklearn import metrics
test_prediction = model.predict_classes(test_tweet_vector)
print(metrics.classification_report(test_data['Sentiment'], test_prediction))

test_pred_prob = model.predict(test_tweet_vector)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(test_data['Sentiment'], test_pred_prob[:, 1], average = 'macro')
precision, recall, fscore, support = score(test_data['Sentiment'], test_prediction, average = 'macro')


#importing the csv module 
import csv

#evaluation data as dictionary object
my_dict4 = [{'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F_score': fscore, 'AUC': auc, 'Model': 'DNN', 'Textmining_model':'TF_IDF'}]
fields = ['Textmining_model', 'Model', 'Accuracy', 'Precision', 'Recall', 'F_score', 'AUC']
filename = "Sentiment1.6M_Tweets.csv"
#writing data to csv file

with open(filename, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writerows(my_dict4)
    
    csvfile.close()