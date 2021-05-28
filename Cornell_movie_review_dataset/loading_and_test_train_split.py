import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)
from wordcloud import WordCloud, STOPWORDS
dataset = pd.read_csv('movie_dataset.csv')
dataset.columns

data = dataset
data.head(10)

class_count = data['Sentiment'].value_counts() # Returned in descending order [1, 0]
plt.figure(figsize = (12, 8))
plt.xticks([1, 0], ['Positive', 'Negative'])
plt.xticks([1, 0])
plt.bar(x = class_count.keys(), 
        height = class_count.values, 
        color = ['g', 'r'])
plt.xlabel("Tweet sentiment")
plt.ylabel("Tweet count")
plt.title("Count of tweets for each sentiment")
plt.legend()

print(data['review'][0])
print(data['review'][1])

positive_tweets = ' '.join(data[data['Sentiment'] == 1]['review'].str.lower())
negative_tweets = ' '.join(data[data['Sentiment'] == 0]['review'].str.lower())


wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", max_words = 1000).generate(positive_tweets)
plt.figure(figsize = (12, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Positive tweets Wordcloud")


wordcloud = WordCloud(stopwords = STOPWORDS, background_color = "white", max_words = 1000).generate(negative_tweets)
plt.figure(figsize = (12, 8))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Negative tweets Wordcloud")

from termcolor import colored
from sklearn.model_selection import train_test_split


# Train test split
print(colored("Splitting train and test dataset into 80:20", "yellow"))
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['Sentiment'], test_size = 0.20, random_state = 100)
train_dataset = pd.DataFrame({
	'Tweet': X_train,
	'Sentiment': y_train
	})
print(colored("Train data distribution:", "yellow"))
print(train_dataset['Sentiment'].value_counts())
test_dataset = pd.DataFrame({
	'Tweet': X_test,
	'Sentiment': y_test
	})
print(colored("Test data distribution:", "yellow"))
print(test_dataset['Sentiment'].value_counts())
print(colored("Split complete", "yellow"))

# Save train data
print(colored("Saving train data", "yellow"))

train_dataset.to_csv('train.csv', index = False)
print(colored("Train data saved to train.csv", "green"))

# Save test data
print(colored("Saving test data", "yellow"))
test_dataset.to_csv('test.csv', index = False)
print(colored("Test data saved to test.csv", "green"))
