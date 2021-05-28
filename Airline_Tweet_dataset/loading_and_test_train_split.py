import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import warnings
warnings.filterwarnings('ignore', category = DeprecationWarning)
from wordcloud import WordCloud, STOPWORDS
dataset = pd.read_csv('Tweets.csv')
dataset.columns
plot_size = plt.rcParams["figure.figsize"] 
print(plot_size[0]) 
print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

dataset.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')

dataset.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
airline_sentiment = dataset.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
airline_sentiment.plot(kind='bar')

sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data= dataset)

from termcolor import colored
print(colored("Useful columns: Sentiment and Tweet", "yellow"))
print(colored("Removing other columns", "red"))
dataset.drop(['tweet_id', 'airline_sentiment_confidence',
       'negativereason', 'negativereason_confidence', 'airline',
       'airline_sentiment_gold', 'name', 'negativereason_gold',
       'retweet_count', 'tweet_coord', 'tweet_created',
       'tweet_location', 'user_timezone'], axis = 1, inplace = True)
print(colored("Columns removed", "red"))

index_names = dataset[ dataset['airline_sentiment'] == 'neutral' ].index
dataset.drop(index_names, axis = 0, inplace = True)
air_data = dataset.reset_index(drop=True)

air_data['Sentiment'] = air_data['airline_sentiment'].replace(['positive'], 1)
air_data['Sentiment'] = air_data['Sentiment'].replace(['negative'], 0)
col_to_drop = ['airline_sentiment']
data = air_data.drop(col_to_drop, axis = 1)

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

print(data['text'][0])
print(data['text'][1])

positive_tweets = ' '.join(data[data['Sentiment'] == 1]['text'].str.lower())
negative_tweets = ' '.join(data[data['Sentiment'] == 0]['text'].str.lower())


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

from sklearn.model_selection import train_test_split


# Train test split
print(colored("Splitting train and test dataset into 80:20", "yellow"))
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['Sentiment'], test_size = 0.20, random_state = 100)
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
