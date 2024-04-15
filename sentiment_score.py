import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\acer\Dropbox\PC\Downloads\tfidf_dataset.csv')

df.head(5)

# Replace NaN values with an empty string
df['combined_text'] = df['combined_text'].fillna('')

# Convert all values to string
df['combined_text'] = df['combined_text'].astype(str)

from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Tokenize the combined text
tokens = df['combined_text'].apply(word_tokenize)

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def get_sentiment_score(tokens):
    # Calculate sentiment score if tokens exist
    if tokens:
        scores = [sid.polarity_scores(token)['compound'] for token in tokens]
        # Return the average sentiment score
        return sum(scores) / len(scores)
    else:
        # Return a default value when no tokens are present
        return 0

# Calculate sentiment score for each row
df['sentiment_score'] = tokens.apply(get_sentiment_score)

print(df[['combined_text', 'sentiment_score']])

df.head()

# Saving the processed data to a CSV file
df.to_csv('senti_dataset.csv',index=False)
