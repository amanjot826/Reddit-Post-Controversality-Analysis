# importing the required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# loading the pre-processed dataset
sampled_df = pd.read_csv(r'C:\Users\acer\Dropbox\PC\Downloads\balanced_dataset.csv')

# displaying the dataset
sampled_df.head()

# replacing NaN values with empty strings in 'post_title' and 'self_text' columns
sampled_df['post_title'].fillna('', inplace=True)
sampled_df['self_text'].fillna('', inplace=True)

# combining text from multiple columns into a single Series
combined_text = sampled_df['self_text'] + sampled_df['post_title']

# replacing NaN values with an empty string
combined_text_cleaned = combined_text.fillna('')

combined_text_cleaned

# creating a TF-IDF vectorizer
import scipy.sparse
tfidf_vectorizer = TfidfVectorizer()

# fitting the vectorizer to the documents and transforming them into TF-IDF features
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text_cleaned)

# getting feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# function to get the top 10 words for each document
def get_top_words(matrix_row, feature_names, top_n=7):
    sorted_indices = matrix_row.argsort()[::-1]
    top_words_indices = sorted_indices[:top_n]
    top_words = [feature_names[i] for i in top_words_indices]
    return top_words

# getting the top 10 words for each document
top_words_per_doc = [get_top_words(row.toarray().flatten(), feature_names, top_n=10) for row in tfidf_matrix]

# creating a column named top_words_per_doc in the dataset to store word frequencies
sampled_df['top_words_per_doc'] = top_words_per_doc

# combining 'post_title' and 'self_text' into a single column 'combined_text'
sampled_df['combined_text'] = sampled_df['post_title'] + ' ' + sampled_df['self_text']

# printing the size of the vocabulary and shape of the bag of words matrix
print("Vocabulary size:", len(feature_names))
print("BoW matrix shape:", tfidf_matrix.shape)

from gensim import matutils, corpora, models
import gensim

# converting TF-IDF matrix into a Gensim-compatible sparse matrix
gensim_corpus = matutils.Sparse2Corpus(tfidf_matrix.T)

# creating a mapping from word IDs to words
id2word = dict((v, k) for k, v in tfidf_vectorizer.vocabulary_.items())

# creating a Gensim dictionary
gensim_dict = corpora.Dictionary.from_corpus(gensim_corpus, id2word=id2word)

from gensim.models import LdaModel

# defining the number of topics
num_topics = 3

# training the LDA model
lda_model = LdaModel(
    corpus=gensim_corpus,
    id2word=id2word,
    num_topics=num_topics,
    passes = 2
)

# printing the topics generated by the LDA model
# getting the top words for each topic
for topic_idx in range(num_topics):
    print(f"Topic {topic_idx + 1}:")
    top_words = lda_model.show_topic(topic_idx, topn=15)
    top_words_list = [word for word, _ in top_words]
    print(top_words_list)

# Transform new documents into topic distributions
doc_topic_dist = [lda_model.get_document_topics(doc, minimum_probability=0.65) for doc in gensim_corpus]

# Extract the top two topics for each document
top_topics_per_doc = []
for doc_topics in doc_topic_dist:
    # Sort the document topics by probability and get the most probable ones
    top_topics = sorted(doc_topics, key=lambda x: x[1], reverse=True)[:2]
    top_topics_per_doc.append([topic for topic, _ in top_topics])

# Add the top two topics as new columns to your dataset
sampled_df['top_topics'] = [topics[0] if topics else None for topics in top_topics_per_doc]
# sampled_df['top_topic_2'] = [topics[1] if len(topics) > 1 else None for topics in top_topics_per_doc]

sampled_df.head()

print(sampled_df['top_topics'].isnull().sum())
# print(sampled_df['top_topic_2'].isnull().sum())

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pointbiserialr

# Fill missing values in 'top_topics' with a placeholder
sampled_df['top_topics'].fillna('missing', inplace=True)

# Convert 'top_topics' column to string type
sampled_df['top_topics'] = sampled_df['top_topics'].astype(str)

# Perform one-hot encoding for the top topic
onehot_encoder = OneHotEncoder()
top_topic_encoded = onehot_encoder.fit_transform(sampled_df[['top_topics']])

# Convert the encoded topic into a dataframe
top_topic_encoded_df = pd.DataFrame(top_topic_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(['top_topics']))

# Concatenate controversiality and encoded top topics dataframes
controversiality_and_topic = pd.concat([sampled_df['controversiality'], top_topic_encoded_df], axis=1)

# Calculate the correlation matrix
correlation_matrix = controversiality_and_topic.corr()

# Print the correlation matrix
print(correlation_matrix['controversiality'][1:])

# Pivot the DataFrame to get the count of controversiality values (0 and 1) for each topic
topic_controversiality_count = sampled_df.pivot_table(index='top_topics', columns='controversiality', aggfunc='size', fill_value=0)

# Rename the columns for better understanding
topic_controversiality_count.columns = ['count_0', 'count_1']

# Print the resulting matrix
print("Matrix of controversiality counts for each topic:")
print(topic_controversiality_count)

# Define the topic titles dictionary
topic_titles = {
    1: "Military Operations and Tactics",
    2: "Miscellaneous Discussions and Cultural References",
    0: "Geo-Political Dynamics and Conflict"
}

# Map the numerical topics to their corresponding titles
sampled_df['first_topic_title'] = sampled_df['top_topics'].map(topic_titles)

sampled_df.head()

# Saving the processed data to a CSV file
sampled_df.to_csv('tfidf_dataset.csv',index=False)
