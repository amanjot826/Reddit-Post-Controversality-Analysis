# importing the pandas library for data manipulation and analysis
import pandas as pd
from scipy.stats import mode
import contractions

# reading the reddit_dataset from a CSV file located at the specified path
sampled_df = pd.read_csv(r'/Users/sarabjotsingh/Downloads/sampled_dataset.csv')

# displaying the dataset in a tabular format
sampled_df

# # Data Cleaning

# # Handling Duplicates and Missing Values

# checking for duplicates in the sampled dataframe
duplicate_values = sampled_df[sampled_df.duplicated(subset='comment_id')]

# counting the number of duplicate rows found
num_duplicates = duplicate_values.shape[0]

# printing the number of duplicate rows found
print(num_duplicates)  

# checking for missing values in each column
missing_values_counts = sampled_df.isnull().sum()

# printing the number of missing values for each column
print("Number of missing values in each column:")
print(missing_values_counts)

# dropping columns 'user_account_created_time' and 'post_self_text'
columns_to_drop = ["user_account_created_time", "post_self_text"]
sampled_df.drop(columns=columns_to_drop, inplace=True)

# replacing missing values in 'self_text' and 'post_title' columns with 'no content'
sampled_df['self_text'].fillna('No Content', inplace=True)
sampled_df['post_title'].fillna('No Content', inplace=True)

# filling the missing values with the mean
sampled_df['user_awardee_karma'].fillna(sampled_df['user_awardee_karma'].mean(), inplace=True)
sampled_df['user_awarder_karma'].fillna(sampled_df['user_awarder_karma'].mean(), inplace=True)
sampled_df['user_link_karma'].fillna(sampled_df['user_link_karma'].mean(), inplace=True)
sampled_df['user_comment_karma'].fillna(sampled_df['user_comment_karma'].mean(), inplace=True)
sampled_df['user_total_karma'].fillna(sampled_df['user_total_karma'].mean(), inplace=True)

# displaying information about the DataFrame
sampled_df.info()

# Convert non-numeric values to numeric
sampled_df['user_is_verified'] = sampled_df['user_is_verified'].astype(float)

# Fill missing values in 'user_is_verified' with the mode
mode_result = mode(sampled_df['user_is_verified'], nan_policy='omit')
mode_values = mode_result.mode.tolist() if isinstance(mode_result.mode, list) else [mode_result.mode]  # Convert mode values to a list if necessary

if mode_values:
    mode_value = mode_values[0]  # Get the first mode value
    sampled_df['user_is_verified'].fillna(mode_value, inplace=True)
else:
    print("No mode value found.")


# No duplicate comment_id was found in the dataset. However, 6747 and 162658 missing values were found in the “user_account_created_time” 
# and “post_self_text” columns. Since our dataset contains columns, such as “user_is_verified”, “user_awardee_karma” and “user_awarder_karma” 
# which are closely linked (correlated) to “user_account_creation_time”, the column was dropped. Furthermore, “post_self_text” was also 
# deleted because 80% of the datapoints were missing.  

# # Removing Contractions

# defining a function to remove contractions from text
def remove_contractions(text):
    print("Text before:", text)
    try:
        if not isinstance(text, str) or len(text) < 5:  # Check for empty or very short strings
            return text
        expanded_text = contractions.fix(text)
        print("Text after:", expanded_text)
        return expanded_text
    except IndexError:
        print("IndexError occurred. Returning original text.")
        return text

# Applying the remove_contractions function to the specified columns
sampled_df['self_text'] = sampled_df['self_text'].astype(str).apply(remove_contractions)
sampled_df['post_title'] = sampled_df['post_title'].astype(str).apply(remove_contractions)

# Printing the output
sampled_df

# # Removing URLs and Email Addresses

# importing the regular expression module
import re

# Function to remove irrelevant text from strings
def remove_irrelevant_text(text):
    # removing URLs
    text = re.sub(r'http\S+', '', text)
    
    # removing email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # removing other irrelevant information
    text = re.sub(r'\[.*?\]', '', text)
    
    return text

# applying the remove_irrelevant_text function to the specified columns
sampled_df['self_text'] = sampled_df['self_text'].astype(str).apply(remove_irrelevant_text)
sampled_df['post_title'] = sampled_df['post_title'].astype(str).apply(remove_irrelevant_text)

# displaying the cleaned DataFrame
sampled_df

# An unsuccessful attempt to replace URLs with their respective blog titles was made due to an error 404 message occurring for most of the URLs. 
# Hence, URLs and email addresses were eliminated from the text columns.

# # Removing Non-ASCII Characters, Emojis and Special Characters

# importing the unidecode module for removing non-ASCII characters
from unidecode import unidecode

# function to remove non-ASCII characters, emojis, and special characters
def remove_non_ascii(text):
    # removing non-ASCII characters by translittering them to their closest ASCII equivalents
    text = unidecode(text)
    
    # defining a regular expression pattern to match and remove emojis
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)  # substitute emojis with an empty string
    
    # removing special characters , retaining only letters, digits and whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return text

# applying the remove_non_ascii function to the specified columns
sampled_df['self_text'] = sampled_df['self_text'].apply(remove_non_ascii)
sampled_df['post_title'] = sampled_df['post_title'].apply(remove_non_ascii)

# displaying the cleaned DataFrame
sampled_df

# Used transliteration (that is, replacing a non-ASCII character with its closest ASCII equivalent based on pronunciation or similar phonetic 
# characteristics) to replace non-ASCII characters, and removed the emojis and special characters from data.

# # Replacing HTML Tags

# impoting the BeautifulSoup for HTML parsing 
from bs4 import BeautifulSoup

# function to replace HTML tags with the text they contain
def replace_html_tags_with_content(text):
    # parsing the HTML content
    soup = BeautifulSoup(text, 'html.parser')

    # replacing each HTML tag with its text content
    for tag in soup.find_all():
        if tag.string:
            tag.replace_with(tag.string)

    return str(soup)   # Converting the modified BeautifulSoup object back to the string 

# applying the replace_html_tags_with_content function to the specified columns
sampled_df['self_text'] = sampled_df['self_text'].apply(replace_html_tags_with_content)
sampled_df['post_title'] = sampled_df['post_title'].apply(replace_html_tags_with_content)

# displaying the DataFrame with HTML tags replaced by their content
sampled_df

# Each HTML tag was replaced with its text content to retain the essential information within tags

# # Standardization

# importing the numpy module
import numpy as np

# list of text columns to be converted to lowercase
text_columns = ['self_text', 'subreddit', 'post_title']

# function to convert text to lowercase
def lowercase_text(text):
    if isinstance(text, str):  # Check if text is a string
        return text.lower()    # convert text to lowercase
    else:
        return text  # Return text as it is if it's not a string

# applying lowercase transformation to each text column
for column in text_columns:
    sampled_df[column] = sampled_df[column].apply(lowercase_text)

# displaying the DataFrame after lowercase transformation
sampled_df

# Text from “subreddit”, “self_text”, “post_title” and “post_self_text” was lowercased to avoid mis-interpretation of text while model building.

# # Removing Stopwords and Punctuations

# importing the stopwords module 
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')  # downloading the nltk stopwords dataset

# defining the columns to be cleaned
columns_to_clean = ['self_text', 'post_title']

# defining stop words for english language
stop_words = set(stopwords.words('english'))

# defining function to remove stopwords
def remove_stopwords(tokens):
    if isinstance(tokens, str):       # Checking if the input is a string
        filtered_tokens = [token for token in tokens.split() if token.lower() not in stop_words] # tokenizethe text and filter out stopwords
        return ' '.join(filtered_tokens)
    else:
        return tokens

# apply remove_stopwords function to each column
for col in columns_to_clean:
    sampled_df[col] = sampled_df[col].apply(remove_stopwords)

# displaying the DataFrame after removing stopwords
sampled_df

# Ensured consistency and uniformity in the text data by applying the stopwords removal process to columns “self_text”, “post_title” 
# and “post_self_text” in the DataFrame . By eliminating frequent stopwords, the text data's quality improves.

# # Tokenization

# impoting the word_tokenize function
from nltk.tokenize import word_tokenize

# Tokenizing self_text column and creating a new column 'self_text_tokens' to store tokens
sampled_df['self_text_tokens'] = sampled_df['self_text'].apply(word_tokenize)

# Tokenizing post_title column and creating new column 'post_title_tokens' to store tokens
sampled_df['post_title_tokens'] = sampled_df['post_title'].apply(word_tokenize)

# Checking the DataFrame after tokenization
sampled_df.head()

# Transforming the raw text data into a structured and analyzable format by tokenizing the text into individual words or tokens. 
# Therefore, prepareing the text data for further analysis and modeling tasks, enhancing the capabilities of natural language processing 
# applications.

# # Stemming and Lemmatization

# importing necessary libraries for text processing
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
import pandas as pd
import spacy

# Downloading the 'punkt' tokenizer
nltk.download('punkt')

# Function to perform stemming
def perform_stemming(text):
    stemmer = PorterStemmer()  # Initializing a PorterStemmer object
    
    # Tokenize the input text into individual words and apply stemming to each other
    stemmed_words = [stemmer.stem(word) for word in nltk.word_tokenize(text)] 
    
    # Joining the stemmed words back into a single string
    return ' '.join(stemmed_words)

# Applying stemming to 'self_text' and 'post_title' column
sampled_df['post_title'] = sampled_df['post_title'].apply(perform_stemming)
sampled_df['self_text'] = sampled_df['self_text'].apply(perform_stemming)

# Every tokenized term in the text is subjected to the Porter Stemmer algorithm. Through the removal of affixes (prefixes and suffixes), 
# each word is reduced to its root or base form, normalizing the words in the process.

def perform_lemmatization_spacy(text):
    # Process the text using spaCy
    doc = nlp(text)
    # Lemmatize each token and join them back into a string
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return lemmatized_text

# Loading the spaCy language model
nlp = spacy.load('en_core_web_sm')

# Ensuring that the column contain string values
sampled_df['post_title'] = sampled_df['post_title'].astype(str)
sampled_df['self_text'] = sampled_df['self_text'].astype(str)

# Applying lemmatization to the 'post_self_text' column
sampled_df['post_title'] = sampled_df['post_title'].apply(perform_lemmatization_spacy)
sampled_df['self_text'] = sampled_df['self_text'].apply(perform_lemmatization_spacy)

# The lemmatization is done on the ‘post_self_text’ , ‘post_title’ and ‘self_text’ column. After lemmatization, text data is transformed 
# into a more standardized and semantically meaningful representation, which facilitates better understanding

sampled_df.head()

# Saving the processed data to a CSV file
sampled_df.to_csv('processed_data.csv',index=False)

