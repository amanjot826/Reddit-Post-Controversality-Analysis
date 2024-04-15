#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


df = pd.read_csv(r'C:\Users\acer\Dropbox\PC\Downloads\senti_dataset.csv')
display(df.head(5))


# In[10]:


df.info()


# # Label Encoding of Subreddit

# In[11]:


import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'subreddit' column
df['subreddit_encoded'] = label_encoder.fit_transform(df['subreddit'])

# Get the unique number of values in the 'subreddit' column
unique_subreddits = df['subreddit'].nunique()

# Print the unique number of subreddits
print("Unique number of subreddits:", unique_subreddits)


# In[12]:


df.drop(columns=['subreddit'], inplace=True)


# # DROPPING COLUMNS

# In[13]:


df.drop(columns=['comment_id'], inplace=True)
df.drop(columns=['self_text'], inplace=True)
df.drop(columns=['created_time'],inplace=True)
df.drop(columns=['post_id'],inplace=True)
df.drop(columns=['post_title'],inplace=True)
df.drop(columns=['author_name'], inplace=True)
df.drop(columns=['ups'], inplace=True)
df.drop(columns=['downs'], inplace=True)
df.drop(columns=['post_total_awards_received'], inplace=True)
df.drop(columns=['post_created_time'],inplace=True)
df.drop(columns=['self_text_tokens'], inplace=True)
df.drop(columns=['post_title_tokens'],inplace=True)
df.info()


# In[14]:


import nltk
from nltk.tokenize import word_tokenize

# Tokenize the text in the 'combined_text' column
df['tokens'] = df['combined_text'].apply(lambda x: word_tokenize(x))


# # Renaming Columns

# In[15]:


# Rename the 'score' column to 'comment_score'
df = df.rename(columns={'score': 'comment_score'})


# # Correlation Matrix

# In[16]:


# Select the relevant columns
selected_columns = ['user_awardee_karma', 'user_awarder_karma', 'user_link_karma', 'user_comment_karma', 'user_total_karma']
selected_df = df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_df.corr()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select the relevant columns
selected_columns = ['user_awardee_karma', 'user_awarder_karma', 'user_link_karma', 'user_comment_karma', 'user_total_karma']
selected_df = df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_df.corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


# In[18]:


df.drop(columns=['user_link_karma'], inplace=True)
df.drop(columns=['user_comment_karma'], inplace=True)


# In[19]:


# Calculate the correlation between 'post_score' and 'post_thumbs_ups' columns
correlation = df['post_score'].corr(df['post_thumbs_ups'])

print("Correlation between post_score and post_thumbs_ups:", correlation)


# In[20]:


# Drop the 'post_thumbs_ups' column
df.drop(columns=['post_thumbs_ups'], inplace = True)


# In[21]:


df.info()


# In[22]:


import pandas as pd

# Select the relevant columns
columns = ['user_awardee_karma', 'user_awarder_karma', 'user_total_karma', 'post_score']
relevant_df = df[columns]

# Calculate the correlation matrix
correlation_matrix = relevant_df.corr()


# In[23]:


# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


# In[24]:


import pandas as pd

# Select the relevant columns
columns = ['user_awardee_karma', 'user_awarder_karma', 'user_total_karma', 'post_upvote_ratio']
df_selected = df[columns]

# Calculate the correlation matrix
correlation_matrix = df_selected.corr()


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f")  # Use 'viridis' colormap
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()


# In[26]:


df.info()


# In[27]:


# Encode the 'top_topics' column
df['top_topics_encoded'] = label_encoder.fit_transform(df['top_topics'])


# In[28]:


# Assuming df contains your DataFrame with the specified columns
selected_columns = ['controversiality', 'comment_score', 'user_is_verified', 'user_awardee_karma', 
                    'user_awarder_karma', 'user_total_karma', 'post_score', 'post_upvote_ratio', 
                    'total_comments', 'sentiment_score', 'subreddit_encoded', 'top_topics_encoded']

correlation_matrix = df[selected_columns].corr()

correlation_matrix


# In[29]:


# Split the dataset into features (X) and target variable (y)
X_numeric = df.drop(columns=['controversiality', 'tokens', 'top_words_per_doc', 'combined_text', 'top_topics', 
                             'first_topic_title', 'user_awarder_karma', 'user_awardee_karma','user_total_karma'])
X_text = df['combined_text']
y = df['controversiality']


# In[30]:


X_numeric.info()


# In[31]:


from sklearn.model_selection import train_test_split

# Split the data into train and test sets without shuffling
X_numeric_train, X_numeric_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_numeric, X_text, y, test_size=0.2, random_state=42)


# In[32]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Define transformations for numeric and categorical columns
numeric_features = ['comment_score', 'post_score', 'post_upvote_ratio', 'total_comments', 'sentiment_score']
numeric_categorical_features = ['subreddit_encoded', 'top_topics_encoded']

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Use drop='first' to avoid multicollinearity
])


# In[33]:


from sklearn.compose import ColumnTransformer

# Combine numeric and categorical transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, numeric_categorical_features)
    ])


# In[37]:


from sklearn.manifold import TSNE

# Fit t-SNE after preprocessing
tsne = TSNE(n_components=2, random_state=42, init="random")

# Preprocess and fit the t-SNE model
X_train_processed = preprocessor.fit_transform(X_numeric_train)

# Fit t-SNE on the preprocessed data
X_tsne_train = tsne.fit_transform(X_train_processed)

# Plot class separability
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne_train[:, 0], y=X_tsne_train[:, 1], hue=y_train, palette='Set1', alpha=0.8)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Class Separability Plot using t-SNE')
plt.legend(title='Controversiality')
plt.show()


# In[38]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Define the text transformer
text_transformer = TfidfVectorizer(max_features=12000)


# In[39]:


import numpy as np

# Preprocess text data
X_text_train_transformed = text_transformer.fit_transform(X_text_train)

# Concatenate processed numeric and text data
X_train_processed_concatenated = np.concatenate((X_train_processed.toarray(), X_text_train_transformed.toarray()), axis=1)

# Fit t-SNE on the concatenated preprocessed data
X_tsne_train_concatenated = tsne.fit_transform(X_train_processed_concatenated)

# Plot class separability for the concatenated data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne_train_concatenated[:, 0], y=X_tsne_train_concatenated[:, 1], hue=y_train, palette='Set1', alpha=0.8)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Class Separability Plot using t-SNE (Numeric + Text)')
plt.legend(title='Controversiality')
plt.show()


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_numeric_train, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Sort feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X_numeric.shape[1]), feature_importances[sorted_indices], align='center')
plt.xticks(range(X_numeric.shape[1]), X_numeric.columns[sorted_indices], rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

# Select top k important features (e.g., top 10 features)
k = 11
selected_features = X_numeric.columns[sorted_indices][:k]
print("Selected features:", selected_features)


# In[87]:


# downloading the sampled dataset to your system
df.to_csv('selected_features.csv', index=False)


# In[ ]:




