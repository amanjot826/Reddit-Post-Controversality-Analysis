import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Load the dataset
data = pd.read_csv("/Users/sarabjotsingh/Downloads/selected_features.csv")

# Split the dataset into features (X) and target variable (y)
X_numeric = data.drop(columns=['controversiality', 'tokens', 'top_words_per_doc', 'combined_text', 
                               'user_is_verified', 'user_awarder_karma', 'user_total_karma', 'user_awardee_karma',
                              'first_topic_title'])
X_text = data['combined_text']
y = data['controversiality']

# Split the data into train and test sets without shuffling
X_numeric_train, X_numeric_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_numeric, X_text, y, test_size=0.2, random_state=42)

# Define transformations for numeric and categorical columns
numeric_features = ['comment_score','post_score', 'post_upvote_ratio', 'total_comments', 'sentiment_score']
numeric_categorical_features = [ 'subreddit_encoded']

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine numeric and categorical transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, numeric_categorical_features)
    ])

# Define the text transformer
text_transformer = TfidfVectorizer(max_features=12000)

# Load the combined pickle file containing both models
combined_model = joblib.load("/Users/sarabjotsingh/Downloads/models.pkl")

# Extract the individual models from the combined model
xg_boost= combined_model['XGBoost']
naive_bayes = combined_model['NaiveBayes']

# Compute permutation importances for XGBoost model
perm_importance = permutation_importance(xg_boost, X_numeric_test, y_test, n_repeats=10, random_state=42)

# Get feature names
feature_names = X_numeric_train.columns

# Plot permutation importances
sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(feature_names[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Permutation Feature Importance')
plt.show()

# Initialize LIME explainer
explainer = LimeTextExplainer()

# Choose a random instance for explanation
instance_idx = 0
instance_text = X_text_test.iloc[instance_idx]

# Explain prediction using LIME for Naive Bayes model
explanation = explainer.explain_instance(instance_text, naive_bayes.predict_proba)

# Visualize explanation
explanation.show_in_notebook(text=True)
