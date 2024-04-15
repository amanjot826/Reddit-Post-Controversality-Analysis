
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data = pd.read_csv(r'C:\Users\acer\Dropbox\PC\Downloads\selected_features.csv')
data.info()

# Split the dataset into features (X) and target variable (y)
X_numeric = data.drop(columns=['controversiality', 'tokens', 'top_words_per_doc', 'combined_text', 'top_topics', 
                               'user_is_verified', 'user_awarder_karma', 'user_total_karma', 'user_awardee_karma',
                              'first_topic_title'])
X_text = data['combined_text']
y = data['controversiality']

X_numeric.info()

# Split the data into train and test sets without shuffling
X_numeric_train, X_numeric_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_numeric, X_text, y, test_size=0.2, random_state=42)

# Define transformations for numeric and categorical columns
numeric_features = ['comment_score','post_score', 'total_comments', 'sentiment_score', 'post_upvote_ratio']

numeric_categorical_features = ['subreddit_encoded', 'top_topics_encoded']

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))])  # Use drop='first' to avoid multicollinearity

# Combine numeric and categorical transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer,numeric_categorical_features)
    ])

# Define the text transformer
text_transformer = TfidfVectorizer(max_features=12000)

# Define RandomForestClassifier pipeline
xg_boost_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(scale_pos_weight=1.5)) 
])

naive_bayes_model = Pipeline([
    ('text', text_transformer),
    ('classifier', MultinomialNB())
])

# Define and fit logistic regression model
naive_bayes_model.fit(X_text_train, y_train)
xg_boost_model.fit(X_numeric_train, y_train)

# Train base models and generate predictions on validation set
xg_boost_preds_val = xg_boost_model.predict(X_numeric_test)
naive_bayes_preds_val = naive_bayes_model.predict(X_text_test)

# Calculate accuracy of each base model on validation set
xg_boost_accuracy = accuracy_score(y_test, xg_boost_preds_val)
naive_bayes_accuracy = accuracy_score(y_test, naive_bayes_preds_val)

print(xg_boost_accuracy)
print(naive_bayes_accuracy)

# Assign weights based on performance
total_accuracy = xg_boost_accuracy + naive_bayes_accuracy
xg_boost_weight = xg_boost_accuracy / total_accuracy
naive_bayes_weight = naive_bayes_accuracy / total_accuracy

print(xg_boost_weight)
print(naive_bayes_weight)

# Combine predictions using weighted averaging
weighted_avg_preds = (xg_boost_weight * xg_boost_preds_val) + \
                     (naive_bayes_weight * naive_bayes_preds_val)

# Define a threshold to convert continuous predictions to binary labels
threshold = 0.5  # Adjust the threshold as needed

# Convert continuous predictions to binary labels based on the threshold
binary_preds = (weighted_avg_preds > threshold).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate performance metrics for combined model
combined_accuracy = accuracy_score(y_test, binary_preds)
combined_precision = precision_score(y_test, binary_preds)
combined_recall = recall_score(y_test, binary_preds)
combined_f1 = f1_score(y_test, binary_preds)

print("Combined Accuracy:", combined_accuracy)
print("Combined Precision:", combined_precision)
print("Combined Recall:", combined_recall)
print("Combined F1-score:", combined_f1)

import numpy as np

# Get feature names from TF-IDF vectorizer
text_feature_names = naive_bayes_model.named_steps['text'].get_feature_names_out()

# Get Naive Bayes feature log probabilities
feature_log_prob = naive_bayes_model.named_steps['classifier'].feature_log_prob_

# Get top words for each class
num_top_words = 10
top_positive_words = [text_feature_names[idx] for idx in np.argsort(feature_log_prob[1])[-num_top_words:]]
top_negative_words = [text_feature_names[idx] for idx in np.argsort(feature_log_prob[0])[-num_top_words:]]

print("Top positive words:", top_positive_words)
print("Top negative words:", top_negative_words)

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Fit the model (replace "model" with your trained model)
xg_boost_model.fit(X_numeric_train, y_train)

# Compute permutation importances
perm_importance = permutation_importance(xg_boost_model, X_numeric_test, y_test, n_repeats=30, random_state=42)

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

# Choose a random instance for local interpretation
instance_idx = 500
instance_numeric_features = X_numeric_test.iloc[instance_idx]
instance_text_features = X_text_test.iloc[instance_idx]

# Predict probabilities for the single instance using XGBoost
xgboost_contribution_instance = xg_boost_model.predict_proba(instance_numeric_features.to_frame().transpose())[:, 1]

# Predict probabilities for the single instance using Naive Bayes
naive_bayes_contribution_instance = naive_bayes_model.predict_proba([instance_text_features])[:, 1]

print("XGBoost contribution for the instance:", xgboost_contribution_instance)
print("Naive Bayes contribution for the instance:", naive_bayes_contribution_instance)

from lime import lime_text
from lime.lime_text import LimeTextExplainer

# Initialize LIME explainer
explainer = LimeTextExplainer()

# Choose a random instance index
instance_idx = 500

# Get the text instance
instance_text = X_text_test.iloc[instance_idx]

# Explain prediction using LIME
explanation = explainer.explain_instance(instance_text, naive_bayes_model.predict_proba, num_features=10)

# Visualize explanation
explanation.show_in_notebook(text=True)

import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Train XGBoost model
xg_boost_model = XGBClassifier(scale_pos_weight=1.5)
xg_boost_model.fit(X_numeric_train, y_train)

# Create a SHAP explainer object using the fitted model
explainer = shap.Explainer(xg_boost_model, X_numeric_train)

# Generate SHAP values for test data
shap_values = explainer.shap_values(X_numeric_test)

# Visualize SHAP summary plot
shap.summary_plot(shap_values, X_numeric_test)

import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Assuming you have already trained your XGBoost model and split your data into X_numeric_train and X_numeric_test

# Choose a random instance index from the test data
instance_idx = 500

# Get the features of the chosen instance
instance_numeric_features = X_numeric_test.iloc[[instance_idx]]

# Create a SHAP explainer object
explainer = shap.Explainer(xg_boost_model, X_numeric_train)

# Generate SHAP values for the chosen instance
shap_values_instance = explainer.shap_values(instance_numeric_features)

# Visualize SHAP values for the chosen instance
shap.force_plot(explainer.expected_value, shap_values_instance, instance_numeric_features)

