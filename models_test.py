import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier  
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score



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
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))  # Use drop='first' to avoid multicollinearity
])

# Combine numeric and categorical transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, numeric_categorical_features)
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

# Assign weights based on performance
total_accuracy = xg_boost_accuracy + naive_bayes_accuracy
xg_boost_weight = xg_boost_accuracy / total_accuracy
naive_bayes_weight = naive_bayes_accuracy / total_accuracy

# Combine predictions using weighted averaging
weighted_avg_preds = (xg_boost_weight * xg_boost_preds_val) + \
                     (naive_bayes_weight * naive_bayes_preds_val)

# Define a threshold to convert continuous predictions to binary labels
threshold = 0.5  # Adjust the threshold as needed

# Convert continuous predictions to binary labels based on the threshold
binary_preds = (weighted_avg_preds > threshold).astype(int)

# Evaluate performance of ensemble model using appropriate evaluation metric

# Evaluate precision, recall, F1 score, and support
print("Classification Report:")
print(classification_report(y_test, binary_preds))

# Compute confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, binary_preds))

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, weighted_avg_preds)
auc = roc_auc_score(y_test, weighted_avg_preds)
print("AUC:", auc)
