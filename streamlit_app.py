import streamlit as st
import pickle
import pandas as pd
import os
from data_cleaning import clean_text, tokenize_text, remove_stopwords, lemmatize_text, stem_text
from nltk.sentiment import SentimentIntensityAnalyzer

# Function to load Gensim dictionary
def load_gensim_dict():
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    dict_path = os.path.join(script_dir, 'gensim_dict.pkl')
    try:
        with open(dict_path, 'rb') as f:
            gensim_dict = pickle.load(f)
        return gensim_dict
    except FileNotFoundError:
        st.error("Gensim dictionary file not found. Please ensure the file exists in the correct location.")
        return None

# Load the Gensim dictionary
gensim_dict = load_gensim_dict()

# Load the trained models
def load_models():
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    naive_bayes_model_path = os.path.join(script_dir, 'naive_bayes_model.pkl')
    xgboost_model_path = os.path.join(script_dir, 'xgboost_model.pkl')

    # Load the trained Naive Bayes model
    try:
        with open(naive_bayes_model_path, "rb") as file:
            naive_bayes_model = pickle.load(file)
    except FileNotFoundError:
        st.error("Naive Bayes model file not found. Please ensure the file exists in the correct location.")
        naive_bayes_model = None

    # Load the trained XGBoost model
    try:
        with open(xgboost_model_path, "rb") as file:
            xg_boost_model = pickle.load(file)
    except FileNotFoundError:
        st.error("XGBoost model file not found. Please ensure the file exists in the correct location.")
        xg_boost_model = None

    return naive_bayes_model, xg_boost_model

# Load the trained models
naive_bayes_model, xg_boost_model = load_models()

# Load the LDA model
def load_lda_model():
    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    lda_model_path = os.path.join(script_dir, 'topic_modeling.pkl')
    try:
        with open(lda_model_path, "rb") as file:
            lda_model = pickle.load(file)
    except FileNotFoundError:
        st.error("LDA model file not found. Please ensure the file exists in the correct location.")
        lda_model = None

    return lda_model


# Define Streamlit app
def main():
    # Page title and description
    st.title('Controversiality Prediction App')
    st.write('This app predicts the controversiality of a comment based on numeric features, text content, and topic distribution.')
    st.write('---')

    # Input form for numeric features
    st.sidebar.title('Enter Numeric Features:')
    comment_score = int(st.sidebar.number_input('Comment Score:'))
    post_score = int(st.sidebar.number_input('Post Score:'))
    total_comments = int(st.sidebar.number_input('Total Comments:'))
    post_upvote_ratio = float(st.sidebar.number_input('Post Upvote Ratio:'))

    # Input form for categorical features
    st.sidebar.title('Enter Categorical Features:')
    subreddit_encoded = int(st.sidebar.number_input('Subreddit Encoded:'))

    # Input form for text feature
    st.sidebar.title('Enter Text Feature:')
    combined_text = str(st.sidebar.text_area('Combined Text:'))

    # Clean text data
    cleaned_text = clean_text(combined_text)
    tokens = tokenize_text(cleaned_text)
    tokens_without_stopwords = remove_stopwords(tokens)
    lemmatized_text = lemmatize_text(tokens_without_stopwords)
    stemmed_text = stem_text(lemmatized_text)

    # Convert the list of stemmed tokens into a single string
    stemmed_text_str = ' '.join(stemmed_text)

    # Load the LDA model
    lda_model = load_lda_model()

    # Convert the list of tokens into a bag-of-words representation using the Gensim dictionary
    bow_vector = gensim_dict.doc2bow(stemmed_text)
        
    # Get topic distribution for the text
    topic_distribution = lda_model[bow_vector]
        
    # Extract top topics encoded based on probability threshold
    top_topics_encoded = [topic[0] for topic in topic_distribution if topic[1] > 0.2]
    if not top_topics_encoded:
        top_topics_encoded = [3]  # Default value if no topics have probability greater than 0.35

    # Initialize SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    if stemmed_text:
        scores = [sid.polarity_scores(token)['compound'] for token in stemmed_text]
        # Return the average sentiment score
        sentiment_score = sum(scores) / len(scores)
    else:
        # Return a default value when no tokens are present
        sentiment_score = 0

    # Make prediction when "Predict Controversiality" button is clicked
    if st.sidebar.button('Predict Controversiality'):
        if naive_bayes_model is not None and xg_boost_model is not None:
            input_data = pd.DataFrame({
                'comment_score': [comment_score],
                'post_score': [post_score],
                'total_comments': [total_comments],
                'sentiment_score': [sentiment_score],
                'post_upvote_ratio': [post_upvote_ratio],
                'subreddit_encoded': [subreddit_encoded],
                'top_topics_encoded': [top_topics_encoded[0]]
            })

            # Make prediction using Naive Bayes model
            naive_bayes_pred = naive_bayes_model.predict([stemmed_text_str])[0]

            # Make prediction using XGBoost model
            xg_boost_pred = xg_boost_model.predict(input_data)[0]

            # Take weighted average of predictions
            weighted_avg_pred = (0.46 * naive_bayes_pred) + (0.54 * xg_boost_pred)

            # Convert continuous predictions to binary labels based on the threshold
            threshold = 0.5
            binary_pred = 1 if weighted_avg_pred > threshold else 0

            st.write('Binary Prediction:', binary_pred)
        else:
            st.error("Models not loaded. Please check the model files and try again.")

    # Add a button to predict topics
    if st.sidebar.button('Predict Topics'):
        if top_topics_encoded[0] == 0:
            st.sidebar.write("Topic: Space Technology and Disinformation")
        elif top_topics_encoded[0] == 1:
            st.sidebar.write("Topic: Military Operations and Engagement")
        elif top_topics_encoded[0] == 2:
            st.sidebar.write("Topic: Geopolitical Tensions and Diplomatic Relations")
        else:
            st.sidebar.write("No topic assigned!")
    pass

# Run the Streamlit app
if __name__ == '__main__':
    main()
