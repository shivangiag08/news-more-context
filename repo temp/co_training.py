import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import scipy.sparse
import pickle

# Load the labeled dataset
global_news_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\global news data set\rating.csv")
global_news_data = global_news_data[['article_id', 'title', 'title_sentiment']]
sentiment_mapping = {'Positive': 2, 'Negative': 0, 'Neutral': 1}

global_news_data['title_sentiment'] = global_news_data['title_sentiment'].map(sentiment_mapping)
# Load the unlabeled dataset
bbc_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\bbc_news.csv")
bbc_data['title_sentiment'] = -1  # Mark unlabeled data

new_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\global news data set\uncommon_data.csv", encoding='latin-1', encoding_errors='ignore')
new_data['title_sentiment'] = -1  # Mark unlabeled data
new_data.dropna(subset=['title'], inplace=True)
# Combine the labeled and unlabeled datasets
combined_data = pd.concat([global_news_data, bbc_data[['title', 'title_sentiment']], new_data[['title', 'title_sentiment']]])

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_combined = vectorizer.fit_transform(combined_data['title'])
y_combined = combined_data['title_sentiment'].values

# Split only the labeled part of the data for evaluation
train_size = 0.8
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_combined[:len(global_news_data)], y_combined[:len(global_news_data)], 
    train_size=train_size, random_state=42)

# Initialize the base estimators
logistic_regression = LogisticRegression(max_iter=1000)
naive_bayes = LabelPropagation()

# Assume the unlabeled data is at the end of the combined_data dataframe
unlabeled_data_start_index = len(global_news_data) + len(bbc_data)
X_unlabeled = X_combined[unlabeled_data_start_index:]
y_unlabeled = np.full((X_unlabeled.shape[0],), -1)

# Number of iterations for co-training
num_iterations = 10  # you can change this to your preference
confidence_threshold = 0.8  # the threshold for adding pseudo-labels
y_train_eval = y_train_eval.astype(int)

for iteration in range(num_iterations):
    # Train each classifier on the labeled data
    logistic_regression.fit(X_train_eval, y_train_eval)
    naive_bayes.fit(X_train_eval, y_train_eval)

    # Use each classifier to predict labels for the unlabeled data
    proba_lr = logistic_regression.predict_proba(X_unlabeled)
    proba_nb = naive_bayes.predict_proba(X_unlabeled)

    # Get the most confident predictions for each classifier
    confident_predictions_lr = np.argmax(proba_lr, axis=1)
    confident_prob_lr = np.max(proba_lr, axis=1)
    confident_predictions_nb = np.argmax(proba_nb, axis=1)
    confident_prob_nb = np.max(proba_nb, axis=1)

    # Filter out predictions below the confidence threshold
    confident_mask_lr = confident_prob_lr > confidence_threshold
    confident_mask_nb = confident_prob_nb > confidence_threshold

    # Update the labeled dataset with pseudo-labeled data
    # Make sure the pseudo-labels are integers, to avoid the TypeError
    pseudo_labels_lr = confident_predictions_nb[confident_mask_nb].astype(int)
    pseudo_labels_nb = confident_predictions_lr[confident_mask_lr].astype(int)

    X_train_with_pseudo_lr = scipy.sparse.vstack((X_train_eval, X_unlabeled[confident_mask_nb]))
    y_train_with_pseudo_lr = np.concatenate((y_train_eval, pseudo_labels_lr))

    X_train_with_pseudo_nb = scipy.sparse.vstack((X_train_eval, X_unlabeled[confident_mask_lr]))
    y_train_with_pseudo_nb = np.concatenate((y_train_eval, pseudo_labels_nb))

    # Retrain the classifiers with the augmented datasets
    logistic_regression.fit(X_train_with_pseudo_lr, y_train_with_pseudo_lr)
    naive_bayes.fit(X_train_with_pseudo_nb, y_train_with_pseudo_nb)

    # Remove the pseudo-labeled data from the unlabeled dataset
    X_unlabeled = X_unlabeled[~(confident_mask_lr | confident_mask_nb)]
# Evaluate the models on the reserved 20% labeled test data
predicted_test_lr = logistic_regression.predict(X_test_eval)
predicted_test_nb = naive_bayes.predict(X_test_eval)

print("Evaluation on Test Data:")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_eval, predicted_test_lr)}")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test_eval, predicted_test_nb)}")
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test_eval, predicted_test_lr))
print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(y_test_eval, predicted_test_nb))
print("Logistic Regression Classification Report:")
print(classification_report(y_test_eval, predicted_test_lr))
print("Naive Bayes Classification Report:")
print(classification_report(y_test_eval, predicted_test_nb))

