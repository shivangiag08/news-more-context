import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the labeled dataset
global_news_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\global news data set\rating.csv")
global_news_data = global_news_data[['article_id', 'title', 'title_sentiment']]

# Load the unlabeled dataset
bbc_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\bbc_news.csv")
bbc_data['title_sentiment'] = -1  # Mark unlabeled data

# Combine the labeled and unlabeled datasets
combined_data = pd.concat([global_news_data, bbc_data[['title', 'title_sentiment']]])

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_combined = vectorizer.fit_transform(combined_data['title'])
y_combined = combined_data['title_sentiment'].values
print("X_combined shape:", X_combined.shape)
print("y_combined shape:", y_combined.shape)

# Split only the global_news_data for evaluation (20%)
# The split is based on the original size of global_news_data to maintain the 80-20 split accurately
train_size = len(global_news_data) - len(bbc_data)
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(
    X_combined[:len(global_news_data)], y_combined[:len(global_news_data)], 
    train_size=train_size, random_state=42)

# Initialize the base estimator and the SelfTrainingClassifier with Naive Bayes
base_estimator = MultinomialNB()
self_training_model = SelfTrainingClassifier(base_estimator, threshold=0.7, criterion='threshold')

# Train the SelfTrainingClassifier on combined data (80% labeled + unlabeled)
self_training_model.fit(X_combined, y_combined)

# Evaluate the model on the reserved 20% labeled test data
predicted_test = self_training_model.predict(X_test_eval)
print("Evaluation on Test Data:")
print(f"Accuracy: {accuracy_score(y_test_eval, predicted_test)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_eval, predicted_test))
print("Classification Report:")
print(classification_report(y_test_eval, predicted_test))

# save the model to disk
import pickle
filename = 'finalized_model_naive_bayes.sav'
pickle.dump(self_training_model, open(filename, 'wb'))

# save test data with labels to disk
test_data = pd.DataFrame(X_test_eval)
test_data['title_sentiment'] = y_test_eval
test_data.to_csv('test_data_naive_bayes.csv', index=False)