import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from translate import Translator
import pickle

# Back-translation function
def back_translate(series, to_lang='de', from_lang='en'):
    translator_to = Translator(to_lang=to_lang)
    translator_back = Translator(from_lang=to_lang, to_lang=from_lang)
    return series.apply(lambda x: translator_back.translate(translator_to.translate(x)))

# Load the labeled dataset
global_news_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\global news data set\rating.csv")  # Update with your path

# Split the labeled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    global_news_data[['title']], global_news_data['title_sentiment'], 
    train_size=0.8)

#print("Training Set Size:", len(X_train))
# Apply back-translation to the training set for augmentation
#augmented_titles = back_translate(X_train[y_train.isin(['Positive', 'Negative'])]['title'])
#augmented_data = X_train[y_train.isin(['Positive', 'Negative'])].copy()
#augmented_data['title'] = augmented_titles

#print("Augmented Data Size:", len(augmented_data))
# Append back-translated data to the training set
X_train_augmented = X_train
y_train_augmented = y_train

# Load and prepare the unlabeled dataset
bbc_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\bbc_news.csv")  # Update with your path
bbc_data['title_sentiment'] = -1  # Mark unlabeled data
new_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\global news data set\uncommon_data.csv", encoding='latin-1', encoding_errors='ignore')
new_data['title_sentiment'] = -1  # Mark unlabeled data
new_data.dropna(subset=['title'], inplace=True)

# Combine unlabeled data with the augmented training set
combined_training_titles = pd.concat([X_train_augmented['title'], bbc_data['title'], new_data['title']])
combined_training_labels = pd.concat([y_train_augmented, bbc_data['title_sentiment'], new_data['title_sentiment']])

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_combined = vectorizer.fit_transform(combined_training_titles)
y_combined = combined_training_labels.values

# Initialize the base estimator and the SelfTrainingClassifier
base_estimator = LogisticRegression(max_iter=1000)
self_training_model = SelfTrainingClassifier(base_estimator, threshold=0.999, criterion='threshold')

# Train the SelfTrainingClassifier on combined data
self_training_model.fit(X_combined, y_combined)

# Evaluate the model on the reserved labeled test data
X_test_vectorized = vectorizer.transform(X_test['title'])
predicted_test = self_training_model.predict(X_test_vectorized)
print("Evaluation on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, predicted_test)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_test))
print("Classification Report:")
print(classification_report(y_test, predicted_test))

# Save the model to disk
filename = 'finalized_model_logistic_regression.sav'
pickle.dump(self_training_model, open(filename, 'wb'))