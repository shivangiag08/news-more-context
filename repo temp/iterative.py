
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the labeled dataset
global_news_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\global news data set\rating.csv")
global_news_data = global_news_data[['article_id', 'title', 'title_sentiment']]

# Load the unlabeled dataset
bbc_data = pd.read_csv(r"C:\Shivangi\college\Sem 4\MLPR\project\news-more-context\output_10k.csv")
bbc_data = bbc_data[['article_id', 'title']]

# Split labeled data into training and testing sets
train_data, test_data = train_test_split(global_news_data, test_size=0.2, random_state=42)

# Encode sentiment labels
label_encoder = LabelEncoder()
label_encoder.fit(["Positive", "Negative", "Neutral"])
train_data['title_sentiment'] = label_encoder.transform(train_data['title_sentiment'])
test_data['title_sentiment'] = label_encoder.transform(test_data['title_sentiment'])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data['title'])
y_train = train_data['title_sentiment'].values

# Initialize and train the Label Spreading model
model = LabelSpreading(kernel='knn')
model.fit(X_train, y_train)

# Vectorize unlabeled titles
X_unlabeled = vectorizer.transform(bbc_data['title'])

# Confidence threshold for pseudo-labeling
confidence_threshold = 0.5

# Iterative pseudo-labeling process
while True:
    # Predict probabilities on the unlabeled data
    probs = model.predict_proba(X_unlabeled)
    print(probs)
    max_confidence = np.max(probs, axis=1)
    
    # Determine instances meeting the confidence threshold
    confident_mask = max_confidence > confidence_threshold
    if not np.any(confident_mask):
        break  # Exit if no confident predictions
    
    # Select confident instances and their pseudo-labels
    y_confident = np.argmax(probs[confident_mask], axis=1)
    pseudo_labels = label_encoder.inverse_transform(y_confident)

    # Assign pseudo-labels to bbc_data for confident instances
    bbc_data.loc[confident_mask, 'pseudo_label'] = pseudo_labels

    # Update training set and model for the next iteration
    confident_titles = bbc_data.loc[confident_mask, 'title']
    all_titles_series = pd.concat([train_data['title'], confident_titles])
    all_labels = np.concatenate([y_train, y_confident])
    
    X_train_updated = vectorizer.fit_transform(all_titles_series)
    y_train_updated = all_labels
    model.fit(X_train_updated, y_train_updated)
    
    # Remove confidently labeled instances from X_unlabeled for the next iteration
    bbc_data = bbc_data.loc[~confident_mask]
    X_unlabeled = vectorizer.transform(bbc_data['title'])

# Prepare test data and evaluate the model
X_test = vectorizer.transform(test_data['title'])
y_test = test_data['title_sentiment']
predicted_test = model.predict(X_test)

# Display evaluation metrics
print("Evaluation on Test Data:")
print(f"Accuracy: {accuracy_score(y_test, predicted_test)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted_test))
print("Classification Report:")
print(classification_report(y_test, predicted_test))


# Save the labeled bbc_data with pseudo-labels to a new CSV file
output_path = r"C:\Shivangi\college\Sem 4\MLPR\project/news-more-context\labeled_bbc_data.csv"
print(bbc_data['pseudo_label'].unique())
bbc_data.to_csv(output_path, index=False)