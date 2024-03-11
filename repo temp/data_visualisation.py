import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv('/Users/tanmay/Downloads/rating.csv')

# Check the basic information of the dataset
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Explore categorical features
print(data['source_name'].value_counts())
print(data['category'].value_counts())

# Visualize categorical features
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='source_name',palette='rocket')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='category')
plt.xticks(rotation=90)
plt.show()

# Explore numerical features
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='title_sentiment',palette='rocket')
plt.show()

# Explore text features
print(data['title'].str.len().describe())
print(data['description'].str.len().describe())
print(data['content'].str.len().describe())

data['description'].fillna('', inplace=True)
data['article'].fillna('', inplace=True)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_title = tfidf_vectorizer.fit_transform(data['title'])
X_description = tfidf_vectorizer.fit_transform(data['description'])
X_content = tfidf_vectorizer.fit_transform(data['content'])
X_article = tfidf_vectorizer.fit_transform(data['article'])

# Combine TF-IDF features
X = pd.concat([pd.DataFrame(X_title.toarray()), pd.DataFrame(X_description.toarray()), pd.DataFrame(X_content.toarray())], axis=1)

# Apply PCA
# Select the most important features using PCA
# Update PCA to capture more components based on explained variance ratio
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X)

# Visualize PCA components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()