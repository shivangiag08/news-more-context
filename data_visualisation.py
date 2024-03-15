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
""" 
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
plt.show() """

#print the number of articles with source_name as Daily News and title_sentiment as 0
print(data[(data['source_name'] == 'ETF Daily News') & (data['title_sentiment'] == 'Neutral')].shape[0])


# Stacked bar chart for title_sentiment proportions by source_name
proportions = {
    'Forbes': {'Neutral': 69.432471, 'Positive': 22.808908, 'Negative': 7.758621},
    'CNA': {'Neutral': 68.545994, 'Negative': 18.694362, 'Positive': 12.759644},
    'Time': {'Neutral': 63.666667, 'Negative': 30.833333, 'Positive': 5.500000},
    'Phys.Org': {'Neutral': 66.191607, 'Negative': 22.486144, 'Positive': 11.322249},
    'Digital Trends': {'Positive': 51.785714, 'Neutral': 42.729592, 'Negative': 5.484694},
    'Al Jazeera English': {'Neutral': 55.709135, 'Negative': 42.427885, 'Positive': 1.862981},
    'BBC News': {'Neutral': 56.081401, 'Negative': 40.416469, 'Positive': 3.502130},
    'Deadline': {'Neutral': 85.193133, 'Negative': 9.012876, 'Positive': 5.793991},
    'Euronews': {'Neutral': 54.895105, 'Negative': 39.860140, 'Positive': 5.244755},
    'RT': {'Neutral': 61.879433, 'Negative': 35.726950, 'Positive': 2.393617},
    'The Punch': {'Neutral': 68.277778, 'Negative': 25.277778, 'Positive': 6.444444},
    'International Business Times': {'Neutral': 69.925435, 'Negative': 26.429163, 'Positive': 3.645402},
    'ETF Daily News': {'Neutral': 90.732115, 'Positive': 5.923803, 'Negative': 3.344082},
    'ABC News': {'Neutral': 57.968902, 'Negative': 35.276968, 'Positive': 6.754130},
    'Globalsecurity.org': {'Neutral': 72.298814, 'Negative': 23.949984, 'Positive': 3.751202},
    'Marketscreener.com': {'Neutral': 85.682819, 'Positive': 11.233480, 'Negative': 3.083700},
    'The Times of India': {'Neutral': 68.616738, 'Negative': 16.124733, 'Positive': 15.258529},
    'GlobeNewswire': {'Neutral': 82.813941, 'Positive': 16.688180, 'Negative': 0.497879},
    'CNN': {'Neutral': 53.183521, 'Negative': 38.202247, 'Positive': 8.614232},
    'Business Insider': {'Neutral': 44.591937, 'Negative': 43.756146, 'Positive': 11.651917},
    'Gizmodo.com': {'Neutral': 65.206186, 'Negative': 19.587629, 'Positive': 15.206186},
    'Wired': {'Neutral': 51.111111, 'Positive': 24.814815, 'Negative': 24.074074},
    'The Verge': {'Neutral': 55.140187, 'Positive': 28.504673, 'Negative': 16.355140},
    'NPR': {'Neutral': 62.895005, 'Negative': 28.440367, 'Positive': 8.664628},
    'Boing Boing': {'Neutral': 41.093969, 'Negative': 35.063114, 'Positive': 23.842917},
    'Android Central': {'Positive': 52.107280, 'Neutral': 40.613027, 'Negative': 7.279693},
    'ReadWrite': {'Neutral': 66.666667, 'Positive': 21.604938, 'Negative': 11.728395},
    'AllAfrica - Top Africa News': {'Neutral': 60.0, 'Negative': 40.0}
}

df = pd.DataFrame(proportions).T
df.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='#9C365A')
plt.title('Title Sentiment Proportions by Source Name')
plt.xlabel('Source Name')
plt.ylabel('Proportion (%)')
plt.legend(title='Title Sentiment')
plt.show()