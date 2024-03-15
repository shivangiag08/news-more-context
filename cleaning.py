import pandas as pd

global_news_data_whole = pd.read_csv("C:\Shivangi\college\Sem 4\MLPR\project/news-more-context\global news data set/rating.csv")
# drop the columns that are not required
global_news_data_whole = global_news_data_whole[['article_id', 'source_id', 'source_name', 'title','published_at', 'category', 'article', 'title_sentiment']]
# create a smaller dataset with 10000 random samples
global_news_data = global_news_data_whole.sample(n=20000)
## split global news into training and testing data. use sklearn.model_selection.train_test_split
from sklearn.model_selection import train_test_split
train, test = train_test_split(global_news_data, test_size=0.2)
print(train.shape)

# build a self training semi supervised learning model
# use the title_sentiment as the target variable
# use the article as the feature
# use the confidence score of each prediction as the threshold for the next iteration
# only if the model can predict with a confidence score greater than the threshold, the prediction will be used as a label for the next iteration
# keep iterating until all unlabeled data is labeled

from sklearn.svm import SVC
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# create a pipeline
# use TfidfVectorizer to convert the article into a matrix of TF-IDF features
# use SVC as the classifier
# use ConfidenceBasedSelfTraining as the semi supervised learning model

pipeline = make_pipeline(TfidfVectorizer(), SelfTrainingClassifier(SVC(probability=True)))

# fit the model
pipeline.fit(train['article'], train['title_sentiment'])

# predict the test data
predicted = pipeline.predict(test['article'])

# print the accuracy
print(pipeline.score(test['article'], test['title_sentiment']))

# print the classification report
from sklearn.metrics import classification_report

print(classification_report(test['title_sentiment'], predicted))

# print the confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(test['title_sentiment'], predicted))