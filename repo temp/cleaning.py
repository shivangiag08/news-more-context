import pandas as pd

global_news_data = pd.read_csv("C:\Shivangi\college\Sem 4\MLPR\project/news-more-context\global news data set/rating.csv")
# drop the columns that are not required
global_news_data = global_news_data[['article_id','title', 'title_sentiment']]
# create a smaller dataset with 10000 random samples
## split global news into training and testing data. use sklearn.model_selection.train_test_split
from sklearn.model_selection import train_test_split
global_news_data, test = train_test_split(global_news_data, test_size=0.2)

# build a co training classifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
parameter = 'title'
# create a pipeline
model = make_pipeline(TfidfVectorizer(), LabelSpreading())
model.fit(global_news_data[parameter], global_news_data['title_sentiment'])
# predict the sentiment of the test data
predicted_sentiment = model.predict(test[parameter])
# calculate the accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(parameter)
print(accuracy_score(test['title_sentiment'], predicted_sentiment))
print(confusion_matrix(test['title_sentiment'], predicted_sentiment))
print(classification_report(test['title_sentiment'], predicted_sentiment))