import pandas as pd

file_path = "project/news-more-context/global news data set/rating.csv"

def remove_rows(file_path):

    df = pd.read_csv(file_path, encoding_errors='ignore')
    print(len(df))
    # print rows 75-85
    print(df[9843:9873])
    # only take rows where title_sentiment == Positive or Negative or Neutral. use inplace to modify the dataframe
    return df

# write the dataframe to a new csv file
def write_to_csv(df):
    df.to_csv('project/news-more-context/global news data set/ratings_cleaned.csv', index=False)

df = remove_rows(file_path)
print("Dataframe cleaned.")
print("Dataframe written to csv.")
print(len(df))