import numpy as np
import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

#load csv for cleaning
data = pd.read_csv('data_cleaned.csv')

#clean the text data
data['article'] = data['article'].apply(clean_text)
data['summary'] = data['summary'].apply(clean_text)

#save cleaned data to new csv file
data.to_csv('data_cleaned3.csv', index=False)