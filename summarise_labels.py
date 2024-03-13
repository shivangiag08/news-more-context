import pathlib
import textwrap
import csv
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import pandas as pd
import config
from google.generativeai.types import HarmCategory, HarmBlockThreshold

def to_markdown(text):
    text = text.replace('•', ' *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda __: True))
genai.configure(api_key=config.gemini_api_key)

generation_config = {
"temperature": 0,
"top_p": 1,
"top_k": 1,
"max_output_tokens": 400,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config, safety_settings=safety_settings)

def summarize_content(content):
    try:
        # Use the Gemini API to summarize the content
        response = model.generate_content("Summarize the following content in max 50 words. Write an abstractive summary based only on what is written. Do not retain any other context: " + content,safety_settings=safety_settings,generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def copy_rows(start_row, end_row):
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/Users/tanmay/Downloads/rating.csv', encoding='utf-8')

    # Select rows in the specified range
    df_subset = df.iloc[start_row:end_row]

    return df_subset

def add_summaries(df_subset,mode='a'):
    # Apply the summarize_content function to each row in the 'content' column
    # and store the results in a new 'summary' column
    df_subset['summary'] = df_subset['article'].apply(summarize_content)

    # Write the updated DataFrame back to a new CSV file
    df_subset.to_csv('data_with_summaries.csv', mode=mode, header=False if mode == 'a' else True, index=False, encoding='utf-8')

# Call the functions
df_subset = copy_rows(7500,9500)
add_summaries(df_subset, mode='a')