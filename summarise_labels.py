import pathlib
import textwrap
import csv
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import pandas as pd

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

API_KEY = "AIzaSyBDpSy0QWjY-83z01vSRdSrIKh5cRBZ89M"
genai.configure(api_key=API_KEY)

# Initialize the Gemini API model
model = genai.GenerativeModel('gemini-pro')

def summarize_content(content):
    # Use the Gemini API to summarize the content
    response = model.generate_content("Summarise the follwing content in max 50 words. Write an abstractive summary based only on what is written. Do not retain any other context: " + content)  # Adjust max_tokens as needed
    return response.text

# Function to read a CSV, summarize the content of each row, and add it to a new 'summary' column
def add_summaries():
    with open('Book1.csv', 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        fieldnames = reader.fieldnames + ['summary']  # Add 'summary' to the list of fields

    # Summarize content and add to the 'summary' column
    for row in rows:
        content = row['article']
        summary = summarize_content(content)
        row['summary'] = summary

    # Write the updated data back to the CSV
    with open('Book1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def add_summaries_pandas():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('/Users/tanmay/Downloads/Book1.csv', encoding='utf-8')
    
    # Apply the summarize_content function to each row in the 'content' column
    # and store the results in a new 'summary' column
    df['summary'] = df['content'].apply(summarize_content)
    
    # Write the updated DataFrame back to a new CSV file
    df.to_csv('data_with_summaries.csv', index=False, encoding='utf-8')

# Call the function to add summaries using pandas
add_summaries_pandas()
