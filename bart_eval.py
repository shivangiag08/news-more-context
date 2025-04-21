import pandas as pd
import tensorflow as tf
import keras_nlp
import time
import keras
from keras.models import load_model

BATCH_SIZE = 8
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40

# Load the pre-trained model
preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)

bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

# Load the saved weights
bart_lm = keras.saving.load_model("/Users/tanmay/Downloads/bart_model.keras")

# Load data from CSV file
data = pd.read_csv('data_cleaned3.csv')

# The 'article' and 'summary' columns as a DataFrame
articles = data['article']
summaries = data['summary']

def generate_text(model, input_text, max_length=100, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    if print_time_taken:
        print(f"Total Time Elapsed: {end - start:.2f}s")
    return output

# Generate summaries for the first 10 articles
for i in range(10):
    input_text = articles[i]
    print(f"Article {i+1}:")
    print(input_text)
    print("\nSummary:")
    summary = generate_text(bart_lm, input_text, max_length=MAX_DECODER_SEQUENCE_LENGTH)
    print(summary[0].text)
    print("\n")