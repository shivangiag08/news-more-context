import os
import time
import pandas as pd
import keras_nlp
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy

BATCH_SIZE = 8
NUM_BATCHES = 200
EPOCHS = 1  # Can be set to a higher value for better results
MAX_ENCODER_SEQUENCE_LENGTH = 512
MAX_DECODER_SEQUENCE_LENGTH = 128
MAX_GENERATION_LENGTH = 40

# Load data from CSV file
data = pd.read_csv('/kaggle/input/news-temp-2/data_cleaned3.csv')

# The 'article' and 'summary' columns as a DataFrame
articles = data['article']
summaries = data['summary']

# Create a dictionary dataset.
dataset = {
    "encoder_text": articles,
    "decoder_text": summaries,
}

# Create a tf.data.Dataset object.
train_ds = tf.data.Dataset.from_tensor_slices(dataset)
train_ds = train_ds.batch(BATCH_SIZE)

preprocessor = keras_nlp.models.BartSeq2SeqLMPreprocessor.from_preset(
    "bart_base_en",
    encoder_sequence_length=MAX_ENCODER_SEQUENCE_LENGTH,
    decoder_sequence_length=MAX_DECODER_SEQUENCE_LENGTH,
)
bart_lm = keras_nlp.models.BartSeq2SeqLM.from_preset(
    "bart_base_en", preprocessor=preprocessor
)

bart_lm.summary()

optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    epsilon=1e-6,
    global_clipnorm=1.0,  # Gradient clipping.
)
# Exclude layernorm and bias terms from weight decay.
optimizer.exclude_from_weight_decay(var_names=["bias"])
optimizer.exclude_from_weight_decay(var_names=["gamma"])
optimizer.exclude_from_weight_decay(var_names=["beta"])

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

bart_lm.compile(
    optimizer=optimizer,
    loss=loss,
    weighted_metrics=["accuracy"],
)

bart_lm.fit(train_ds, epochs=EPOCHS)

def generate_text(model, input_text, max_length=200, print_time_taken=False):
    start = time.time()
    output = model.generate(input_text, max_length=max_length)
    end = time.time()
    print(f"Total Time Elapsed: {end - start:.2f}s")
    return output

# Save the entire model
model_path = '/kaggle/working/bart_model.keras'
bart_lm.save(model_path)

# Save the weights separately
weights_path = '/kaggle/working/bart_model_weights.keras'
bart_lm.save_weights(weights_path)