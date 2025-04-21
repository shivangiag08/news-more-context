import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import wandb

# Initialize wandb
wandb.init(project="scan-summarizer")

class PointerGenerator(tf.keras.layers.Layer):
    def __init__(self, units):
        super(PointerGenerator, self).__init__()
        self.W_gen = Dense(units, activation='sigmoid')
    
    def call(self, context_vector, decoder_state, decoder_input):
        p_gen = self.W_gen(tf.concat([context_vector, decoder_state, decoder_input], axis=-1))
        return p_gen

# Define Encoder
class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, max_len):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True)
        self.lstm = LSTM(self.enc_units, return_sequences=True, return_state=True)

    def call(self, inputs):
        x = self.embedding(inputs)
        encoder_output, state_h, state_c = self.lstm(x)
        encoder_states = [state_h, state_c]
        return x, encoder_output, encoder_states  # Return x, encoder_output, and encoder_states


# Define Decoder
class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, max_len):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim, input_length=max_len, mask_zero=True)
        self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True)
        self.pointer_generator = PointerGenerator(dec_units)

    def build(self, input_shape):
        self.lstm.build(input_shape)
        super(Decoder, self).build(input_shape)

    def call(self, data):
        input_seq, target, encoder_output, encoder_states = data
        decoder_input = self.embedding(target)
        decoder_states = encoder_states
        outputs = []
        for t in range(target.shape[1] - 1):  # Exclude the last token "<end>"
            decoder_output, state_h, state_c = self.lstm(decoder_input, initial_state=decoder_states)
            p_gen = self.pointer_generator(encoder_output, decoder_states[0], decoder_input)
            outputs.append([decoder_output, decoder_states, p_gen])
            decoder_input = self.embedding(tf.argmax(decoder_output, axis=-1))  # Update decoder input
            decoder_states = [state_h, state_c]
        return outputs


# Define Model
class Seq2Seq(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, dec_units, max_len_inp, max_len_targ):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, enc_units, max_len_inp)
        self.decoder = Decoder(vocab_size, embedding_dim, dec_units, max_len_targ)
        self.final_layer = Dense(vocab_size)

    def call(self, inputs):
        input_seq, target_seq = inputs  # Unpack the inputs
        batch_size = tf.shape(input_seq)[0]
        encoder_output, encoder_states = self.encoder(input_seq)

        # Initialize decoder input as "<start>" token
        decoder_input = tf.expand_dims([tokenizer_target.word_index['<start>']] * batch_size, 1)

        # Initialize empty list to store decoder outputs
        outputs = []

        # Teacher forcing - feeding the target as the next input
        for t in range(1, target_seq.shape[1]):
            # Pass encoder output and states to the decoder
            decoder_outputs = self.decoder([decoder_input, encoder_output, encoder_states])
            decoder_output, decoder_states, p_gen = decoder_outputs[0]  # Unpack decoder outputs

            # Append decoder output to outputs
            outputs.append(decoder_output)

            # Use teacher forcing
            decoder_input = tf.expand_dims(target_seq[:, t], 1)

        # Stack the outputs
        outputs = tf.stack(outputs, axis=1)

        return outputs


# Load data from CSV
data = pd.read_csv('data_cleaned.csv')

# Prepare input and target sequences
input_sequences = data['article'].values
target_sequences = data['summary'].values

# Tokenize input and target sequences
tokenizer_input = tf.keras.preprocessing.text.Tokenizer()
tokenizer_input.fit_on_texts(input_sequences)
input_sequences = tokenizer_input.texts_to_sequences(input_sequences)

tokenizer_target = tf.keras.preprocessing.text.Tokenizer()
tokenizer_target.fit_on_texts(target_sequences)
target_sequences = tokenizer_target.texts_to_sequences(target_sequences)

# Padding sequences
max_len_inp = max([len(seq) for seq in input_sequences])
max_len_targ = max([len(seq) for seq in target_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_len_inp, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_len_targ, padding='post')

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(input_sequences, target_sequences, test_size=0.2)

# Hyperparameters
vocab_size = len(tokenizer_input.word_index) + 1
embedding_dim = 256
enc_units = 512
dec_units = 512
batch_size = 64
epochs = 10

# Build and compile the model
model = Seq2Seq(vocab_size, embedding_dim, enc_units, dec_units, max_len_inp, max_len_targ)
optimizer = Adam()
loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

model.compile(optimizer=optimizer, loss=loss_function)

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

# Function to decode sequences
def decode_sequence(input_seq):
    encoder_output, encoder_states = model.encoder(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer_target.word_index['<start>']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        decoder_outputs = model.decoder([target_seq, encoder_output, encoder_states])
        decoder_output, decoder_states, p_gen = decoder_outputs[0]  # Unpack decoder outputs
        sampled_token_index = np.argmax(decoder_output[0, -1, :])
        sampled_token = tokenizer_target.index_word[sampled_token_index]
        if sampled_token != '<end>':
            decoded_sentence += ' ' + sampled_token
        if (sampled_token == '<end>' or len(decoded_sentence.split()) >= (max_len_targ-1)):
            stop_condition = True
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        encoder_states = decoder_states
    return decoded_sentence.strip()

# Generate summaries for test data
generated_summaries = []
for i in range(len(x_test)):
    input_seq = x_test[i:i+1]
    generated_summary = decode_sequence(input_seq)
    generated_summaries.append(generated_summary)

# Convert indices back to words
original_summaries = []
for summary in y_test:
    original_summary = ' '.join([tokenizer_target.index_word[idx] for idx in summary if idx > 0])
    original_summaries.append(original_summary)

# Calculate ROUGE score
rouge = Rouge()
scores = rouge.get_scores(generated_summaries, original_summaries, avg=True)

# Log ROUGE scores to wandb
wandb.log(scores)
wandb.finish()