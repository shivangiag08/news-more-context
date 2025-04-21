import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import random
import numpy as np
from rouge import Rouge
import nltk
import wandb
import gensim.downloader as api

# Download NLTK resources
nltk.download('punkt')

# Initialize wandb for logging
wandb.init(project="scan-summarizer")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained Word2Vec embeddings
word_vectors = api.load("word2vec-google-news-300")

# Initialize embedding matrix with pre-trained embeddings
def create_embedding_matrix(word_vectors, vocab, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.stoi.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]
    return torch.tensor(embedding_matrix, dtype=torch.float32)

# Configuration parameters
config = {
    'vocab_size': 50000,
    'embedding_dim': 300,  # Assuming you're using 300-dimensional Word2Vec embeddings
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.5,
    'pad_idx': 0
}

# Max summary length
max_summary_length = 100

# Define fields for input and output sequences
article_field = Field(tokenize=nltk.word_tokenize, lower=True, batch_first=False)
summary_field = Field(tokenize=nltk.word_tokenize, lower=True, init_token='<sos>', eos_token='<eos>', fix_length=None)

fields = [('article', article_field), ('summary', summary_field)]

# Load the data from the CSV file
dataset = TabularDataset(
    path='/content/drive/MyDrive/data_cleaned3.csv',
    format='csv',
    fields=fields,
    skip_header=True
)

# Split the dataset into train and test
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(42))

# Combine the training and test datasets
combined_dataset = train_data + test_data

# Build vocabulary
article_field.build_vocab(combined_dataset, max_size=config['vocab_size'])
summary_field.build_vocab(combined_dataset, max_size=config['vocab_size'])

# Define data iterators
batch_size = 16
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data),
    batch_sizes=(batch_size, batch_size),
    device=device,
    sort_key=lambda x: len(x.article),
    sort_within_batch=True,
    shuffle=True
)

# Define Seq2Seq model
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        # Initialize embedding layer with pre-trained embeddings
        embedding_matrix = create_embedding_matrix(word_vectors, article_field.vocab, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=False)

    def forward(self, x):
        embedded = self.embedding(x)
        encoder_outputs, (hidden, cell) = self.lstm(embedded)
        return encoder_outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_dim)
        # Initialize embedding layer with pre-trained embeddings
        embedding_matrix = create_embedding_matrix(word_vectors, summary_field.vocab, embedding_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input = trg[0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

# Initialize model
input_size = len(article_field.vocab)
output_size = len(summary_field.vocab)
embedding_dim = config['embedding_dim']
hidden_dim = config['hidden_dim']

encoder = Encoder(input_size, embedding_dim, hidden_dim).to(device)
decoder = Decoder(output_size, embedding_dim, hidden_dim).to(device)

# Update the encoder embedding layer
encoder.embedding = nn.Embedding(len(article_field.vocab), embedding_dim)
encoder.embedding.weight.data.copy_(create_embedding_matrix(word_vectors, article_field.vocab, embedding_dim))

# Update the decoder embedding layer
decoder.embedding = nn.Embedding(len(summary_field.vocab), embedding_dim)
decoder.embedding.weight.data.copy_(create_embedding_matrix(word_vectors, summary_field.vocab, embedding_dim))

model = Seq2Seq(encoder, decoder, device).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=summary_field.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters())

# Rouge metric
rouge = Rouge()

import torch.nn.utils.rnn as rnn_utils

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch in train_iter:
        optimizer.zero_grad()
        output = model(batch.article, batch.summary)
        mask = (batch.summary != summary_field.vocab.stoi['<pad>'])
        loss = criterion(output[mask], batch.summary[mask])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_iter)

    print(f'Epoch: {epoch+1}, Train Loss: {average_loss}')
    wandb.log({'train_loss': average_loss})

# Save model
wandb.save("model_v2.pth")
torch.save(model.state_dict(), '/content/drive/MyDrive/model.pth')
wandb.finish()

# Load the saved model
device = 'cpu'
loaded_model = Seq2Seq(encoder, decoder, device).to(device)
loaded_model.load_state_dict(torch.load('/content/drive/MyDrive/model-2.pth', map_location=torch.device('cpu')))
loaded_model.eval()  # Set the model to evaluation mode

# Initialize ROUGE
rouge = Rouge()

# Define the testing loop
test_loss = 0
rouge_scores = {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
               'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
               'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}

# Define the testing loop for the first 5 summaries
num_summaries_to_evaluate = 100
num_batches_evaluated = 0

with torch.no_grad():
   for batch in test_iter:
       if num_batches_evaluated >= num_summaries_to_evaluate:
           break  # Stop evaluation after evaluating the first 5 summaries

       output = loaded_model(batch.article, batch.summary)
       mask = (batch.summary != summary_field.vocab.stoi['<pad>'])
       loss = criterion(output[mask], batch.summary[mask])
       test_loss += loss.item()

       # Convert indices to tokens
       predicted_tokens = []
       for idx_list in output.argmax(2).squeeze().tolist():
           tokens = []
           for idx in idx_list:
               tokens.append(summary_field.vocab.itos[int(idx)])
           predicted_tokens.append(tokens)

       target_tokens = []
       for idx_list in batch.summary.squeeze().tolist():
           tokens = []
           for idx in idx_list:
               tokens.append(summary_field.vocab.itos[int(idx)])
           target_tokens.append(tokens)

       # Calculate ROUGE scores
       for predicted, target in zip(predicted_tokens, target_tokens):
           predicted_str = ' '.join(predicted)
           target_str = ' '.join(target)
           scores = rouge.get_scores(predicted_str, target_str)[0]
           for key, value in scores.items():
               rouge_scores[key]['f'] += value['f']
               rouge_scores[key]['p'] += value['p']
               rouge_scores[key]['r'] += value['r']

       # Print the summaries
       print("Predicted Summary:", ' '.join(predicted_tokens[0]))
       print("Target Summary:", ' '.join(target_tokens[0]))
       print()

       num_batches_evaluated += 1

# Calculate average ROUGE scores
for key in rouge_scores.keys():
   for metric in ['f', 'p', 'r']:
       rouge_scores[key][metric] /= num_summaries_to_evaluate

# Print ROUGE scores
for key, value in rouge_scores.items():
   print(f'{key}:')
   print(f'  F1: {value["f"]}')
   print(f'  Precision: {value["p"]}')
   print(f'  Recall: {value["r"]}')

# Print the first few tokens of the target summaries
for idx_list in batch.summary.squeeze().tolist()[:5]:
   tokens = []
   for idx in idx_list:
       tokens.append(summary_field.vocab.itos[int(idx)])
   print("Target Summary:", ' '.join(tokens[:10]))  # Print the first 10 tokens