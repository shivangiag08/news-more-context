import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, BucketIterator
import nltk
from rouge import Rouge
import random
import gensim.downloader as api
import numpy as np

# Download NLTK resources
nltk.download('punkt')

# Define fields for input and output sequences
article_field = Field(tokenize=nltk.word_tokenize, lower=True, batch_first=False)
summary_field = Field(tokenize=nltk.word_tokenize, lower=True, init_token='<sos>', eos_token='<eos>', fix_length=None)

fields = [('article', article_field), ('summary', summary_field)]

# Load the data from the CSV file
dataset = TabularDataset(
    path='data_cleaned3.csv',
    format='csv',
    fields=fields,
    skip_header=True
)

# Load pre-trained Word2Vec embeddings
word_vectors = api.load("word2vec-google-news-300")

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = len(article_field.vocab)
output_size = len(summary_field.vocab)
embedding_dim = 300
hidden_dim = 256
num_layers = 1
dropout = 0.5
pad_idx = 0

def create_embedding_matrix(word_vectors, vocab, embedding_dim):
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.stoi.items():
        if word in word_vectors:
            embedding_matrix[idx] = word_vectors[word]
    return torch.tensor(embedding_matrix, dtype=torch.float32)

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

# Load the model
encoder = Encoder(input_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)
decoder = Decoder(output_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)

# Update the encoder embedding layer
encoder.embedding = nn.Embedding(input_size, embedding_dim)
encoder.embedding.weight.data.copy_(create_embedding_matrix(word_vectors, article_field.vocab, embedding_dim))

# Update the decoder embedding layer
decoder.embedding = nn.Embedding(output_size, embedding_dim)
decoder.embedding.weight.data.copy_(create_embedding_matrix(word_vectors, summary_field.vocab, embedding_dim))

model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/model-word2vec.pth', map_location=device))
model.eval()

# Initialize Rouge
rouge = Rouge()

# Define data iterator for evaluation
batch_size = 1  # Set batch size to 1 for evaluation
test_iter = BucketIterator(dataset, batch_size=batch_size, device=device,
                            sort_key=lambda x: len(x.article), sort_within_batch=True, shuffle=False)

# Evaluate on the first 10 summaries
total_rouge_score = {"rouge-1": {"f": 0, "p": 0, "r": 0}, "rouge-2": {"f": 0, "p": 0, "r": 0}, "rouge-l": {"f": 0, "p": 0, "r": 0}}
num_examples = 0
for batch in test_iter:
    if num_examples >= 10:
        break

    src = batch.article
    trg = batch.summary
    output = model(src, trg, teacher_forcing_ratio=0)  # Turn off teacher forcing during evaluation

    # Convert token IDs to tokens
    generated_tokens = [summary_field.vocab.itos[idx] for idx in output.argmax(dim=-1).squeeze().tolist()]
    target_tokens = [summary_field.vocab.itos[idx] for idx in trg.squeeze().tolist()]

    # Remove padding tokens and join tokens into strings
    generated_summary = ' '.join([token for token in generated_tokens if token not in ['<pad>', '<eos>', '<sos>']])
    target_summary = ' '.join([token for token in target_tokens if token not in ['<pad>', '<eos>', '<sos>']])

    # Compute Rouge scores
    scores = rouge.get_scores(generated_summary, target_summary)

    # Accumulate Rouge scores
    for key in total_rouge_score.keys():
        for metric in ['f', 'p', 'r']:
            total_rouge_score[key][metric] += scores[0][key][metric]

    print(f'Generated Summary: {generated_summary}')
    print(f'Target Summary: {target_summary}')
    print(f'Rouge Scores: {scores}\n')

    num_examples += 1

# Calculate average Rouge score
average_rouge_score = {}
for key in total_rouge_score.keys():
    average_rouge_score[key] = {}
    for metric in ['f', 'p', 'r']:
        average_rouge_score[key][metric] = total_rouge_score[key][metric] / 10

print("Average Rouge Score:", average_rouge_score)