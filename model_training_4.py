import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import random
from rouge import Rouge
import nltk
import wandb
import torch.nn.functional as F

# Download NLTK resources
nltk.download('punkt')

# Initialize wandb for logging
wandb.init(project="scan-summarizer")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration parameters
config = {
    'vocab_size': 10000,
    'embedding_dim': 256,
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.5,
    'pad_idx': 0,
    'num_heads': 8  # Number of attention heads
}

# Max summary length
max_summary_length = 100

print("checkpoint 1")
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

# Split the dataset into train and test
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(42))

print("checkpoint 2")
# Build vocabulary
article_field.build_vocab(train_data, max_size=config['vocab_size'])
summary_field.build_vocab(train_data, max_size=config['vocab_size'])

# Define data iterators
batch_size = 16
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data),
    batch_sizes=(batch_size, batch_size),
    device=device,
    sort_key=lambda x: (len(x.article), len(x.summary)),  # Sort by article length and summary length
    sort_within_batch=True,
    shuffle=True
)

print("checkpoint 3")
# Define Seq2Seq model
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=False, batch_first=True)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return encoder_outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, hidden, encoder_outputs):
        # Permute the hidden state to (batch_size, 1, hidden_dim)
        hidden = hidden.permute(1, 0, 2)

        # Apply multihead attention
        attn_output, _ = self.multihead_attn(hidden, encoder_outputs, encoder_outputs)

        # Permute the output back to (1, batch_size, hidden_dim)
        attn_output = attn_output.permute(1, 0, 2)

        return attn_output

class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim, num_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.attention = Attention(hidden_dim, num_heads=config['num_heads'])  # Use multihead attention
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        input = input.unsqueeze(1)  # Add an extra dimension for the sequence length
        embedded = self.embedding(input)

        # Permute the hidden tensor to (batch_size, 1, hidden_dim)
        hidden = hidden.permute(1, 0, 2)

        # Pass the hidden state to the Attention module
        top_hidden = hidden

        attn_weights = self.attention(top_hidden, encoder_outputs)

        # Permute the attention weights back to (1, batch_size, hidden_dim)
        attn_weights = attn_weights.permute(1, 0, 2)

        context = attn_weights.unsqueeze(1).bmm(encoder_outputs.permute(0, 1, 2))
        context = context.repeat(1, embedded.shape[1], 1)  # Repeat the context vector for each sequence step
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden.permute(1, 0, 2), cell))
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)
        prediction = self.fc(output)
        return prediction, hidden.permute(1, 0, 2), cell, attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pointer_gen=True):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pointer_gen = pointer_gen

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_size
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Get the lengths of the input sequences
        src_lengths = [len(s) for s in src]

        # Pass the input sequences and their lengths to the Encoder
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        # Initialize the hidden and cell states with the correct batch size
        hidden = hidden.repeat(1, batch_size, 1)
        cell = cell.repeat(1, batch_size, 1)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            if self.pointer_gen:
                # Implement pointer generator switch
                # Compute pointer distribution
                pointer_dist = attn_weights.bmm(encoder_outputs.permute(0, 1, 2))
                pointer_dist = pointer_dist.permute(0, 2, 1)  # (batch_size, trg_len, src_len)
                vocab_dist = output  # (batch_size, trg_vocab_size)

               # Compute combined distribution
                combined_dist = torch.zeros_like(vocab_dist).scatter_add_(1, top1.unsqueeze(1), pointer_dist.gather(2, top1.unsqueeze(2)).squeeze(2))

               # Sample from the combined distribution
                input = combined_dist.multinomial(1).squeeze(1)
            else:
                input = trg[:, t] if teacher_force else top1
        return outputs

print("checkpoint 4")
# Initialize model
input_size = len(article_field.vocab)
output_size = len(summary_field.vocab)
embedding_dim = config['embedding_dim']
hidden_dim = config['hidden_dim']

encoder = Encoder(input_size, embedding_dim, hidden_dim).to(device)
encoder_hidden_dim = config['hidden_dim']
decoder = Decoder(output_size, embedding_dim, encoder_hidden_dim, hidden_dim).to(device)

model = Seq2Seq(encoder, decoder, device, pointer_gen=True).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=summary_field.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters())

print("checkpoint 5")
# Rouge metric
rouge = Rouge()

import torch.nn.utils.rnn as rnn_utils

print("checkpoint 6")
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
   print("starting training")
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

# Evaluation loop using rouge metric
model.eval()
wandb.save("model_v3.pth")
torch.save(model.state_dict(), 'model.pth')
all_generated_summaries = []
all_reference_summaries = []

num_evaluated_batches = 0  # Track the number of evaluated batches

for batch in test_iter:
   output = model(batch.article, batch.summary, 0)
   output_dim = output.shape[-1]
   output = output[:, :, 1:].reshape(-1, output_dim)
   trg = batch.summary[:, 1:].reshape(-1)
   generated_summaries = output.argmax(1).reshape(batch_size, -1)
   all_generated_summaries.extend(generated_summaries.tolist())
   all_reference_summaries.extend(trg.tolist())

   num_evaluated_batches += 1
   if num_evaluated_batches >= 10:
       break  # Break out of the loop after 10 batches

# Convert indices to tokens
generated_summaries = [' '.join([summary_field.vocab.itos[t] for t in summary]) for summary in all_generated_summaries]
reference_summaries = [' '.join([summary_field.vocab.itos[t] for t in summary]) for summary in all_reference_summaries]

# Compute rouge scores
scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
print(scores)
wandb.log(scores)
wandb.finish()