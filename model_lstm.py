import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import random
from rouge import Rouge
import wandb

wandb.init(project="scan-summarizer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
  'vocab_size': 10000,  # reduced from 15000
  'embedding_dim': 200,  # reduced from 300
  'hidden_dim': 64,  # reduced from 128
  'num_layers': 1,  # kept the same
  'dropout': 0.5,  # kept the same
  'pad_idx': 0  # kept the same
}

max_summary_length = 50  # reduced from 100

class FeatureRichEncoder(nn.Module):
  def __init__(self, input_sizes, hidden_size, num_layers=1, dropout=0.1):
    super(FeatureRichEncoder, self).__init__()
    self.input_sizes = input_sizes
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)

    self.embeddings = nn.ModuleList([nn.Embedding(input_size, hidden_size, padding_idx=0)
                                    for input_size in input_sizes])

    self.encoder = nn.LSTM(hidden_size * len(input_sizes), hidden_size, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

  def forward(self, inputs, hidden):
    embedded = [emb(inp) for emb, inp in zip(self.embeddings, inputs)]
    embedded = torch.cat(embedded, dim=-1)
    embedded = self.dropout(embedded)

    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, inputs[1], batch_first=True, enforce_sorted=False)
    packed_output, hidden = self.encoder(packed_embedded, hidden)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

    return output, hidden

  def initHidden(self, batch_size):
    return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device))


class LSTMDecoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
    super(LSTMDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.dropout = nn.Dropout(dropout)

    self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
    self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden):
    embedded = self.embedding(input)
    embedded = self.dropout(embedded)

    output, hidden = self.lstm(embedded, hidden)
    output = self.out(output)
    output = torch.log_softmax(output, dim=-1)

    return output, hidden

  def initHidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))


class Summarizer(nn.Module):
  def __init__(self, input_sizes, hidden_size, output_size, num_layers=1, dropout=0.1):
    super(Summarizer, self).__init__()
    self.encoder = FeatureRichEncoder(input_sizes, hidden_size, num_layers, dropout)
    self.decoder = LSTMDecoder(output_size, hidden_size, output_size, num_layers, dropout)

  def forward(self, inputs, decoder_input, decoder_hidden, encoder_hidden=None):
    batch_size = inputs[0].size(0)
    if encoder_hidden is None:
      encoder_hidden = self.encoder.initHidden(batch_size)
    encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden)

    # Decoder loop
    decoder_outputs = []
    for i in range(max_summary_length):
      output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
      decoder_outputs.append(output)
      decoder_input = torch.argmax(output, dim=-1).unsqueeze(1)  # Greedy search

    decoder_outputs = torch.cat(decoder_outputs, dim=1)
    return decoder_outputs, encoder_hidden

num_epochs = 10

nlp = spacy.load("en_core_web_sm")

# Define the fields for input and output sequences
article_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, include_lengths=True, batch_first=True)
summary_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, init_token='<sos>', eos_token='<eos>', include_lengths=True, batch_first=True)

# Load the data from the CSV file
dataset = TabularDataset(
  path='data_cleaned.csv',  # assuming the CSV file is in the current directory
  format='csv',
  fields=[('article', article_field), ('summary', summary_field)]
)

# Perform train-test split
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(42))

# Reduce maximum vocabulary size
article_field.build_vocab(train_data, max_size=10000)  # Adjust max_size as needed
summary_field.build_vocab(train_data, max_size=5000)  # Adjust max_size as needed

# Define the data iterators with static padding
batch_size = 32
partial_train_data = [train_data[i:i+batch_size] for i in range(0, len(train_data), batch_size)]
partial_test_data = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]

# Decrease batch size
new_batch_size = 16  # Adjust batch size as needed
train_iter = [BucketIterator(partial, batch_size=new_batch_size, sort_key=lambda x: len(x.article), sort_within_batch=True, device=device) for partial in partial_train_data]
test_iter = [BucketIterator(partial, batch_size=new_batch_size, sort_key=lambda x: len(x.article), sort_within_batch=True, device=device, train=False) for partial in partial_test_data]

# Define the model
input_sizes = [len(article_field.vocab)] * len(article_field.vocab.stoi)
hidden_size = config['hidden_dim']
output_size = len(summary_field.vocab)
model = Summarizer(input_sizes, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss(ignore_index=summary_field.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
  model.train()
  epoch_loss = 0

  for batch in train_iter:
    articles, article_lengths = batch.article
    summaries, summary_lengths = batch.summary

    decoder_input = summaries[:, :-1].to(device)
    decoder_target = summaries[:, 1:].to(device)

    decoder_hidden = model.decoder.initHidden(1)
    optimizer.zero_grad()

    decoder_outputs, _ = model(articles.to(device), decoder_input, decoder_hidden)

    loss = criterion(decoder_outputs.view(-1, output_size), decoder_target.contiguous().view(-1))
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
  print(f'Epoch: {epoch+1}, Train Loss: {epoch_loss/len(train_iter)}')
  wandb.log({"train_loss": epoch_loss/len(train_iter)})

# Evaluate the model on the test set
model.eval()
rouge = Rouge()
all_generated_summaries = []
all_reference_summaries = []

def generate_batch_summaries(batch):
  articles, article_lengths = batch.article
  summaries, summary_lengths = batch.summary

  batch_size = articles.size(0)
  encoder_hidden = model.encoder.initHidden(batch_size)

  generated_summaries = []
  with torch.no_grad():
    for i in range(batch_size):
      encoder_outputs, encoder_hidden = model.encoder([articles[i].to(device)], encoder_hidden)  # Move tensors to GPU
      decoder_input = torch.tensor([[summary_field.vocab.stoi['<sos>']]], device=device)
      decoder_hidden = encoder_hidden

      summary = []
      for _ in range(max_summary_length):
        output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
        word_id = torch.argmax(output, dim=-1).item()

        if word_id == summary_field.vocab.stoi['<eos>']:
          break
        else:
          summary.append(word_id)
          decoder_input = torch.tensor([[word_id]], device=device)

      summary = [summary_field.vocab.itos[word_id] for word_id in summary]
      generated_summaries.append(' '.join(summary))

  return generated_summaries

for batch in test_iter:
  generated_summaries = generate_batch_summaries(batch)
  reference_summaries = [' '.join(summary_field.vocab.itos[token_id] for token_id in summary) for summary in batch.summary]

  all_generated_summaries.extend(generated_summaries)
  all_reference_summaries.extend(reference_summaries)

scores = rouge.get_scores(all_generated_summaries, all_reference_summaries, avg=True)
print(scores)

# Save the model
torch.save(model.state_dict(), 'summarizer.pth')
wandb.save('summarizer.pth')
wandb.finish()