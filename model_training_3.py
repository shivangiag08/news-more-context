import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
from rouge import Rouge
import wandb
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import random_split
import torch.nn.functional as F

wandb.init(project="scan-summarizer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'vocab_size': 10000,
    'embedding_dim': 128,
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.5,
    'pad_idx': 0
}

max_summary_length = 100

class FeatureRichEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(FeatureRichEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers,
                               bidirectional=True, dropout=dropout, batch_first=True)

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.encoder(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        return output, hidden

class HierarchicalAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(HierarchicalAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, dropout=dropout)
        self.decoder = nn.LSTM(hidden_size + output_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)

        self.generator = nn.Linear(hidden_size * 2, output_size)
        self.pointer_switch = nn.Linear(hidden_size * 2, 1)

    def forward(self, decoder_input, encoder_outputs, decoder_hidden):
        batch_size, seq_len, _ = encoder_outputs.size()

        context_vector, _ = self.attention(query=decoder_input.unsqueeze(1),
                                            key=encoder_outputs,
                                            value=encoder_outputs)

        context_vector = context_vector.squeeze(1)

        decoder_input = torch.cat((decoder_input, context_vector), dim=-1)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        combined_vector = torch.cat((decoder_output, context_vector), dim=-1)
        pointer_switch = torch.sigmoid(self.pointer_switch(combined_vector))

        generator_output = torch.log_softmax(self.generator(combined_vector), dim=-1)
        pointer_distribution = F.softmax(context_vector, dim=-1)

        final_distribution = pointer_switch * pointer_distribution + (1 - pointer_switch) * generator_output

        return final_distribution, decoder_hidden

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Summarizer, self).__init__()
        self.encoder = FeatureRichEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = HierarchicalAttentionDecoder(hidden_size, output_size, num_layers, dropout)

    def forward(self, inputs, sequence_lengths, decoder_input, decoder_target, encoder_hidden=None):
        batch_size = inputs.size(0)
        if encoder_hidden is None:
            encoder_hidden = self.encoder.initHidden(batch_size)
        
        if decoder_input is None:  # If decoder_input is not provided, create it
            decoder_input = torch.zeros(batch_size, 1, self.decoder.output_size, device=device)  # Start token
        
        encoder_outputs, encoder_hidden = self.encoder(inputs, sequence_lengths)
        final_distribution, decoder_hidden = self.decoder(decoder_input, encoder_outputs, encoder_hidden)
        return final_distribution, decoder_hidden

# Define other parameters
num_epochs = 10
nlp = spacy.load("en_core_web_sm")

# Load data
data_df = pd.read_csv('/Users/tanmay/Downloads/data_cleaned.csv')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Tokenize data
data_df['article'] = data_df['article'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512, return_tensors='pt')[0])
data_df['summary'] = data_df['summary'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512, return_tensors='pt')[0])

# Create datasets and data loaders
article_data = [article for article in data_df['article'].values]
summary_data = [summary for summary in data_df['summary'].values]
dataset = list(zip(article_data, summary_data))
train_data, test_data = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

def collate_fn(batch):
    articles, summaries = zip(*batch)
    articles = torch.nn.utils.rnn.pad_sequence(articles, batch_first=True, padding_value=tokenizer.pad_token_id)
    summaries = torch.nn.utils.rnn.pad_sequence(summaries, batch_first=True, padding_value=tokenizer.pad_token_id)
    article_lengths = [torch.nonzero(article).size(0) for article in articles]  # Compute true lengths
    return articles, summaries, article_lengths

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the model, loss function, and optimizer
max_token_id = max(tokenizer.vocab.values()) + 1
model = Summarizer(max_token_id, config['hidden_dim'], max_token_id).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for articles, summaries, article_lengths in train_loader:
        articles = articles.to(device)
        summaries = summaries.to(device)
        decoder_input = summaries[:, :-1]
        decoder_target = summaries[:, 1:]

        optimizer.zero_grad()
        final_distribution, _ = model(articles, article_lengths, decoder_input, decoder_target)

        loss = criterion(final_distribution.view(-1, max_token_id), decoder_target.contiguous().view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch: {epoch+1}, Train Loss: {epoch_loss/len(train_loader)}')
    wandb.log({"train_loss": epoch_loss/len(train_loader)})

# Evaluate the model
model.eval()
rouge = Rouge()
all_generated_summaries = []
all_reference_summaries = []

def generate_batch_summaries(batch):
    articles, summaries = batch
    articles = articles.to(device)
    summaries = summaries.to(device)

    batch_size = articles.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size)

    generated_summaries = []
    with torch.no_grad():
        for i in range(batch_size):
            encoder_outputs, encoder_hidden = model.encoder(articles[i].unsqueeze(0))
            decoder_input = torch.tensor([[tokenizer.cls_token_id]], device=device)
            decoder_hidden = encoder_hidden

            summary = []
            for _ in range(max_summary_length):
                final_distribution, decoder_hidden = model.decoder(decoder_input, encoder_outputs, decoder_hidden)
                word_id = final_distribution.argmax(dim=-1).item()

                if word_id == tokenizer.sep_token_id:
                    break
                else:
                    summary.append(word_id)
                    decoder_input = torch.tensor([[word_id]], device=device)

            summary = tokenizer.decode(summary, skip_special_tokens=True)
            generated_summaries.append(summary)

    return generated_summaries

for batch in test_loader:
    generated_summaries = generate_batch_summaries(batch)
    reference_summaries = [tokenizer.decode(summary, skip_special_tokens=True) for summary in batch[1]]

    all_generated_summaries.extend(generated_summaries)
    all_reference_summaries.extend(reference_summaries)

scores = rouge.get_scores(all_generated_summaries, all_reference_summaries, avg=True)
print(scores)

# Save the model
torch.save(model.state_dict(), 'summarizer.pth')
wandb.save('summarizer.pth')
wandb.finish()
