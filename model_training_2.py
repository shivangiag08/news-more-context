import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy
import random
from rouge import Rouge
import wandb
import gensim.downloader as api

wandb.init(project="scan-summarizer")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {
    'vocab_size': 25000,
    'embedding_dim': 100,
    'hidden_dim': 128,
    'num_layers': 1,
    'dropout': 0.5,
    'pad_idx': 0
}

word2vec_model = api.load('word2vec-google-news-300')

max_summary_length = 100

print("checkpoint 1")

class FeatureRichEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(FeatureRichEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word2vec_model.vectors), freeze=False)

        self.encoder = nn.GRU(hidden_size, hidden_size, num_layers,
                               batch_first=True, bidirectional=True, dropout=dropout)

    def forward(self, inputs, sequence_lengths):
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.encoder(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=device))

class HierarchicalAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(HierarchicalAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.word_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.sentence_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.GRU(hidden_size + output_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
        self.generator = nn.Linear(hidden_size * 3, output_size)
        self.pointer_switch = nn.Linear(hidden_size * 3, 1)

    def forward(self, decoder_input, encoder_outputs, decoder_hidden):
        batch_size = decoder_input.size(0)
        seq_len = encoder_outputs.size(1)

        # Hierarchical attention computation
        word_attns = self.compute_word_attention(encoder_outputs)
        sentence_attns = self.compute_sentence_attention(encoder_outputs)
        combined_attns = word_attns * sentence_attns
        combined_attns = combined_attns.sum(dim=1)
        context_vector = torch.bmm(combined_attns.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)

        # Decoder step
        decoder_input = torch.cat((decoder_input, context_vector), dim=-1)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        # Pointer switch calculation
        combined_vector = torch.cat((decoder_output, context_vector), dim=-1)
        pointer_switch = torch.sigmoid(self.pointer_switch(combined_vector))

        # Generator and pointer distributions
        generator_output = torch.log_softmax(self.generator(combined_vector), dim=-1)
        pointer_distribution = torch.softmax(word_attns.view(batch_size, -1), dim=-1)

        # Combine generator and pointer
        final_distribution = pointer_switch * pointer_distribution + (1 - pointer_switch) * generator_output

        return final_distribution, decoder_hidden

    def compute_word_attention(self, encoder_outputs):
        word_attns = torch.tanh(self.word_attention(encoder_outputs))
        word_attns = word_attns.view(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        word_attns = torch.softmax(word_attns, dim=2)
        return word_attns

    def compute_sentence_attention(self, encoder_outputs):
        sentence_attns = torch.tanh(self.sentence_attention(encoder_outputs))
        sentence_attns = torch.softmax(sentence_attns, dim=1)
        sentence_attns = sentence_attns.unsqueeze(2)
        return sentence_attns

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Summarizer, self).__init__()
        self.encoder = FeatureRichEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = HierarchicalAttentionDecoder(hidden_size, output_size, num_layers, dropout)

    def forward(self, inputs, sequence_lengths, decoder_input, encoder_hidden=None):
        batch_size = inputs.size(0)
        if encoder_hidden is None:
            encoder_hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(inputs, sequence_lengths)
        final_distribution, decoder_hidden = self.decoder(decoder_input, encoder_outputs, encoder_hidden)
        return final_distribution, decoder_hidden

num_epochs = 10

nlp = spacy.load("en_core_web_sm")

print("checkpoint 2")

# Define the fields for input and output sequences
max_article_length = 100
article_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, batch_first=True, fix_length=max_article_length)
summary_field = Field(tokenize=lambda x: [token.text for token in nlp.tokenizer(x)], lower=True, init_token='<sos>', eos_token='<eos>')

fields = [('article', article_field), ('summary', summary_field)]
# Load the data from the CSV file
dataset = TabularDataset(
    path='data_cleaned.csv',
    format='csv',
    fields=fields,
    skip_header=True  # Add this line if your CSV file has a header row
)
train_data, test_data = dataset.split(split_ratio=0.8, random_state=random.seed(42))
article_field.build_vocab(train_data, max_size=25000)
summary_field.build_vocab(train_data, max_size=5000)

print("checkpoint 3")

# Define the data iterators with GPU, static padding, and custom sort_key
batch_size = 32
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data),
    batch_sizes=(batch_size, batch_size),
    device=device,
    sort_within_batch=True,
    shuffle=True
)

print("checkpoint 4")

# Define the model
input_size = len(article_field.vocab)
hidden_size = config['hidden_dim']
output_size = len(summary_field.vocab)
model = Summarizer(input_size, hidden_size, output_size).to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss(ignore_index=summary_field.vocab.stoi['<pad>'])
optimizer = optim.Adam(model.parameters())

print("checkpoint 5")

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in train_iter:
        articles = batch.article[0]
        article_lengths = batch.article[1]
        summaries = batch.summary[0]
        summary_lengths = batch.summary[1]

        decoder_input = torch.tensor([[summary_field.vocab.stoi['<sos>']] * articles.size(0)], device=device)
        encoder_hidden = model.encoder.initHidden(articles.size(0))
        optimizer.zero_grad()

        final_distribution, decoder_hidden = model(articles, article_lengths, decoder_input, encoder_hidden)

        # Compute loss
        loss = criterion(final_distribution.view(-1, output_size), summaries.view(-1))
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

print("checkpoint 6")

def generate_batch_summaries(batch):
    articles, article_lengths = batch.article
    summaries, _ = batch.summary

    batch_size = articles.size(0)
    encoder_hidden = model.encoder.initHidden(batch_size)

    generated_summaries = []
    with torch.no_grad():
        for i in range(batch_size):
            encoder_outputs, encoder_hidden = model.encoder(articles[i].unsqueeze(0), [article_lengths[i]])
            decoder_input = torch.tensor([[summary_field.vocab.stoi['<sos>']]], device=device)
            decoder_hidden = encoder_hidden

            summary = []
            for _ in range(max_summary_length):
                final_distribution, decoder_hidden = model.decoder(decoder_input, encoder_outputs, decoder_hidden)
                word_id = final_distribution.argmax(dim=-1).item()

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
