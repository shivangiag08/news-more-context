import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import nltk
from rouge import Rouge

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

# Build vocabulary
article_field.build_vocab(dataset)
summary_field.build_vocab(dataset)

# Define data iterator for evaluation
batch_size = 1  # Set batch size to 1 for evaluation
eval_iter = BucketIterator(dataset, batch_size=batch_size, device=torch.device('cpu'),
                            sort_key=lambda x: len(x.article), sort_within_batch=True, shuffle=True)

# Load the saved model
device = torch.device('cpu')  # Load model on CPU
input_size = len(article_field.vocab)
output_size = len(summary_field.vocab)
embedding_dim = 256
hidden_dim = 128

from model_training_5 import Encoder, Decoder, Seq2Seq

encoder = Encoder(input_size, embedding_dim, hidden_dim).to(device)
decoder = Decoder(output_size, embedding_dim, hidden_dim).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('/Users/tanmay/Downloads/model.pth', map_location=device))
model.eval()

# Initialize Rouge
rouge = Rouge()

# Evaluate on the first 10 examples
num_examples = 10
for i, batch in enumerate(eval_iter):
    if i >= num_examples:
        break

    src = batch.article
    trg = batch.summary
    output = model(src, trg, teacher_forcing_ratio=0)  # Turn off teacher forcing during evaluation

    # Convert token IDs to tokens
    generated_tokens = [summary_field.vocab.itos[idx] for idx in output.argmax(dim=-1).squeeze().tolist()]
    target_tokens = [summary_field.vocab.itos[idx] for idx in trg.squeeze().tolist()]

    # Remove padding tokens and join tokens into strings
    generated_summary = ' '.join([token for token in generated_tokens if token not in ['<pad>', '<eos>']])
    target_summary = ' '.join([token for token in target_tokens if token not in ['<pad>', '<eos>']])

    # Compute Rouge scores
    scores = rouge.get_scores(generated_summary, target_summary)

    print(f'Example {i+1}')
    print(f'Generated Summary: {generated_summary}')
    print(f'Target Summary: {target_summary}')
    print(f'Rouge Scores: {scores}\n')
