import torch
import torch.nn as nn
import torch.nn.functional as F

config = {
    'vocab_size': 10000,  
    'embedding_dim': 300,  
    'hidden_dim': 256,
    'num_layers': 1,  
    'dropout': 0.5,  
    'pad_idx': 0 
}

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.vocab_size = config['vocab_size']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.pad_idx = config['pad_idx']

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)

        self.lstm = nn.LSTM(self.embedding_dim,
                            self.hidden_dim,
                            num_layers=self.num_layers,
                            batch_first=True,
                            dropout=self.dropout if self.num_layers > 1 else 0,
                            bidirectional=True) 

    def forward(self, text, text_lengths):
        
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        return output, hidden