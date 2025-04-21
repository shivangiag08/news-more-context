import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, inputs):
        embedded = [emb(inp) for emb, inp in zip(self.embeddings, inputs)]
        embedded = torch.cat(embedded, dim=-1)
        embedded = self.dropout(embedded)
        outputs, (hidden, cell) = self.encoder(embedded)
        return outputs, (hidden, cell)

class HierarchicalAttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(HierarchicalAttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.word_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.sentence_attention = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(hidden_size + output_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
        self.generator = nn.Linear(hidden_size * 3, output_size)
        self.pointer_switch = nn.Linear(hidden_size * 3, 1)

    def forward(self, decoder_input, encoder_outputs, decoder_hidden):
        batch_size = decoder_input.size(0)
        seq_len = encoder_outputs.size(1)

        # Hierarchical attention computation
        word_attns = torch.tanh(self.word_attention(encoder_outputs))
        word_attns = word_attns.view(batch_size, seq_len, -1)
        word_attns = F.softmax(word_attns, dim=2)

        sentence_attns = torch.tanh(self.sentence_attention(encoder_outputs))
        sentence_attns = F.softmax(sentence_attns, dim=1)
        sentence_attns = sentence_attns.unsqueeze(2)

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
        generator_output = F.log_softmax(self.generator(combined_vector), dim=-1)
        pointer_distribution = F.softmax(word_attns.view(batch_size, -1), dim=-1)

        # Combine generator and pointer
        final_distribution = pointer_switch * pointer_distribution + (1 - pointer_switch) * generator_output

        return final_distribution, decoder_hidden

class Summarizer(nn.Module):
    def __init__(self, input_sizes, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Summarizer, self).__init__()
        self.encoder = FeatureRichEncoder(input_sizes, hidden_size, num_layers, dropout)
        self.decoder = HierarchicalAttentionDecoder(hidden_size, output_size, num_layers, dropout)

    def forward(self, inputs, decoder_input, decoder_hidden):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        final_distribution, decoder_hidden = self.decoder(decoder_input, encoder_outputs, decoder_hidden)
        return final_distribution, decoder_hidden
  

"""
Possible improvements:
- Add attention mechanism to the encoder
- Add dropout to the encoder
- Add dropout to the decoder
- Add layer normalization to the encoder and decoder
"""