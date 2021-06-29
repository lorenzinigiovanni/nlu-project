
import torch.nn as nn
import torch.nn.functional as F
from LSTM import LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Model(nn.Module):
    def __init__(
        self,
        n_token,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.5,
        pytorch=False,
    ):
        super(Model, self).__init__()

        self.nhid = hidden_size
        self.nlayers = num_layers
        self.ntoken = n_token
        self.pytorch = pytorch

        # dropout, to be used on input, between layers and on output
        self.drop = nn.Dropout(dropout)

        # embedding encoder
        self.encoder = nn.Embedding(n_token, input_size, padding_idx=0)

        if not self.pytorch:
            # list of LSTM, one for each layer
            self.rnn = nn.ModuleList()
            for _ in range(self.nlayers):
                self.rnn.append(LSTM(input_size, hidden_size))

        if self.pytorch:
            # instantiate PyTorch LSTM
            self.rnn = nn.LSTM(input_size, hidden_size,
                               num_layers, dropout=dropout)

        # linear layer decoder
        self.decoder = nn.Linear(hidden_size, n_token)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )

    def forward(self, input, hidden, lenghts):
        # encode the input
        emb = self.encoder(input)

        # apply dropout on the input
        dropped = self.drop(emb)

        if not self.pytorch:
            output = dropped

            # cycle over the layers of the LSTM
            for i in range(self.nlayers):
                # pass the data to the LSTM
                output, hidden = self.rnn[i](
                    output,
                    hidden
                )
                # apply dropout on the output
                output = self.drop(output[i, :, :, :])

        if self.pytorch:
            packed = pack_padded_sequence(dropped, lenghts, enforce_sorted=False)
            output, hidden = self.rnn(packed, hidden)
            output, _ = pad_packed_sequence(output)
            output = self.drop(output)

        # decode the output
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)

        # apply the log softmax to the output and return it alongside hidden states
        return F.log_softmax(decoded, dim=1), hidden
