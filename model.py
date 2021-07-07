
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

        self.num_layers = num_layers
        self.n_token = n_token
        self.pytorch = pytorch

        # dropout, to be used on input, between layers and on output
        self.drop = nn.Dropout(dropout)

        # embedding encoder
        self.encoder = nn.Embedding(n_token, input_size, padding_idx=0)

        if not self.pytorch:
            # list of LSTM, one for each layer
            self.rnn = nn.ModuleList()
            for _ in range(self.num_layers):
                self.rnn.append(LSTM(input_size, hidden_size))

        if self.pytorch:
            # instantiate PyTorch LSTM
            self.rnn = nn.LSTM(input_size, hidden_size,
                               num_layers, dropout=dropout)

        # linear layer decoder
        self.decoder = nn.Linear(hidden_size, n_token)

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def forward(self, input, lenghts, hidden_states):
        # encode the input
        emb = self.encoder(input)

        # apply dropout on the input
        dropped = self.drop(emb)

        if not self.pytorch:
            output = dropped

            # cycle over the layers of the LSTM
            for i in range(self.num_layers):
                # pass the data to the LSTM
                output, hidden = self.rnn[i](output, hidden_states[i])
                # update the hidden states
                hidden_states[i] = hidden
                # apply dropout on the output
                output = self.drop(output)

        if self.pytorch:
            # pack the padded sequence
            packed = pack_padded_sequence(dropped, lenghts, enforce_sorted=False)
            # pass the data to the LSTM
            output, hidden_states = self.rnn(packed, hidden_states if hidden_states[0] is not None else None)
            # upack by padding, the packed sequence
            output, _ = pad_packed_sequence(output)
            # apply dropout on the output
            output = self.drop(output)

        # decode the output
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.n_token)

        # apply the log softmax to the output and return it
        return F.log_softmax(decoded, dim=1), hidden_states
