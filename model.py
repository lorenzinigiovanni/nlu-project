
import torch.nn as nn
import torch.nn.functional as F
from LSTM import LSTM


class Model(nn.Module):
    def __init__(
        self,
        n_token,
        input_size,
        hidden_size,
        num_layers,
        dropout=0.5,
    ):
        super(Model, self).__init__()

        self.nhid = hidden_size
        self.nlayers = num_layers
        self.ntoken = n_token

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(n_token, input_size)

        self.rnn = nn.ModuleList()
        for _ in range(self.nlayers):
            self.rnn.append(LSTM(input_size, hidden_size))

        self.decoder = nn.Linear(hidden_size, n_token)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, input, hidden):
        output = self.drop(self.encoder(input))

        for i in range(self.nlayers):
            output, hidden = self.rnn[i](
                output,
                hidden
            )
            output = self.drop(output[i, :, :, :])

        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)

        return F.log_softmax(decoded, dim=0), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid)
        )
