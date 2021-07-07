import torch
import torch.nn as nn
import math


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # forget layer
        self.W_f = nn.Parameter(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(self.hidden_size))

        # input layer
        self.W_i = nn.Parameter(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(self.hidden_size))
        self.W_Ctilde = nn.Parameter(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        self.b_Ctilde = nn.Parameter(torch.Tensor(self.hidden_size))

        # output layer
        self.W_o = nn.Parameter(torch.Tensor(self.input_size + self.hidden_size, self.hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(self.hidden_size))

        self.init_weights()

    def forward(self, x, hidden_states):
        # list with the outputs
        h = []

        # initialize the hidden_states if a empty array is received
        if hidden_states is None:
            h_t1 = torch.zeros(x.size()[1], self.hidden_size, device='cuda')
            C_t1 = torch.zeros(x.size()[1], self.hidden_size, device='cuda')
        else:
            h_t1 = hidden_states[0]
            C_t1 = hidden_states[1]

        for t in range(x.size()[0]):
            # the input
            x_t = x[t, :, :]

            # concatenate x_t with h_{t-1}
            x_t_h_t1 = torch.cat((h_t1, x_t), dim=1)

            # forget layer
            f_t = torch.sigmoid(torch.matmul(x_t_h_t1, self.W_f) + self.b_f)

            # input layer
            i_t = torch.sigmoid(torch.matmul(x_t_h_t1, self.W_i) + self.b_i)
            Ctilde_t = torch.tanh(torch.matmul(x_t_h_t1, self.W_Ctilde) + self.b_Ctilde)

            # update cell state
            C_t = f_t * C_t1 + i_t * Ctilde_t

            # output layer
            o_t = torch.sigmoid(torch.matmul(x_t_h_t1, self.W_o) + self.b_o)
            h_t = o_t * torch.tanh(C_t)

            # append the current output to the list of outputs
            h.append(h_t.unsqueeze(0))

            # the actual cell state should go in the previous cell state for the next iteration
            h_t1 = h_t
            C_t1 = C_t

        return torch.cat(h), [h_t, C_t]

    def init_weights(self):
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -0.05, 0.05)
